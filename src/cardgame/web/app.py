"""A basic Flask + Flask-SocketIO front end for playing Cross Kings online
in a 'room', with game state synced live between players over WebSockets.

Two players can play a game together by both visiting

    https://<site>/room/<4 letter room code>

Visiting a room makes you a spectator by default. Spectators can claim
either vacant seat (Player 1 / Player 2) before the game starts, and seated
players can give their seat back up right up until the first move is made.
Once the game is under way, seats are locked in for the rest of that game.

Every visitor picks a display name (stored in their session, so it carries
across rooms) the first time they try to claim a seat. That name is what
shows up in place of "Player 1" / "Player 2" for everyone else watching.

A `/rooms` page lists all rooms currently in memory, live-updated the same
way, for matchmaking or spectating.

Game state is kept in memory on the server, keyed by room code. This is
intentionally simple: it is fine for running a single-process demo/dev
server, but it means state is lost on restart and won't be shared across
multiple worker processes. For a production deployment you would want to
move `_rooms` out into a shared store (e.g. Redis, which flask-socketio can
also use as a "message queue" to fan out events across multiple workers).
"""

import os
import random
import re
import secrets
import string
import threading
import uuid
from dataclasses import dataclass, field

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_socketio import SocketIO
from flask_socketio import join_room as sio_join_room
from flask_socketio import leave_room as sio_leave_room
from jinja2 import ChoiceLoader, PackageLoader

from ..game import Game

_CODE_RE = re.compile(r"^[A-Za-z]{4}$")
_MAX_NAME_LENGTH = 24

# Socket.IO broadcast group used for the /rooms lobby listing. This is a
# genuine flask-socketio "room" (its group-of-connections concept, distinct
# from our own 4-letter game room codes) since every viewer of the lobby
# sees identical content, unlike a game room's live state, which is
# personalised per player/spectator and therefore sent individually instead
# of through a broadcast group. See _broadcast_state() below.
_LOBBY_GROUP = "lobby"


def _normalize_code(raw_code):
    """Return an upper-cased 4 letter room code, or None if invalid."""
    if not raw_code or not _CODE_RE.match(raw_code):
        return None
    return raw_code.upper()


@dataclass
class RoomState:
    game: Game
    # Which player id currently occupies each seat: {1: player_id, 2: player_id}.
    # A seat with no entry is vacant. Unlike the old "first two visitors are
    # auto-assigned" model, claiming and leaving a seat are now explicit
    # actions (see claim_seat/vacate_seat), so this only changes when a
    # player asks it to.
    seats: dict = field(default_factory=dict)
    # Display name for each player id who has set one while in this room
    # (player_id -> name). The name itself lives in that player's session
    # too, so it carries across rooms; this is just this room's copy of it
    # for rendering to *other* viewers.
    player_names: dict = field(default_factory=dict)
    # Connected Socket.IO session ids currently watching this room, mapped
    # to the player id using that connection, so a change can be pushed out
    # to everyone watching.
    sid_players: dict = field(default_factory=dict)
    # player_id of whoever has asked for a rematch, if a request is
    # currently pending a response from the other seated player. None if
    # there's no pending request. Cleared on accept, decline, or a fresh deal.
    rematch_requested_by: str | None = None
    # Per-viewer position while stepping back and forth through a finished
    # game's move history (player_id -> move index, 0..len(moves)). Only
    # meaningful once the game is over; absence means "viewing the latest
    # (final) position". Cleared whenever a new game is dealt.
    history_index: dict = field(default_factory=dict)
    # player_ids who have already been shown the "you won/lost" modal for
    # the *current* game, so it only pops up once per game per viewer
    # (rather than re-appearing every time they step through history).
    # Cleared whenever a new game is dealt.
    game_over_seen: set = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def seat_of(self, player_id):
        for seat, occupant in self.seats.items():
            if occupant == player_id:
                return seat
        return None

    @property
    def game_started(self):
        return len(self.game.moves) > 0

    @property
    def game_over(self):
        return not self.game.legal_moves

    @property
    def current_turn_seat(self):
        if self.game.is_p1_turn:
            return 1
        if self.game.is_p2_turn:
            return 2
        return None

    def claim_seat(self, seat, player_id):
        """Try to claim a vacant seat. Returns (success, error_message)."""
        with self.lock:
            if seat not in (1, 2):
                return False, "Invalid seat."
            if self.game_started and not self.game_over:
                return False, "The game has already started."
            if any(occupant == player_id for occupant in self.seats.values()):
                return False, "You already have a seat."
            if seat in self.seats:
                return False, "That seat is already taken."
            self.seats[seat] = player_id
            return True, None

    def vacate_seat(self, player_id):
        """Give up whichever seat player_id holds. Returns (success, error_message)."""
        with self.lock:
            if self.game_started and not self.game_over:
                return False, "You can't leave while a game is in progress."
            for seat, occupant in list(self.seats.items()):
                if occupant == player_id:
                    del self.seats[seat]
                    return True, None
            return False, "You don't have a seat."

    def displayed_game(self, player_id):
        """The Game to actually render for this viewer: the live game while
        it's in progress, or - once it's over - whichever point in its
        history they're currently stepping through (default: the final
        position). Returns (game_to_show, history_index, history_total).
        """
        total = len(self.game.moves)
        if not self.game_over:
            return self.game, total, total
        index = self.history_index.get(player_id, total)
        index = max(0, min(total, index))
        if index == total:
            return self.game, index, total
        return self.game.undo(total - index), index, total


_rooms: dict[str, RoomState] = {}
_rooms_lock = threading.Lock()

# Maps a connected Socket.IO session id to the (code, player_id) it belongs
# to, so the disconnect handler knows what to clean up.
_sid_index: dict[str, tuple[str, str]] = {}
_sid_index_lock = threading.Lock()


def _get_or_create_room(code):
    with _rooms_lock:
        room = _rooms.get(code)
        if room is None:
            room = RoomState(game=Game.deal())
            _rooms[code] = room
        return room


def _random_unused_code():
    while True:
        code = "".join(random.choices(string.ascii_uppercase, k=4))
        with _rooms_lock:
            if code not in _rooms:
                return code


def _ensure_player_id():
    if "player_id" not in session:
        session["player_id"] = uuid.uuid4().hex
    return session["player_id"]


def _room_context(code, room, player_id):
    """Build the (viewer-specific) template context for a room's state."""
    final_game = room.game
    game_over = room.game_over
    game_started = room.game_started
    my_seat = room.seat_of(player_id)

    display_game, history_index, history_total = room.displayed_game(player_id)
    viewing_history = game_over and history_index != history_total

    turn_seat = None if game_over else room.current_turn_seat

    winner = None
    if game_over:
        p1_score, p2_score = final_game.p1.score(), final_game.p2.score()
        if p1_score > p2_score:
            winner = 1
        elif p2_score > p1_score:
            winner = 2
        else:
            winner = 0  # draw

    def name_for_seat(seat):
        occupant = room.seats.get(seat)
        if occupant is None:
            return None
        return room.player_names.get(occupant, f"Player {seat}")

    both_seated = 1 in room.seats and 2 in room.seats
    # Seats are only locked in while a game is actively being played; once
    # it's over (however that happened - finished naturally, or a rematch
    # was declined) seats free up again, same as before the game started.
    seats_locked = game_started and not game_over

    show_game_over_modal = False
    if game_over and player_id not in room.game_over_seen:
        show_game_over_modal = True
        room.game_over_seen.add(player_id)

    rematch_requested_by_name = None
    rematch_requested_by_me = False
    if room.rematch_requested_by is not None:
        rematch_requested_by_me = room.rematch_requested_by == player_id
        rematch_requested_by_name = name_for_seat(room.seat_of(room.rematch_requested_by))

    return {
        "code": code,
        "game": display_game,
        "my_seat": my_seat,
        "my_name": room.player_names.get(player_id),
        "is_spectator": my_seat is None,
        "both_seated": both_seated,
        "your_turn": my_seat is not None and my_seat == turn_seat and both_seated,
        "legal_moves": set() if (game_over or viewing_history) else display_game.legal_moves,
        "game_over": game_over,
        "game_started": game_started,
        "seats_locked": seats_locked,
        "winner": winner,
        "p1_name": name_for_seat(1),
        "p2_name": name_for_seat(2),
        "p1_seated": 1 in room.seats,
        "p2_seated": 2 in room.seats,
        "p1_score": final_game.p1.score(),
        "p2_score": final_game.p2.score(),
        "seats_taken": len(room.seats),
        "taken_card": display_game.taken_card if display_game.moves else None,
        "history_index": history_index,
        "history_total": history_total,
        "viewing_history": viewing_history,
        "show_game_over_modal": show_game_over_modal,
        "rematch_pending": room.rematch_requested_by is not None,
        "rematch_requested_by_me": rematch_requested_by_me,
        "rematch_requested_by_name": rematch_requested_by_name,
    }


def _render_state(code, room, player_id):
    context = _room_context(code, room, player_id)
    return render_template("_game_state.html.jinja2", **context)


def _broadcast_state(code, room):
    """Push a freshly rendered, viewer-specific board to everyone in the room.

    Each viewer's status bar/seat buttons depend on who *they* are (their
    own seat, whether they can join a vacant one, etc.), so unlike the lobby
    listing below this can't be sent as one shared broadcast - it's rendered
    and sent individually per connected socket.
    """
    with room.lock:
        sid_players = dict(room.sid_players)
    for sid, player_id in sid_players.items():
        html = _render_state(code, room, player_id)
        socketio.emit("state", {"html": html}, to=sid)


def _lobby_room_summaries():
    with _rooms_lock:
        codes = list(_rooms.keys())
    summaries = []
    for code in codes:
        room = _rooms.get(code)
        if room is None:
            continue
        with room.lock:
            game = room.game
            game_over = not game.legal_moves
            game_started = room.game_started
            p1_name = room.player_names.get(room.seats.get(1)) if 1 in room.seats else None
            p2_name = room.player_names.get(room.seats.get(2)) if 2 in room.seats else None
            seated_ids = set(room.seats.values())
            spectators = sum(
                1 for pid in room.sid_players.values() if pid not in seated_ids
            )
            connected = len(room.sid_players) > 0
            if game_over:
                status = "finished"
            elif game_started:
                status = "in progress"
            else:
                status = "waiting for players"
            summaries.append(
                {
                    "code": code,
                    "p1_name": p1_name,
                    "p2_name": p2_name,
                    "status": status,
                    "spectators": spectators,
                    "connected": connected,
                    "p1_score": game.p1.score() if (game_started or game_over) else None,
                    "p2_score": game.p2.score() if (game_started or game_over) else None,
                }
            )
    status_order = {"waiting for players": 0, "in progress": 1, "finished": 2}
    summaries.sort(key=lambda r: (status_order[r["status"]], r["code"]))
    return summaries


def _render_lobby():
    return render_template("_lobby_state.html.jinja2", rooms=_lobby_room_summaries())


def _broadcast_lobby():
    socketio.emit("lobby_state", {"html": _render_lobby()}, to=_LOBBY_GROUP)


def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("CARDGAME_SECRET_KEY", secrets.token_hex(16))

    # Reuse the same card/board/hand rendering macros the package already
    # uses for its Jupyter _repr_html_ output (templates/components.html.jinja2
    # at the repo root), instead of maintaining a separate copy here. Flask's
    # own template loader only looks in this package's templates/ folder by
    # default, so we chain in the package-wide one too.
    app.jinja_loader = ChoiceLoader(
        [
            app.jinja_loader,
            PackageLoader(package_name="cardgame", package_path="../../templates"),
        ]
    )

    @app.context_processor
    def inject_current_name():
        # Makes the player's current display name (if any) available to
        # every template without each route needing to pass it explicitly.
        return {"current_name": session.get("player_name")}

    @app.get("/")
    def index():
        return render_template("index.html.jinja2")

    @app.get("/rooms")
    def rooms_view():
        return render_template("rooms.html.jinja2", rooms=_lobby_room_summaries())

    @app.post("/create-room")
    def create_room():
        code = _random_unused_code()
        return redirect(url_for("room_view", code=code))

    @app.post("/join-room")
    def join_room_form():
        code = _normalize_code(request.form.get("code", ""))
        if code is None:
            return redirect(url_for("index"))
        return redirect(url_for("room_view", code=code))

    @app.get("/room/<code>")
    def room_view(code):
        normalized = _normalize_code(code)
        if normalized is None:
            abort(404, description="Room codes must be 4 letters, e.g. /room/ABCD")
        room = _get_or_create_room(normalized)
        player_id = _ensure_player_id()
        context = _room_context(normalized, room, player_id)
        return render_template("room.html.jinja2", **context)

    return app


app = create_app()
socketio = SocketIO(app, async_mode="threading")


@socketio.on("join")
def handle_join(data):
    code = _normalize_code((data or {}).get("code", ""))
    if code is None:
        socketio.emit("error_message", {"message": "Invalid room code."}, to=request.sid)
        return

    room = _get_or_create_room(code)
    player_id = _ensure_player_id()

    sid = request.sid
    with room.lock:
        room.sid_players[sid] = player_id
        # Pick up this player's globally-set name (if any) for this room's
        # display, in case they set it while visiting a different room.
        name = session.get("player_name")
        if name:
            room.player_names[player_id] = name
    with _sid_index_lock:
        _sid_index[sid] = (code, player_id)

    socketio.emit("state", {"html": _render_state(code, room, player_id)}, to=sid)
    _broadcast_lobby()


@socketio.on("set_name")
def handle_set_name(data):
    name = ((data or {}).get("name") or "").strip()[:_MAX_NAME_LENGTH]
    if not name:
        socketio.emit("error_message", {"message": "Please enter a name."}, to=request.sid)
        return

    session["player_name"] = name
    player_id = _ensure_player_id()

    sid = request.sid
    with _sid_index_lock:
        info = _sid_index.get(sid)
    if info is not None:
        code, _player_id = info
        room = _rooms.get(code)
        if room is not None:
            with room.lock:
                room.player_names[player_id] = name
            _broadcast_state(code, room)
            _broadcast_lobby()

    socketio.emit("name_set", {"name": name}, to=sid)


@socketio.on("claim_seat")
def handle_claim_seat(data):
    data = data or {}
    code = _normalize_code(data.get("code", ""))
    if code is None:
        return
    room = _rooms.get(code)
    if room is None:
        return

    if not session.get("player_name"):
        socketio.emit("need_name", {}, to=request.sid)
        return

    player_id = _ensure_player_id()
    try:
        seat = int(data.get("seat"))
    except (TypeError, ValueError):
        return

    ok, error = room.claim_seat(seat, player_id)
    if not ok:
        socketio.emit("error_message", {"message": error}, to=request.sid)
        return

    _broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("vacate_seat")
def handle_vacate_seat(data):
    code = _normalize_code((data or {}).get("code", ""))
    if code is None:
        return
    room = _rooms.get(code)
    if room is None:
        return

    player_id = _ensure_player_id()
    ok, error = room.vacate_seat(player_id)
    if not ok:
        socketio.emit("error_message", {"message": error}, to=request.sid)
        return

    _broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("move")
def handle_move(data):
    data = data or {}
    code = _normalize_code(data.get("code", ""))
    if code is None:
        socketio.emit("error_message", {"message": "Invalid room code."}, to=request.sid)
        return
    room = _rooms.get(code)
    if room is None:
        socketio.emit("error_message", {"message": "That room doesn't exist."}, to=request.sid)
        return

    player_id = _ensure_player_id()
    my_seat = room.seat_of(player_id)

    with room.lock:
        game = room.game
        if not (1 in room.seats and 2 in room.seats):
            socketio.emit(
                "error_message",
                {"message": "Waiting for both players to join before the game can start."},
                to=request.sid,
            )
            return
        if my_seat is None or my_seat != room.current_turn_seat:
            socketio.emit("error_message", {"message": "It isn't your turn."}, to=request.sid)
            return
        try:
            row = int(data["row"])
            col = int(data["col"])
        except (KeyError, TypeError, ValueError):
            socketio.emit("error_message", {"message": "Invalid move."}, to=request.sid)
            return
        if (row, col) not in game.legal_moves:
            socketio.emit("error_message", {"message": "That isn't a legal move."}, to=request.sid)
            return

        outcomes = game.move(row, col)
        # Moving onto a face-down card reveals one of the remaining unseen
        # cards. As in Game.random_move(), any of the remaining face-down
        # cards is an equally likely identity for it, so we pick one
        # uniformly at random to reveal.
        room.game = random.choice(outcomes)

    _broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("history_step")
def handle_history_step(data):
    data = data or {}
    code = _normalize_code(data.get("code", ""))
    if code is None:
        return
    room = _rooms.get(code)
    if room is None:
        return
    if not room.game_over:
        return
    player_id = _ensure_player_id()
    try:
        delta = int(data.get("delta", 0))
    except (TypeError, ValueError):
        return

    with room.lock:
        total = len(room.game.moves)
        current = room.history_index.get(player_id, total)
        room.history_index[player_id] = max(0, min(total, current + delta))

    socketio.emit("state", {"html": _render_state(code, room, player_id)}, to=request.sid)


@socketio.on("history_goto")
def handle_history_goto(data):
    data = data or {}
    code = _normalize_code(data.get("code", ""))
    if code is None:
        return
    room = _rooms.get(code)
    if room is None:
        return
    if not room.game_over:
        return
    player_id = _ensure_player_id()
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        return

    with room.lock:
        total = len(room.game.moves)
        room.history_index[player_id] = max(0, min(total, index))

    socketio.emit("state", {"html": _render_state(code, room, player_id)}, to=request.sid)


@socketio.on("request_rematch")
def handle_request_rematch(data):
    code = _normalize_code((data or {}).get("code", ""))
    if code is None:
        return
    room = _rooms.get(code)
    if room is None:
        return

    player_id = _ensure_player_id()
    my_seat = room.seat_of(player_id)
    if my_seat is None:
        socketio.emit(
            "error_message", {"message": "Only players can request a rematch."}, to=request.sid
        )
        return

    with room.lock:
        if not room.game_over:
            socketio.emit(
                "error_message", {"message": "The game hasn't finished yet."}, to=request.sid
            )
            return
        if room.rematch_requested_by is not None and room.rematch_requested_by != player_id:
            # The other player already asked - treat this as accepting.
            room.game = Game.deal()
            room.rematch_requested_by = None
            room.history_index.clear()
            room.game_over_seen.clear()
        else:
            room.rematch_requested_by = player_id

    _broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("respond_rematch")
def handle_respond_rematch(data):
    data = data or {}
    code = _normalize_code(data.get("code", ""))
    if code is None:
        return
    room = _rooms.get(code)
    if room is None:
        return

    player_id = _ensure_player_id()
    my_seat = room.seat_of(player_id)
    accept = bool(data.get("accept"))

    with room.lock:
        if my_seat is None:
            return
        if room.rematch_requested_by is None or room.rematch_requested_by == player_id:
            return  # nothing to respond to, or you're the one who asked
        if accept:
            room.game = Game.deal()
            room.history_index.clear()
            room.game_over_seen.clear()
        room.rematch_requested_by = None

    _broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("join_lobby")
def handle_join_lobby():
    sio_join_room(_LOBBY_GROUP)
    socketio.emit("lobby_state", {"html": _render_lobby()}, to=request.sid)


@socketio.on("leave_lobby")
def handle_leave_lobby():
    sio_leave_room(_LOBBY_GROUP)


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    with _sid_index_lock:
        info = _sid_index.pop(sid, None)
    if info is not None:
        code, _player_id = info
        room = _rooms.get(code)
        if room is not None:
            with room.lock:
                room.sid_players.pop(sid, None)
            _broadcast_lobby()


def main():
    """Entry point for running a local dev server: `cardgame-web`."""
    host = os.environ.get("CARDGAME_HOST", "127.0.0.1")
    port = int(os.environ.get("CARDGAME_PORT", "5000"))
    debug = os.environ.get("CARDGAME_DEBUG", "").lower() in ("1", "true", "yes")
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
