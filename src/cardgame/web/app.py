"""A basic Flask + Flask-SocketIO front end for playing Cross Kings online
in a 'room', with game state synced live between players over WebSockets.

Two players can play a game together by both visiting

    https://<site>/room/<4 letter room code>

Visiting a room auto-claims a vacant seat for you (see the "join" handler
below); once both seats are taken, later visitors become spectators.
Spectators can still claim either vacant seat, up until the game starts;
seated players can give their seat back up right up until the first move.
Once the game is under way, seats are locked in for the rest of that game.

Every visitor can pick a display name (via the settings button) at any
point; unnamed seated players just show as "Player 1"/"Player 2".

There's no server-side login or session here: each browser generates its
own player id (a UUID) on first visit and keeps it - along with the chosen
display name - in localStorage. Both are sent up with every relevant
Socket.IO event (see `withIdentity()` in base.html.jinja2), which is what
this server uses to recognise "the same visitor" across page loads and
reconnects. This means the initial (non-JS) page render can't know who's
asking - routes render a generic shell, and the real, personalised board
is filled in moments later once the page's own JS reads its stored identity
and asks for it over the socket.

A `/rooms` page lists all rooms currently in memory, live-updated the same
way, for matchmaking or spectating.

`/play` is the same underlying room model, just keyed by the visiting
player's own id instead of a shared code, with the computer opponent
(see `cardgame.ai`) automatically seated in the other seat, and rematches
auto-accepted on its behalf - see `rooms.RoomState.computer_seat`. There's
deliberately no separate "solo game" concept here: a solo game against the
computer *is* a room, just one where the second seat happens to be
computer-played instead of human-played.

Both finished rooms and finished solo games can also be reviewed after the
fact from a frozen snapshot (see the "Recently finished" sections on
/rooms), independent of whatever the room/player has done since.

This module wires up the actual HTTP routes and Socket.IO event handlers.
The supporting state/storage/rendering logic behind them is split out by
concern into sibling modules, kept in memory and per-process (this is
intentionally simple - fine for a single-process demo/dev server, but state
is lost on restart and won't be shared across multiple worker processes;
for production you'd want to move it into a shared store, e.g. Redis, which
flask-socketio can also use as a "message queue" to fan out events across
multiple workers):

- `identity.py` - validating the client-supplied player id/name described above.
- `rooms.py` - room state (`RoomState`, keyed by room code or player id).
- `extensions.py` - the shared `socketio` instance `rooms.py` emits through.
"""

import importlib.metadata
import os
import random
import secrets

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_socketio import join_room as sio_join_room
from flask_socketio import leave_room as sio_leave_room
from jinja2 import ChoiceLoader, PackageLoader

from ..game import Game
from .extensions import app_holder, socketio
from .identity import _MAX_NAME_LENGTH, _normalize_player_id
from .rooms import (
    _broadcast_state,
    _ensure_analysis_worker,
    _ensure_live_eval,
    _finished_rooms_lock,
    _find_finished_room_entry,
    _get_or_create_room,
    _lobby_active_summaries,
    _lobby_finished_summaries,
    _maybe_play_computer_move,
    _normalize_code,
    _random_unused_code,
    _record_room_finished_locked,
    _render_room_review_state,
    _render_state,
    _resolve_or_create_room_for_join,
    _resolve_room_key,
    _rooms,
    _sid_index,
    _sid_index_lock,
)

# Socket.IO broadcast group used for the /rooms lobby listing. This is a
# genuine flask-socketio "room" (its group-of-connections concept, distinct
# from our own 4-letter game room codes) since every viewer of the lobby
# sees identical content, unlike a game room's live state, which is
# personalised per player/spectator and therefore sent individually instead
# of through a broadcast group. See rooms._broadcast_state() for that case.
_LOBBY_GROUP = "lobby"


def _render_lobby():
    return render_template(
        "_lobby_state.html.jinja2",
        rooms=_lobby_active_summaries(),
        finished_games=_lobby_finished_summaries(),
    )


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
    def inject_app_version():
        return {"app_version": importlib.metadata.version("cardgame")}

    @app.get("/")
    def index():
        return render_template("index.html.jinja2")

    @app.get("/rooms")
    def rooms_view():
        return render_template(
            "rooms.html.jinja2",
            rooms=_lobby_active_summaries(),
            finished_games=_lobby_finished_summaries(),
        )

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

    @app.get("/play")
    def play():
        # No known player id yet at this point - see module docstring. The
        # page's own JS reads/creates its client-side identity and asks for
        # its actual game over the socket (see "join" below) moments after
        # this loads. ?show_live_eval=true additionally turns on the
        # anytime search (cardgame.search_alt), continuously evaluating the
        # current position and streaming a live move ranking to the page -
        # the mode flag travels with the page's "join" (see handler below)
        # since the room itself is only resolved/created there.
        show_live_eval = request.args.get("show_live_eval", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        return render_template("play.html.jinja2", live_eval=show_live_eval)

    @app.get("/play/<player_id>")
    def spectate_solo(player_id):
        target_id = _normalize_player_id(player_id)
        if target_id is None:
            abort(404, description="Invalid player id.")
        return render_template("play.html.jinja2", spectate_player_id=target_id)

    @app.get("/review/<entry_id>")
    def review(entry_id):
        # Checked eagerly here (unlike spectate_solo, which only validates
        # the id's *shape*) since finished entries are content-addressed
        # and never change, so a 404 now is a reliable, permanent answer.
        # The page's own JS still re-asks over the socket - see "join"
        # below - using the exact same live/spectate flow, just pointed at
        # this frozen entry instead of a live game. Solo and room games are
        # otherwise different templates (play.html.jinja2 needs no "code";
        # room.html.jinja2 shows one), so dispatch on the frozen entry's own
        # "is_solo" flag rather than needing two separate routes for it.
        entry = _find_finished_room_entry(entry_id)
        if entry is None:
            abort(404, description="That finished game could no longer be found.")
        if entry["is_solo"]:
            return render_template("play.html.jinja2", review_entry_id=entry_id)
        return render_template("room.html.jinja2", code=entry["code"], review_entry_id=entry_id)

    @app.get("/room/<code>")
    def room_view(code):
        normalized = _normalize_code(code)
        if normalized is None:
            abort(404, description="Room codes must be 4 letters, e.g. /room/ABCD")
        _get_or_create_room(normalized)
        return render_template("room.html.jinja2", code=normalized)

    return app


app = create_app()
socketio.init_app(app, async_mode="threading")
# Background threads in rooms.py render templates; they need the app to
# enter an app context (see extensions.app_holder).
app_holder["app"] = app


@socketio.on("join")
def handle_join(data):
    data = data or {}
    player_id = _normalize_player_id(data.get("player_id"))
    if player_id is None:
        socketio.emit("error_message", {"message": "Missing player id."}, to=request.sid)
        return

    # review_entry_id names a *frozen, finished* room to read-only review -
    # a completely separate, stateless path from live rooms below, since
    # there's no live RoomState to join once the result's been recorded
    # (the room itself may have long since gone into a rematch).
    review_entry_id = (data.get("review_entry_id") or "").strip()
    if review_entry_id:
        entry = _find_finished_room_entry(review_entry_id)
        if entry is None:
            socketio.emit(
                "error_message",
                {"message": "That finished game could no longer be found."},
                to=request.sid,
            )
            return
        socketio.emit(
            "state", {"html": _render_room_review_state(entry, player_id)}, to=request.sid
        )
        return

    name = (data.get("name") or "").strip()[:_MAX_NAME_LENGTH]
    code, room, error = _resolve_or_create_room_for_join(data.get("code", ""), player_id)
    if room is None:
        socketio.emit("error_message", {"message": error}, to=request.sid)
        return

    sid = request.sid
    with room.lock:
        room.sid_players[sid] = player_id
        # Live-eval mode is an owner's choice of entry point
        # (/play?show_live_eval=true vs /play) for their own solo room;
        # other pages send no flag at all
        # and leave the mode as it is.
        if room.computer_seat is not None and code == player_id and "live_eval" in data:
            room.live_eval = bool(data.get("live_eval"))
        # Pick up this player's client-stored name (if any), in case they
        # set it while visiting a different room.
        if name:
            room.player_names[player_id] = name

        # Auto-claim a vacant seat
        #
        # A *multiplayer* seat needs a name first
        # needs_name below tells the client to prompt for one and retry,
        # same as the manual "claim seat" button already does. A *solo*
        # room's seat is exempt - the computer opponent doesn't care who
        # you are, so you can still play anonymously (name_for_seat already
        # falls back to "Player 1" either way).
        vacant_seat = next(
            (seat for seat in (1, 2) if not room.occupied(seat)), None
        )
        eligible = (
            vacant_seat is not None
            and room.seat_of(player_id) is None
            and not (room.game_started and not room.game_over)
        )
        can_auto_seat = name or room.computer_seat is not None
        needs_name = eligible and not can_auto_seat
        if eligible and can_auto_seat:
            room.seats[vacant_seat] = player_id
    with _sid_index_lock:
        _sid_index[sid] = (code, player_id)

    if needs_name:
        # Don't show them a board at all while we're withholding their
        # seat pending a name - the client's own "Loading..." placeholder
        # stays put behind the name prompt. Once they provide one and the
        # retry lands here again, can_auto_seat is true and this branch
        # isn't hit, so the *first* board they ever see is the real,
        # already-seated one - and everyone else already in the room (not
        # just this sid) gets refreshed too, below.
        socketio.emit("need_name", {}, to=sid)
    else:
        _broadcast_state(code, room)
    _ensure_live_eval(code, room)
    _ensure_analysis_worker(code, room)
    _broadcast_lobby()


@socketio.on("set_name")
def handle_set_name(data):
    data = data or {}
    name = (data.get("name") or "").strip()[:_MAX_NAME_LENGTH]
    if not name:
        socketio.emit("error_message", {"message": "Please enter a name."}, to=request.sid)
        return
    player_id = _normalize_player_id(data.get("player_id"))
    if player_id is None:
        socketio.emit("error_message", {"message": "Missing player id."}, to=request.sid)
        return

    sid = request.sid
    with _sid_index_lock:
        info = _sid_index.get(sid)
    if info is not None:
        code, _player_id = info
        room = _rooms.get(code)
        if room is not None:
            with room.lock:
                room.player_names[player_id] = name
                is_seated = room.seat_of(player_id) is not None
            # Only worth telling everyone else if this name is actually
            # visible to them (i.e. this player is seated) - broadcasting
            # for an unseated visitor would render them a premature
            # spectator view of the board a moment before the pending
            # "join" retry (see handle_join) seats them and renders it for
            # real, defeating the point of withholding it until then.
            if is_seated:
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

    name = (data.get("name") or "").strip()
    if not name:
        socketio.emit("need_name", {}, to=request.sid)
        return

    player_id = _normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
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
    data = data or {}
    code = _normalize_code(data.get("code", ""))
    if code is None:
        return
    room = _rooms.get(code)
    if room is None:
        return

    player_id = _normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    ok, error = room.vacate_seat(player_id)
    if not ok:
        socketio.emit("error_message", {"message": error}, to=request.sid)
        return

    _broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("move")
def handle_move(data):
    data = data or {}
    player_id = _normalize_player_id(data.get("player_id"))
    if player_id is None:
        socketio.emit("error_message", {"message": "Missing player id."}, to=request.sid)
        return
    code = _resolve_room_key(data.get("code", ""), player_id)
    room = _rooms.get(code)
    if room is None:
        socketio.emit("error_message", {"message": "That room doesn't exist."}, to=request.sid)
        return
    my_seat = room.seat_of(player_id)

    with room.lock:
        game = room.game
        if not room.both_seated:
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
        if room.game_over:
            _record_room_finished_locked(code, room)

    _broadcast_state(code, room)
    _broadcast_lobby()
    _maybe_play_computer_move(code, room)
    _ensure_analysis_worker(code, room)
    _broadcast_lobby()


@socketio.on("history_step")
def handle_history_step(data):
    data = data or {}
    player_id = _normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    try:
        delta = int(data.get("delta", 0))
    except (TypeError, ValueError):
        return

    review_entry_id = (data.get("review_entry_id") or "").strip()
    if review_entry_id:
        entry = _find_finished_room_entry(review_entry_id)
        if entry is None:
            socketio.emit(
                "error_message",
                {"message": "That finished game could no longer be found."},
                to=request.sid,
            )
            return
        with _finished_rooms_lock:
            total = len(entry["game"].moves)
            current = entry["history_index"].get(player_id, total)
            entry["history_index"][player_id] = max(0, min(total, current + delta))
        socketio.emit(
            "state", {"html": _render_room_review_state(entry, player_id)}, to=request.sid
        )
        return

    code = _resolve_room_key(data.get("code", ""), player_id)
    room = _rooms.get(code)
    if room is None or not room.game_over:
        return

    with room.lock:
        total = len(room.game.moves)
        current = room.history_index.get(player_id, total)
        room.history_index[player_id] = max(0, min(total, current + delta))

    socketio.emit("state", {"html": _render_state(code, room, player_id)}, to=request.sid)


@socketio.on("history_goto")
def handle_history_goto(data):
    data = data or {}
    player_id = _normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        return

    review_entry_id = (data.get("review_entry_id") or "").strip()
    if review_entry_id:
        entry = _find_finished_room_entry(review_entry_id)
        if entry is None:
            socketio.emit(
                "error_message",
                {"message": "That finished game could no longer be found."},
                to=request.sid,
            )
            return
        with _finished_rooms_lock:
            total = len(entry["game"].moves)
            entry["history_index"][player_id] = max(0, min(total, index))
        socketio.emit(
            "state", {"html": _render_room_review_state(entry, player_id)}, to=request.sid
        )
        return

    code = _resolve_room_key(data.get("code", ""), player_id)
    room = _rooms.get(code)
    if room is None or not room.game_over:
        return

    with room.lock:
        total = len(room.game.moves)
        room.history_index[player_id] = max(0, min(total, index))

    socketio.emit("state", {"html": _render_state(code, room, player_id)}, to=request.sid)


@socketio.on("request_rematch")
def handle_request_rematch(data):
    data = data or {}
    player_id = _normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    code = _resolve_room_key(data.get("code", ""), player_id)
    room = _rooms.get(code)
    if room is None:
        return

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
        if room.computer_seat is not None:
            # The computer always accepts immediately - there's no one to
            # ask, and no point waiting.
            room.game = Game.deal()
            room.game_id = secrets.token_hex(8)
            room.rematch_requested_by = None
            room.history_index.clear()
            room.game_over_seen.clear()
            room.analysis = None
            room.analysis_inflight = set()
        elif room.rematch_requested_by is not None and room.rematch_requested_by != player_id:
            # The other player already asked - treat this as accepting.
            room.game = Game.deal()
            room.game_id = secrets.token_hex(8)
            room.rematch_requested_by = None
            room.history_index.clear()
            room.game_over_seen.clear()
            room.analysis = None
            room.analysis_inflight = set()
        else:
            room.rematch_requested_by = player_id

    _broadcast_state(code, room)
    _broadcast_lobby()


@socketio.on("respond_rematch")
def handle_respond_rematch(data):
    data = data or {}
    player_id = _normalize_player_id(data.get("player_id"))
    if player_id is None:
        return
    code = _resolve_room_key(data.get("code", ""), player_id)
    room = _rooms.get(code)
    if room is None:
        return

    my_seat = room.seat_of(player_id)
    accept = bool(data.get("accept"))

    with room.lock:
        if my_seat is None:
            return
        if room.rematch_requested_by is None or room.rematch_requested_by == player_id:
            return  # nothing to respond to, or you're the one who asked
        if accept:
            room.game = Game.deal()
            room.game_id = secrets.token_hex(8)
            room.history_index.clear()
            room.game_over_seen.clear()
            room.analysis = None
            room.analysis_inflight = set()
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
