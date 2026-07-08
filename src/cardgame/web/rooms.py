"""Room state and its supporting logic - covers both multiplayer rooms and
solo (vs. computer) games, which are the same thing under the hood: a
`RoomState` with two seats, one game, and a shared 4-letter-code-or-player-id
key namespace in `_rooms`.

A multiplayer room is keyed by a 4-letter room code and has two human seats.
A solo game is keyed by the owning player's own id and has the computer
sitting in the second seat (`RoomState.computer_seat`) instead of a second
human - tracked as a plain seat flag, never as an entry in `seats` or any
other player_id-keyed structure, so there's no identity a client could ever
send to impersonate "the computer".

Game state here is in-memory and per-process - see app.py's module
docstring for the tradeoffs.

app.py owns the actual HTTP routes and `@socketio.on(...)` handlers; this
module is the supporting model + storage + rendering logic those handlers
call into, kept separate so app.py can stay focused on request/event
wiring.
"""

import random
import re
import secrets
import string
import threading
from dataclasses import dataclass, field

from flask import render_template

from ..ai import choose_move
from ..game import Game
from .extensions import socketio
from .identity import _normalize_player_id

_CODE_RE = re.compile(r"^[A-Za-z]{4}$")
_COMPUTER_NAME = "Computer"

# How many finished games (rooms and solo games together) the /rooms
# "Recently finished games" table keeps around - see
# _record_room_finished_locked.
_MAX_FINISHED_ENTRIES = 20


def _normalize_code(raw_code):
    """Return an upper-cased 4 letter room code, or None if invalid."""
    if not raw_code or not _CODE_RE.match(raw_code):
        return None
    return raw_code.upper()


@dataclass
class RoomState:
    game: Game
    # Which player id currently occupies each human seat: {1: player_id, 2: player_id}.
    # A seat with no entry is vacant, unless it's the computer's (see
    # computer_seat) - claiming/leaving are still explicit actions
    # (claim_seat/vacate_seat) for a spectator reacting to a seat opening
    # up after they've already connected, but the common case is a visitor
    # auto-claiming a vacant seat the moment they join (see app.py's
    # "join" handler).
    seats: dict = field(default_factory=dict)
    # Which seat (if any) the computer occupies, for a solo-vs-computer
    # room. None for an ordinary multiplayer room. Deliberately not stored
    # in `seats` - the computer never has a player id, so nothing here is
    # ever confused with (or spoofable as) a real visitor's identity.
    computer_seat: int | None = None
    # Display name for each player id who has set one while in this room
    # (player_id -> name). The name itself lives in the player's own
    # localStorage too, so it carries across rooms; this is just this
    # room's copy of it for rendering to *other* viewers.
    player_names: dict = field(default_factory=dict)
    # Connected Socket.IO session ids currently watching this room, mapped
    # to the player id using that connection, so a change can be pushed out
    # to everyone watching.
    sid_players: dict = field(default_factory=dict)
    # player_id of whoever has asked for a rematch, if a request is
    # currently pending a response from the other seated player. None if
    # there's no pending request, or if the other seat is computer-played
    # (the computer always accepts immediately - see handle_request_rematch
    # - so this never actually gets set in a solo room). Cleared on accept,
    # decline, or a fresh deal.
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

    def occupied(self, seat):
        return seat in self.seats or self.computer_seat == seat

    @property
    def both_seated(self):
        return self.occupied(1) and self.occupied(2)

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

    def name_for_seat(self, seat):
        if self.computer_seat == seat:
            return _COMPUTER_NAME
        occupant = self.seats.get(seat)
        if occupant is None:
            return None
        return self.player_names.get(occupant, f"Player {seat}")

    def claim_seat(self, seat, player_id):
        """Try to claim a vacant seat. Returns (success, error_message)."""
        with self.lock:
            if seat not in (1, 2):
                return False, "Invalid seat."
            if self.computer_seat == seat:
                return False, "That seat is taken by the computer."
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
            if self.computer_seat is not None:
                return False, "You can't leave a solo game against the computer."
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

# Maps a connected Socket.IO session id to the (room key, player_id) it
# belongs to, so the disconnect handler knows what to clean up. One shared
# index for both multiplayer rooms and solo games - review viewers (see
# _find_finished_room_entry) are never added here, since a frozen entry
# never changes and so never needs to push anything out to them.
_sid_index: dict[str, tuple[str, str]] = {}
_sid_index_lock = threading.Lock()

# Frozen summary snapshots of rooms as they finished, oldest first. Each
# entry also keeps the actual (immutable) Game object as it stood at that
# moment, a per-viewer history-scrubbing position, a unique id, and whether
# it was a solo (vs. computer) game - so it can be reviewed later even
# after the room has gone into a rematch, and so the /rooms lobby's
# "Recently finished games" table can link each entry to the right route
# ("review_room" vs "review_solo") - see _room_review_context, which reads
# from this frozen entry rather than the live RoomState.
_finished_rooms: list[dict] = []
_finished_rooms_lock = threading.Lock()


def _record_room_finished_locked(code, room):
    """Freeze this room's just-finished result into history. Caller must
    already hold room.lock, so the snapshot (game object included)
    reflects exactly the move that just finished it - not a later one.
    """
    entry = _room_summary_locked(code, room)
    entry["id"] = secrets.token_hex(8)
    entry["game"] = room.game
    entry["history_index"] = {}
    with _finished_rooms_lock:
        _finished_rooms.append(entry)
        while len(_finished_rooms) > _MAX_FINISHED_ENTRIES:
            _finished_rooms.pop(0)


def _find_finished_room_entry(entry_id):
    with _finished_rooms_lock:
        for entry in _finished_rooms:
            if entry["id"] == entry_id:
                return entry
    return None


def _get_or_create_room(code, solo=False):
    """Look up a room by its key, creating it if necessary. `solo=True`
    only takes effect on creation - it seats the computer in seat 2, for a
    brand new solo-vs-computer room. Looking up an existing room ignores
    it (a room's type is fixed at creation).
    """
    with _rooms_lock:
        room = _rooms.get(code)
        if room is None:
            room = RoomState(game=Game.deal())
            if solo:
                room.computer_seat = 2
            _rooms[code] = room
        return room


def _random_unused_code():
    while True:
        code = "".join(random.choices(string.ascii_uppercase, k=4))
        with _rooms_lock:
            if code not in _rooms:
                return code


def _resolve_room_key(raw_code, player_id):
    """Normalize a client-supplied room key for a lookup-only event (move,
    history step/goto, rematch) - it's either a 4-letter multiplayer room
    code or another (or your own) player's id naming their solo-vs-computer
    room. Falls back to your own id if nothing usable was given.
    """
    return _normalize_code(raw_code) or _normalize_player_id(raw_code) or player_id


def _resolve_or_create_room_for_join(raw_code, player_id):
    """Like _resolve_room_key, but may create the room:
    - a valid 4-letter code always gets-or-creates a multiplayer room.
    - your own id (given explicitly, or implied by giving nothing at all)
      gets-or-creates your solo room, with the computer seated, if missing.
    - anyone else's id is looked up only - it must already exist.
    Returns (code, room, error_message) - room and error_message are
    mutually exclusive.
    """
    code = _normalize_code(raw_code)
    if code is not None:
        return code, _get_or_create_room(code), None
    target_id = _normalize_player_id(raw_code) or player_id
    if target_id == player_id:
        return target_id, _get_or_create_room(target_id, solo=True), None
    room = _rooms.get(target_id)
    if room is None:
        return target_id, None, "That solo game doesn't exist any more."
    return target_id, room, None


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

    seats_locked = game_started and not game_over

    show_game_over_modal = False
    if game_over and player_id not in room.game_over_seen:
        show_game_over_modal = True
        room.game_over_seen.add(player_id)

    rematch_requested_by_name = None
    rematch_requested_by_me = False
    if room.rematch_requested_by is not None:
        rematch_requested_by_me = room.rematch_requested_by == player_id
        rematch_requested_by_name = room.name_for_seat(room.seat_of(room.rematch_requested_by))

    return {
        "code": code,
        "vs_computer": room.computer_seat is not None,
        "game": display_game,
        "my_seat": my_seat,
        "my_name": room.player_names.get(player_id),
        "is_spectator": my_seat is None,
        "both_seated": room.both_seated,
        "your_turn": my_seat is not None and my_seat == turn_seat and room.both_seated,
        "legal_moves": set() if (game_over or viewing_history) else display_game.legal_moves,
        "game_over": game_over,
        "game_started": game_started,
        "seats_locked": seats_locked,
        "winner": winner,
        "p1_name": room.name_for_seat(1),
        "p2_name": room.name_for_seat(2),
        "p1_seated": room.occupied(1),
        "p2_seated": room.occupied(2),
        "p1_score": final_game.p1.score(),
        "p2_score": final_game.p2.score(),
        "seats_taken": len(room.seats) + (1 if room.computer_seat else 0),
        "taken_card": display_game.taken_card if display_game.moves else None,
        "history_index": history_index,
        "history_total": history_total,
        "viewing_history": viewing_history,
        "show_game_over_modal": show_game_over_modal,
        "rematch_pending": room.rematch_requested_by is not None,
        "rematch_requested_by_me": rematch_requested_by_me,
        "rematch_requested_by_name": rematch_requested_by_name,
    }


def _room_review_context(entry, viewer_id):
    """Build the template context for one viewer reviewing a frozen,
    finished room (a _finished_rooms entry) - as opposed to _room_context,
    which reads the live, possibly-since-rematched RoomState. The Game
    object itself never changes here; only which move *this viewer* is
    currently scrubbed to does, tracked the same way as everywhere else -
    a per-viewer position in entry["history_index"] - so review reuses the
    exact same join/history_step/history_goto socket flow as live play and
    spectating, just pointed at a frozen entry instead of a live room.
    """
    game = entry["game"]
    total = len(game.moves)
    with _finished_rooms_lock:
        history_index = entry["history_index"].get(viewer_id, total)
    history_index = max(0, min(total, history_index))
    display_game = game if history_index == total else game.undo(total - history_index)

    p1_score, p2_score = entry["p1_score"], entry["p2_score"]
    if p1_score > p2_score:
        winner = 1
    elif p2_score > p1_score:
        winner = 2
    else:
        winner = 0  # draw

    return {
        "code": entry["code"],
        "vs_computer": entry["is_solo"],
        "review": True,
        "game": display_game,
        "my_seat": None,
        "my_name": None,
        "is_spectator": True,
        "both_seated": True,
        "your_turn": False,
        "legal_moves": set(),
        "game_over": True,
        "game_started": True,
        "seats_locked": True,
        "winner": winner,
        "p1_name": entry["p1_name"],
        "p2_name": entry["p2_name"],
        "p1_seated": True,
        "p2_seated": True,
        "p1_score": p1_score,
        "p2_score": p2_score,
        "seats_taken": 2,
        "taken_card": display_game.taken_card if display_game.moves else None,
        "history_index": history_index,
        "history_total": total,
        "viewing_history": history_index != total,
        "show_game_over_modal": False,
        "rematch_pending": False,
        "rematch_requested_by_me": False,
        "rematch_requested_by_name": None,
        "entry_id": entry["id"],
    }


def _render_room_review_state(entry, viewer_id):
    context = _room_review_context(entry, viewer_id)
    return render_template("_game_state.html.jinja2", **context)


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


def _maybe_play_computer_move(code, room):
    """If it's now the computer's turn (only ever true for a solo-vs-
    computer room), play its move and broadcast the result. Loops in case
    that leaves it the computer's turn again, though a single move always
    hands the turn back in practice. A harmless no-op for any room without
    a computer seat, so callers can invoke this unconditionally after every
    move.
    """
    while True:
        with room.lock:
            if room.computer_seat is None or room.game_over:
                return
            if room.current_turn_seat != room.computer_seat:
                return
            row, col = choose_move(room.game)
            outcomes = room.game.move(row, col)
            room.game = random.choice(outcomes)
            if room.game_over:
                _record_room_finished_locked(code, room)
        _broadcast_state(code, room)


def _room_summary_locked(code, room):
    """Like _room_summary, but assumes the caller already holds room.lock
    (needed so a finished-game snapshot can be taken atomically with the
    move that just finished it - see _record_room_finished_locked).
    """
    game = room.game
    game_over = not game.legal_moves
    game_started = room.game_started
    seated_ids = set(room.seats.values())
    spectators = sum(1 for pid in room.sid_players.values() if pid not in seated_ids)
    connected = len(room.sid_players) > 0
    if game_over:
        status = "finished"
    elif game_started:
        status = "in progress"
    else:
        status = "waiting for players"
    return {
        "code": code,
        "is_solo": room.computer_seat is not None,
        "has_moves": game_started,
        "p1_name": room.name_for_seat(1),
        "p2_name": room.name_for_seat(2),
        "status": status,
        "spectators": spectators,
        "connected": connected,
        "p1_score": game.p1.score() if (game_started or game_over) else None,
        "p2_score": game.p2.score() if (game_started or game_over) else None,
    }


def _room_summary(code, room):
    with room.lock:
        return _room_summary_locked(code, room)


def _lobby_active_summaries():
    """Active games - multiplayer rooms and solo-vs-computer games alike -
    for the /rooms list's "Ongoing games" table. A room/solo game are the
    same underlying thing (see module docstring), so one list covers both;
    the template distinguishes them via each summary's "is_solo" flag
    (e.g. to link to room_view vs spectate_solo). Solo games with no moves
    yet are left off - there's nothing meaningful to watch, and every
    /play visit otherwise creates one of these; a multiplayer room with no
    moves yet still shows (as "waiting for players"), since that's exactly
    what a second player needs to see to find it.
    """
    with _rooms_lock:
        codes = list(_rooms.keys())
    summaries = []
    for code in codes:
        room = _rooms.get(code)
        if room is None:
            continue
        summary = _room_summary(code, room)
        if summary["status"] == "finished":
            continue
        if summary["is_solo"] and not summary["has_moves"]:
            continue
        summaries.append(summary)
    status_order = {"waiting for players": 0, "in progress": 1}
    summaries.sort(key=lambda r: (status_order[r["status"]], r["code"]))
    return summaries


def _lobby_finished_summaries():
    """The most recently finished games - rooms and solo games alike, most-
    recent first - frozen snapshots taken as each one finished (see
    _record_room_finished_locked), so a later rematch/new game doesn't
    change or remove its entry here.
    """
    with _finished_rooms_lock:
        return list(reversed(_finished_rooms))
