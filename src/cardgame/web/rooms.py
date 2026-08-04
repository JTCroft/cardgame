"""Room state model, registries and lifecycle helpers for multiplayer rooms
and solo (vs. computer) games - the same `RoomState`, keyed in `rooms` by a
4-letter code or the owning player's id.

A multiplayer room has two human seats; a solo game seats the computer in one
(`RoomState.computer_seat`), never stored in `seats`. State is in-memory and
per-process. app.py owns the routes and socket handlers.
"""

import base64
import random
import re
import secrets
import string
import threading
from dataclasses import dataclass, field

from ..game import Game

__all__ = (
    "RoomState",
    "rooms",
    "rooms_lock",
    "sid_index",
    "sid_index_lock",
    "finished_rooms",
    "finished_rooms_lock",
    "normalize_code",
    "normalize_player_id",
    "MAX_NAME_LENGTH",
    "status_counts",
    "get_or_create_room",
    "deal_fresh_locked",
    "random_unused_code",
    "resolve_room_key",
    "resolve_or_create_room_for_join",
    "find_finished_room_entry",
    "encode_game_state",
    "decode_game_state",
    "create_solo_room_from_state",
)

_CODE_RE = re.compile(r"^[A-Za-z]{4}$")
_COMPUTER_NAME = "Computer"


def normalize_code(raw_code):
    """Return an upper-cased 4 letter room code, or None if invalid."""
    if not raw_code or not _CODE_RE.match(raw_code):
        return None
    return raw_code.upper()


# Client-supplied visitor identity (a UUID from getPlayerId()), doubling as a
# solo room's key. Any short id-shaped token is accepted.
_PLAYER_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
MAX_NAME_LENGTH = 24


def normalize_player_id(raw):
    """Return a validated client-supplied player id, or None if missing/malformed."""
    if not raw or not isinstance(raw, str) or not _PLAYER_ID_RE.match(raw):
        return None
    return raw


@dataclass
class RoomState:
    game: Game
    # Stable id for the current match, fresh per dealt game; also the frozen
    # finished_rooms entry id.
    game_id: str = field(default_factory=lambda: secrets.token_hex(8))
    # Human seat occupants, {1: player_id, 2: player_id}. Missing = vacant.
    seats: dict = field(default_factory=dict)
    # Seat the computer occupies in a solo room, else None. Kept out of `seats`.
    computer_seat: int | None = None
    # This room's copy of each player's display name (player_id -> name).
    player_names: dict = field(default_factory=dict)
    # Connected socket ids watching this room -> the player id on each.
    sid_players: dict = field(default_factory=dict)
    # player_id awaiting the other seat's rematch response, else None.
    rematch_requested_by: str | None = None
    # Per-viewer history position (player_id -> move index); absent = latest.
    history_index: dict = field(default_factory=dict)
    # player_ids already shown the game-over modal.
    game_over_seen: set = field(default_factory=set)
    # Post-game move analyses, {move_index: analyse_moves(...)}. Shared with the
    # frozen finished_rooms entry (same dict). Reset to None on a fresh deal.
    analysis: dict | None = None
    # Move indices currently being analysed.
    analysis_inflight: set = field(default_factory=set)
    # monotonic() start time of each in-flight "Calculate" click.
    analysis_calc_started: dict = field(default_factory=dict)
    # At most one analysis worker runs per room while this is set.
    analysis_running: bool = False
    # monotonic() deadline for the worker's post-game grace period, set at game
    # end. None while a game is in progress.
    analysis_deadline: float | None = None
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
        # Guard on the marker: an unplaced game also has no legal moves.
        return not self.game.needs_marker and not self.game.legal_moves

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

    def _seat2_occupant(self):
        # Who holds seat 2 (the marker placer): a player id, "computer", or None.
        if self.computer_seat == 2:
            return "computer"
        return self.seats.get(2)

    def _reset_marker_if_orphaned(self, previous_seat2):
        """Clear the marker if seat 2's occupant changed since placing it, so
        the new occupant chooses. No-op once play has begun or if unplaced.
        Callers hold the lock and pass the pre-change seat-2 occupant."""
        if self.game.needs_marker or self.game.moves:
            return
        if self._seat2_occupant() != previous_seat2:
            self.game = self.game.clear_marker()

    def claim_seat(self, seat, player_id):
        """Try to claim a vacant seat. Returns (success, error_message).

        No game_started guard: a seat is only vacant mid-game if kick_seat
        emptied it, and claiming it then is wanted.
        """
        with self.lock:
            if seat not in (1, 2):
                return False, "Invalid seat."
            if self.computer_seat == seat:
                return False, "That seat is taken by the computer."
            if any(occupant == player_id for occupant in self.seats.values()):
                return False, "You already have a seat."
            if seat in self.seats:
                return False, "That seat is already taken."
            previous_seat2 = self._seat2_occupant()
            self.seats[seat] = player_id
            self._reset_marker_if_orphaned(previous_seat2)
            return True, None

    def vacate_seat(self, player_id):
        """Give up whichever seat player_id holds. Returns (success, error_message)."""
        with self.lock:
            if self.computer_seat is not None:
                return False, "You can't leave a solo game against the computer."
            if self.game_started and not self.game_over:
                return False, "You can't leave while a game is in progress."
            previous_seat2 = self._seat2_occupant()
            for seat, occupant in list(self.seats.items()):
                if occupant == player_id:
                    del self.seats[seat]
                    self._reset_marker_if_orphaned(previous_seat2)
                    return True, None
            return False, "You don't have a seat."

    def swap_seats(self, player_id):
        """Swap the two seats' occupants (including the computer's, for a solo
        room). Returns (success, error_message). Refused once the game has
        started, since seat number then decides whose turn it is.
        """
        with self.lock:
            if self.game_started:
                return False, "You can't swap seats after the game has started."
            if player_id not in self.seats.values():
                return False, "You don't have a seat."
            previous_seat2 = self._seat2_occupant()
            self.seats = {3 - seat: occupant for seat, occupant in self.seats.items()}
            if self.computer_seat is not None:
                self.computer_seat = 3 - self.computer_seat
            self._reset_marker_if_orphaned(previous_seat2)
            return True, None

    def seat_disconnected(self, seat):
        """Whether `seat` is held by a human with no connected socket - the
        signal to offer a "Kick" button so someone else can take over a seat
        abandoned mid-game. Never true for the computer's seat.
        """
        occupant = self.seats.get(seat)
        return occupant is not None and occupant not in self.sid_players.values()

    def kick_seat(self, seat):
        """Forcibly empty `seat`, but only while it's a disconnected player's
        (see seat_disconnected). Allowed mid-game, unlike vacate_seat.
        Returns (success, error_message).
        """
        with self.lock:
            if seat not in (1, 2):
                return False, "Invalid seat."
            if self.computer_seat is not None:
                return False, "You can't kick a seat in a solo game."
            if not self.seat_disconnected(seat):
                return False, "That seat isn't a disconnected player's."
            previous_seat2 = self._seat2_occupant()
            del self.seats[seat]
            self._reset_marker_if_orphaned(previous_seat2)
            return True, None

    def displayed_game(self, player_id):
        """The Game to render for this viewer: the live game while in
        progress, else the history position they're viewing (default final).
        Returns (game_to_show, history_index, history_total).
        """
        total = len(self.game.moves)
        if not self.game_over:
            return self.game, total, total
        index = self.history_index.get(player_id, total)
        index = max(0, min(total, index))
        if index == total:
            return self.game, index, total
        return self.game.undo(total - index), index, total


# Shared registries, read across the room subsystem's modules.
rooms: dict[str, RoomState] = {}
rooms_lock = threading.Lock()

# Connected socket id -> (room key, player_id), for disconnect cleanup.
# Review viewers aren't added.
sid_index: dict[str, tuple[str, str]] = {}
sid_index_lock = threading.Lock()

# Frozen snapshots of finished rooms, oldest first. See
# rendering._room_review_context.
finished_rooms: list[dict] = []
finished_rooms_lock = threading.Lock()


def status_counts():
    """Lightweight, lock-safe health counts for the /status endpoint:
    live rooms, recorded finished games, and connected sockets."""
    with rooms_lock:
        active_rooms = len(rooms)
    with finished_rooms_lock:
        finished = len(finished_rooms)
    with sid_index_lock:
        connected_clients = len(sid_index)
    return {
        "active_rooms": active_rooms,
        "finished_rooms": finished,
        "connected_clients": connected_clients,
    }


def find_finished_room_entry(entry_id):
    with finished_rooms_lock:
        for entry in finished_rooms:
            if entry["id"] == entry_id:
                return entry
    return None


def get_or_create_room(code, solo=False):
    """Look up a room by key, creating it if missing. `solo=True` seats the
    computer in seat 2 on creation; it's ignored for an existing room (a
    room's type is fixed at creation).
    """
    with rooms_lock:
        room = rooms.get(code)
        if room is None:
            # Dealt unplaced: seat 2 places the marker before play.
            room = RoomState(game=Game.deal(marker=None))
            if solo:
                room.computer_seat = 2
            rooms[code] = room
        return room


def deal_fresh_locked(room):
    """Reset a room to a brand-new dealt game (marker unplaced). Caller holds
    room.lock. Clears the per-game rematch / history / analysis state that must
    not carry over. Shared by the rematch handlers and the solo new-game path."""
    room.game = Game.deal(marker=None)
    room.game_id = secrets.token_hex(8)
    room.rematch_requested_by = None
    room.history_index.clear()
    room.game_over_seen.clear()
    room.analysis = None
    room.analysis_inflight = set()
    room.analysis_calc_started = {}


def random_unused_code():
    while True:
        code = "".join(random.choices(string.ascii_uppercase, k=4))
        with rooms_lock:
            if code not in rooms:
                return code


def resolve_room_key(raw_code, player_id):
    """Normalize a client-supplied room key for a lookup-only event: a
    4-letter code or a player id naming a solo room. Falls back to your own
    id if nothing usable was given.
    """
    return normalize_code(raw_code) or normalize_player_id(raw_code) or player_id


def resolve_or_create_room_for_join(raw_code, player_id):
    """Like resolve_room_key, but may create the room:
    - a valid 4-letter code always gets-or-creates a multiplayer room.
    - your own id (given explicitly, or implied by giving nothing at all)
      gets-or-creates your solo room, with the computer seated, if missing.
    - anyone else's id is looked up only - it must already exist.
    Returns (code, room, error_message) - room and error_message are
    mutually exclusive.
    """
    code = normalize_code(raw_code)
    if code is not None:
        return code, get_or_create_room(code), None
    target_id = normalize_player_id(raw_code) or player_id
    if target_id == player_id:
        return target_id, get_or_create_room(target_id, solo=True), None
    room = rooms.get(target_id)
    if room is None:
        return target_id, None, "That solo game doesn't exist any more."
    return target_id, room, None


def encode_game_state(game):
    """URL-safe token carrying a full game position (the "Play from here"
    payload). Wraps Game.save's alnum form in urlsafe base64 so its '/' and
    '?' separators survive in a URL path."""
    raw = game.save(alnum=True).encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_game_state(token):
    """Inverse of encode_game_state; returns a Game, or None if `token`
    isn't a valid saved position (so callers can 404 on a mangled link)."""
    try:
        pad = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(token + pad).decode("ascii")
        return Game.load(raw)
    except Exception:
        return None


def create_solo_room_from_state(player_id, game):
    """Create the player's solo room (replacing any existing) seeded at
    `game`, seating them on the side to move and the computer in the other,
    so a "Play from here" link always opens with its user to move. Returns
    (code, room), code being the player's own id."""
    mover_seat = 1 if len(game.moves) % 2 == 0 else 2
    room = RoomState(game=game)
    room.computer_seat = 3 - mover_seat
    room.seats = {mover_seat: player_id}
    with rooms_lock:
        rooms[player_id] = room
    return player_id, room
