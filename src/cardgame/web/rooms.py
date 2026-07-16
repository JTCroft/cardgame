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
import time
from collections import Counter
from dataclasses import dataclass, field

from flask import render_template

from ..ai import choose_move
from ..analysis import AnalysisAborted, analyse_moves
from ..game import Game
from ..search_alt import _snapshot as _search_snapshot
from ..search_alt import move_search_iterator
from .extensions import app_holder, socketio
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
    # Post-game move-comparison analyses, {move_index: analyse_moves(...)}.
    # Filled from two cooperating sources sharing this dict: a live worker
    # that analyses already-played positions *during* the game (each
    # position is fixed the moment its move is made, and a long game
    # donates its thinking time to the deep, expensive ones - see
    # _live_analysis_loop), and the post-game drain that fills the cheap
    # tail at game end (_precompute_analysis). Shared with the frozen
    # _finished_rooms entry recorded at game end - same dict object, so
    # late-finishing live analyses still land in the review. Reset to None
    # on a fresh deal (the frozen entry keeps the old dict).
    analysis: dict | None = None
    # Move indices currently being analysed by the live worker, so the
    # post-game drain doesn't duplicate a minutes-long computation.
    # Replaced (not cleared) on a fresh deal, since the frozen entry
    # shares the old set.
    analysis_inflight: set = field(default_factory=set)
    # Guard so at most one live analysis worker runs per room.
    analysis_running: bool = False
    # Cooperative abort for the live worker's current analysis: set when
    # the game ends (see _record_room_finished_locked) so an unbounded
    # backtracking analysis stops burning CPU on a game nobody is playing
    # any more. Replaced with a fresh event on a new deal.
    analysis_abort: threading.Event = field(default_factory=threading.Event)
    # Whether this (solo) room is in "live eval" mode: a background thread
    # runs the anytime search (cardgame.search_alt) on whatever the current
    # position is, pushing a continuously-updating move ranking to every
    # connected socket. Set by the owner's entry point (/play/live vs
    # /play) at join time.
    live_eval: bool = False
    # Guard so at most one live-eval thread runs per room; the thread
    # clears it when it exits (no viewers / mode switched off) so a later
    # join can start a fresh one.
    live_eval_running: bool = False
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


# Total background time spent analysing any one finished game. Positions
# are analysed newest-first with monotonically growing cost (going
# backwards only ever adds cards and unknowns) - observed growth is
# roughly 3-15x per ply, so the walk stops once the next (costlier)
# position can no longer plausibly fit the remaining budget.
_ANALYSIS_TIME_CAP = 120.0
_ANALYSIS_GROWTH_FACTOR = 6


def _analysis_feasible(game):
    """Hard ceiling on which positions `analyse_moves` may even attempt -
    calibrated against worst cases over sampled random games (recalibrated
    2026-07 after the cached-scoring / single-pass-walk optimisations,
    which bought roughly 8-30x here). The adaptive soft stop in
    _precompute_analysis does the fine-grained cost control; this only
    rules out the combinations whose worst cases run into minutes. It
    walks the full remaining tree with no pruning, so it's viable only
    near the end of the game; as with Game.evaluate, the chance-node
    branching from face-down cards is the main cost driver.
    """
    # The next combinations out - (14, 5), (15, 4), (16, 4), (17, 3) -
    # were measured at 100-336s for a single position, past any sensible
    # budget; this ceiling sits exactly at that cliff edge.
    cards_left = 36 - len(game.moves)
    facedown = len(game.board.facedown_cards)
    if cards_left <= 9:
        return True
    if cards_left <= 10:
        return facedown <= 7
    if cards_left <= 11:
        return facedown <= 6
    if cards_left <= 13:
        return facedown <= 5
    if cards_left <= 14:
        return facedown <= 4
    if cards_left <= 16:
        return facedown <= 3
    return False


def _analysis_attemptable(game):
    """The live worker's wider ceiling: during play there are minutes of
    thinking time rather than a post-game budget, so it may attempt one
    ring beyond _analysis_feasible - the combinations measured in the
    ~2-6 minute range ((14,5): 104s, (15,4): 100s, (16,4): 336s,
    (17,3): 319s). Anything deeper runs into tens of minutes.
    """
    if _analysis_feasible(game):
        return True
    cards_left = 36 - len(game.moves)
    facedown = len(game.board.facedown_cards)
    if cards_left <= 14:
        return facedown <= 5
    if cards_left <= 16:
        return facedown <= 4
    if cards_left <= 17:
        return facedown <= 3
    return False


def _precompute_analysis(game, cache, inflight=frozenset(), on_done=None):
    """Fill `cache[move_index]` with `analyse_moves` of the position after
    `move_index` moves, walking backwards from the last move played until
    a position is past the hard ceiling, the previous position already
    took long enough that the next (always costlier) one shouldn't be
    started, or the total budget is spent. Skips indices already analysed
    (typically by the live worker during the game) or currently in flight
    there. Runs on a background thread - see _record_room_finished_locked.
    """
    deadline = time.monotonic() + _ANALYSIS_TIME_CAP
    total = len(game.moves)
    for index in range(total - 1, -1, -1):
        if index in cache or index in inflight:
            continue
        position = game.undo(total - index)
        if not _analysis_feasible(position):
            break
        position_start = time.monotonic()
        cache[index] = analyse_moves(position)
        now = time.monotonic()
        if now >= deadline or (now - position_start) * _ANALYSIS_GROWTH_FACTOR > deadline - now:
            break
    if on_done is not None:
        on_done()


def _live_analysis_loop(code, room):
    """Analyse the current game's already-played positions while it is
    still being played: every position is fixed the moment its move is
    made. _analysis_attemptable decides where the work *starts*; once
    every position inside that band is done, the worker backtracks one
    position deeper at a time with no ceiling at all - the running game
    itself is the budget, so a long, thoughtful game buys itself review
    depth a fixed gate never could. Results go into the same cache the
    post-game review reads; a backtracking analysis still in flight when
    the game ends is aborted cooperatively (room.analysis_abort, set at
    game end) rather than left burning CPU. Exits when the game ends (the
    post-game drain in _precompute_analysis owns the tail) or the room
    empties."""
    try:
        while True:
            with room.lock:
                if not room.sid_players:
                    return
                game = room.game
                if room.analysis is None:
                    room.analysis = {}
                cache = room.analysis
                inflight = room.analysis_inflight
                abort = room.analysis_abort
            if not game.legal_moves:
                return
            total = len(game.moves)
            done = cache.keys() | inflight
            candidates = [k for k in range(total + 1) if k not in done]
            index = None
            # Newest attemptable position first: the band's members are all
            # cheap (minutes at worst), so clear them before going deeper.
            for k in reversed(candidates):
                if _analysis_attemptable(game.undo(total - k)):
                    index = k
                    break
            if index is None and done:
                # Band finished - backtrack one position deeper, unbounded.
                frontier = min(done)
                if frontier > 0 and frontier - 1 in candidates:
                    index = frontier - 1
            if index is None:
                # Either the game hasn't reached the sensible starting
                # point yet, or every position back to the deal is done.
                time.sleep(1.0)
                continue
            position = game.undo(total - index)
            with room.lock:
                inflight.add(index)
            try:
                cache[index] = analyse_moves(position, abort=abort)
            except AnalysisAborted:
                pass
            finally:
                with room.lock:
                    inflight.discard(index)
    finally:
        with room.lock:
            room.analysis_running = False


def _ensure_analysis_worker(code, room):
    """Start the room's live analysis worker if the game is in progress,
    someone is watching, and none is running. Callers invoke this after
    joins and moves; it's a cheap no-op otherwise."""
    with room.lock:
        if (
            room.analysis_running
            or not room.sid_players
            or not room.game.legal_moves
            or not room.game.moves
        ):
            return
        room.analysis_running = True
    threading.Thread(target=_live_analysis_loop, args=(code, room), daemon=True).start()


def _in_app_context(fn):
    """Run fn inside the Flask app's context - required for
    render_template on a background thread. A no-op if the app hasn't been
    constructed (unit tests poking internals directly)."""
    app = app_holder.get("app")
    if app is None:
        return
    with app.app_context():
        fn()


def _record_room_finished_locked(code, room):
    """Freeze this room's just-finished result into history. Caller must
    already hold room.lock, so the snapshot (game object included)
    reflects exactly the move that just finished it - not a later one.
    """
    entry = _room_summary_locked(code, room)
    entry["id"] = secrets.token_hex(8)
    entry["game"] = room.game
    entry["history_index"] = {}
    # One shared cache for the room's own post-game review and this frozen
    # entry's /review page - usually already largely filled by the live
    # worker during the game; the drain adds the cheap tail (skipping
    # anything the worker finished or is still finishing). Viewers already
    # scrubbing get a refresh when it's done; anyone arriving later just
    # finds it ready.
    if room.analysis is None:
        room.analysis = {}
    # Stop any unbounded backtracking analysis still in flight - the game
    # it was buying time for is over. The cheap drain below takes over.
    room.analysis_abort.set()
    cache = room.analysis
    entry["analysis"] = cache
    entry["analysis_inflight"] = room.analysis_inflight
    threading.Thread(
        target=_precompute_analysis,
        args=(entry["game"], cache),
        kwargs={
            "inflight": room.analysis_inflight,
            "on_done": lambda: _in_app_context(lambda: _broadcast_state(code, room)),
        },
        daemon=True,
    ).start()
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


# Sequential ramp for the outcome heatmap's cells: dark (0% of outcomes,
# matching .move-analysis's own background so an empty cell reads as
# "nothing here") up to this site's existing accent blue (100%). A single
# hue carries likelihood; which side of the axis a cell sits on is what
# carries who it favours, so the two are never conflated in one channel.
_HEATMAP_BASE_RGB = (0x2B, 0x2B, 0x2B)
_HEATMAP_ACCENT_RGB = (0x7C, 0xB8, 0xFF)


def _heatmap_style(pct):
    """The cell's background - interpolated along the sequential ramp - and
    a text color for the percentage label printed on top of it, picked by
    the background's luminance so the label stays legible at both ends."""
    t = min(1.0, max(0.0, pct / 100))
    r, g, b = (
        round(base + (accent - base) * t)
        for base, accent in zip(_HEATMAP_BASE_RGB, _HEATMAP_ACCENT_RGB)
    )
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    text = "#161616" if luminance > 140 else "#f2f2f2"
    return f"#{r:02x}{g:02x}{b:02x}", text


# The heatmap's axis is fixed rather than sized to each position's actual
# range, so positions can be compared at a glance instead of each drawing
# its own scale. Outcomes beyond it collapse into the two end cells.
_HEATMAP_RANGE = 4


def _outcome_heatmap(distribution, mover_seat):
    """Turn a move's {mover_diff: weight} distribution (see analyse_moves)
    into the "who's winning" heatmap's template context: a fixed row of
    cells from -_HEATMAP_RANGE to +_HEATMAP_RANGE (P1 - P2), each annotated
    with its likelihood, plus two end cells pooling everything beyond that
    range - and the single most likely outcome, called out as the headline
    text. Face-down cards mean even optimal play from a fixed position can
    end in a spread of scores, not one number, which is what this is for.
    """
    total = sum(distribution.values())
    # distribution is in the mover's own perspective (self - opponent);
    # flip it onto the fixed P1 - P2 axis the template renders.
    sign = 1 if mover_seat == 1 else -1
    p1_distribution = Counter({sign * diff: weight for diff, weight in distribution.items()})

    def cell(label, pct, zero=False):
        color, text_color = _heatmap_style(pct)
        return {"label": label, "pct": pct, "zero": zero, "color": color, "text_color": text_color}

    low_pct = sum(w for d, w in p1_distribution.items() if d <= -_HEATMAP_RANGE - 1) / total * 100
    high_pct = sum(w for d, w in p1_distribution.items() if d >= _HEATMAP_RANGE + 1) / total * 100
    cells = [cell(f"≤-{_HEATMAP_RANGE + 1}", low_pct)]
    for diff in range(-_HEATMAP_RANGE, _HEATMAP_RANGE + 1):
        pct = p1_distribution.get(diff, 0) / total * 100
        cells.append(cell(f"{diff:+d}" if diff else "0", pct, zero=(diff == 0)))
    cells.append(cell(f"≥+{_HEATMAP_RANGE + 1}", high_pct))

    mode_diff, mode_weight = max(p1_distribution.items(), key=lambda item: (item[1], -abs(item[0])))
    return {
        "cells": cells,
        "mode_diff": mode_diff,
        "mode_pct": mode_weight / total * 100,
    }


def _move_analysis_context(cache, full_game, display_game, history_index, p1_name, p2_name,
                           inflight=frozenset()):
    """Build the move-comparison panel's template context for the position
    currently being reviewed, or None when there's nothing to show (final
    position, no analysis recorded, or an intractable position). Returns
    {"pending": True, ...} while the background computation hasn't reached
    a tractable position yet - or is mid-flight on this one - so the
    template can say it's on its way.
    """
    if cache is None or history_index >= len(full_game.moves):
        return None
    mover_seat = 1 if history_index % 2 == 0 else 2
    mover_name = (p1_name if mover_seat == 1 else p2_name) or f"Player {mover_seat}"
    opponent_name = (p2_name if mover_seat == 1 else p1_name) or f"Player {3 - mover_seat}"
    data = cache.get(history_index)
    if data is None:
        if history_index not in inflight and not _analysis_feasible(display_game):
            return None
        return {"pending": True, "mover_name": mover_name}
    played_marker = full_game.moves[history_index]
    rows = []
    for marker, move in sorted(
        data.items(), key=lambda item: (-item[1]["combined"], item[0])
    ):
        card = move["card"]
        rows.append(
            {
                "marker": marker,
                "card": card,
                # A face-down move's analysis averages over what it might
                # have been, but the one actually played has a known
                # outcome - the final board holds its revealed identity.
                "revealed": (
                    full_game.board[marker[0]][marker[1]]
                    if card.facedown and marker == played_marker
                    else None
                ),
                "combined": move["combined"],
                "defensive": move["defensive"],
                "offensive": move["offensive"],
                "best": move["combined"] == 0,
                "played": marker == played_marker,
            }
        )
    best = rows[0]
    return {
        "pending": False,
        "mover_name": mover_name,
        "opponent_name": opponent_name,
        "best_player_mean": data[best["marker"]]["player_mean"],
        "best_opponent_mean": data[best["marker"]]["opponent_mean"],
        "rows": rows,
        "heatmap": _outcome_heatmap(data[best["marker"]]["distribution"], mover_seat),
    }


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

    move_analysis = None
    if game_over:
        move_analysis = _move_analysis_context(
            room.analysis,
            final_game,
            display_game,
            history_index,
            room.name_for_seat(1),
            room.name_for_seat(2),
            inflight=room.analysis_inflight,
        )

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
        "move_analysis": move_analysis,
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
        "move_analysis": _move_analysis_context(
            entry.get("analysis"),
            game,
            display_game,
            history_index,
            entry["p1_name"],
            entry["p2_name"],
            inflight=entry.get("analysis_inflight", frozenset()),
        ),
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


# Live-eval pacing: per-position caps keep an early-game search (which can
# never finish - the opening's tree is astronomically large) from pinning
# the CPU and growing its node tree forever; deeper positions solve well
# inside them. Batches are bounded by wall clock, not quantum count - a
# single quantum ranges from microseconds (an expansion) to a few hundred
# milliseconds (an exact leaf solve), so only time-batching keeps the
# emit/position-change checks responsive. Small sleeps between batches
# keep the request threads breathing under the GIL.
_LIVE_EVAL_BUDGET = 60.0
_LIVE_EVAL_WORK_CAP = 30000
_LIVE_EVAL_BATCH_SECONDS = 0.25
_LIVE_EVAL_EMIT_INTERVAL = 0.8


def _render_live_eval(room, game, snapshot, status):
    mover_seat = 1 if len(game.moves) % 2 == 0 else 2
    return render_template(
        "_live_eval.html.jinja2",
        snapshot=snapshot,
        mover_name=room.name_for_seat(mover_seat) or f"Player {mover_seat}",
        status=status,
    )


def _emit_live_eval(room, game, snapshot, status):
    with room.lock:
        sids = list(room.sid_players)
    if not sids:
        return
    html = []
    _in_app_context(
        lambda: html.append(_render_live_eval(room, game, snapshot, status))
    )
    if not html:
        return
    for sid in sids:
        socketio.emit("live_eval", {"html": html[0]}, to=sid)


def _live_eval_status(root, capped):
    if root.resolved:
        return "solved - exact value known"
    if root.proven:
        return "best move proven"
    if capped:
        return "paused - search limit for this position reached"
    return f"searching ({root.ctx.work} solves)"


def _live_eval_loop(code, room):
    """Continuously evaluate the room's current position with the anytime
    search, pushing ranking updates to everyone connected. The search on a
    position runs for as long as that position stays current (or until it
    is solved / hits its caps); a move or a fresh deal abandons it and
    starts over on the new position. Exits when the room empties or leaves
    live-eval mode - a later join starts a new thread."""
    try:
        while True:
            with room.lock:
                if not room.live_eval or not room.sid_players:
                    return
                game = room.game
            if not game.legal_moves:
                # game over - the post-game analysis panel takes over; wait
                # here for a rematch to swap in a new game
                time.sleep(1.0)
                continue
            root = move_search_iterator(game)
            started = time.monotonic()
            last_emit = 0.0
            final_emitted = False
            while True:
                with room.lock:
                    if not room.live_eval or not room.sid_players:
                        return
                    if room.game is not game:
                        break  # position moved on - restart on the new one
                capped = (
                    time.monotonic() - started > _LIVE_EVAL_BUDGET
                    or root.ctx.work >= _LIVE_EVAL_WORK_CAP
                )
                if root.resolved or capped:
                    if not final_emitted:
                        _emit_live_eval(
                            room,
                            game,
                            _search_snapshot(root, time.monotonic() - started),
                            _live_eval_status(root, capped),
                        )
                        final_emitted = True
                    time.sleep(0.5)
                    continue
                batch_end = time.monotonic() + _LIVE_EVAL_BATCH_SECONDS
                while time.monotonic() < batch_end and not root.resolved:
                    try:
                        next(root)
                    except StopIteration:
                        break
                now = time.monotonic()
                if root.moves is not None and now - last_emit >= _LIVE_EVAL_EMIT_INTERVAL:
                    _emit_live_eval(
                        room,
                        game,
                        _search_snapshot(root, now - started),
                        _live_eval_status(root, capped=False),
                    )
                    last_emit = now
                time.sleep(0.02)
    finally:
        with room.lock:
            room.live_eval_running = False


def _ensure_live_eval(code, room):
    """Start the room's live-eval thread if its mode calls for one and none
    is running. Callers invoke this unconditionally after joins; it's a
    no-op for ordinary rooms."""
    with room.lock:
        if not room.live_eval or room.live_eval_running or not room.sid_players:
            return
        room.live_eval_running = True
    threading.Thread(target=_live_eval_loop, args=(code, room), daemon=True).start()


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
