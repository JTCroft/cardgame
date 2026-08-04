"""Game-flow glue spanning model, view, and analysis: driving the computer
opponent's turn and freezing a finished game into review history.

Sits above the other web modules (rooms, rendering, analysis_worker) and is
called from app.py's socket handlers.
"""

import random
import time

from ..ai import choose_move, choose_placement
from .analysis_worker import start_analysis_worker_locked
from .rendering import broadcast_state, room_summary_locked
from .rooms import finished_rooms, finished_rooms_lock

__all__ = ("maybe_play_computer_move", "record_room_finished_locked")

# Pad a fast computer move up to this so it reads as a deliberate turn.
_COMPUTER_MIN_MOVE_SECONDS = 1.0

# Finished games kept in the /rooms "Recently finished games" table.
_MAX_FINISHED_ENTRIES = 20

# Background time the post-game worker keeps backtracking after a game ends.
_ANALYSIS_TIME_CAP = 120.0


def record_room_finished_locked(code, room):
    """Freeze this room's just-finished result into history.

    Caller must hold room.lock, so the snapshot reflects the move that just
    finished the game. Starts the post-game analysis grace period.
    """
    entry = room_summary_locked(code, room)
    # Reuse the match's own id so the /review URL stays fixed from deal time.
    entry["id"] = room.game_id
    entry["game"] = room.game
    entry["history_index"] = {}
    # One shared analysis cache for the room's review and this frozen entry.
    if room.analysis is None:
        room.analysis = {}
    # Give the worker _ANALYSIS_TIME_CAP more seconds to keep backtracking.
    room.analysis_deadline = time.monotonic() + _ANALYSIS_TIME_CAP
    entry["analysis"] = room.analysis
    entry["analysis_inflight"] = room.analysis_inflight
    entry["analysis_calc_started"] = room.analysis_calc_started
    # Copied by value: a frozen entry's deadline refers to this match only,
    # even after a rematch repurposes the live room's field.
    entry["analysis_deadline"] = room.analysis_deadline
    start_analysis_worker_locked(code, room)
    with finished_rooms_lock:
        finished_rooms.append(entry)
        while len(finished_rooms) > _MAX_FINISHED_ENTRIES:
            finished_rooms.pop(0)


def maybe_play_computer_move(code, room):
    """Play the computer's move and broadcast it if it's now its turn.

    Only ever acts in a solo-vs-computer room; a harmless no-op otherwise,
    so callers can invoke it unconditionally after every move. Loops in case
    the move leaves it the computer's turn again.
    """
    while True:
        with room.lock:
            if room.computer_seat is None or room.game_over:
                return
            if room.game.needs_marker:
                # Computer placer is seat 2 only; seat 1 placement is the human's.
                if not room.both_seated or room.computer_seat != 2:
                    return
                placing = True
            elif room.current_turn_seat != room.computer_seat:
                return
            else:
                placing = False
            game, game_id = room.game, room.game_id
        # Think and pad outside the lock so a slow think doesn't block
        # broadcasts or the analysis worker.
        start = time.monotonic()
        if placing:
            placement = choose_placement(game)
        else:
            row, col = choose_move(game)
        remaining = _COMPUTER_MIN_MOVE_SECONDS - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)
        with room.lock:
            # Only apply if the position we solved is still current (a rematch,
            # seat swap, or another caller may have moved on meanwhile).
            if (
                room.game is not game
                or room.game_id != game_id
                or room.computer_seat is None
                or room.game_over
            ):
                return
            if placing:
                # Re-check under the lock: a seat swap can hand seat 2 to the human.
                if not room.both_seated or room.computer_seat != 2:
                    return
                room.game = room.game.place_marker(*placement)
            else:
                if room.current_turn_seat != room.computer_seat:
                    return
                room.game = random.choice(room.game.move(row, col))
                if room.game_over:
                    record_room_finished_locked(code, room)
        broadcast_state(code, room)
        if placing:
            # Loop in case it's the computer's turn again (never is in practice).
            continue
