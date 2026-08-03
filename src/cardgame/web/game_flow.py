"""Game-flow glue that spans the model, view, and analysis subsystems:
driving the computer opponent's turn, and freezing a finished game into
review history (which also kicks off the post-game analysis grace period).

These sit above the other web modules - they call into the model (rooms),
the view (rendering), and the worker (analysis_worker) - so they live here
rather than in any one of those, and app.py's socket handlers call them.
"""

import random
import time

from ..ai import choose_move, choose_placement
from .analysis_worker import start_analysis_worker_locked
from .rendering import broadcast_state, room_summary_locked
from .rooms import finished_rooms, finished_rooms_lock

__all__ = ("maybe_play_computer_move", "record_room_finished_locked")

# Floor on how long the computer takes to play, so a move it solves near-
# instantly (e.g. a forced move or a shallow endgame) still reads as a
# deliberate turn rather than snapping onto the board. Only ever pads a fast
# move up to this; a longer think is left alone.
_COMPUTER_MIN_MOVE_SECONDS = 1.0

# How many finished games (rooms and solo games together) the /rooms
# "Recently finished games" table keeps around - see
# record_room_finished_locked.
_MAX_FINISHED_ENTRIES = 20

# Total background time the post-game worker keeps backtracking after a game
# ends (see record_room_finished_locked / analysis_worker).
_ANALYSIS_TIME_CAP = 120.0


def record_room_finished_locked(code, room):
    """Freeze this room's just-finished result into history. Caller must
    already hold room.lock, so the snapshot (game object included)
    reflects exactly the move that just finished it - not a later one.
    """
    entry = room_summary_locked(code, room)
    # Reuse the match's own id (assigned at deal time - see RoomState.game_id)
    # rather than minting a fresh one: the /review URL for this match was
    # already fixed the moment it was dealt.
    entry["id"] = room.game_id
    entry["game"] = room.game
    entry["history_index"] = {}
    # One shared cache for the room's own post-game review and this frozen
    # entry's /review page - usually already largely filled during the game;
    # the worker's post-game grace period (below) adds whatever more it can
    # in the time it has left. Viewers already scrubbing get a refresh when
    # it's done; anyone arriving later just finds it ready.
    if room.analysis is None:
        room.analysis = {}
    # Give the worker _ANALYSIS_TIME_CAP more seconds to keep backtracking -
    # long enough to let anything already in flight finish rather than
    # aborting it outright, and to fill in a bit more depth besides - before
    # it stops burning CPU on a game nobody is playing any more. See
    # analysis_worker and analysis_deadline's field comment.
    room.analysis_deadline = time.monotonic() + _ANALYSIS_TIME_CAP
    entry["analysis"] = room.analysis
    entry["analysis_inflight"] = room.analysis_inflight
    entry["analysis_calc_started"] = room.analysis_calc_started
    # Copied by value, not shared like the dicts above: the live room's own
    # analysis_deadline gets repurposed (or cleared - see its field comment)
    # by a later rematch, but a frozen entry's "has the automatic grace
    # period passed, so is a 'Calculate' button appropriate" question always
    # refers to *this* match's own deadline, fixed at the moment it froze.
    entry["analysis_deadline"] = room.analysis_deadline
    start_analysis_worker_locked(code, room)
    with finished_rooms_lock:
        finished_rooms.append(entry)
        while len(finished_rooms) > _MAX_FINISHED_ENTRIES:
            finished_rooms.pop(0)


def maybe_play_computer_move(code, room):
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
            if room.game.needs_marker:
                # Computer placer (seat 2 only): choose a starting cell once
                # both seats are filled. Placing from seat 1 is never the
                # computer's call - that's the human's.
                if not room.both_seated or room.computer_seat != 2:
                    return
                placing = True
            elif room.current_turn_seat != room.computer_seat:
                return
            else:
                placing = False
            game, game_id = room.game, room.game_id
        # Think (and enforce the minimum move time) outside the lock, so a
        # slow think no longer holds the room's lock and the padding sleep
        # never blocks broadcasts or the analysis worker. Placement runs the
        # same budgeted search over the four starting cells.
        start = time.monotonic()
        if placing:
            placement = choose_placement(game)
        else:
            row, col = choose_move(game)
        remaining = _COMPUTER_MIN_MOVE_SECONDS - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)
        with room.lock:
            # The game may have moved on while we were thinking (a rematch
            # dealt a fresh game, a seat was swapped, the move/placement was
            # applied by another caller): only apply if it is still exactly
            # the position we solved. `room.game is not game` already covers a
            # marker placed meanwhile (the game object would have changed).
            if (
                room.game is not game
                or room.game_id != game_id
                or room.computer_seat is None
                or room.game_over
            ):
                return
            if placing:
                # Re-check the placement preconditions under the lock: a seat
                # swap during the think can hand seat 2 (the placer) to the
                # human, in which case the computer must not place after all.
                # (The turn re-check below is the move-branch equivalent.)
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
            # The move turn is now P1's; loop in case it becomes the
            # computer's turn again (it never is in practice).
            continue
