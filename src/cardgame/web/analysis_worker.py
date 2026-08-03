"""The post-game move-analysis worker and its thread pools.

A single worker per room (see _analysis_worker_loop) analyses already-played
positions both during the game and for a bounded grace period after it ends,
streaming per-move results into the room's shared analysis cache and
broadcasting them as they land. Explicit "Calculate" clicks go through
start_ondemand_analysis on a separate pool.

Depends on the room-state model (rooms), the view layer (rendering, for
broadcast_state), and the pure feasibility predicates (analysis_policy).
Nothing here builds template context itself.
"""

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from ..analysis import AnalysisAborted, analyse_moves_by_deadline
from ..analysis_native import FINAL, NATIVE_AVAILABLE, iter_move_analyses
from .analysis_policy import analysis_attemptable
from .extensions import app_holder
from .rendering import broadcast_state

__all__ = (
    "start_ondemand_analysis",
    "start_analysis_worker_locked",
    "ensure_analysis_worker",
)


# _analysis_worker_loop runs exactly two concurrent analyse_moves calls,
# each with a fixed, reserved role (see the function's docstring): one
# always chasing the newest attemptable position, the other always
# extending one step further back into history. Neither role ever
# borrows the other's slot, even when idle - a backward call can run for
# a very long time, and if it were allowed to occupy both slots, a fresh
# move played in the meantime would have to wait behind it. The reserved
# forward slot guarantees that never happens, at the cost of sitting idle
# whenever there's nothing new to catch up on.

# Hard per-call backstop, regardless of live play or the post-game grace
# period: bounds a single analyse_moves call in case its actual cost
# doesn't match what analysis_attemptable (or the unbounded backtrack)
# expected, so a bad estimate can only ever tie up its own slot for a
# bounded time - not indefinitely, and not at the expense of the other
# slot's role.
_ANALYSIS_CALL_SAFETY_CAP = 30 * 60

# Deadline for an explicit, one-off "Calculate" request (see
# start_ondemand_analysis) - only reachable once the automatic worker has
# already given up on a position (see analysis_feasible/_ANALYSIS_TIME_CAP),
# so this is a much longer leash than that unattended budget: a user who
# clicks the button is deliberately opting into the wait for one single
# position, not leaving something running unsupervised. Comfortably past the
# ~2-6 minute range analysis_attemptable's docstring measures for the ring
# just beyond analysis_feasible's ceiling, with headroom for deeper ones.
_ONDEMAND_ANALYSIS_DEADLINE = 30 * 60

# Shared, app-wide pools that run analyse_moves calls. Threads, not
# processes: the native solver (cardgame_native) releases the GIL for the
# whole walk, so concurrent calls genuinely use multiple cores, and running
# in-process lets a call stream its per-move results straight into the room
# cache and broadcast them as they land - something a ProcessPoolExecutor
# Future (all-or-nothing) can't do. Without the native core the walk is
# pure-Python and holds the GIL, so those calls serialise; that path is the
# untuned fallback and streams nothing (analyse_moves_by_deadline returns the
# whole dict at once).
#
# The forward and backward slots (see _analysis_worker_loop) get *separate*
# pools rather than sharing one. A single shared pool only reserves the two
# roles per-worker, not at the point of execution: forward submissions still
# queue behind whatever backward calls are already occupying the pool's
# threads - across *all* rooms - and a backward call can run for the whole
# grace period. On a small box (os.cpu_count() is 1-2) or after many games
# have piled leftover backward calls into the pool, the cheap, feasible deep
# positions a finishing game hands to the forward slot then never get a thread
# before their grace expires, and are left stuck showing "still being computed"
# forever. Dedicating a pool to each role keeps forward work from ever queuing
# behind expensive backward work, so the reserved-role guarantee holds in
# execution too. Floored at two workers each so both roles get a real slot even
# on a single-CPU host (native calls release the GIL, so oversubscribing one
# core just time-slices - a cheap forward call still finishes promptly rather
# than waiting out a backward call ahead of it in a queue).
# A third pool serves explicit "Calculate" requests (start_ondemand_analysis),
# kept separate again so a user-clicked calculation is neither blocked by the
# automatic slots nor able to starve them in turn.
_analysis_forward_pool = None
_analysis_backward_pool = None
_analysis_ondemand_pool = None
_analysis_pool_lock = threading.Lock()


def _init_analysis_pools_locked():
    global _analysis_forward_pool, _analysis_backward_pool, _analysis_ondemand_pool
    if _analysis_forward_pool is None:
        workers = max(2, os.cpu_count() or 2)
        _analysis_forward_pool = ThreadPoolExecutor(max_workers=workers)
        _analysis_backward_pool = ThreadPoolExecutor(max_workers=workers)
        _analysis_ondemand_pool = ThreadPoolExecutor(max_workers=workers)


def _get_analysis_pools():
    """Return (forward_pool, backward_pool), creating them on first use."""
    with _analysis_pool_lock:
        _init_analysis_pools_locked()
        return _analysis_forward_pool, _analysis_backward_pool


def _get_ondemand_pool():
    """Pool for explicit "Calculate" requests - see start_ondemand_analysis."""
    with _analysis_pool_lock:
        _init_analysis_pools_locked()
        return _analysis_ondemand_pool


def _in_app_context(fn):
    """Run fn inside the Flask app's context - required for
    render_template on a background thread. A no-op if the app hasn't been
    constructed (unit tests poking internals directly)."""
    app = app_holder.get("app")
    if app is None:
        return
    with app.app_context():
        fn()


def _stream_analysis(cache, lock, position, index, deadline, broadcast):
    """Analyse one position, folding results into cache[index] and calling
    `broadcast` after each update - the single per-position work unit both
    the automatic worker and the on-demand path submit to the pool.

    Native path: streams one root move at a time (see iter_move_analyses),
    writing a provisional, re-sorting table into cache[index] as each move
    lands and the winner's heatmap once it's all in. Whatever completed
    before a deadline stays visible. Fallback path: computes the whole dict
    and writes it once. Returns the finished move_data, or None if the
    deadline cut it short (the streamed partial, if any, is left in place).
    """
    if not NATIVE_AVAILABLE:
        result = analyse_moves_by_deadline(position, deadline)
        if result is not None:
            with lock:
                cache[index] = result
            broadcast()
        return result

    total = len(position.legal_moves)
    try:
        final = None
        for marker, data in iter_move_analyses(position, deadline):
            with lock:
                if marker is FINAL:
                    final = data
                    cache[index] = data  # complete move_data (marker -> stats)
                else:
                    # A distinct shape from a finished move_data dict (whose
                    # keys are (row, col) tuples) - see _move_analysis_context.
                    cache[index] = {
                        "streaming": True,
                        "moves": dict(data),
                        "done": len(data),
                        "total": total,
                    }
            broadcast()
        return final
    except AnalysisAborted:
        # Keep whatever streamed, but settle it so the UI stops spinning.
        with lock:
            existing = cache.get(index)
            if isinstance(existing, dict) and existing.get("streaming"):
                existing["streaming"] = False
        broadcast()
        return None


def start_ondemand_analysis(cache, inflight, calc_started, lock, game, index, on_done=None):
    """Kick off a one-off analysis of a single already-played position (at
    `index` moves into `game`) that the automatic worker gave up on,
    running it on a background thread with _ONDEMAND_ANALYSIS_DEADLINE to
    work with instead of the worker's own stingier budget. Returns True if
    a fresh calculation was actually started, False if `index` is already
    cached or already being calculated (by a previous on-demand request, or
    in principle still by the automatic worker itself) - in which case the
    caller has nothing further to do, the existing "pending"/inflight state
    already covers it.

    `cache`/`inflight`/`calc_started`/`lock` are a live room's or a frozen
    finished_rooms entry's own analysis/analysis_inflight/
    analysis_calc_started plus whichever lock guards them (room.lock, or
    finished_rooms_lock for a frozen entry - see the "calculate_move"
    socket handler) - passed in rather than a RoomState/entry directly so
    this one function serves both.

    `on_done`, if given, runs (inside a Flask app context, so it's safe to
    render_template) once the result - or lack of one - has already been
    folded into `cache`/`inflight`/`calc_started` above; it's how the
    caller pushes a fresh render out to whoever's watching, since nothing
    here knows about sockets or which room/entry this even belongs to.
    """
    total = len(game.moves)
    if not (0 <= index < total):
        return False
    with lock:
        if index in cache or index in inflight:
            return False
        inflight.add(index)
        calc_started[index] = time.monotonic()

    def worker():
        position = game.undo(total - index)
        pool = _get_ondemand_pool()
        call_deadline = time.monotonic() + _ONDEMAND_ANALYSIS_DEADLINE
        broadcast = (lambda: _in_app_context(on_done)) if on_done is not None else (lambda: None)
        # _stream_analysis folds results (and streamed partials) into cache
        # itself; just wait for it to settle before clearing the inflight
        # markers and pushing the final render.
        pool.submit(
            _stream_analysis, cache, lock, position, index, call_deadline, broadcast
        ).result()
        with lock:
            inflight.discard(index)
            calc_started.pop(index, None)
        if on_done is not None:
            _in_app_context(on_done)

    threading.Thread(target=worker, daemon=True).start()
    return True


def _newest_attemptable_index(game, total, done, upper):
    """The newest index in range(upper) not already in `done` whose
    position satisfies analysis_attemptable, or None if there isn't one.
    Shared by _analysis_worker_loop's backward slot (bootstrapping itself
    when nothing's been analysed yet at all) and forward slot (its
    ordinary catch-up scan)."""
    for k in reversed(range(upper)):
        if k in done:
            continue
        if analysis_attemptable(game.undo(total - k)):
            return k
    return None


def _abandon_futures(room, cache, inflight, futures):
    """Stop tracking this room's outstanding submissions against `cache`/
    `inflight` - called whenever the worker is giving up on that game's
    epoch: a rematch has swapped in a new game (see the epoch check in
    _analysis_worker_loop) or the worker itself is exiting (room emptied,
    grace period elapsed, or any other reason). Anything already finished
    is harvested into `cache` first, so a result that landed moments before
    the decision to stop isn't wasted; whatever's still running gets
    `.cancel()`'d, which only actually prevents it from starting if the
    pool hasn't gotten to it yet - a task already running in a pool thread
    can't be interrupted from here. It's left to run to its own baked-in
    deadline in that case (the native walk polls that deadline itself, so it
    does stop), streaming into `cache` as it goes but with the slot no longer
    tracked. Either way `inflight` always gets cleared, so a later worker can
    pick the same index up again rather than seeing it "pending" forever."""
    for index, fut in futures.items():
        if fut.done():
            result = fut.result()
            if result is not None:
                cache[index] = result
        else:
            fut.cancel()
        with room.lock:
            inflight.discard(index)
    futures.clear()


def _submit(pool, code, room, cache, inflight, game, index, ended, deadline):
    """Submit the streaming analysis of the position `index` moves into
    `game`, marking it inflight. Shared by both of _analysis_worker_loop's
    reserved slots - only what candidate to submit differs between them.
    Results (and streamed partials) are folded into `cache` and pushed to
    viewers via broadcast_state as they land."""
    total = len(game.moves)
    position = game.undo(total - index)
    safety = time.monotonic() + _ANALYSIS_CALL_SAFETY_CAP
    call_deadline = min(deadline, safety) if ended else safety
    with room.lock:
        inflight.add(index)
    broadcast = lambda: _in_app_context(lambda: broadcast_state(code, room))
    return pool.submit(
        _stream_analysis, cache, room.lock, position, index, call_deadline, broadcast
    )


def _analysis_worker_loop(code, room):
    """Analyse the room's already-played positions, live during the game and
    for a bounded grace period after it ends - one continuous walk rather
    than two, so nothing already in flight at the moment the game ends gets
    thrown away.

    Runs exactly two concurrent analyse_moves calls, each on its own pool
    thread (see _get_analysis_pools - the two slots get separate pools so the
    forward slot can never queue behind expensive backward work; the native
    walk releases the GIL, so the two genuinely run in parallel and stream
    their per-move results straight into the shared cache), with a fixed,
    reserved role that never borrows
    the other's slot:

    - The backward slot always extends one position further back into
      history than anything analysed so far, with no ceiling of its own -
      including the very first position ever analysed for this game: with
      nothing analysed yet, it bootstraps itself by finding the newest
      attemptable position directly (see _newest_attemptable_index), so
      it - not the forward slot - is the one that ends up owning the
      earliest analysed position and everything behind it. While the game
      is in progress the running game itself is the budget - a long,
      thoughtful game buys itself review depth a fixed gate never could.
    - The forward slot always chases the newest not-yet-analysed position
      that satisfies analysis_attemptable. In practice this only ever
      finds anything once there's a position newer than whatever the
      backward slot has already claimed - a fresh move is the only thing
      that can be attemptable and not already spoken for at that point.
      This is the slot guaranteed to never fall behind: a backward call
      can legitimately run for a very long time, and if that call were
      ever allowed to occupy *both* slots, a freshly played move would
      have to wait behind it before its own analysis could even start.
      Reserving this slot rules that out entirely, at the cost of it
      sitting idle whenever there's nothing new to catch up on (including
      once the game has ended - there will never be another new move to
      stay ready for).

    A single call unexpectedly running long only ties up its own slot, and
    _ANALYSIS_CALL_SAFETY_CAP bounds it regardless - it can't block the
    other slot's role.

    Once the game ends, room.analysis_deadline (set by
    game_flow.record_room_finished_locked) gives it _ANALYSIS_TIME_CAP more
    seconds to keep backtracking regardless of viewers, so the frozen review
    still gets filled in even if the room empties the instant the game
    finishes. Each call submitted from then on carries that deadline (and the
    safety cap, whichever is sooner) baked in up front - the native walk polls
    it and stops on its own - so a call already in flight when the deadline is
    set still gets cut off at it, rather than running to completion regardless.

    Exits when the room empties (game still in progress), the post-game
    grace period elapses, or there's nothing left to analyse.
    """
    forward_pool, backward_pool = _get_analysis_pools()
    futures = {}  # move_index -> Future, this room's own outstanding submissions
    forward_index = None  # key in `futures` currently owned by the forward slot
    backward_index = None  # key in `futures` currently owned by the backward slot
    # Which match (RoomState.game_id) the above futures - and the cache/
    # inflight below - actually belong to. A rematch can swap in a fresh
    # game_id (and reset room.analysis/room.analysis_inflight to match)
    # while a wait() below is in progress (not holding room.lock), so
    # re-reading room.analysis/room.analysis_inflight at the top of the next
    # iteration isn't safe to assume they still belong to the match
    # `futures` was submitted for - checking game_id (a single, explicit
    # equality check on the one thing that's the actual source of truth for
    # "is this still the same match") is what tells them apart.
    futures_game_id = None
    futures_cache = None
    futures_inflight = None
    try:
        while True:
            # _abandon_futures acquires room.lock itself (it's a plain,
            # non-reentrant Lock), so any exit that needs it is decided
            # inside this block but only actually called after it's
            # released, below.
            stop, broadcast_on_exit = False, False
            deadline = None  # only meaningful once ended - see _submit
            with room.lock:
                game = room.game
                game_id = room.game_id
                # An unplaced game (dealt with no marker yet, e.g. just after
                # a rematch) also has no legal moves, but it's pre-game, not
                # ended - treating it as ended drives the backward slot to a
                # negative index (undo past the start). Mirror game_over.
                ended = not game.needs_marker and not game.legal_moves
                if ended:
                    deadline = room.analysis_deadline
                    if deadline is None or time.monotonic() >= deadline:
                        stop = broadcast_on_exit = True
                else:
                    if room.analysis_deadline is not None:
                        # Leftover from a previous game a rematch has since
                        # replaced - safe to clear now since nothing can
                        # still be relying on it (see analysis_deadline's
                        # field comment for why a rematch itself never
                        # touches this).
                        room.analysis_deadline = None
                    if not room.sid_players:
                        # Nobody's watching a game that's still going - stop
                        # here rather than keep filling slots with
                        # potentially unbounded backward positions nobody
                        # will ever see. Abandon rather than wait for
                        # in-flight work to drain: while live there's no
                        # deadline bounding it, so waiting could mean
                        # waiting indefinitely.
                        stop = True
                if not stop:
                    if room.analysis is None:
                        room.analysis = {}
                    cache = room.analysis
                    inflight = room.analysis_inflight

            if stop:
                _abandon_futures(room, futures_cache, futures_inflight, futures)
                forward_index = backward_index = None
                if broadcast_on_exit:
                    _in_app_context(lambda: broadcast_state(code, room))
                return

            if futures and game_id != futures_game_id:
                # A rematch swapped in a new match while we were in wait()
                # below - `futures` is for a position from a match that's no
                # longer this room's current one. It can't contribute to
                # the *new* match's cache, so stop waiting on it rather than
                # risk harvesting a stale result into the wrong match's slot.
                _abandon_futures(room, futures_cache, futures_inflight, futures)
                forward_index = backward_index = None
            futures_game_id, futures_cache, futures_inflight = game_id, cache, inflight

            # Harvest whatever finished since the last time round.
            for index in [k for k, fut in futures.items() if fut.done()]:
                result = futures.pop(index).result()
                with room.lock:
                    inflight.discard(index)
                if result is not None:
                    cache[index] = result
                if index == forward_index:
                    forward_index = None
                if index == backward_index:
                    backward_index = None

            total = len(game.moves)
            # Live, `game` itself (index `total`, i.e. undo(0)) is always a
            # legal-move position worth analysing - there's always a next
            # move to make. Once the game has ended it isn't: it has no
            # legal moves left at all, so there's nothing for analyse_moves
            # to compare there.
            upper = total + 1 if not ended else total

            # Backward slot: one position further back into history than
            # anything analysed so far, unconditionally. With nothing
            # analysed yet at all there's no existing frontier to extend, so
            # it bootstraps one:
            # - Once the game has ended, from the last playable position
            #   (upper - 1) regardless of whether anything is "attemptable" -
            #   post-game every call is bounded by the grace deadline, so it
            #   can safely backtrack without a cost ceiling, exactly as it
            #   does when a live worker carries a frontier into the endgame.
            #   This is what lets a game that ended early (the marker trapped
            #   with only deep, un-attemptable positions left, so live
            #   analysis never got started) still get filled in afterwards
            #   rather than the worker finding nothing to do and exiting.
            # - While still live, from the newest attemptable position (the
            #   same search the forward slot uses below), so it waits for a
            #   position to come within the affordable ceiling rather than
            #   kicking off unbounded work mid-game.
            if backward_index is None:
                done = cache.keys() | inflight
                if done:
                    frontier = min(done)
                    index = frontier - 1 if frontier > 0 else None
                elif ended:
                    index = upper - 1
                else:
                    index = _newest_attemptable_index(game, total, done, upper)
                if index is not None:
                    futures[index] = _submit(
                        backward_pool, code, room, cache, inflight, game, index, ended, deadline
                    )
                    backward_index = index

            # Forward slot: only the newest not-yet-done attemptable
            # position - never falls back to backtracking, so it's always
            # free the instant a fresh move needs analysing rather than
            # waiting on whatever the backward slot is doing. In practice
            # this only ever finds anything once there's a position newer
            # than whatever the backward slot has already claimed above -
            # a fresh move is the only thing that can be attemptable and
            # not already spoken for at that point.
            if forward_index is None:
                done = cache.keys() | inflight
                index = _newest_attemptable_index(game, total, done, upper)
                if index is not None:
                    futures[index] = _submit(
                        forward_pool, code, room, cache, inflight, game, index, ended, deadline
                    )
                    forward_index = index

            if not futures:
                if ended:
                    _in_app_context(lambda: broadcast_state(code, room))
                    return
                # Either the game hasn't reached the sensible starting
                # point yet, or every position back to the deal is done.
                time.sleep(1.0)
                continue

            wait(futures.values(), timeout=1.0, return_when=FIRST_COMPLETED)
    finally:
        _abandon_futures(room, futures_cache, futures_inflight, futures)
        with room.lock:
            room.analysis_running = False


def start_analysis_worker_locked(code, room):
    """Start the analysis worker if it isn't already running and there's
    currently a reason to: live viewers for a game in progress, or still
    within the post-game grace period for one that just ended. Caller must
    already hold room.lock (see ensure_analysis_worker for the version
    that acquires it, and game_flow.record_room_finished_locked, which is
    already holding it when the game ends)."""
    if room.analysis_running or not room.game.moves:
        return
    if room.game.legal_moves:
        if not room.sid_players:
            return
    elif room.analysis_deadline is None or time.monotonic() >= room.analysis_deadline:
        return
    room.analysis_running = True
    threading.Thread(target=_analysis_worker_loop, args=(code, room), daemon=True).start()


def ensure_analysis_worker(code, room):
    """Start the room's analysis worker if there's currently a reason to
    and none is running - see start_analysis_worker_locked. Callers invoke
    this after joins and moves; it's a cheap no-op otherwise."""
    with room.lock:
        start_analysis_worker_locked(code, room)
