"""The post-game move-analysis worker and its thread pools.

A single worker per room (see _analysis_worker_loop) analyses already-played
positions during the game and for a bounded grace period after it ends,
streaming per-move results into the room's shared analysis cache and
broadcasting them as they land. Explicit "Calculate" clicks go through
start_ondemand_analysis on a separate pool.

Depends on rooms (room-state model), rendering (broadcast_state), and
analysis_policy (feasibility predicates).
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


# Hard per-call backstop, so a bad cost estimate can only tie up its own slot
# for a bounded time.
_ANALYSIS_CALL_SAFETY_CAP = 30 * 60

# Deadline for an explicit "Calculate" request - a much longer leash than the
# unattended worker budget, since the user is opting into the wait.
_ONDEMAND_ANALYSIS_DEADLINE = 30 * 60

# Shared pools running analyse_moves calls. Threads, not processes: the native
# solver releases the GIL, so calls genuinely parallelise and can stream
# per-move results into the room cache as they land. The forward and backward
# slots get separate pools so forward work never queues behind expensive
# backward work; a third pool serves "Calculate" requests in isolation. Floored
# at two workers each so both roles get a slot on a single-CPU host.
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
    """Run fn inside the Flask app context (needed for render_template on a
    background thread). No-op if the app hasn't been constructed."""
    app = app_holder.get("app")
    if app is None:
        return
    with app.app_context():
        fn()


def _stream_analysis(cache, lock, position, index, deadline, broadcast):
    """Analyse one position, folding results into cache[index] and calling
    `broadcast` after each update. The per-position work unit both the
    automatic worker and the on-demand path submit to the pool.

    Native path streams one root move at a time; fallback path computes the
    whole dict and writes it once. Returns the finished move_data, or None if
    the deadline cut it short (any streamed partial is left in place).
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
                    # Distinct shape from a finished move_data (keyed by (row, col)).
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
    """Kick off a one-off analysis of a single already-played position the
    automatic worker gave up on, on a background thread with the longer
    _ONDEMAND_ANALYSIS_DEADLINE. Returns True if a calculation was started,
    False if `index` is already cached or already inflight.

    `cache`/`inflight`/`calc_started`/`lock` are a live room's or a frozen
    finished_rooms entry's analysis fields plus their guarding lock, passed in
    so this one function serves both. `on_done`, if given, runs in a Flask app
    context once the result has been folded in, to push a fresh render.
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
        # _stream_analysis folds results into cache itself; just wait for it to
        # settle before clearing inflight and pushing the final render.
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
    """Newest index in range(upper) not in `done` whose position satisfies
    analysis_attemptable, or None. Shared by the backward slot's bootstrap and
    the forward slot's catch-up scan."""
    for k in reversed(range(upper)):
        if k in done:
            continue
        if analysis_attemptable(game.undo(total - k)):
            return k
    return None


def _abandon_futures(room, cache, inflight, futures):
    """Stop tracking this room's outstanding submissions - called when giving
    up on a game's epoch (a rematch swapped in a new game, or the worker is
    exiting). Finished futures are harvested into `cache`; the rest are
    cancelled (a running task runs on to its own deadline). `inflight` is
    always cleared so a later worker can pick the same index up again."""
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
    """Submit the streaming analysis of the position `index` moves into `game`,
    marking it inflight. Shared by both reserved slots; results are folded into
    `cache` and pushed to viewers as they land."""
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
    """Analyse the room's already-played positions, live and for a bounded
    grace period after the game ends - one continuous walk, so in-flight work
    isn't thrown away when the game ends.

    Runs two concurrent analyse_moves calls, each on its own pool thread with a
    fixed role:
    - Backward slot: always extends one position further back than anything
      analysed so far, with no ceiling. Bootstraps from the newest attemptable
      position when nothing is analysed yet.
    - Forward slot: always chases the newest not-yet-analysed attemptable
      position, so it's free the instant a fresh move needs analysing rather
      than waiting on the backward slot.

    _ANALYSIS_CALL_SAFETY_CAP bounds any single call. Once the game ends,
    room.analysis_deadline gives it _ANALYSIS_TIME_CAP more seconds to
    backtrack regardless of viewers; each call submitted then carries that
    deadline. Exits when the room empties (live game), the grace period
    elapses, or there's nothing left to analyse.
    """
    forward_pool, backward_pool = _get_analysis_pools()
    futures = {}  # move_index -> Future, this room's outstanding submissions
    forward_index = None  # key in `futures` owned by the forward slot
    backward_index = None  # key in `futures` owned by the backward slot
    # Which match (game_id) `futures`, cache, and inflight belong to. A rematch
    # can swap in a fresh game_id during a wait() below, so this equality check
    # tells a stale set apart from the current match.
    futures_game_id = None
    futures_cache = None
    futures_inflight = None
    try:
        while True:
            # _abandon_futures acquires room.lock itself, so any exit needing it
            # is decided in this block but called after it's released, below.
            stop, broadcast_on_exit = False, False
            deadline = None  # only meaningful once ended - see _submit
            with room.lock:
                game = room.game
                game_id = room.game_id
                # An unplaced game has no legal moves but is pre-game, not
                # ended; treating it as ended would drive the backward slot to
                # a negative index. Mirror game_over.
                ended = not game.needs_marker and not game.legal_moves
                if ended:
                    deadline = room.analysis_deadline
                    if deadline is None or time.monotonic() >= deadline:
                        stop = broadcast_on_exit = True
                else:
                    if room.analysis_deadline is not None:
                        # Leftover from a game a rematch has replaced.
                        room.analysis_deadline = None
                    if not room.sid_players:
                        # Nobody watching a live game - stop rather than fill
                        # slots with unbounded backward work nobody will see.
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
                # A rematch swapped in a new match during wait() - `futures`
                # can't contribute to the new match's cache, so drop it.
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
            # Live, `game` itself (index `total`) is worth analysing - there's a
            # next move to make. Ended, it has no legal moves to compare.
            upper = total + 1 if not ended else total

            # Backward slot: one position further back than anything analysed,
            # unconditionally. With nothing analysed yet it bootstraps - from
            # the last playable position once ended (post-game calls are
            # deadline-bounded, so no cost ceiling needed), or from the newest
            # attemptable position while still live.
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

            # Forward slot: only the newest not-yet-done attemptable position;
            # never backtracks, so it's free the instant a fresh move arrives.
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
                # Not at the starting point yet, or everything back to the deal is done.
                time.sleep(1.0)
                continue

            wait(futures.values(), timeout=1.0, return_when=FIRST_COMPLETED)
    finally:
        _abandon_futures(room, futures_cache, futures_inflight, futures)
        with room.lock:
            room.analysis_running = False


def start_analysis_worker_locked(code, room):
    """Start the analysis worker if it isn't running and there's a reason to:
    live viewers for a game in progress, or still within the post-game grace
    period. Caller must hold room.lock (see ensure_analysis_worker to acquire
    it)."""
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
    """Start the room's analysis worker if there's a reason to and none is
    running - see start_analysis_worker_locked. Called after joins and moves;
    a cheap no-op otherwise."""
    with room.lock:
        start_analysis_worker_locked(code, room)
