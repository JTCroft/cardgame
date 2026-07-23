"""Rust-backed per-move analysis: analysis.analyse_moves via the native core.

Mirrors analysis.py field-for-field, but drives the walk one root move at a
time through cardgame_native (analyse_move for each move's exact aggregate,
distribution for the winner's outcome histogram). The per-move granularity
is what lets iter_move_analyses stream results to the UI as each move lands;
analyse_moves_native just drains that generator into the same dict the pure-
Python analyse_moves returns. Requires the optional cardgame-native package;
analysis.py keeps the pure-Python implementation as the reference fallback.
"""

import time
from collections import Counter

from .game import Eval
from .solver import _root_state, _legal_cells

try:
    from cardgame_native import analyse_move as _analyse_move, distribution as _distribution
except ImportError:
    _analyse_move = _distribution = None

NATIVE_AVAILABLE = _analyse_move is not None

# Sentinel yielded by iter_move_analyses once the full, finalised move_data
# (best move, deltas, winner distribution) is ready - distinct from any
# (row, col) marker.
FINAL = object()

__all__ = (
    "analyse_moves_native",
    "iter_move_analyses",
    "move_eval_native",
    "NATIVE_AVAILABLE",
    "FINAL",
)


def move_eval_native(game, marker, deadline=None):
    """Exact Eval(m, w, d, s) for a single legal move, current player's
    perspective - the native per-move aggregate without the full-slate walk
    iter_move_analyses does. Used to break the exact solver's (2w+d, s) ties
    by Game.evaluate's full (2w+d, w, s) order (see ai._exact_move)."""
    if _analyse_move is None:
        raise ImportError("cardgame-native is not installed")
    cells, cell, rows, cols, mi, mk, oi, ok, unknowns = _root_state(game)
    cells_i = [-1 if c is None else c for c in cells]
    target = marker[0] * 6 + marker[1]
    agg = _analyse_move(
        cells_i, target, rows, cols, mi, mk, oi, ok, list(unknowns), _remaining(deadline)
    )
    m, w, d, s, _mover_sum = agg
    return Eval(m, w, d, s)


def _remaining(deadline):
    """Seconds left until `deadline` (a time.monotonic() value), or None for
    an unbounded call. Raised into AnalysisAborted by the caller once <= 0."""
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _move_record(game, marker, agg):
    """The absolute per-move stats analyse_moves reports, from one native
    aggregate (m, w, d, s, mover_sum) in the analysed player's perspective.
    The relative fields (offensive/defensive/combined/best) are filled in by
    _apply_deltas once every move is known."""
    m, w, d, s, mover_sum = agg
    return {
        "card": game.board[marker[0]][marker[1]],
        "player_mean": mover_sum / m,
        "opponent_mean": (mover_sum - s) / m,
        "mean_diff": s / m,
        "distribution": None,  # winner only, filled in at the end
        "win_pct": 100 * w / m,
        "draw_pct": 100 * d / m,
        "loss_pct": 100 * (m - w - d) / m,
        "eval": Eval(m, w, d, s),
    }


def _best_marker(move_data):
    """The best move by the same (eval, marker) tie-break analyse_moves and
    the rest of the codebase use."""
    best_marker = best_key = None
    for marker, data in move_data.items():
        candidate = (data["eval"], marker)
        if best_key is None or candidate > best_key:
            best_key, best_marker = candidate, marker
    return best_marker


def _apply_deltas(move_data):
    """Fill in each move's offensive/defensive/combined deltas (against the
    best move) and its `best` flag - identical to analyse_moves' finalisation.
    Safe to call on a partial move_data for provisional streaming display:
    the deltas are simply relative to the best move seen so far."""
    best = move_data[_best_marker(move_data)]
    baseline_player = best["player_mean"]
    baseline_opponent = best["opponent_mean"]
    baseline_diff = best["mean_diff"]
    for data in move_data.values():
        data["offensive"] = data["opponent_mean"] - baseline_opponent
        data["defensive"] = data["player_mean"] - baseline_player
        data["combined"] = data["mean_diff"] - baseline_diff
        data["best"] = data["eval"] == best["eval"]


def iter_move_analyses(game, deadline=None):
    """Solve each legal move in turn, yielding (marker, move_data-so-far) as
    each one lands, then (FINAL, move_data) once the best move, deltas and
    the winner's outcome distribution are all in.

    Each intermediate yield carries the full accumulated dict with
    provisional deltas applied (the best move can still change as later moves
    arrive - see _apply_deltas), so a streaming caller can render a live,
    re-sorting table. The final yield is field-for-field what analyse_moves
    returns.

    Raises AnalysisAborted if `deadline` passes mid-walk; whatever was
    already yielded stays valid (the caller keeps it).
    """
    from .analysis import AnalysisAborted

    if _analyse_move is None:
        raise ImportError("cardgame-native is not installed")
    if not game.legal_moves:
        raise ValueError("Game is already over — no legal moves to analyse.")

    cells, cell, rows, cols, mi, mk, oi, ok, unknowns = _root_state(game)
    cells_i = [-1 if c is None else c for c in cells]
    uk = list(unknowns)
    move_data = {}
    for target in _legal_cells(cell, rows, cols):
        budget = _remaining(deadline)
        if budget is not None and budget <= 0:
            raise AnalysisAborted
        agg = _analyse_move(cells_i, target, rows, cols, mi, mk, oi, ok, uk, budget)
        if agg is None:
            raise AnalysisAborted
        marker = divmod(target, 6)
        move_data[marker] = _move_record(game, marker, agg)
        _apply_deltas(move_data)
        yield marker, move_data

    best_marker = _best_marker(move_data)
    budget = _remaining(deadline)
    if budget is not None and budget <= 0:
        raise AnalysisAborted
    best_target = best_marker[0] * 6 + best_marker[1]
    dist = _distribution(cells_i, best_target, rows, cols, mi, mk, oi, ok, uk, budget)
    if dist is None:
        raise AnalysisAborted
    move_data[best_marker]["distribution"] = Counter({diff: w for diff, w in dist})
    yield FINAL, move_data


def analyse_moves_native(game, deadline=None):
    """Drain iter_move_analyses into the finalised move_data dict - the same
    return value as analysis.analyse_moves, computed natively."""
    final = None
    for marker, data in iter_move_analyses(game, deadline):
        if marker is FINAL:
            final = data
    return final
