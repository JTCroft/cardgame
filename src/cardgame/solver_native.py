"""Native-backed exact solvers: `best_move` and `evaluate_native`.

The native exact solver is an iterative-deepening search with a gate and a
bound cache (port of experiments/id_best). It agrees with the pure-Python
`solver._solve_python` reference on the best move and its value; rival-move
values are valid upper bounds but, being order- and deepening-dependent, may
differ numerically. Requires the optional cardgame-native package (`pip
install ./native`); the pure-Python solver remains the reference / fallback.
"""

from .game import ProbEval
from .solver import SIGN_SCALE, _root_state, _FACT, _solve_python

try:
    from cardgame_native import solve_root as _solve_root
    from cardgame_native import evaluate_root as _evaluate_root
except ImportError:
    _solve_root = None
    _evaluate_root = None

NATIVE_AVAILABLE = _solve_root is not None

__all__ = ("best_move", "evaluate_native", "NATIVE_AVAILABLE")


def evaluate_native(game, deadline=None):
    """Exact value of `game` as (ProbEval, best_marker) - the score-difference
    distribution under optimal play in the mover's own perspective, plus the
    best move (None at a terminal). One native `evaluate_root` walk, using the
    same (2w+d, w, s)-then-marker selection Game.evaluate uses. Backs
    Game.evaluate on the native path; returns (None, None) if `deadline` trips.
    """
    if _evaluate_root is None:
        raise ImportError(
            "cardgame-native is not installed - build it with "
            "`uv pip install ./native` (requires a Rust toolchain)"
        )
    cells, cell, rows, cols, mi, mk, oi, ok, unknowns = _root_state(game)
    result = _evaluate_root(
        [-1 if c is None else c for c in cells],
        cell, rows, cols, mi, mk, oi, ok, list(unknowns), deadline,
    )
    if result is None:               # deadline tripped before completion
        return None, None
    marker, (_m, _w, _d, _s), pairs = result
    prob = ProbEval(
        multiplicity=game.multiplicity,
        initial_counts={diff: weight for diff, weight in pairs},
    )
    return prob, (tuple(marker) if marker is not None else None)


def _run(engine, game, deadline):
    if engine is None:
        raise ImportError(
            "cardgame-native is not installed - build it with "
            "`uv pip install ./native` (requires a Rust toolchain)"
        )
    cells, cell, rows, cols, mi, mk, oi, ok, unknowns = _root_state(game)
    result = engine(
        [-1 if c is None else c for c in cells],
        cell, rows, cols, mi, mk, oi, ok, list(unknowns), deadline,
    )
    if result is None:               # deadline tripped before completion
        return None
    marker, (p, s), moves = result
    return {
        "marker": marker,
        "value": p * SIGN_SCALE + s,
        "sign_sum": p,
        "score_sum": s,
        "multiplicity": _FACT[len(unknowns)],
        "moves": {
            mk_: (mp * SIGN_SCALE + ms, exact) for mk_, (mp, ms), exact in moves
        },
    }


def best_move(game, deadline=None):
    """Exact best move for `game`, as a dict {marker, value, sign_sum,
    score_sum, multiplicity, moves} (value ranks by 2w+d then score-sum).
    THE route when you want the move to play.

    Native-backed by the iterative-deepening + gate engine, falling back to the
    pure-Python `_solve_python` reference when the native core is unavailable.
    A `deadline` (seconds) requires the native core - the pure-Python search
    can't be interrupted - and the call returns None if it trips before
    finishing (the caller keeps whatever it already had). The best move and its
    value match the pure-Python reference; rival-move values are valid upper
    bounds that may differ numerically.

    For the full outcome distribution use `Game.evaluate`; for one specific
    move's exact value use `analysis.move_value`."""
    if NATIVE_AVAILABLE:
        return _run(_solve_root, game, deadline)
    if deadline is not None:
        return None  # the pure-Python search cannot be interrupted
    return _solve_python(game)
