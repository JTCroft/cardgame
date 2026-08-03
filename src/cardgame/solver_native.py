"""solve_native: cardgame.solver.solve backed by the Rust core.

The native search replicates solve()'s deterministic order and window
arithmetic exactly, so the returned dict is field-for-field identical -
including the upper-bound values for pruned rival moves. Requires the
optional cardgame-native package (`pip install ./native`); the pure
Python solver remains the reference implementation.
"""

from .solver import SIGN_SCALE, _root_state, _FACT

try:
    from cardgame_native import solve_root as _solve_root
    from cardgame_native import solve_id_root as _solve_id_root
except ImportError:
    _solve_root = None
    _solve_id_root = None

NATIVE_AVAILABLE = _solve_root is not None

__all__ = ("solve_native", "solve_id_native", "NATIVE_AVAILABLE")


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


def solve_native(game, deadline=None):
    """Exact solve of `game`. With `deadline` (seconds), the search aborts once
    it elapses and returns None instead of a result - the caller keeps whatever
    move it already had. Without a deadline the solve is exhaustive (the search
    is bit-identical to the deadline-free path when it is not aborted)."""
    return _run(_solve_root, game, deadline)


def solve_id_native(game, deadline=None):
    """Exact solve via iterative deepening + gate/bound cache (port of
    experiments/id_best). Faster than solve_native on deeper, facedown-heavy
    positions. The best move and its value are identical to solve_native;
    rival-move values are valid upper bounds but may differ numerically."""
    return _run(_solve_id_root, game, deadline)
