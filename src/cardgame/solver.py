"""Exact best-move solver: expectiminimax on an ordered-group value.

Replaces distribution-tuple comparison for move selection. Every node
value is one integer, summed over the leaves of its subtree:

    value = sum( m_leaf * (SIGN_SCALE * sign(diff) + diff) )

where m_leaf is the leaf's chance multiplicity and diff the final score
difference from the leaf mover's perspective. Ranking by this value is
ranking by win/draw/loss expectation first (equivalent to the 2w+d term
of Game.evaluate's (2w+d, w, s) order, since 2w+d = sign_sum + m), with
the cumulative score as tie-break. Unlike (2w+d, w, s), integers form an
ordered abelian group - negation reverses order and addition preserves
it - so negamax alpha-beta and Star1 chance-node windows are provably
sound under any move ordering. The (2w+d, w, s) middle tie-break term is
deliberately dropped: it is what made the old comparator unsound.
"""

from math import factorial
from .cards import Rank
from .game import _cached_score

__all__ = ("solve", "solve_plain", "decode", "SIGN_SCALE")

# Must exceed 2 * 26 * 12! so any sign-count difference outweighs any
# possible score-sum difference at the maximum root multiplicity.
SIGN_SCALE = 1 << 35
_LEAF_UNIT_BOUND = SIGN_SCALE + 26
_FACT = tuple(factorial(n) for n in range(13))
_COL_BIT = tuple(1 << ((i % 6) * 6 + i // 6) for i in range(36))

# (marker_cell, row occupancy, col occupancy) -> tuple of target cells,
# face-up ordering applied per node since it depends on the board.
_LEGAL = {}


def _legal_cells(cell, rows, cols):
    row, col = divmod(cell, 6)
    row_bits = ((rows >> (row * 6)) | (1 << col)) & 63
    col_bits = ((cols >> (col * 6)) | (1 << row)) & 63
    key = (cell << 12) | (row_bits << 6) | col_bits
    cells = _LEGAL.get(key)
    if cells is None:
        base = row * 6
        cells = tuple(
            [base + j for j in range(6) if not row_bits >> j & 1]
            + [i * 6 + col for i in range(6) if not col_bits >> i & 1]
        )
        _LEGAL[key] = cells
    return cells


def _leaf(mi, mk, oi, ok, n_unknown):
    diff = _cached_score(mi, mk) - _cached_score(oi, ok)
    m = _FACT[n_unknown]
    if diff > 0:
        return m * SIGN_SCALE + m * diff
    if diff < 0:
        return -m * SIGN_SCALE + m * diff
    return 0


def _walk(cells, cell, rows, cols, mi, mk, oi, ok, unknowns):
    """Exhaustive expectiminimax reference - no pruning."""
    legal = _legal_cells(cell, rows, cols)
    if not legal:
        return _leaf(mi, mk, oi, ok, len(unknowns))
    best = None
    for target in legal:
        nrows = rows | 1 << target
        ncols = cols | _COL_BIT[target]
        payload = cells[target]
        if payload is None:
            v = 0
            for i in range(len(unknowns)):
                card = unknowns[i]
                rest = unknowns[:i] + unknowns[i + 1 :]
                if card:
                    v -= _walk(cells, target, nrows, ncols, oi, ok, mi | card, mk, rest)
                else:
                    v -= _walk(cells, target, nrows, ncols, oi, ok, mi, mk + 1, rest)
        elif payload:
            v = -_walk(cells, target, nrows, ncols, oi, ok, mi | payload, mk, unknowns)
        else:
            v = -_walk(cells, target, nrows, ncols, oi, ok, mi, mk + 1, unknowns)
        if best is None or v > best:
            best = v
    return best


def _search(cells, cell, rows, cols, mi, mk, oi, ok, unknowns, alpha, beta):
    """Fail-soft negamax: exact within (alpha, beta), a lower bound if
    >= beta, an upper bound if <= alpha."""
    legal = _legal_cells(cell, rows, cols)
    if not legal:
        return _leaf(mi, mk, oi, ok, len(unknowns))
    best = None
    for target in legal:
        nrows = rows | 1 << target
        ncols = cols | _COL_BIT[target]
        payload = cells[target]
        if payload is None:
            v = _chance(
                cells, target, nrows, ncols, mi, mk, oi, ok, unknowns, alpha, beta
            )
        elif payload:
            v = -_search(
                cells, target, nrows, ncols, oi, ok, mi | payload, mk, unknowns,
                -beta, -alpha,
            )
        else:
            v = -_search(
                cells, target, nrows, ncols, oi, ok, mi, mk + 1, unknowns,
                -beta, -alpha,
            )
        if best is None or v > best:
            best = v
            if v >= beta:
                return v
            if v > alpha:
                alpha = v
    return best


def _chance(cells, target, nrows, ncols, mi, mk, oi, ok, unknowns, alpha, beta):
    """Star1: each resolution searched in the window that could still
    swing the summed total across (alpha, beta), given +/- bounds for
    the unresolved siblings."""
    n = len(unknowns)
    child_bound = _FACT[n - 1] * _LEAF_UNIT_BOUND
    done = 0
    for i in range(n):
        slack = (n - 1 - i) * child_bound
        a_i = alpha - done - slack
        b_i = beta - done + slack
        card = unknowns[i]
        rest = unknowns[:i] + unknowns[i + 1 :]
        if card:
            r = -_search(
                cells, target, nrows, ncols, oi, ok, mi | card, mk, rest, -b_i, -a_i
            )
        else:
            r = -_search(
                cells, target, nrows, ncols, oi, ok, mi, mk + 1, rest, -b_i, -a_i
            )
        if r >= b_i:
            return done + r - slack
        if r <= a_i:
            return done + r + slack
        done += r
    return done


def _search_o(cells, cell, rows, cols, mi, mk, oi, ok, unknowns, alpha, beta):
    """_search with heuristic ordering: face-up moves by the mover's
    marginal score gain (cached DP probes), chance moves last. Sound for
    any ordering - this only affects speed."""
    legal = _legal_cells(cell, rows, cols)
    if not legal:
        return _leaf(mi, mk, oi, ok, len(unknowns))
    base = _cached_score(mi, mk)
    face = []
    chance = []
    for target in legal:
        payload = cells[target]
        if payload is None:
            chance.append(target)
        elif payload:
            face.append((base - _cached_score(mi | payload, mk), target))
        else:
            face.append((base - _cached_score(mi, mk + 1), target))
    face.sort()
    best = None
    for _, target in face:
        payload = cells[target]
        if payload:
            v = -_search_o(
                cells, target, rows | 1 << target, cols | _COL_BIT[target],
                oi, ok, mi | payload, mk, unknowns, -beta, -alpha,
            )
        else:
            v = -_search_o(
                cells, target, rows | 1 << target, cols | _COL_BIT[target],
                oi, ok, mi, mk + 1, unknowns, -beta, -alpha,
            )
        if best is None or v > best:
            best = v
            if v >= beta:
                return v
            if v > alpha:
                alpha = v
    for target in chance:
        v = _chance_o(
            cells, target, rows | 1 << target, cols | _COL_BIT[target],
            mi, mk, oi, ok, unknowns, alpha, beta,
        )
        if best is None or v > best:
            best = v
            if v >= beta:
                return v
            if v > alpha:
                alpha = v
    return best


def _chance_o(cells, target, nrows, ncols, mi, mk, oi, ok, unknowns, alpha, beta):
    n = len(unknowns)
    child_bound = _FACT[n - 1] * _LEAF_UNIT_BOUND
    done = 0
    for i in range(n):
        slack = (n - 1 - i) * child_bound
        a_i = alpha - done - slack
        b_i = beta - done + slack
        card = unknowns[i]
        rest = unknowns[:i] + unknowns[i + 1 :]
        if card:
            r = -_search_o(
                cells, target, nrows, ncols, oi, ok, mi | card, mk, rest, -b_i, -a_i
            )
        else:
            r = -_search_o(
                cells, target, nrows, ncols, oi, ok, mi, mk + 1, rest, -b_i, -a_i
            )
        if r >= b_i:
            return done + r - slack
        if r <= a_i:
            return done + r + slack
        done += r
    return done


def _root_state(game):
    board = game.board
    cells = [None] * 36
    for r in range(6):
        row = board[r]
        for c in range(6):
            card = row[c]
            if card.facedown:
                continue
            if card[0] is Rank.K:
                cells[r * 6 + c] = 0
            else:
                cells[r * 6 + c] = 1 << ((card[1] * 8) + card[0] - 1)
    rows = cols = 0
    for r, c in game.moves:
        rows |= 1 << (r * 6 + c)
        cols |= 1 << (c * 6 + r)
    mi, mk, oi, ok = game._hand_state()
    unknowns = tuple(
        0 if card[0] is Rank.K else 1 << ((card[1] * 8) + card[0] - 1)
        for card in board.facedown_cards
    )
    mr, mc = game.marker
    return cells, mr * 6 + mc, rows, cols, mi, mk, oi, ok, unknowns


def decode(value):
    """value -> (sign_sum, score_sum). 2w+d = sign_sum + multiplicity."""
    p = (value + (SIGN_SCALE >> 1)) // SIGN_SCALE
    return p, value - p * SIGN_SCALE


def _finish(best_v, best_marker, moves, multiplicity):
    p, s = decode(best_v) if best_v is not None else (None, None)
    return {
        "marker": best_marker,
        "value": best_v,
        "sign_sum": p,
        "score_sum": s,
        "multiplicity": multiplicity,
        "moves": moves,
    }


def solve_plain(game):
    """Exhaustive root solve - every move's exact value, no pruning."""
    cells, cell, rows, cols, mi, mk, oi, ok, unknowns = _root_state(game)
    legal = _legal_cells(cell, rows, cols)
    m = _FACT[len(unknowns)]
    if not legal:
        return _finish(_leaf(mi, mk, oi, ok, len(unknowns)), None, {}, m)
    best_v = best_marker = None
    moves = {}
    for target in legal:
        nrows = rows | 1 << target
        ncols = cols | _COL_BIT[target]
        payload = cells[target]
        if payload is None:
            v = 0
            for i in range(len(unknowns)):
                card = unknowns[i]
                rest = unknowns[:i] + unknowns[i + 1 :]
                if card:
                    v -= _walk(cells, target, nrows, ncols, oi, ok, mi | card, mk, rest)
                else:
                    v -= _walk(cells, target, nrows, ncols, oi, ok, mi, mk + 1, rest)
        elif payload:
            v = -_walk(cells, target, nrows, ncols, oi, ok, mi | payload, mk, unknowns)
        else:
            v = -_walk(cells, target, nrows, ncols, oi, ok, mi, mk + 1, unknowns)
        marker = divmod(target, 6)
        moves[marker] = (v, True)
        if best_v is None or (v, marker) > (best_v, best_marker):
            best_v, best_marker = v, marker
    return _finish(best_v, best_marker, moves, m)


def solve(game, ordered=False):
    """Pruned root solve. The best move's value is exact; rival moves
    carry (value, exact_flag) - a non-exact value is an upper bound.
    Root alpha sits one below the incumbent so equal-valued rivals stay
    exact and the (value, marker) tie-break matches solve_plain.
    ordered=True uses heuristic move ordering - measured neutral overall
    (wins ~15-25% on most positions, loses similar on some), kept for
    experimentation."""
    cells, cell, rows, cols, mi, mk, oi, ok, unknowns = _root_state(game)
    legal = _legal_cells(cell, rows, cols)
    m = _FACT[len(unknowns)]
    if not legal:
        return _finish(_leaf(mi, mk, oi, ok, len(unknowns)), None, {}, m)
    search = _search_o if ordered else _search
    chance = _chance_o if ordered else _chance
    inf = m * _LEAF_UNIT_BOUND + 1
    best_v = best_marker = None
    moves = {}
    # Face-up moves first (cheap subtrees narrow the window before any
    # chance move is expanded); heuristic mode also sorts them by the
    # mover's marginal score gain.
    if ordered:
        base = _cached_score(mi, mk)
        key = lambda t: (
            (2, 0)
            if cells[t] is None
            else (1, base - _cached_score(mi | cells[t], mk))
            if cells[t]
            else (1, base - _cached_score(mi, mk + 1))
        )
    else:
        key = lambda t: cells[t] is None
    for target in sorted(legal, key=key):
        alpha = -inf if best_v is None else best_v - 1
        nrows = rows | 1 << target
        ncols = cols | _COL_BIT[target]
        payload = cells[target]
        if payload is None:
            v = chance(
                cells, target, nrows, ncols, mi, mk, oi, ok, unknowns, alpha, inf
            )
        elif payload:
            v = -search(
                cells, target, nrows, ncols, oi, ok, mi | payload, mk, unknowns,
                -inf, -alpha,
            )
        else:
            v = -search(
                cells, target, nrows, ncols, oi, ok, mi, mk + 1, unknowns,
                -inf, -alpha,
            )
        marker = divmod(target, 6)
        moves[marker] = (v, best_v is None or v > alpha)
        if best_v is None or (v, marker) > (best_v, best_marker):
            best_v, best_marker = v, marker
    return _finish(best_v, best_marker, moves, m)
