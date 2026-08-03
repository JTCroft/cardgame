"""A computer opponent for Cross Kings.

`AlphaBetaBot` plays in two regimes:

* **Endgame** — once the exact solver's measured worst case for the
  position's (cards left, face-down) cell fits the per-move budget, it
  defers to `cardgame.best_move` (native, pure-Python fallback), which
  optimises win/draw/loss expectation directly - reaching exact play
  from ~16-19 cards at normal budgets.
* **Opening/midgame** — iterative-deepening expectiminimax under a time
  budget: negamax with alpha-beta at decision nodes, expectation with
  Star1 cutoffs at face-down (chance) nodes — each resolution searched
  only in the window of contributions that could still move the
  expectation into the parent's (alpha, beta) — and a heuristic leaf
  evaluation built from:

  - the score difference between the two hands (via the scoring DP),
  - "potential": the discounted marginal score of every card still on the
    board against each hand — taking a card the opponent needs denies it, so
    denial is valued automatically,
  - rank centrality, decaying as the game progresses (middle ranks fit into
    more runs, worth more early),
  - a small mobility term, and a terminal win/loss bonus so the bot steers
    towards ending the game while ahead.

Every behavioral tunable lives in `SearchParams`; a bot is an `AlphaBetaBot`
(params + time budget), so variants can be matched against each other via
`cardgame.validation.arena`. Module-level `choose_move` is the default-param
convenience API.

Averaging over the remaining face-down cards is legitimate card counting:
the deck and every face-up card are public, so both players can deduce the
face-down multiset — no knowledge of which cell hides which card is used.
"""

import time
from dataclasses import dataclass
from functools import lru_cache

from .analysis import move_value
from .cards import Card, Rank
from .scoring import score_dp
from .solver import _root_state, decode
from .solver_native import best_move, NATIVE_AVAILABLE

try:
    from cardgame_native import heuristic_root as _heuristic_root
except ImportError:
    _heuristic_root = None

try:
    from cardgame_native import heuristic_placement as _heuristic_placement
except ImportError:
    _heuristic_placement = None

HEURISTIC_NATIVE = _heuristic_root is not None

__all__ = ("choose_move", "choose_placement", "AlphaBetaBot", "SearchParams")

_score = lru_cache(maxsize=1 << 18)(score_dp)

_RANK_MASK = 0x01010101  # one bit per suit for a single rank


def _centrality_table():
    # How many runs of length 3-8 include each rank, scaled to 0..1.
    counts = [
        sum(
            1
            for length in range(3, 9)
            for start in range(1, 10 - length)
            if start <= rank <= start + length - 1
        )
        for rank in range(1, 9)
    ]
    lo, hi = min(counts), max(counts)
    return tuple((c - lo) / (hi - lo) for c in counts)


_CENTRALITY = _centrality_table()
# King centrality inside the *move-ordering* key only: the _ORDER_*
# weights below were jointly fitted with kings pinned at 1.0, so this
# stays a constant alongside them. The evaluation's king centrality is
# SearchParams.king_centrality (same 1.0 default, independently tunable).
_KING_CENTRALITY = 1.0

# Move-ordering score weights, fitted by within-position least squares on
# oracle-labeled positions (cardgame.validation.oracle). Ordering never
# changes what a search returns, only how early it cuts. `replies` is how
# many moves the opponent is left with (restricting mobility orders well).
_ORDER_ME = 0.5883
_ORDER_OPP = 0.1114
_ORDER_CENT = 2.8549
_ORDER_KING = 0.2874
_ORDER_FD = -0.1927
_ORDER_REPLIES = -0.4922

# Bitmask of the cells reachable from each marker (its row and column,
# self excluded), so opponent reply counts cost one AND + popcount
# against the taken-cell mask.
_REPLY_MASK = {
    (i, j): sum(
        1 << (r * 6 + c)
        for r, c in (
            {(i, j2) for j2 in range(6)} | {(i2, j) for i2 in range(6)}
        )
        - {(i, j)}
    )
    for i in range(6)
    for j in range(6)
}

# Hands are carried through the search as (hand_int, num_kings) pairs in the
# same encoding as Hand.as_int, so scores come from the cached DP directly.


def _add_card(hand, card):
    hand_int, kings = hand
    if card[0] is Rank.K:
        return hand_int, kings + 1
    return hand_int | (1 << (card[1] * 8 + card[0] - 1)), kings


def _marginal(hand, card):
    """Points added to `hand`'s score by acquiring `card` (never negative)."""
    return _score(*_add_card(hand, card)) - _score(*hand)


@lru_cache(maxsize=1 << 16)
def _centrality_sum(hand, king_centrality=1.0):
    hand_int, kings = hand
    total = kings * king_centrality
    for rank in range(8):
        total += _CENTRALITY[rank] * ((hand_int >> rank) & _RANK_MASK).bit_count()
    return total



# Cards pre-encoded for the leaf potential loop: a king is -1, anything else
# is its Hand.as_int bit. The remaining-card multiset is threaded down the
# search as a tuple of these tokens, shrinking by the taken card each ply.
_TOKEN = {
    card: (-1 if card[0] is Rank.K else 1 << (card[1] * 8 + card[0] - 1))
    for card in Card.deck()
}


def _without(remaining, token):
    """`remaining` minus one occurrence of `token`."""
    i = remaining.index(token)
    return remaining[:i] + remaining[i + 1 :]


def _remaining_cards(game):
    """All cards still on the board; face-down placeholders are replaced by
    the (known) multiset of remaining face-down cards."""
    taken = set(game.moves)
    faceup = [
        game.board[row][col]
        for row in range(6)
        for col in range(6)
        if (row, col) not in taken and not game.board[row][col].facedown
    ]
    return faceup + list(game.board.facedown_cards)


class _Timeout(Exception):
    pass


# Skip the exact-upgrade attempt with less budget than this left - too little
# to finish anything, just wasted setup.
_EXACT_UPGRADE_MIN_SECONDS = 0.05


def _exact_move(game, deadline=None):
    """Exact best move, or None if there is no move or the solve was abandoned
    (deadline tripped). A deadline requires the native solver - the pure-Python
    fallback cannot be interrupted (best_move returns None then)."""
    result = best_move(game, deadline=deadline)
    if result is None:
        return None
    marker = result["marker"]
    if marker is None:
        return None
    # The solver ranks by (2w+d, s), dropping evaluate's middle "prefer
    # decisive" term (w) for sound pruning (see solver.py). Among moves tied on
    # 2w+d, re-rank by the full (2w+d, w, s) so a guaranteed draw never beats an
    # equal-expectation move that can still win. Untied winners skip this.
    moves = result["moves"]
    best_sign = max(decode(value)[0] for value, _exact in moves.values())
    tied = [mk for mk, (value, _exact) in moves.items() if decode(value)[0] == best_sign]
    if len(tied) > 1:
        best_key = None
        for mk in tied:
            candidate = (move_value(game, mk), mk)
            if best_key is None or candidate > best_key:
                best_key, marker = candidate, mk
    return marker if marker in game.legal_moves else None


@dataclass(frozen=True)
class SearchParams:
    """Every *behavioral* tunable of the midgame search - anything that can
    change what it returns - in one immutable object, so bot variants can be
    built side by side and A/B tested (see `cardgame.validation.arena`).
    Speed-only or feature-shape constants (the `_ORDER_*` weights, the
    `_CENTRALITY` table) live at module level: they are offline-fit outputs,
    only coherent to change by refitting, and arena-invisible individually."""

    # Upper bound on any value the search can return, used for Star1 cutoffs
    # at chance nodes (tighter = stronger). Sound by construction: terminal
    # and leaf values are clamped to this range on return.
    value_bound: float = 36.0
    win_bonus: float = 6.0
    # Heuristic leaf weights, fitted by least squares against exact oracle
    # values (cardgame.validation.oracle) and arena-validated. The centrality
    # weight is fitted under the linear decay below.
    potential_weight: float = 0.238
    centrality_weight: float = 7.562
    mobility_weight: float = 0.378
    # King centrality relative to the 0..1 rank table (4/5 = 1.0) in the
    # evaluation's centrality sum. Pinned at 1.0; fittable via fit_weights.
    king_centrality: float = 1.0
    # Phase slopes: each base weight varies linearly in cards left above the
    # pivot, w_eff = weight + slope * max(cards_left - pivot, 0), so play at
    # <= pivot cards is exactly the arena-validated champion and the
    # correction only ramps in toward the opening. `centrality_base` adds a
    # flat floor to the decayed centrality term. Slopes = 0 recovers the
    # phase-flat champion. Fitted on the exact + bootstrap corpora.
    potential_slope: float = -0.01199
    centrality_base: float = 0.0
    mobility_slope: float = 0.00037
    tempo_slope: float = -0.1182
    # Where the phase ramp starts, in cards left. The slopes are only valid
    # at the pivot they were fitted against; refitting them (fit_weights'
    # phase-fit section) is what changes it, not a free knob.
    phase_pivot: float = 12.0
    # Constant added to every heuristic leaf: the value of being on move.
    # Affects only comparisons against terminal values, encoding that a live
    # position is worth more than its bare score.
    tempo_bonus: float = 2.367
    # Face-down moves branch into one child per possible hidden card; deeper
    # in the tree the expectation is approximated with this many evenly-spaced
    # samples of the rank-ordered possibilities.
    resolution_cap: int = 5
    # Positions with at most this many cards left are solved exactly by
    # Game.evaluate instead of leaf-evaluated. Affordable but arena-neutral,
    # so off by default (suspected value-scale mismatch vs heuristic leaves).
    # 0 disables.
    exact_leaf_cards: int = 0
    # Don't start another deepening iteration past this fraction of the
    # budget. 1.0 keeps deepening until the deadline aborts mid-iteration:
    # the previous best is searched first and improvements kept, so an
    # interrupted iteration is never worse than idling.
    deepen_fraction: float = 1.0
    # Defer to the exact solver inside the calibrated endgame region.
    exact_endgame: bool = True
    # Run the midgame search in the Rust core (heuristic_root) instead of
    # Python. Same search, bit-identical at equal depth; the win is speed and
    # it is arena-neutral, so it is the default. Falls back to Python when the
    # native core is unavailable or exact_leaf_cards is set (native port is
    # exact_leaf_cards=0 only). native=False forces the Python reference.
    native: bool = True


class AlphaBetaBot:
    """A configured player: `choose_move(game)` returns a (row, col) move.

    Stateless between moves (the scoring cache is shared module state), so
    one instance can be reused across positions and games.
    """

    def __init__(self, time_budget=6.0, params=None, name="AlphaBetaBot"):
        self.time_budget = time_budget
        self.params = params if params is not None else SearchParams()
        self.name = name

    def choose_move(self, game):
        """Pick a legal (row, col) move for `game`'s current player.

        Runs the iterative-deepening heuristic search first (always yields a
        move). If it converged with budget to spare (the endgame), the leftover
        time attempts an exact solve under a hard deadline - it enumerates every
        face-down resolution the heuristic only samples. If it finishes its move
        wins, otherwise the deadline trips and the heuristic move stands."""
        legal_moves = game.legal_moves
        if not legal_moves:
            raise ValueError("Game is over - no legal moves to choose from")
        if len(legal_moves) == 1:
            return next(iter(legal_moves))
        move = self._search_root(game)
        if self.params.exact_endgame and NATIVE_AVAILABLE:
            leftover = self.time_budget - self.last_search.get(
                "elapsed", self.time_budget
            )
            if leftover > _EXACT_UPGRADE_MIN_SECONDS:
                exact = _exact_move(game, deadline=leftover)
                if exact is not None:
                    return exact
        return move

    def _terminal_value(self, me, opp):
        params = self.params
        diff = _score(*me) - _score(*opp)
        if diff > 0:
            return min(diff + params.win_bonus, params.value_bound)
        if diff < 0:
            return max(diff - params.win_bonus, -params.value_bound)
        return 0.0

    def _full_potential(self, hand):
        """Sum of `hand`'s marginals over the *root* remaining multiset.
        Tokens the hand already holds contribute zero (the OR is a no-op)
        and are skipped without a probe; every king has the same marginal,
        solved once. Cached per hand in `_fullpot` for the whole root
        search, so this O(cards-left) pass runs once per distinct hand."""
        score = _score
        hand_int, kings = hand
        base = score(hand_int, kings)
        total = 0
        king_marginal = None
        for token in self._root_tokens:
            if token < 0:
                if king_marginal is None:
                    # A 4-king hand has no kings left to gain: its king
                    # marginal is only ever multiplied by zero net remaining
                    # kings, so any value cancels — use 0 (the DP does not
                    # go past 4 kings). Must match _evaluate_leaf's guard.
                    king_marginal = (
                        score(hand_int, kings + 1) - base if kings < 4 else 0
                    )
                total += king_marginal
            elif not hand_int & token:
                total += score(hand_int | token, kings) - base
        return total

    def _evaluate_leaf(self, game, me, opp, taken, remaining):
        # The search's hot spot. The potential term (each hand's summed
        # marginals over cards still on the board) is computed the shorter of
        # two equivalent ways per leaf: direct (one probe per hand per
        # remaining card) or incremental (the _fullpot-cached root potential
        # minus the path's taken marginals). Shallow horizons go incremental,
        # deep endgame horizons direct; integer sums, so identical either way.
        params = self.params
        score = _score
        me_int, me_kings = me
        opp_int, opp_kings = opp
        my_base = score(me_int, me_kings)
        opp_base = score(opp_int, opp_kings)
        king_mine = king_opp = None
        if len(taken) < len(remaining):
            fullpot = self._fullpot
            my_potential = fullpot.get(me)
            if my_potential is None:
                my_potential = fullpot[me] = self._full_potential(me)
            opp_potential = fullpot.get(opp)
            if opp_potential is None:
                opp_potential = fullpot[opp] = self._full_potential(opp)
            for token in taken:
                if token < 0:
                    if king_mine is None:
                        # 4-king guard mirrors _full_potential: the terms
                        # cancel.
                        king_mine = (
                            score(me_int, me_kings + 1) - my_base
                            if me_kings < 4 else 0
                        )
                        king_opp = (
                            score(opp_int, opp_kings + 1) - opp_base
                            if opp_kings < 4 else 0
                        )
                    my_potential -= king_mine
                    opp_potential -= king_opp
                else:
                    if not me_int & token:
                        my_potential -= score(me_int | token, me_kings) - my_base
                    if not opp_int & token:
                        opp_potential -= score(opp_int | token, opp_kings) - opp_base
        else:
            my_potential = opp_potential = 0
            for token in remaining:
                if token < 0:
                    if king_mine is None:
                        king_mine = score(me_int, me_kings + 1) - my_base
                        king_opp = score(opp_int, opp_kings + 1) - opp_base
                    my_potential += king_mine
                    opp_potential += king_opp
                else:
                    my_potential += score(me_int | token, me_kings) - my_base
                    opp_potential += score(opp_int | token, opp_kings) - opp_base
        cards_left = 36.0 - len(game.moves)
        pivot = params.phase_pivot
        phase = cards_left - pivot if cards_left > pivot else 0.0
        value = float(my_base - opp_base) + params.tempo_bonus
        value += params.tempo_slope * phase
        value += (
            params.potential_weight + params.potential_slope * phase
        ) * (my_potential - opp_potential)
        decay = cards_left / 36.0
        if decay > 0:
            kc = params.king_centrality
            value += (
                params.centrality_base + params.centrality_weight * decay
            ) * (_centrality_sum(me, kc) - _centrality_sum(opp, kc))
        value += (
            params.mobility_weight + params.mobility_slope * phase
        ) * len(game.legal_moves)
        return min(params.value_bound, max(-params.value_bound, value))

    def _sample_resolutions(self, resolutions):
        cap = self.params.resolution_cap
        if len(resolutions) <= cap:
            return resolutions
        ordered = sorted(resolutions, key=lambda g: g.taken_card)
        last = len(ordered) - 1
        indices = [round(i * last / (cap - 1)) for i in range(cap)]
        return tuple(ordered[i] for i in indices)

    def _ordered_markers(self, game, me, opp, mask):
        """Legal moves as (marker, facedown), best-looking first for the
        mover, with no child games constructed - callers build resolutions
        only for moves that are actually searched, so moves behind a cutoff
        cost nothing. The key is the fitted _ORDER_* score: the card's
        marginal to each hand (cached DP), its static centrality/king
        value, the face-down flag (means over the hidden multiset), and
        how many replies the move leaves the opponent (one popcount
        against `mask`, the taken-cell bitmask)."""
        facedown_key = None
        board = game.board
        facedown_positions = board.facedown_positions
        moves = []
        for marker in game.legal_moves:
            replies = (_REPLY_MASK[marker] & ~mask).bit_count()
            if marker in facedown_positions:
                if facedown_key is None:
                    hidden = board.facedown_cards
                    total = 0.0
                    for card in hidden:
                        total += _ORDER_ME * _marginal(me, card)
                        total += _ORDER_OPP * _marginal(opp, card)
                        if card[0] is Rank.K:
                            total += _ORDER_CENT * _KING_CENTRALITY + _ORDER_KING
                        else:
                            total += _ORDER_CENT * _CENTRALITY[card[0] - 1]
                    facedown_key = total / len(hidden) + _ORDER_FD
                key, facedown = facedown_key, True
            else:
                cell = board[marker[0]][marker[1]]
                key = _ORDER_ME * _marginal(me, cell) + _ORDER_OPP * _marginal(
                    opp, cell
                )
                if cell[0] is Rank.K:
                    key += _ORDER_CENT * _KING_CENTRALITY + _ORDER_KING
                else:
                    key += _ORDER_CENT * _CENTRALITY[cell[0] - 1]
                facedown = False
            moves.append((key + _ORDER_REPLIES * replies, marker, facedown))
        moves.sort(key=lambda entry: (-entry[0], entry[1]))
        return [(marker, facedown) for _, marker, facedown in moves]

    def _resolutions(self, game, marker, facedown):
        """The child games of one move, face-down ones sampled to the cap."""
        resolutions = game.move(*marker)
        if facedown:
            resolutions = self._sample_resolutions(resolutions)
        return resolutions

    def _chance_value(
        self, resolutions, depth, alpha, beta, me, opp, taken, remaining,
        mask, deadline,
    ):
        """Expected value over equally-likely face-down resolutions (Star1):
        each child is searched only in the window of contributions that could
        still move the expectation into (alpha, beta), and the loop returns a
        bound as soon as the running mean can no longer enter the window. A
        child cut off by its narrowed window returns a fail-soft bound, which
        makes the running total a bound in exactly the direction that triggers
        the corresponding cutoff check below, so the returned value stays
        sound."""
        bound = self.params.value_bound
        n = len(resolutions)
        total = 0.0
        for i, child in enumerate(resolutions):
            spread = (n - i - 1) * bound
            lo = max(n * alpha - total - spread, -bound)
            hi = min(n * beta - total + spread, bound)
            card = child.taken_card
            child_me = _add_card(me, card)
            token = _TOKEN[card]
            total -= self._search(
                child, depth - 1, -hi, -lo, opp, child_me,
                taken + (token,), _without(remaining, token), mask, deadline,
            )
            upper = (total + spread) / n
            if upper <= alpha:
                return upper
            lower = (total - spread) / n
            if lower >= beta:
                return lower
        return total / n

    def _exact_leaf(self, game):
        """Exact value of a small position: the expectation of the terminal
        scoring (score difference with the win bonus) over Game.evaluate's
        outcome distribution, from the mover's perspective. Cached per root
        search keyed by move prefix, so the frontier subtrees revisited by
        every deepening iteration are solved once."""
        value = self._exact_cache.get(game.moves)
        if value is None:
            prob, _best = game.evaluate()
            m = prob.multiplicity
            w, d, s = prob.wds
            value = (s + self.params.win_bonus * (2 * w + d - m)) / m
            self._exact_cache[game.moves] = value
        return value

    def _search(
        self, game, depth, alpha, beta, me, opp, taken, remaining, mask, deadline
    ):
        if time.perf_counter() > deadline:
            raise _Timeout
        if not game.legal_moves:
            return self._terminal_value(me, opp)
        if 36 - len(game.moves) <= self.params.exact_leaf_cards:
            return self._exact_leaf(game)
        # Transposition probe. The taken-cell mask, marker and both hands
        # fully determine the subgame (hands alone don't). Entries are
        # (depth, flag, value, best_marker), flag 0 exact / 1 lower / -1 upper
        # (fail-soft); a shallower entry can't answer a deeper request but its
        # best move still improves ordering.
        key = (mask, game.moves[-1], me, opp)
        entry = self._tt.get(key)
        tt_move = None
        if entry is not None:
            tt_depth, flag, value, tt_move = entry
            if tt_depth >= depth and (
                flag == 0
                or (flag == 1 and value >= beta)
                or (flag == -1 and value <= alpha)
            ):
                self._tt_cuts += 1
                return value
        if depth == 0:
            return self._evaluate_leaf(game, me, opp, taken, remaining)
        markers = self._ordered_markers(game, me, opp, mask)
        if tt_move is not None and markers[0][0] != tt_move:
            for i, item in enumerate(markers):
                if item[0] == tt_move:
                    markers.insert(0, markers.pop(i))
                    break
        alpha0 = alpha
        best = -self.params.value_bound
        best_marker = None
        for marker, facedown in markers:
            child_mask = mask | (1 << (marker[0] * 6 + marker[1]))
            resolutions = self._resolutions(game, marker, facedown)
            if len(resolutions) == 1:
                child = resolutions[0]
                card = child.taken_card
                child_me = _add_card(me, card)
                token = _TOKEN[card]
                value = -self._search(
                    child, depth - 1, -beta, -alpha, opp, child_me,
                    taken + (token,), _without(remaining, token),
                    child_mask, deadline,
                )
            else:
                value = self._chance_value(
                    resolutions, depth, alpha, beta, me, opp, taken,
                    remaining, child_mask, deadline,
                )
            if value > best:
                best = value
                best_marker = marker
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
        if entry is None or entry[0] <= depth:
            flag = -1 if best <= alpha0 else (1 if best >= beta else 0)
            self._tt[key] = (depth, flag, best, best_marker)
        return best

    def _use_native(self):
        return (
            self.params.native
            and HEURISTIC_NATIVE
            and self.params.exact_leaf_cards == 0
        )

    def _native_params(self):
        p = self.params
        return [
            p.value_bound, p.win_bonus, p.potential_weight, p.potential_slope,
            p.centrality_weight, p.centrality_base, p.king_centrality,
            p.mobility_weight, p.mobility_slope, p.tempo_bonus, p.tempo_slope,
            p.phase_pivot,
        ]

    def _search_root_native(self, game):
        cells, cell, rows, cols, mi, mk, oi, ok, _ = _root_state(game)
        codes = [int(c[0]) * 4 + int(c[1]) for c in game.board.facedown_cards]
        marker, completed, elapsed = _heuristic_root(
            [-1 if c is None else c for c in cells], cell, rows, cols,
            mi, mk, oi, ok, codes, self._native_params(),
            self.params.resolution_cap, self.params.deepen_fraction,
            self.time_budget,
        )
        self.last_search = {"completed_depth": completed, "elapsed": elapsed, "native": True}
        return marker

    def _search_root(self, game):
        if self._use_native():
            return self._search_root_native(game)
        return self._search_root_python(game)

    def _search_root_python(self, game):
        params = self.params
        bound = params.value_bound
        self._exact_cache = {}
        # Per-hand cache of the full-board potential (see _full_potential);
        # valid for exactly one root search, whose remaining multiset is
        # fixed in _root_tokens.
        self._fullpot = {}
        # Fresh table per root search: keys are per-deal (card -> cell
        # mappings differ between deals) and per-position entries go stale
        # as the game advances anyway (the taken-cell mask only grows).
        self._tt = {}
        self._tt_cuts = 0
        start = time.perf_counter()
        deadline = start + self.time_budget
        if len(game.moves) % 2 == 0:
            me, opp = game.p1.as_int, game.p2.as_int
        else:
            me, opp = game.p2.as_int, game.p1.as_int

        remaining = self._root_tokens = tuple(
            _TOKEN[card] for card in _remaining_cards(game)
        )
        mask = 0
        for row, col in game.moves:
            mask |= 1 << (row * 6 + col)
        # The root is one node: materialise its children once and reuse
        # them across every deepening iteration.
        moves = [
            (marker, self._resolutions(game, marker, facedown))
            for marker, facedown in self._ordered_markers(game, me, opp, mask)
        ]
        best_marker = moves[0][0]
        # Value of best_marker to the player to move (higher = better for the
        # mover). Tracked so choose_placement can compare positions, not just
        # pick a move. Set once depth 1 completes.
        best_value = -bound
        max_depth = 36 - len(game.moves)
        depth = 1
        completed_depth = 0
        interrupted = False
        while depth <= max_depth:
            alpha = -bound
            iteration_best = None
            scores = {}
            try:
                for marker, resolutions in moves:
                    child_mask = mask | (1 << (marker[0] * 6 + marker[1]))
                    if len(resolutions) == 1:
                        child = resolutions[0]
                        card = child.taken_card
                        child_me = _add_card(me, card)
                        token = _TOKEN[card]
                        value = -self._search(
                            child, depth - 1, -bound, -alpha, opp, child_me,
                            (token,), _without(remaining, token),
                            child_mask, deadline,
                        )
                    else:
                        value = self._chance_value(
                            resolutions, depth, alpha, bound, me, opp,
                            (), remaining, child_mask, deadline,
                        )
                    scores[marker] = value
                    if value > alpha:
                        alpha = value
                        iteration_best = marker
            except _Timeout:
                # Keep a mid-iteration improvement: the previous best was
                # searched first, so a move that raised alpha at this depth
                # has proven itself better under the deeper search (its
                # value is a sound fail-soft lower bound).
                interrupted = True
                if iteration_best is not None:
                    best_marker = iteration_best
                    best_value = scores[iteration_best]
                break
            if iteration_best is not None:
                best_marker = iteration_best
                best_value = scores[iteration_best]
            completed_depth = depth
            # Search the previous iteration's best moves first next time round.
            moves.sort(key=lambda entry: (-scores[entry[0]], entry[0]))
            depth += 1
            if time.perf_counter() - start > params.deepen_fraction * self.time_budget:
                break
        self.last_search = {
            "completed_depth": completed_depth,
            "interrupted": interrupted,
            "elapsed": time.perf_counter() - start,
            "tt_entries": len(self._tt),
            "tt_cuts": self._tt_cuts,
            "value": best_value,
        }
        return best_marker

    def choose_placement(self, game):
        """Best starting cell for the placer (the player NOT moving first):
        the one MINIMISING the mover's best reply.

        Shares one search across the four placements. The starting cell is
        never collected, so the position after the mover's first move depends
        only on the target cell, and the transposition table keys exactly on
        that - every shared subtree is searched once. Joint iterative deepening
        keeps all four values at equal, comparable depth."""
        if self._use_native() and _heuristic_placement is not None:
            return self._choose_placement_native(game)
        return self._choose_placement_python(game)

    def _choose_placement_native(self, game):
        starts = game._valid_starting_positions
        placed0 = game.place_marker(*starts[0])
        cells, _cell, rows, cols, mi, mk, oi, ok, _ = _root_state(placed0)
        codes = [int(c[0]) * 4 + int(c[1]) for c in placed0.board.facedown_cards]
        (r, c), completed, elapsed, values = _heuristic_placement(
            [-1 if x is None else x for x in cells],
            [row * 6 + col for row, col in starts],
            rows, cols, mi, mk, oi, ok, codes, self._native_params(),
            self.params.resolution_cap, self.params.deepen_fraction,
            self.time_budget,
        )
        self.last_search = {
            "completed_depth": completed, "elapsed": elapsed, "native": True,
            "values": {pos: v for pos, v in zip(starts, values)},
        }
        return (r, c)

    def _choose_placement_python(self, game):
        params = self.params
        bound = params.value_bound
        self._exact_cache = {}
        self._fullpot = {}
        self._tt = {}
        self._tt_cuts = 0
        start = time.perf_counter()
        deadline = start + self.time_budget
        starts = game._valid_starting_positions
        # All placements share one board and one remaining multiset (no moves
        # yet); the mover is always P1 after a placement.
        placed0 = game.place_marker(*starts[0])
        me, opp = placed0.p1.as_int, placed0.p2.as_int
        remaining = self._root_tokens = tuple(
            _TOKEN[card] for card in _remaining_cards(placed0)
        )
        # Each placement's first-move children (marker + facedown resolutions).
        placements = []
        for pos in starts:
            placed = game.place_marker(*pos)
            children = [
                (marker, self._resolutions(placed, marker, facedown))
                for marker, facedown in self._ordered_markers(placed, me, opp, 0)
            ]
            placements.append((pos, children))
        best = {pos: -bound for pos, _ in placements}
        max_depth = 36 - len(placed0.moves)
        depth = 1
        completed_depth = 0
        while depth <= max_depth:
            values = {}
            try:
                for pos, children in placements:
                    value = -bound
                    for marker, resolutions in children:
                        child_mask = 1 << (marker[0] * 6 + marker[1])
                        # Full window: exact child values (comparable across
                        # placements) and exact table entries (maximal reuse).
                        if len(resolutions) == 1:
                            child = resolutions[0]
                            token = _TOKEN[child.taken_card]
                            child_value = -self._search(
                                child, depth - 1, -bound, bound, opp,
                                _add_card(me, child.taken_card), (token,),
                                _without(remaining, token), child_mask, deadline,
                            )
                        else:
                            child_value = self._chance_value(
                                resolutions, depth, -bound, bound, me, opp,
                                (), remaining, child_mask, deadline,
                            )
                        if child_value > value:
                            value = child_value
                    values[pos] = value
            except _Timeout:
                break
            best = values
            completed_depth = depth
            depth += 1
            if time.perf_counter() - start > params.deepen_fraction * self.time_budget:
                break
        self.last_search = {
            "completed_depth": completed_depth,
            "elapsed": time.perf_counter() - start,
            "values": best,
        }
        # Placer minimises the mover's value; ties break to the lower/righter
        # cell via the natural order of _valid_starting_positions.
        return min(best, key=lambda pos: (best[pos], pos))


def choose_move(game, time_budget=6.0):
    """Pick a legal (row, col) move for `game`'s current player, using a
    default-parameter `AlphaBetaBot`; construct one directly to customise."""
    return AlphaBetaBot(time_budget=time_budget).choose_move(game)


def choose_placement(game, time_budget=6.0):
    """Choose the marker's starting cell for the placer (the player NOT taking
    the first move), using a default `AlphaBetaBot`; construct one directly to
    customise. Returns a (row, col)."""
    return AlphaBetaBot(time_budget=time_budget).choose_placement(game)
