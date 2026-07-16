"""A computer opponent for Cross Kings.

`AlphaBetaBot` plays in two regimes:

* **Endgame** — once the remaining game tree is small enough (measured by
  cards left on the board and unresolved face-down cards, calibrated
  empirically), it defers to the exact expectiminimax in `Game.evaluate`,
  which optimises win/draw/loss directly.
* **Opening/midgame** — iterative-deepening expectiminimax under a time
  budget: negamax with alpha-beta at decision nodes, expectation with
  Star1 cutoffs at face-down (chance) nodes — each resolution searched
  only in the window of contributions that could still move the
  expectation into the parent's (alpha, beta) — and a heuristic leaf
  evaluation built from:

  - the actual score difference between the two hands (via the scoring DP),
  - "potential": the discounted marginal score of every card still on the
    board against each hand — taking a card the opponent needs removes it
    from *their* potential, so denial is valued automatically,
  - rank centrality, decaying as the game progresses: middle ranks (4, 5)
    fit into more possible runs, so they are worth more early on both to
    build with and to deny,
  - a small mobility term, and a terminal win/loss bonus so the bot steers
    towards ending the game while it is ahead rather than maximising a
    speculative score.

Every tunable lives in `SearchParams`, and a bot is an `AlphaBetaBot`
instance (params + a time budget), so differently-configured bots can be
built side by side and matched against each other — `cardgame.arena`
plays duplicate-deal matches between two bot versions to validate that a
change actually gains strength. The module-level `choose_move` keeps the
original convenience API with default parameters.

Averaging over the remaining face-down cards is legitimate card counting,
not cheating: the deck composition and every face-up card are public, so
both players can deduce the face-down *multiset* — no knowledge of which
position hides which card is used.
"""

import time
from dataclasses import dataclass
from functools import lru_cache

from .cards import Card, Rank
from .scoring import score_dp

__all__ = ("choose_move", "AlphaBetaBot", "SearchParams")

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
_KING_CENTRALITY = 1.0

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
def _centrality_sum(hand):
    hand_int, kings = hand
    total = kings * _KING_CENTRALITY
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


# Regions where a full `Game.evaluate` stays comfortably inside the ~10s
# move ceiling (observed worst cases a few seconds, leaving headroom for
# unsampled tails), calibrated over random games. The chance-node branching
# from face-down cards is the main cost driver.
def _exact_feasible(game):
    cards_left = 36 - len(game.moves)
    facedown = len(game.board.facedown_cards)
    return (
        cards_left <= 8
        or (cards_left == 9 and facedown <= 5)
        or (cards_left == 10 and facedown <= 4)
        or (cards_left <= 12 and facedown <= 3)
        or (cards_left == 13 and facedown <= 2)
    )


def _exact_move(game):
    sequence = game.evaluate()["Deterministic optimal moves"]
    first = sequence[0] if sequence else None
    if isinstance(first, Card):
        # Face-up optimal moves are recorded as the card taken; cards are
        # unique, so locate it among the legal cells.
        for marker in game.legal_moves:
            if game.board[marker[0]][marker[1]] == first:
                return marker
        return None
    return first if first in game.legal_moves else None


@dataclass(frozen=True)
class SearchParams:
    """Every tunable of the midgame search, in one immutable object so bot
    variants can be constructed side by side and A/B tested (see
    `cardgame.arena`)."""

    # Upper bound on any value the search can return, used for Star1 cutoffs
    # at chance nodes - their strength scales directly with how tight this
    # is. Sound by construction: terminal and leaf values are clamped to
    # this range on return. Terminals reach at most ~32 (score difference +
    # win bonus) and heuristic leaves stay within ~±13 in practice, so
    # clamping essentially never fires outside overwhelmingly decided
    # positions.
    value_bound: float = 36.0
    win_bonus: float = 6.0
    potential_weight: float = 0.35
    centrality_weight: float = 0.6
    mobility_weight: float = 0.05
    # Move ordering weight on denial: a cell's ordering key is the mover's
    # marginal gain plus this times the opponent's marginal for the same
    # card. Ordering-only - it never changes what a search returns.
    order_denial_weight: float = 1.0
    # Face-down moves branch into one child per possible hidden card; deeper
    # in the tree the expectation is approximated with this many
    # evenly-spaced samples of the rank-ordered possibilities.
    resolution_cap: int = 5
    # Positions with at most this many cards left are solved exactly by
    # Game.evaluate wherever the midgame search meets them, instead of
    # being searched heuristically or leaf-evaluated: the expectation of
    # the terminal scoring over the exact outcome distribution replaces
    # the horizon-truncated guess. Cost-wise 6 is affordable (~0.6ms median
    # solves, cached across deepening iterations, and measurably *deeper*
    # searches in the 9-14 card facedown-heavy band) but arena-tested
    # strength-neutral at 6 (100 pairs, 50.0%), so it stays off by default;
    # a suspected cause is the value-scale mismatch against heuristic
    # leaves (exact leaves carry the win-bonus expectation, heuristic
    # leaves carry none). 0 disables.
    exact_leaf_cards: int = 0
    # Don't start another deepening iteration once this fraction of the
    # time budget has been spent. 1.0 means keep deepening until the
    # deadline aborts mid-iteration: with the previous best searched first
    # and sound mid-iteration improvements kept, an interrupted iteration
    # is never worse than idling, so there is no opportunity cost.
    deepen_fraction: float = 1.0
    # Defer to the exact solver inside the calibrated endgame region.
    exact_endgame: bool = True


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
        """Pick a legal (row, col) move for `game`'s current player."""
        legal_moves = game.legal_moves
        if not legal_moves:
            raise ValueError("Game is over - no legal moves to choose from")
        if len(legal_moves) == 1:
            return next(iter(legal_moves))
        if self.params.exact_endgame and _exact_feasible(game):
            move = _exact_move(game)
            if move is not None:
                return move
        return self._search_root(game)

    def _terminal_value(self, me, opp):
        params = self.params
        diff = _score(*me) - _score(*opp)
        if diff > 0:
            return min(diff + params.win_bonus, params.value_bound)
        if diff < 0:
            return max(diff - params.win_bonus, -params.value_bound)
        return 0.0

    def _evaluate_leaf(self, game, me, opp, remaining):
        # The search's hot spot (~3/4 of all time goes here, almost all of
        # it in the potential loop), hence the single hand-inlined pass:
        # one cached-DP probe per hand per remaining card, kings solved
        # once and reused (every king has the same marginal).
        params = self.params
        score = _score
        me_int, me_kings = me
        opp_int, opp_kings = opp
        my_base = score(me_int, me_kings)
        opp_base = score(opp_int, opp_kings)
        my_potential = opp_potential = 0
        king_mine = king_opp = None
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
        value = float(my_base - opp_base)
        value += params.potential_weight * (my_potential - opp_potential)
        decay = 1.0 - len(game.moves) / 36.0
        if decay > 0:
            value += (
                params.centrality_weight
                * decay
                * (_centrality_sum(me) - _centrality_sum(opp))
            )
        value += params.mobility_weight * len(game.legal_moves)
        return min(params.value_bound, max(-params.value_bound, value))

    def _sample_resolutions(self, resolutions):
        cap = self.params.resolution_cap
        if len(resolutions) <= cap:
            return resolutions
        ordered = sorted(resolutions, key=lambda g: g.taken_card)
        last = len(ordered) - 1
        indices = [round(i * last / (cap - 1)) for i in range(cap)]
        return tuple(ordered[i] for i in indices)

    def _ordered_markers(self, game, me, opp):
        """Legal moves as (marker, facedown), best-looking first for the
        mover, with no child games constructed - callers build resolutions
        only for moves that are actually searched, so moves behind a cutoff
        cost nothing. A card's pull is what it adds to the mover's hand
        plus what taking it denies the opponent - both marginals come from
        the same cached DP, so denial-awareness is nearly free."""
        denial = self.params.order_denial_weight
        facedown_mean = None
        board = game.board
        facedown_positions = board.facedown_positions
        moves = []
        for marker in game.legal_moves:
            if marker in facedown_positions:
                if facedown_mean is None:
                    hidden = board.facedown_cards
                    facedown_mean = sum(
                        _marginal(me, card) + denial * _marginal(opp, card)
                        for card in hidden
                    ) / len(hidden)
                key, facedown = facedown_mean, True
            else:
                cell = board[marker[0]][marker[1]]
                key = _marginal(me, cell) + denial * _marginal(opp, cell)
                facedown = False
            moves.append((key, marker, facedown))
        moves.sort(key=lambda entry: (-entry[0], entry[1]))
        return [(marker, facedown) for _, marker, facedown in moves]

    def _resolutions(self, game, marker, facedown):
        """The child games of one move, face-down ones sampled to the cap."""
        resolutions = game.move(*marker)
        if facedown:
            resolutions = self._sample_resolutions(resolutions)
        return resolutions

    def _chance_value(
        self, resolutions, depth, alpha, beta, me, opp, remaining, mask, deadline
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
            total -= self._search(
                child, depth - 1, -hi, -lo, opp, child_me,
                _without(remaining, _TOKEN[card]), mask, deadline,
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
            prob = game.evaluate()["Evaluation"]
            m = prob.multiplicity
            w, d, s = prob.wds
            value = (s + self.params.win_bonus * (2 * w + d - m)) / m
            self._exact_cache[game.moves] = value
        return value

    def _search(self, game, depth, alpha, beta, me, opp, remaining, mask, deadline):
        if time.perf_counter() > deadline:
            raise _Timeout
        if not game.legal_moves:
            return self._terminal_value(me, opp)
        if 36 - len(game.moves) <= self.params.exact_leaf_cards:
            return self._exact_leaf(game)
        # Transposition probe. `mask` is the taken-cell bitmask; with the
        # marker and both hands it fully determines the subgame (hands alone
        # don't - the same revealed card can have come from different
        # face-down cells). Entries are (depth, flag, value, best_marker)
        # with flag 0 exact / 1 lower bound / -1 upper bound (fail-soft);
        # a shallower entry can't answer for a deeper request but its best
        # move still improves ordering, which is where an iterative
        # deepener earns most of its table hits.
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
            return self._evaluate_leaf(game, me, opp, remaining)
        markers = self._ordered_markers(game, me, opp)
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
                value = -self._search(
                    child, depth - 1, -beta, -alpha, opp, child_me,
                    _without(remaining, _TOKEN[card]), child_mask, deadline,
                )
            else:
                value = self._chance_value(
                    resolutions, depth, alpha, beta, me, opp, remaining,
                    child_mask, deadline,
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

    def _search_root(self, game):
        params = self.params
        bound = params.value_bound
        self._exact_cache = {}
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

        remaining = tuple(_TOKEN[card] for card in _remaining_cards(game))
        mask = 0
        for row, col in game.moves:
            mask |= 1 << (row * 6 + col)
        # The root is one node: materialise its children once and reuse
        # them across every deepening iteration.
        moves = [
            (marker, self._resolutions(game, marker, facedown))
            for marker, facedown in self._ordered_markers(game, me, opp)
        ]
        best_marker = moves[0][0]
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
                        value = -self._search(
                            child, depth - 1, -bound, -alpha, opp, child_me,
                            _without(remaining, _TOKEN[card]), child_mask, deadline,
                        )
                    else:
                        value = self._chance_value(
                            resolutions, depth, alpha, bound, me, opp,
                            remaining, child_mask, deadline,
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
                break
            if iteration_best is not None:
                best_marker = iteration_best
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
        }
        return best_marker


def choose_move(game, time_budget=6.0):
    """Pick a legal (row, col) move for `game`'s current player, using a
    default-parameter `AlphaBetaBot`; construct one directly to customise."""
    return AlphaBetaBot(time_budget=time_budget).choose_move(game)
