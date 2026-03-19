"""
A DP implementation of scoring, important for efficiency as we have >1 king to allocate
This implementation uses pre-computes scores form runs/sets, lru_caching
"""

from functools import reduce
from itertools import combinations, product
from operator import or_
from .cards import Suit, Rank, Card

# ---------------------------------------------------------------------------
# Module-level precomputation (runs once on import, ~40ms)
# ---------------------------------------------------------------------------

_pow5 = [5**r for r in range(8)]

# Set score table: (base_card_count, king_count) -> points.
# Explicit threshold check is required — max(0, 2b+k-3) is wrong because
# it gives 1 for (b=2, k=0) even though a pair (total < 3) should score 0.
_set_score_table = {
    (b, k): (2 * b + k - 3) if (b + k >= 3) else 0
    for b in range(5)
    for k in range(5 - b)
}


# Suit run scores: (card_mask, king_mask) -> run points.
# card_mask and king_mask are 8-bit; bit r is set if rank r+1 is present.
def _calculate_score_lookups():
    suit_scores = {}
    for pmut in product([False, True], repeat=8):
        for kings in product(*(([False] if card else [False, True]) for card in pmut)):
            if sum(kings) > 4:
                continue
            score = 0
            run = 0
            kings_in_run = 0
            for card, king in zip(pmut, kings):
                if card:
                    run += 1
                elif king:
                    run += 1
                    kings_in_run += 1
                else:
                    if run >= 3:
                        score += run * 2 - 3 - kings_in_run
                    run = kings_in_run = 0
            if run >= 3:
                score += run * 2 - 3 - kings_in_run
            suit_scores[
                (
                    reduce(or_, (1 << i for i, card in enumerate(pmut) if card), 0),
                    reduce(or_, (1 << i for i, card in enumerate(kings) if card), 0),
                )
            ] = score
    rank_scores = {
        (
            reduce(or_, (1 << (8 * i) for i, card in enumerate(pmut) if card), 0),
            reduce(or_, (1 << (8 * i) for i, card in enumerate(kings) if card), 0),
        ): (2 * sum(pmut) - 3 + sum(kings) if sum(pmut) + sum(kings) >= 3 else 0)
        for pmut in product([False, True], repeat=4)
        for kings in product(*(([False] if card else [False, True]) for card in pmut))
    }
    return rank_scores, suit_scores


rank_scores, suit_scores = _calculate_score_lookups()


def _int_to_bits(n):
    components = []
    while n > 0:
        lsb = n & -n
        components.append(lsb)
        n &= n - 1
    return tuple(reversed(components))


# Transition table: (suit_mask, max_kings_available) -> list of
# (kings_placed, run_score_contribution, rank_delta).
# rank_delta is the base-5 increment to add to the DP state integer.
def _build_transitions():
    transitions = {}
    for mask in range(256):
        empty_ranks = [r for r in range(8) if not ((mask >> r) & 1)]
        for max_kings in range(5):
            trans = []
            for k_s in range(min(max_kings, len(empty_ranks)) + 1):
                for chosen in combinations(empty_ranks, k_s):
                    king_mask = sum(1 << r for r in chosen)
                    run_score = suit_scores[(mask, king_mask)]
                    rank_delta = sum(_pow5[r] for r in chosen)
                    trans.append((k_s, run_score, rank_delta))
            transitions[(mask, max_kings)] = trans
    return transitions


_transitions = _build_transitions()

# ---------------------------------------------------------------------------
# Public scoring function
# ---------------------------------------------------------------------------


def score_dp(hand_int: int, number_of_kings: int) -> int:
    """Return the maximum score for a hand, with optimal king placement.

    Parameters
    ----------
    hand_int : int
        32-bit integer from Hand.as_int. Bit (suit*8 + rank-1) is set for
        each non-king card in the hand.
    number_of_kings : int
        Number of kings in the hand (second element of Hand.as_int).

    Returns
    -------
    int
        Maximum achievable score — identical to Hand.score() but faster.

    Examples
    --------
        score_dp(*hand.as_int)          # unpack Hand.as_int directly
        score_dp(hand_int, num_kings)   # or pass integers explicitly
    """
    suit_mask = 0b11111111
    rank_mask = 0b1000000010000000100000001  # bits 0, 8, 16, 24 (one per suit)

    suit_cards = [(hand_int >> (8 * s)) & suit_mask for s in range(4)]
    rank_base = [bin((hand_int >> r) & rank_mask).count("1") for r in range(8)]

    # dp maps state_int -> (best_run_score, kings_used).
    # state_int encodes the number of kings placed at each rank in base-5:
    #   state_int = sum(king_count_at_rank_r * 5^r  for r in range(8))
    dp = {0: (0, 0)}

    for s in range(4):
        m = suit_cards[s]
        new_dp = {}
        for state_int, (run_score, kings_used) in dp.items():
            kings_left = number_of_kings - kings_used
            for k_s, run_contrib, rank_delta in _transitions[(m, kings_left)]:
                new_state = state_int + rank_delta
                new_score = run_score + run_contrib
                existing = new_dp.get(new_state)
                if existing is None or existing[0] < new_score:
                    new_dp[new_state] = (new_score, kings_used + k_s)
        dp = new_dp

    best = 0
    for state_int, (run_score, kings_used) in dp.items():
        if kings_used != number_of_kings:
            continue
        tmp = state_int
        set_score = 0
        for r in range(8):
            set_score += _set_score_table[(rank_base[r], tmp % 5)]
            tmp //= 5
        total = run_score + set_score
        if total > best:
            best = total

    return best


def score_with_king_allocation(
    hand_int: int, number_of_kings: int
) -> tuple[int, tuple[Card, ...]]:

    possible_kings = _int_to_bits(0b11111111111111111111111111111111 ^ hand_int)
    best_score, best_kings = 0, 0
    suit_mask = 0b11111111
    rank_mask = 0b1000000010000000100000001
    for king_allocation in combinations(possible_kings, number_of_kings):
        king_int = reduce(or_, king_allocation, 0)
        score = 0
        for suit in Suit:
            base_suit = (hand_int >> (8 * suit)) & suit_mask
            king_suit = (king_int >> (8 * suit)) & suit_mask
            score += suit_scores[(base_suit, king_suit)]
        for rank in list(Rank)[:-1]:
            base_rank = (hand_int >> (rank - 1)) & rank_mask
            king_rank = (king_int >> (rank - 1)) & rank_mask
            score += rank_scores[(base_rank, king_rank)]
        best_score, best_kings = max((best_score, best_kings), (score, king_int))
    king_cards = tuple(
        Card(((king_int.bit_length() - 1) % 8) + 1, (king_int.bit_length() - 1) // 8)
        for king_int in _int_to_bits(best_kings)
    )
    return best_score, king_cards
