"""
move_analysis.py — per-move offensive/defensive analysis for Cross Kings
-------------------------------------------------------------------------
For each legal move from a position, computes how much it helps the current
player (offensive value), how much it hurts the opponent (defensive value),
and the combined score-difference swing — all under optimal play by both
sides from that point forward.
"""

import time
from collections import Counter

from .game import Eval, _cached_score

__all__ = ("analyse_moves", "analyse_moves_by_deadline", "AnalysisAborted")


class AnalysisAborted(Exception):
    """Raised out of analyse_moves when its `abort` event is set - the
    walk is unbounded in general (an early-game position can take hours),
    so long-running callers need a way to abandon one cooperatively."""


class _Deadline:
    """Plain, picklable stand-in for a live abort signal: is_set() fires
    once the given time.monotonic() deadline has passed. Used to bound a
    call running in a separate process, where there's no shared object a
    caller could set to signal it cooperatively - the cutoff has to be
    decided up front and travel with the call instead."""

    __slots__ = ("deadline",)

    def __init__(self, deadline):
        self.deadline = deadline

    def is_set(self):
        return self.deadline is not None and time.monotonic() >= self.deadline


def analyse_moves_by_deadline(game, deadline):
    """Like analyse_moves, but takes a plain time.monotonic() deadline (or
    None for unbounded) instead of a live abort object - the entry point
    for running a call in a worker process via a ProcessPoolExecutor, which
    can only receive plain, picklable arguments up front. Returns None
    instead of raising if the deadline passes before finishing."""
    try:
        return analyse_moves(game, abort=_Deadline(deadline))
    except AnalysisAborted:
        return None


class _Agg:
    """Cheap, allocation-free stand-in for (ProbEval, pairs Counter) - just
    the aggregate win/draw/score-sum/mover's-own-score-sum a position
    resolves to under optimal play, with no per-outcome histogram. Mirrors
    validation/exact_simple.py's _SAgg philosophy (aggregates are all any
    comparison ever needs), extended with `mover_sum` so player_mean/
    opponent_mean stay recoverable without a distribution: after one more
    negation (bringing it into the analysed player's own perspective),
    `mover_sum` is that player's own score-sum and `mover_sum - s` is the
    opponent's - see analyse_moves.

    Never partial (this walk never prunes, so `key` is a plain tuple
    comparison - no bounds needed the way ProbEval/Eval's fail-soft
    bookkeeping requires for pruned search).
    """

    __slots__ = ("m", "w", "d", "s", "mover_sum")

    def __init__(self, m, w=0, d=0, s=0, mover_sum=0):
        self.m, self.w, self.d, self.s, self.mover_sum = m, w, d, s, mover_sum

    @property
    def key(self):
        return (2 * self.w + self.d, self.w, self.s)

    def __add__(self, other):
        return _Agg(
            self.m + other.m,
            self.w + other.w,
            self.d + other.d,
            self.s + other.s,
            self.mover_sum + other.mover_sum,
        )

    def __neg__(self):
        return _Agg(self.m, self.m - self.w - self.d, self.d, -self.s, self.mover_sum - self.s)


def _collect_aggregate(game, _state=None, abort=None):
    """Like _collect_terminals, but returns only the winning line's _Agg -
    no pairs/Counter, no per-outcome histogram. Same base case, same
    per-move/per-possibility structure, same (candidate, marker) tie-break,
    so it selects the identical optimal line _collect_terminals does - this
    is what makes every move's win/draw/score-sum/mean cheap to compute
    for every legal move, not just the winner (see analyse_moves)."""
    if abort is not None and abort.is_set():
        raise AnalysisAborted
    if _state is None:
        _state = game._hand_state()
    if not game.legal_moves:
        mover_score = _cached_score(_state[0], _state[1])
        other_score = _cached_score(_state[2], _state[3])
        diff = mover_score - other_score
        m = game.multiplicity
        return _Agg(
            m,
            w=(m if diff > 0 else 0),
            d=(m if diff == 0 else 0),
            s=diff * m,
            mover_sum=mover_score * m,
        )
    child_state = game._child_hand_state
    best_key = None
    best_agg = None
    for move in game.all_moves():
        combined = None
        for possibility in move:
            child_agg = _collect_aggregate(
                possibility, child_state(_state, possibility.taken_card), abort
            )
            combined = child_agg if combined is None else combined + child_agg
        candidate_agg = -combined
        candidate_key = (candidate_agg.key, move[0].marker)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_agg = candidate_agg
    return best_agg


def _collect_terminals(game, _state=None, abort=None):
    """Traverse the game tree under optimal play, returning a weighted
    score frequency map of (other, mover) score pairs.

    Single pass: the optimal-play pair distribution is carried up
    alongside the negamax value, instead of re-running a full score_walk
    at every level of the optimal line as the original formulation did.
    Move selection uses the same cheap _Agg tie-break _collect_aggregate
    does (proven to pick the identical line score_walk/evaluate would,
    since this walk never prunes - see _Agg's docstring) rather than
    ProbEval, which nothing here needs: analyse_moves only ever reads the
    pairs half of this function's return value, never the eval half, and
    ProbEval's bounds-consistent __eq__/__gt__ cost a full Counter pass
    per comparison that a plain tuple compare doesn't.
    """
    if abort is not None and abort.is_set():
        raise AnalysisAborted
    if _state is None:
        _state = game._hand_state()
    if not game.legal_moves:
        # (other, mover): the original returned (p2, p1) with p1 to move
        # and (p1, p2) with p2 to move - both are (other, mover).
        mover_score = _cached_score(_state[0], _state[1])
        other_score = _cached_score(_state[2], _state[3])
        diff = mover_score - other_score
        m = game.multiplicity
        agg = _Agg(
            m,
            w=(m if diff > 0 else 0),
            d=(m if diff == 0 else 0),
            s=diff * m,
            mover_sum=mover_score * m,
        )
        return agg, Counter({(other_score, mover_score): m})
    child_state = game._child_hand_state
    best_key = None
    best_pairs = None
    best_agg = None
    for move in game.all_moves():
        combined = None
        pairs = Counter()
        for possibility in move:
            child_agg, child_pairs = _collect_terminals(
                possibility, child_state(_state, possibility.taken_card), abort
            )
            combined = child_agg if combined is None else combined + child_agg
            # flip the child's (other, mover) into this node's perspective
            for (a, b), v in child_pairs.items():
                pairs[(b, a)] += v
        candidate_agg = -combined
        candidate = (candidate_agg.key, move[0].marker)
        if best_key is None or candidate > best_key:
            best_key = candidate
            best_agg = candidate_agg
            best_pairs = pairs
    return best_agg, best_pairs


def analyse_moves(game, abort=None):
    """Compute offensive, defensive, and combined values for every legal move.

    Parameters
    ----------
    game : Game
        Current game state. Must have at least one legal move.
    abort : threading.Event, optional
        When set, the walk raises AnalysisAborted at the next node - the
        cooperative escape hatch for long-running background analyses.

    Returns
    -------
    dict
        {marker: {...}} - see the fields assembled below. Exactly one move
        has "best": True, chosen the same way Game.evaluate chooses its own
        best move (see "eval" below) - not by whichever has the highest mean
        score-difference, which can disagree with it. Every move gets exact
        win/draw/loss/mean stats cheaply (via _collect_aggregate); only the
        *best* move's "distribution" is populated (the one thing that's
        actually expensive to obtain - see _collect_terminals) - it's `None`
        for every other move, since nothing in this codebase ever reads a
        non-winner's distribution (only the web review page's heatmap does,
        and only for the winner).

    Raises
    ------
    ValueError
        If the game has no legal moves (already terminal).
    AnalysisAborted
        If `abort` was set while the walk was in progress.
    """
    if not game.legal_moves:
        raise ValueError("Game is already over — no legal moves to analyse.")

    state = game._hand_state()
    move_data = {}
    for move_tuple in game.all_moves():
        marker = move_tuple[0].marker
        card = game.board[marker[0]][marker[1]]

        combined = None
        for resolution in move_tuple:
            child_agg = _collect_aggregate(
                resolution, game._child_hand_state(state, resolution.taken_card), abort
            )
            combined = child_agg if combined is None else combined + child_agg
        # One negation brings this from the resolutions' own (opponent's)
        # perspective back to the analysed player's - same convention
        # _collect_terminals's negation uses.
        agg = -combined

        total_weight = agg.m
        player_sum = agg.mover_sum
        opponent_sum = agg.mover_sum - agg.s

        move_data[marker] = {
            "card": card,
            "player_mean": player_sum / total_weight,
            "opponent_mean": opponent_sum / total_weight,
            "mean_diff": agg.s / total_weight,
            "distribution": None,  # filled in for the winner only, below
            "win_pct": 100 * agg.w / total_weight,
            "draw_pct": 100 * agg.d / total_weight,
            "loss_pct": 100 * (total_weight - agg.w - agg.d) / total_weight,
            "eval": Eval(total_weight, agg.w, agg.d, agg.s),
        }

    # Baseline = best move for the current player, by the same (eval, marker)
    # tie-break already used throughout the codebase (_collect_terminals,
    # score_walk, Game.evaluate, evaluate_simple) - not by mean_diff, which
    # can tie or disagree between moves with different win/draw shapes even
    # though "eval" tells them apart (a guaranteed draw vs. a 50/50 win-or-
    # matching-loss can have identical means).
    best_marker, best_key = None, None
    for marker, data in move_data.items():
        candidate = (data["eval"], marker)
        if best_key is None or candidate > best_key:
            best_key, best_marker = candidate, marker
    best = move_data[best_marker]
    baseline_player = best["player_mean"]
    baseline_opponent = best["opponent_mean"]
    baseline_diff = best["mean_diff"]

    # Compute deltas for every move
    for data in move_data.values():
        data["offensive"] = data["opponent_mean"] - baseline_opponent
        data["defensive"] = data["player_mean"] - baseline_player
        data["combined"] = data["mean_diff"] - baseline_diff
        # Not the same as "combined == 0": two moves can tie on mean_diff
        # (and therefore both compute combined == 0) while differing in
        # win/draw shape, in which case only one of them is actually best.
        data["best"] = data is best

    # The one expensive part (the full outcome distribution, needed only for
    # the heatmap) - built only for the winner, via the unchanged pairs walk.
    best_move_tuple = next(m for m in game.all_moves() if m[0].marker == best_marker)
    acc = Counter()
    for resolution in best_move_tuple:
        acc += _collect_terminals(
            resolution, game._child_hand_state(state, resolution.taken_card), abort
        )[1]
    distribution = Counter()
    for (p, q), w in acc.items():
        distribution[p - q] += w
    best["distribution"] = distribution

    return move_data
