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

from .game import ProbEval, _cached_score

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


def _collect_terminals(game, _state=None, abort=None):
    """Traverse the game tree under optimal play, returning a weighted
    score frequency map of (other, mover) score pairs.

    Single pass: the optimal-play pair distribution is carried up
    alongside the negamax value, instead of re-running a full score_walk
    at every level of the optimal line as the original formulation did.
    Move selection replicates score_walk exactly - the same evaluation
    comparison and the same marker tie-break - so the chosen line, and
    therefore the returned distribution, is identical.
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
        return (
            ProbEval(game.multiplicity, {mover_score - other_score: game.multiplicity}),
            Counter({(other_score, mover_score): game.multiplicity}),
        )
    child_state = game._child_hand_state
    best_key = None
    best_pairs = None
    for move in game.all_moves():
        evals = []
        pairs = Counter()
        for possibility in move:
            child_eval, child_pairs = _collect_terminals(
                possibility, child_state(_state, possibility.taken_card), abort
            )
            evals.append(child_eval)
            # flip the child's (other, mover) into this node's perspective
            for (a, b), v in child_pairs.items():
                pairs[(b, a)] += v
        candidate = (-ProbEval.combine(evals), move[0].marker)
        if best_key is None or candidate > best_key:
            best_key = candidate
            best_pairs = pairs
    return best_key[0], best_pairs


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
    MoveAnalysis
        Object with .summary() and .narrative() methods.

    Raises
    ------
    ValueError
        If the game has no legal moves (already terminal).
    AnalysisAborted
        If `abort` was set while the walk was in progress.
    """
    if not game.legal_moves:
        raise ValueError("Game is already over — no legal moves to analyse.")

    # Collect terminal (p1, p2) score distributions for each legal move
    state = game._hand_state()
    move_data = {}
    for move_tuple in game.all_moves():
        marker = move_tuple[0].marker
        card = game.board[marker[0]][marker[1]]

        acc = Counter()
        for resolution in move_tuple:
            acc += _collect_terminals(
                resolution, game._child_hand_state(state, resolution.taken_card), abort
            )[1]

        # Weighted mean: each (p1, p2) score pair is weighted by the number of
        # face-down card orderings that produce it (game.multiplicity at that terminal).
        total_weight = sum(acc.values())
        mean_player = sum(p * w for (p, _), w in acc.items()) / total_weight
        mean_opponent = sum(q * w for (_, q), w in acc.items()) / total_weight
        mean_diff = mean_player - mean_opponent  # always P1 - P2

        # Same (p, q) pairs, collapsed to the score-difference distribution
        # under optimal play - the face-down orderings this move doesn't
        # resolve are exactly why a single mean hides real spread.
        distribution = Counter()
        for (p, q), w in acc.items():
            distribution[p - q] += w

        move_data[marker] = {
            "card": card,
            "player_mean": mean_player,
            "opponent_mean": mean_opponent,
            "mean_diff": mean_diff,
            "distribution": distribution,
        }

    # Baseline = best move for the current player
    best = max(move_data.values(), key=lambda d: d["mean_diff"])
    baseline_player = best["player_mean"]
    baseline_opponent = best["opponent_mean"]
    baseline_diff = best["mean_diff"]

    # Compute deltas for every move
    for data in move_data.values():
        data["offensive"] = data["opponent_mean"] - baseline_opponent
        data["defensive"] = data["player_mean"] - baseline_player
        data["combined"] = data["mean_diff"] - baseline_diff

    return move_data


class MoveAnalysis:
    def __init__(
        self,
        move_data,
        baseline_p1,
        baseline_p2,
        baseline_diff,
        current_player,
        opponent,
        is_p1_turn,
        current_p1_score,
        current_p2_score,
    ):
        self.move_data = move_data
        self.baseline_p1 = baseline_p1
        self.baseline_p2 = baseline_p2
        self.baseline_diff = baseline_diff
        self.current_player = current_player
        self.opponent = opponent
        self.is_p1_turn = is_p1_turn
        self.current_p1_score = current_p1_score
        self.current_p2_score = current_p2_score

    def _sorted_moves(self):
        """Return moves sorted best-to-worst for the current player."""
        return sorted(
            self.move_data.items(),
            key=lambda x: x[1]["combined"],
            reverse=True,
        )

    def summary(self):
        """Return a human-readable table of all moves with their values."""
        lines = []
        cp = self.current_player
        op = self.opponent

        lines.append(f"Move analysis — {cp}'s turn")
        lines.append(
            f"Current score: P1={self.current_p1_score}  P2={self.current_p2_score}"
        )
        lines.append(
            f"Under optimal play: best move leads to "
            f"P1={self.baseline_p1:.1f}  P2={self.baseline_p2:.1f}  "
            f"diff={self.baseline_diff:+.1f}"
        )
        lines.append(f"Deltas below are relative to the best move (best move = 0.0).")
        lines.append("")

        hdr = (
            f"  {'Move':<8}  {'Card':<6}  {'Combined':>9}  "
            f"{'Offensive':>10}  {'Defensive':>10}  {'n':>5}"
        )
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))

        for marker, data in self._sorted_moves():
            comb = data["combined"]
            off = data["offensive"]
            defv = data["defensive"]
            n = data["n"]
            card = data["card"]

            # Tag the move type
            tag = ""
            if comb == 0.0:
                tag = "  ← best"
            elif abs(off) > abs(defv) + 0.5:
                tag = "  [mainly offensive cost]"
            elif abs(defv) > abs(off) + 0.5:
                tag = "  [mainly defensive cost]"

            lines.append(
                f"  {str(marker):<8}  {str(card):<6}  {comb:>+9.2f}  "
                f"{off:>+10.2f}  {defv:>+10.2f}  {n:>5}{tag}"
            )

        lines.append("")
        lines.append(f"  Combined  = swing in ({cp} score − {op} score) vs best move")
        lines.append(f"  Offensive = change in {cp}'s own final score vs best move")
        lines.append(f"  Defensive = change in {op}'s final score vs best move")
        lines.append(
            f"              (negative defensive = {op} scores less = good for {cp})"
        )
        lines.append(
            f"  n         = number of equally-likely face-down card orderings represented"
        )

        return "\n".join(lines)

    def narrative(self):
        """Return natural-language sentences explaining the key move contrasts."""
        lines = []
        cp = self.current_player
        op = self.opponent
        sorted_moves = self._sorted_moves()

        best_marker, best_data = sorted_moves[0]
        worst_marker, worst_data = sorted_moves[-1]

        # Best move description
        best_card = best_data["card"]
        if best_card.facedown:
            lines.append(
                f"Best move: take the face-down card at {best_marker}. "
                f"Under optimal play from here, {cp} leads by "
                f"{self.baseline_diff:+.1f} pts on average "
                f"(P1={self.baseline_p1:.1f}, P2={self.baseline_p2:.1f})."
            )
        else:
            lines.append(
                f"Best move: take {best_card} at {best_marker}. "
                f"Under optimal play from here, {cp} leads by "
                f"{self.baseline_diff:+.1f} pts on average "
                f"(P1={self.baseline_p1:.1f}, P2={self.baseline_p2:.1f})."
            )

        # Worst move description with offensive/defensive breakdown
        if len(sorted_moves) > 1:
            w_card = worst_data["card"]
            w_comb = worst_data["combined"]
            w_off = worst_data["offensive"]
            w_def = worst_data["defensive"]

            card_str = (
                f"the face-down card at {worst_marker}"
                if w_card.facedown
                else f"{w_card} at {worst_marker}"
            )

            lines.append(
                f"\nWorst move: take {card_str} ({w_comb:+.2f} pts combined vs best). "
            )

            # Explain why it's bad
            if abs(w_off) > abs(w_def) + 0.5:
                lines.append(
                    f"This is mainly an offensive cost: {cp} scores "
                    f"{abs(w_off):.1f} pts less on average. "
                    f"The impact on {op}'s score is smaller ({abs(w_def):.1f} pts)."
                )
            elif abs(w_def) > abs(w_off) + 0.5:
                lines.append(
                    f"This is mainly a defensive cost: it allows {op} to score "
                    f"{abs(w_def):.1f} more pts on average. "
                    f"{cp}'s own score drops by less ({abs(w_off):.1f} pts)."
                )
            else:
                lines.append(
                    f"The cost is roughly split: {cp} scores {abs(w_off):.1f} pts less "
                    f"and {op} scores {abs(w_def):.1f} pts more."
                )

        # Any interesting middle moves
        for marker, data in sorted_moves[1:-1]:
            off = data["offensive"]
            defv = data["defensive"]
            card = data["card"]
            card_str = f"face-down at {marker}" if card.facedown else f"{card}"

            if abs(defv) > abs(off) + 1.5:
                lines.append(
                    f"\n{card_str}: primarily a denial move — "
                    f"taking it prevents {op} from scoring {abs(defv):.1f} extra pts, "
                    f"while adding {abs(off):.1f} pts to {cp}'s own score."
                )
            elif abs(off) > abs(defv) + 1.5:
                lines.append(
                    f"\n{card_str}: primarily an offensive move — "
                    f"adds {abs(off):.1f} pts to {cp}'s score "
                    f"but only denies {op} {abs(defv):.1f} pts."
                )

        return "\n".join(lines)

    def __repr__(self):
        return self.summary()
