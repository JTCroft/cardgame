"""
move_analysis.py — per-move offensive/defensive analysis for Cross Kings
-------------------------------------------------------------------------
For each legal move from a position, computes how much it helps the current
player (offensive value), how much it hurts the opponent (defensive value),
and the combined score-difference swing — all under optimal play by both
sides from that point forward.
"""

from collections import Counter

__all__ = ("analyse_moves",)


def _collect_terminals(game):
    """Traverse the game tree under optimal play, returning a weighted score frequency map."""
    if not game.legal_moves:
        if len(game.moves) % 2 == 0:
            return Counter({(game.p2.score(), game.p1.score()): game.multiplicity})
        return Counter({(game.p1.score(), game.p2.score()): game.multiplicity})
    _, best_pos = game.score_walk()
    result = Counter()
    for resolution in game.move(*best_pos):
        result.update(
            {(k2, k1): v for (k1, k2), v in _collect_terminals(resolution).items()}
        )
    return result


def analyse_moves(game):
    """Compute offensive, defensive, and combined values for every legal move.

    Parameters
    ----------
    game : Game
        Current game state. Must have at least one legal move.

    Returns
    -------
    MoveAnalysis
        Object with .summary() and .narrative() methods.

    Raises
    ------
    ValueError
        If the game has no legal moves (already terminal).
    """
    if not game.legal_moves:
        raise ValueError("Game is already over — no legal moves to analyse.")

    # Collect terminal (p1, p2) score distributions for each legal move
    move_data = {}
    for move_tuple in game.all_moves():
        marker = move_tuple[0].marker
        card = game.board[marker[0]][marker[1]]

        acc = Counter()
        for resolution in move_tuple:
            acc += _collect_terminals(resolution)

        # Weighted mean: each (p1, p2) score pair is weighted by the number of
        # face-down card orderings that produce it (game.multiplicity at that terminal).
        total_weight = sum(acc.values())
        mean_player = sum(p * w for (p, _), w in acc.items()) / total_weight
        mean_opponent = sum(q * w for (_, q), w in acc.items()) / total_weight
        mean_diff = mean_player - mean_opponent  # always P1 - P2

        move_data[marker] = {
            "card": card,
            "player_mean": mean_player,
            "opponent_mean": mean_opponent,
            "mean_diff": mean_diff,
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
