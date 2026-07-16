"""Best-first, interval-bounded search for Cross Kings.

`Game.evaluate` is depth-first: it can say almost nothing about the best
move until it has nearly finished. This module explores the same game tree
best-first, keeping every partially explored branch's value pinned inside
provable bounds, so that at any moment it can rank the root moves, prune
any whose upper bound falls below a rival's lower bound, and stop as soon
as a single move survives - typically long before the position is fully
valued. `run()` is a generator of ranking snapshots, so a caller can show
a live best-to-worst view that updates as branches are explored, pruned,
or better bounded.

How the bounds work
-------------------
Any completed game's score difference lies within the global
+/-_SCORE_DIFFERENCE_BOUND (inherited assumption - see game.py), so a
branch observed to multiplicity m of n is bounded by filling the missing
n - m outcomes with -26 (pessimistic) or +26 (optimistic), via
`ProbEval.bound`. Such a partial distribution bounds the true one
outcome-for-outcome. That pointwise coupling is the load-bearing choice:
it survives negation (the bare eval ordering does not, because its middle
prefer-decisive term is not symmetric between the players) and it
survives summation at chance nodes, so bounds cascade soundly up the
tree:

* a face-down move's distribution accumulates one resolved possibility at
  a time - resolving one of k unknowns cuts the filled mass by a factor k;
* a position exports its value upward as soon as its best move is
  *proven* (every rival's upper bound at or below its lower bound, ties
  broken by marker as in `Game.evaluate`), even while that move's own
  subtree is still partly unexplored;
* subtrees small enough for `Game.evaluate` to crack quickly are solved
  exactly and export exact values immediately.

Expansion policy is prove-best / disprove-rest: descend the alive move
with the highest upper bound (either raising the leader's floor or
cutting a challenger's ceiling - whichever can still change the root
decision), and inside a chance node keep working on one unresolved child
until it exports, since nothing tightens until a child proves or solves.

Usage
-----
    search = BestFirstSearch(game)
    for snapshot in search.run(budget=30):
        print(format_snapshot(snapshot))
    best_move = search.result()["best"]
"""

import time

from .game import Eval, ProbEval

__all__ = ("BestFirstSearch", "format_snapshot")


# From evaluate()'s cost calibration: positions with at most 8 cards left
# on the board solve exactly in at most ~0.2s whatever the face-down count,
# 9 in ~0.7s, 10 in a few seconds. Each extra card here removes a full ply
# from the explicit best-first tree (exponentially fewer expansions) in
# exchange for pricier individual leaf solves.
_DEFAULT_LEAF_CARDS = 9


def _terminal_dist(game):
    multiplicity = game.multiplicity
    return ProbEval(multiplicity, {game.negamax_score: multiplicity})


class _Arc:
    """One move out of a node: the (possibly several, for a face-down
    move) resolution children, and the running combination of what they
    have exported so far."""

    __slots__ = ("marker", "card", "children", "acc", "lo", "hi", "dead")

    def __init__(self, marker, card, children):
        self.marker = marker
        self.card = card
        self.children = children
        self.dead = False
        self.rebuild()

    def rebuild(self):
        # Children export in their own mover's perspective; negating flips
        # each observed outcome, so the pointwise bounding survives, and
        # combining sums the (equally likely) resolutions.
        self.acc = ProbEval.combine([-child.export() for child in self.children])
        self.lo = self.acc.lower_bound.eval
        self.hi = self.acc.upper_bound.eval

    @property
    def complete(self):
        return self.acc.observed == self.acc.multiplicity

    @property
    def observed_eval(self):
        """Average over the resolved share of this move's outcomes - the
        search's current point estimate of the move's value, on the logic
        that outcomes near each other in the tree are correlated, so the
        exactly-known resolutions are representative of the unresolved
        ones. None until something has resolved."""
        observed = self.acc.observed
        if not observed:
            return None
        return Eval(observed, *self.acc.observed_wds)


class _Node:
    """A position in the explicit tree."""

    __slots__ = ("game", "multiplicity", "expanded", "solved", "dist", "arcs", "proven")

    def __init__(self, game):
        self.game = game
        self.multiplicity = game.multiplicity
        self.expanded = False
        self.solved = False
        self.dist = None  # exact ProbEval once solved
        self.arcs = None
        self.proven = None  # the single surviving arc, once known

    def export(self):
        """The tightest value distribution this node can currently promise
        its parent: exact when solved, the proven best move's partial
        (exact observed mass, rest implicit fill) once the best move is
        known, pure fill otherwise. Only proven mass may be exported - a
        merely-leading move's mass could belong to the wrong move."""
        if self.solved:
            return self.dist
        if self.proven is not None:
            return self.proven.acc
        return ProbEval(self.multiplicity)

    def expand(self, leaf_cards, allow_exact=True):
        game = self.game
        if not game.legal_moves:
            self.solved = True
            self.dist = _terminal_dist(game)
            return
        if allow_exact and 36 - len(game.moves) <= leaf_cards:
            self.solved = True
            self.dist = game.evaluate()["Evaluation"]
            return
        arcs = []
        for move in game.all_moves():
            marker = move[0].marker
            children = [_Node(g) for g in move]
            for child in children:
                if not child.game.legal_moves:
                    child.solved = True
                    child.dist = _terminal_dist(child.game)
            arcs.append(_Arc(marker, game.board[marker[0]][marker[1]], children))
        arcs.sort(key=lambda arc: arc.marker)
        self.arcs = arcs
        self.expanded = True
        self.refresh()

    def refresh(self):
        """Re-derive deadness / proven / solved from current arc bounds.
        Bounds only ever tighten, so deadness is sticky and `proven` never
        reverts."""
        alive = [arc for arc in self.arcs if not arc.dead]
        best = max((arc.lo, arc.marker) for arc in alive)
        for arc in alive:
            # Strictly below a rival's guarantee can never be best; a tie
            # with a larger-marker rival loses the deterministic tie-break
            # (the same larger-marker-wins rule Game.evaluate uses).
            if (arc.hi, arc.marker) < best:
                arc.dead = True
        alive = [arc for arc in alive if not arc.dead]
        if len(alive) == 1:
            self.proven = alive[0]
            if self.proven.complete:
                self.solved = True
                self.dist = self.proven.acc


class BestFirstSearch:
    def __init__(self, game, leaf_cards=_DEFAULT_LEAF_CARDS):
        if not game.legal_moves:
            raise ValueError("Game is already over - nothing to search.")
        self.root = _Node(game)
        self.leaf_cards = leaf_cards
        self.expansions = 0
        self._start = None

    # -- expansion ---------------------------------------------------------

    def _descend(self):
        """Walk from the root to the frontier node whose expansion is most
        likely to change the root decision. Returns (path, leaf) where
        path is the [(node, arc)] chain to rebuild afterwards."""
        path = []
        node = self.root
        while node.expanded and not node.solved:
            if node.proven is not None:
                arc = node.proven
            else:
                candidates = [
                    arc for arc in node.arcs if not arc.dead and not arc.complete
                ]
                arc = max(candidates, key=lambda a: (a.hi, a.marker))
            # Concentrate on one unresolved possibility: nothing tightens
            # until a child proves its best move or solves outright.
            child = max(
                (c for c in arc.children if not c.solved),
                key=lambda c: c.multiplicity - c.export().observed,
            )
            path.append((node, arc))
            node = child
        return path, node

    def _step(self):
        path, leaf = self._descend()
        leaf.expand(self.leaf_cards)
        for node, arc in reversed(path):
            arc.rebuild()
            node.refresh()
        self.expansions += 1

    # -- driving -----------------------------------------------------------

    def run(self, budget=None, max_expansions=None, snapshot_every=25, stop="proven"):
        """Search until the best move is proven (stop="proven"), the root
        value is exact (stop="solved"), or a budget runs out - yielding a
        ranking snapshot every `snapshot_every` expansions and once at the
        end. Safe to call again after an early stop: it resumes."""
        self._start = time.perf_counter()
        deadline = None if budget is None else self._start + budget
        if not self.root.expanded:
            # Never exact-solve the root silently - expanding it is what
            # gives the caller a per-move ranking.
            self.root.expand(self.leaf_cards, allow_exact=False)
        while not self._done(stop):
            self._step()
            if self.expansions % snapshot_every == 0:
                yield self.snapshot()
            if deadline is not None and time.perf_counter() > deadline:
                break
            if max_expansions is not None and self.expansions >= max_expansions:
                break
        yield self.snapshot()

    def _done(self, stop):
        if self.root.solved:
            return True
        return stop == "proven" and self.root.proven is not None

    # -- reporting ---------------------------------------------------------

    @property
    def proven(self):
        return self.root.proven is not None or self.root.solved

    @property
    def best_move(self):
        """The move to play if we stop now: the proven survivor when there
        is one, otherwise the best point estimate (average of resolved
        outcomes - see _Arc.observed_eval), falling back to optimism while
        nothing at all has resolved."""
        alive = [arc for arc in self.root.arcs if not arc.dead]
        if len(alive) == 1:
            return alive[0].marker
        estimated = [
            (arc.observed_eval, arc.marker)
            for arc in alive
            if arc.observed_eval is not None
        ]
        if estimated:
            return max(estimated)[1]
        return max((arc.hi, arc.marker) for arc in alive)[1]

    @staticmethod
    def _ranked(arcs):
        """Best-to-worst for display: point estimates first (resolved
        average, descending), then still-unestimated moves by optimism."""
        estimated = [arc for arc in arcs if arc.observed_eval is not None]
        unestimated = [arc for arc in arcs if arc.observed_eval is None]
        estimated.sort(key=lambda a: (a.observed_eval, a.marker), reverse=True)
        unestimated.sort(key=lambda a: (a.hi, a.marker), reverse=True)
        return estimated + unestimated

    def snapshot(self):
        root = self.root
        ranked = self._ranked(
            [arc for arc in root.arcs if not arc.dead]
        ) + self._ranked([arc for arc in root.arcs if arc.dead])
        moves = []
        for arc in ranked:
            lo, hi = arc.acc.lower_bound.eval, arc.acc.upper_bound.eval
            mean = arc.observed_eval
            moves.append(
                {
                    "marker": arc.marker,
                    "card": str(arc.card),
                    "alive": not arc.dead,
                    "explored": arc.acc.observed / root.multiplicity,
                    # normed_eval = (win rate counting draws half, win rate,
                    # mean score difference)
                    "win_bounds": (lo.normed_eval[0], hi.normed_eval[0]),
                    "score_bounds": (lo.normed_eval[2], hi.normed_eval[2]),
                    "mean_win": None if mean is None else mean.normed_eval[0],
                    "mean_score": None if mean is None else mean.normed_eval[2],
                }
            )
        return {
            "best": self.best_move,
            "proven": self.proven,
            "solved": root.solved,
            "expansions": self.expansions,
            "elapsed": 0.0 if self._start is None else time.perf_counter() - self._start,
            "moves": moves,
        }

    def result(self):
        snapshot = self.snapshot()
        snapshot["best_card"] = next(
            m["card"] for m in snapshot["moves"] if m["marker"] == snapshot["best"]
        )
        return snapshot


def format_snapshot(snapshot):
    """Render a snapshot as a live best-to-worst text table."""
    status = "SOLVED" if snapshot["solved"] else ("PROVEN" if snapshot["proven"] else "searching")
    lines = [
        f"[{status}] best so far: {snapshot['best']}  "
        f"({snapshot['expansions']} expansions, {snapshot['elapsed']:.1f}s)"
    ]
    for move in snapshot["moves"]:
        w_lo, w_hi = (100 * w for w in move["win_bounds"])
        s_lo, s_hi = move["score_bounds"]
        if move["mean_score"] is None:
            avg = "avg    --  "
        else:
            avg = f"avg {100 * move['mean_win']:3.0f}% {move['mean_score']:+5.2f}"
        tag = "best" if (move["alive"] and move["marker"] == snapshot["best"]) else (
            "alive" if move["alive"] else "dead"
        )
        lines.append(
            f"  {move['card']:>3} {str(move['marker']):>6}  {avg}  "
            f"win {w_lo:5.1f}..{w_hi:5.1f}%  pts {s_lo:+6.2f}..{s_hi:+6.2f}  "
            f"explored {100 * move['explored']:5.1f}%  {tag}"
        )
    return "\n".join(lines)
