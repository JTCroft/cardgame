"""An iterative search that yields partial results

An implementation of a search algorithm that is specifically designed for
games with elements of randomness (Chance nodes) and a bounded score.

The move_search_iterator takes in a Game position and returns a PositionNode
iterator which will, when iterated over, yield information about the partial
node evaluation up until it is exhausted after which the information can be
accessed in the class attributes.

The logic is that every time next() is called on a PositionNode or ChanceNode
it will
- figure out which child node(s) should be prioritised to explore based on
prioritising unexplored, and then child nodes with the best known (explored) outcomes
- calculate any subalpha/subbeta bounds for that move that would cause pruning
for that child node and assign them to the child node iterator
- call next on that child node which will yield a result when it has improved its bounds
- recalculate it's own bounds, and yield the new result if they have changed

The first result from a ChanceNode will be the result of calling next() on all of
its children nodes.

Any unknown bounds are set to the maximum score difference between 2 players, 26

So each update (yield) from the main position node iterator will return
- a minimum and maximum possible evaluation
- the current best to worst move rankings

Implementation notes
--------------------
This is a *resumable, reorderable Game.evaluate*: every piece of window
logic is lifted verbatim from the depth-first solver so its correctness
carries over -

* unknown outcome mass is filled at +/-26 (`ProbEval.bound`), and a move's
  partial distribution tightens one resolved face-down possibility at a
  time;
* sub-windows for a chance possibility come from `Game._get_bounds`, and a
  child is evaluated under the negated window exactly as evaluate would
  call it;
* a move is abandoned (pruned) when its optimistic completion falls below
  alpha, best-move switches use the same
  `(move_score, marker) > (best_score, best_marker)` bound comparison
  (which never regrets: it only fires when the challenger's floor clears
  the incumbent's ceiling), and fail-highs use the same
  `alpha > beta and not -alpha > -beta` guard that patches the eval
  ordering's asymmetry under negation.

What is new is *when* things run, not what runs:

* each next() advances the tree by one quantum (one expansion or one exact
  leaf solve) instead of recursing to completion, so the root can report
  bounds and rankings continuously;
* windows are re-derived on every resume - they only ever tighten (alpha
  rises monotonically), which is sound for the same reason evaluate's
  fixed windows are;
* positions small enough for the calibrated exact gate are solved by
  Game.evaluate *under the current window*, so ancestors' constraints cut
  inside leaf solves too;
* the search can stop with the best move *proven* - every rival abandoned
  or completed worse - before the best move's own value is exact, a stop
  the all-or-nothing depth-first solver cannot make;
* children are instantiated lazily (no resolution Games or child nodes
  exist until a branch is first descended into) and deleted eagerly (a
  merged, abandoned, or resolved subtree is released immediately), so live
  memory tracks what is in contention, not the whole tree.
"""

import time

from .game import Eval, Game, ProbEval

__all__ = ("move_search_iterator", "live_search", "PositionNode", "ChanceNode")

_DEFAULT_LEAF_CARDS = 9


class _Context:
    """Search-wide state: the exact-solve gate, work accounting (one unit
    per expansion or exact leaf solve), and node accounting so memory
    behaviour is observable."""

    __slots__ = ("leaf_cards", "work", "live_nodes", "peak_nodes", "created_nodes")

    def __init__(self, leaf_cards):
        self.leaf_cards = leaf_cards
        self.work = 0
        self.live_nodes = 0
        self.peak_nodes = 0
        self.created_nodes = 0

    def node_created(self):
        self.created_nodes += 1
        self.live_nodes += 1
        if self.live_nodes > self.peak_nodes:
            self.peak_nodes = self.live_nodes

    def node_released(self):
        self.live_nodes -= 1


class _Move:
    """One move out of a position: its (lazily created) subtree while
    being explored, and its accumulated partial score for display and for
    the parent's evaluate-identical bookkeeping."""

    __slots__ = ("marker", "card", "facedown", "node", "score", "done", "abandoned")

    def __init__(self, marker, card, facedown):
        self.marker = marker
        self.card = card
        self.facedown = facedown
        self.node = None  # PositionNode (face-up) / ChanceNode (face-down)
        self.score = None  # ProbEval once anything is known
        self.done = False
        self.abandoned = False

    @property
    def pending(self):
        return not self.done and not self.abandoned

    def upper_eval(self, base_upper):
        if self.score is not None:
            return self.score.upper_bound.eval
        return base_upper

    @property
    def observed_eval(self):
        """Average over the resolved share of this move's outcomes - the
        ranking point estimate (nearby leaves correlate, so the resolved
        share is representative of the rest)."""
        if self.score is None or not self.score.observed:
            return None
        return Eval(self.score.observed, *self.score.observed_wds)

    def release_subtree(self):
        if self.node is not None:
            self.node.release()
            self.node = None


class PositionNode:
    """A position being evaluated exactly as Game.evaluate would, but one
    quantum per next() and with moves resumable in any order. Exhausts when
    resolved; `result` then holds the same Evaluation evaluate would have
    returned under this node's window."""

    def __init__(self, game, ctx, allow_exact=True):
        self.game = game
        self.ctx = ctx
        self.multiplicity = game.multiplicity
        self.allow_exact = allow_exact
        base = ProbEval(self.multiplicity)
        self.base_lower = base.lower_bound
        self.base_upper_eval = base.upper_bound.eval
        # (alpha, beta) Evals; parents reassign before every next(). Only
        # ever tightens, which is as sound as evaluate's fixed windows.
        self.window = (self.base_lower.eval, self.base_upper_eval)
        self.moves = None
        self.best_score = self.base_lower
        self.best_marker = (-1, -1)
        self.resolved = False
        self.result = None
        self._released = False
        ctx.node_created()

    # -- lifecycle ---------------------------------------------------------

    def release(self):
        if self._released:
            return
        self._released = True
        if self.moves is not None:
            for move in self.moves:
                move.release_subtree()
        self.ctx.node_released()

    def _resolve(self, result):
        self.result = result
        self.resolved = True
        if self.moves is not None:
            for move in self.moves:
                move.release_subtree()

    def _expand(self):
        game = self.game
        if not game.legal_moves:
            self._resolve(
                ProbEval(self.multiplicity, {game.negamax_score: self.multiplicity})
            )
            return
        if self.allow_exact and 36 - len(game.moves) <= self.ctx.leaf_cards:
            self.ctx.work += 1
            # Solved by the exact depth-first engine *under our window* -
            # its fail-soft result is exactly what evaluate's recursive
            # call would have produced here.
            self._resolve(game.evaluate(self.window[0], self.window[1])["Evaluation"])
            return
        self.ctx.work += 1
        board = game.board
        moves = [
            _Move(marker, board[marker[0]][marker[1]], marker in board.facedown_positions)
            for marker in game.legal_moves
        ]
        # deterministic (cheap, information-dense) first, as evaluate does
        moves.sort(key=lambda move: (move.facedown, move.marker))
        self.moves = moves

    # -- evaluate-identical bookkeeping -------------------------------------

    def _current_alpha(self):
        return max(self.best_score.lower_bound.eval, self.window[0])

    def _guard_fail_high(self, alpha, beta):
        return alpha > beta and not (-alpha) > (-beta)

    def _select(self):
        """Untouched moves first (face-up before face-down, marker order -
        the expansion order evaluate uses), then keep pushing the current
        best move, then the strongest remaining challenger."""
        pending = [move for move in self.moves if move.pending]
        if not pending:
            return None
        untouched = [move for move in pending if move.node is None]
        if untouched:
            return untouched[0]
        for move in pending:
            if move.marker == self.best_marker:
                return move
        return max(pending, key=lambda m: (m.upper_eval(self.base_upper_eval), m.marker))

    def _advance(self, move, alpha, beta):
        if not move.facedown:
            if move.node is None:
                move.node = PositionNode(self.game.move(*move.marker)[0], self.ctx)
            child = move.node
            child.window = (-beta, -alpha)
            try:
                next(child)
            except StopIteration:
                pass
            if child.resolved:
                score = -(child.result)
                move.score = score
                move.done = True
                child.release()
                move.node = None
                if (score, move.marker) > (self.best_score, self.best_marker):
                    self.best_score = score
                    self.best_marker = move.marker
        else:
            if move.node is None:
                move.node = ChanceNode(self.game, move.marker, self.ctx)
            chance = move.node
            chance.window = (alpha, beta)
            try:
                next(chance)
            except StopIteration:
                pass
            move.score = chance.move_score
            # Bookkeeping in evaluate's order: best-switch check, then the
            # optimistic-completion abandonment against the alpha this
            # quantum started with.
            if (chance.move_score, move.marker) > (self.best_score, self.best_marker):
                # May alias a still-growing move_score - as in evaluate.
                self.best_score = chance.move_score
                self.best_marker = move.marker
            if chance.done:
                move.done = True
                chance.release()
                move.node = None
            elif chance.move_score.upper_bound.eval < alpha:
                move.abandoned = True
                move.release_subtree()

    # -- iteration -----------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self):
        if self.resolved:
            raise StopIteration
        if self.moves is None:
            self._expand()
            return self
        alpha = self._current_alpha()
        beta = self.window[1]
        if self._guard_fail_high(alpha, beta):
            self._resolve(self.best_score)
            return self
        move = self._select()
        if move is None:
            self._resolve(self.best_score)
            return self
        self._advance(move, alpha, beta)
        return self

    # -- reporting -----------------------------------------------------------

    @property
    def proven(self):
        """True once no move other than the current best can still act:
        every rival is abandoned or completed (completions never overtake -
        best-switches require clearing the incumbent's ceiling)."""
        if self.resolved:
            return True
        if self.moves is None or self.best_marker == (-1, -1):
            return False
        return not any(
            move.pending and move.marker != self.best_marker for move in self.moves
        )

    @property
    def best_move(self):
        """The move to play if we stop now: evaluate's incumbent when one
        exists, else the best average over resolved outcomes, else marker
        order."""
        if self.best_marker != (-1, -1):
            return self.best_marker
        candidates = [move for move in self.moves if not move.abandoned]
        estimated = [
            (move.observed_eval, move.marker)
            for move in candidates
            if move.observed_eval is not None
        ]
        if estimated:
            return max(estimated)[1]
        return max(move.marker for move in candidates)


class ChanceNode:
    """The resolution possibilities of one face-down move. move_score
    accumulates -(resolved child Evaluation) exactly as evaluate's inner
    possibility loop does; children materialise on first touch and are
    deleted as they merge."""

    __slots__ = (
        "game", "marker", "ctx", "multiplicity", "window",
        "children", "branch_multiplicity", "move_score", "done", "_released",
    )

    def __init__(self, game, marker, ctx):
        self.game = game
        self.marker = marker
        self.ctx = ctx
        self.multiplicity = game.multiplicity
        self.window = None  # (alpha, beta) in the parent's perspective
        self.children = None
        self.branch_multiplicity = None
        self.move_score = ProbEval(self.multiplicity)
        self.done = False
        self._released = False
        ctx.node_created()

    def release(self):
        if self._released:
            return
        self._released = True
        if self.children is not None:
            for child in self.children:
                if child is not None:
                    child.release()
            self.children = None
        self.ctx.node_released()

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        if self.children is None:
            resolutions = self.game.move(*self.marker)
            self.children = [PositionNode(g, self.ctx) for g in resolutions]
            self.branch_multiplicity = self.children[0].multiplicity
            return self
        alpha, beta = self.window
        index = next(i for i, child in enumerate(self.children) if child is not None)
        child = self.children[index]
        # evaluate's own sub-window arithmetic, from the *current* partial
        # move_score - recomputing on resume only ever tightens.
        subalpha, subbeta = Game._get_bounds(
            self.branch_multiplicity, self.move_score, alpha, beta
        )
        child.window = (-subbeta, -subalpha)
        try:
            next(child)
        except StopIteration:
            pass
        if child.resolved:
            self.move_score.update(-(child.result))
            child.release()
            self.children[index] = None
            if all(c is None for c in self.children):
                self.done = True
                self.children = None
        return self


def move_search_iterator(game, leaf_cards=_DEFAULT_LEAF_CARDS):
    """The root PositionNode iterator for `game`. next() it to advance the
    search one quantum at a time; it exhausts when the position is resolved
    (result == what Game.evaluate would return), and at any point before
    that exposes bounds, rankings, the incumbent best move, and whether the
    best move is already proven."""
    if not game.legal_moves:
        raise ValueError("Game is already over - nothing to search.")
    ctx = _Context(leaf_cards)
    return PositionNode(game, ctx, allow_exact=False)


def _snapshot(root, elapsed):
    """A ranking snapshot in the same shape as search.BestFirstSearch's, so
    search.format_snapshot renders it unchanged."""
    base_lower_eval = root.base_lower.eval
    base_upper_eval = root.base_upper_eval

    def bounds(move):
        if move.score is None:
            return base_lower_eval, base_upper_eval
        return move.score.lower_bound.eval, move.score.upper_bound.eval

    def ranked(moves):
        estimated = [m for m in moves if m.observed_eval is not None]
        rest = [m for m in moves if m.observed_eval is None]
        estimated.sort(key=lambda m: (m.observed_eval, m.marker), reverse=True)
        rest.sort(key=lambda m: m.marker, reverse=True)
        return estimated + rest

    def in_contention(move):
        # pending moves can still become best; a completed non-best or an
        # abandoned move never can (switches require the challenger's
        # floor to clear the incumbent's ceiling)
        return move.pending or move.marker == root.best_marker

    entries = []
    alive = ranked([m for m in root.moves if in_contention(m)])
    dead = ranked([m for m in root.moves if not in_contention(m)])
    for move in alive + dead:
        lo, hi = bounds(move)
        mean = move.observed_eval
        observed = move.score.observed if move.score is not None else 0
        entries.append(
            {
                "marker": move.marker,
                "card": str(move.card),
                "alive": in_contention(move),
                "explored": observed / root.multiplicity,
                "win_bounds": (lo.normed_eval[0], hi.normed_eval[0]),
                "score_bounds": (lo.normed_eval[2], hi.normed_eval[2]),
                "mean_win": None if mean is None else mean.normed_eval[0],
                "mean_score": None if mean is None else mean.normed_eval[2],
            }
        )
    return {
        "best": root.best_move,
        "proven": root.proven,
        "solved": root.resolved,
        "expansions": root.ctx.work,
        "elapsed": elapsed,
        "moves": entries,
    }


def live_search(game, budget=None, stop="proven", leaf_cards=_DEFAULT_LEAF_CARDS,
                snapshot_every=25):
    """Convenience generator over move_search_iterator yielding display
    snapshots (compatible with search.format_snapshot) until the best move
    is proven (stop="proven"), the value is exact (stop="solved"), or the
    budget runs out."""
    root = move_search_iterator(game, leaf_cards=leaf_cards)
    start = time.perf_counter()
    deadline = None if budget is None else start + budget
    next(root)  # the expansion quantum: establishes the move list
    yield _snapshot(root, time.perf_counter() - start)
    steps = 0
    while not (root.resolved or (stop == "proven" and root.proven)):
        try:
            next(root)
        except StopIteration:
            break
        steps += 1
        if steps % snapshot_every == 0:
            yield _snapshot(root, time.perf_counter() - start)
        if deadline is not None and time.perf_counter() > deadline:
            break
    yield _snapshot(root, time.perf_counter() - start)
