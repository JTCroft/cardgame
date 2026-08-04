from jinja2 import Environment, PackageLoader
from random import shuffle, choice
from operator import itemgetter, lt, le, gt, ge, eq
from itertools import groupby
from math import factorial
from collections import Counter
from functools import lru_cache
from .scoring import score_dp, score_with_king_allocation
from .cards import Card, Rank

__all__ = ("Board", "Hand", "Game", "ProbEval", "Eval")

env = Environment(
    loader=PackageLoader(package_name="cardgame", package_path="../../templates")
)

# Approximate (unproven) upper bound on the score difference between two hands;
# these two disjoint hands reach it: 31 vs 5 points.
_SCORE_DIFFERENCE_BOUND = 26

# Global scoring cache, shared across the tree search where hands recur
# constantly. Hand.score routes through this too.
_cached_score = lru_cache(maxsize=1 << 20)(score_dp)

_FACTORIAL = tuple(factorial(n) for n in range(13))


# Legal-move memo shared by every game, keyed by marker cell plus its row and
# column occupancy (at most 36 * 32 * 32 keys). Values are shared frozensets.
_LEGAL_MEMO = {}


def _common_prefix(move_sequences):
    prefix = []
    for elements in zip(*move_sequences):
        if len(set(elements)) == 1:
            prefix.append(elements[0])
        else:
            break
    return tuple(prefix)


class Board(tuple):
    template = env.get_template("board.html.jinja2")
    facedown_indices = set(range(0, 36, 5)) | set(range(0, 36, 7))
    facedown_positions = {(i // 6, i % 6) for i in facedown_indices}

    def __new__(cls, cards, facedown_cards):
        instance = super().__new__(cls, tuple(tuple(row) for row in cards))
        instance.facedown_cards = tuple(facedown_cards)
        return instance

    def __getnewargs__(self):
        # __new__ takes the row layout and facedown_cards separately; unwrap to
        # a plain tuple so pickling doesn't recurse back into this Board.
        return (tuple(self), self.facedown_cards)

    def save(self, alnum=False):
        func = repr if alnum else str
        board_str = "/".join("".join(func(card) for card in row) for row in self)
        fd_str = "".join(func(card) for card in self.facedown_cards)
        return f"{board_str}//{fd_str}"

    @classmethod
    def load(cls, save):
        board, fd = save.split("//")

        def get_str_pairs(input_str):
            while input_str:
                yield input_str[:2]
                input_str = input_str[2:]

        board = [
            tuple(Card.from_str(card) for card in get_str_pairs(row))
            for row in board.split("/")
        ]
        fd = tuple(Card.from_str(card) for card in get_str_pairs(fd))
        return cls(board, fd)

    @classmethod
    def deal(cls):
        deck = list(Card.deck())
        shuffle(deck)
        facedown_cards = itemgetter(*cls.facedown_indices)(deck)
        facedown_card = Card(facedown=True)
        for index in cls.facedown_indices:
            deck[index] = facedown_card
        return cls([deck[i : i + 6] for i in range(0, 36, 6)], facedown_cards)

    def __repr__(self):
        return "\n".join(" ".join(str(card) for card in row) for row in self)

    def _repr_html_(self):
        return self.template.render(board=self)

    def resolve(self, row, col):
        # Reuses the five unchanged row tuples and bypasses __new__'s re-tupling.
        before, target, after = self[:row], self[row], self[row + 1 :]
        left, right = target[:col], target[col + 1 :]
        facedown_cards = self.facedown_cards
        for i, card in enumerate(facedown_cards):
            board = tuple.__new__(
                self.__class__, before + (left + (card,) + right,) + after
            )
            board.facedown_cards = facedown_cards[:i] + facedown_cards[i + 1 :]
            yield board


class Hand(tuple):
    template = env.get_template("hand.html.jinja2")

    def __new__(cls, cards):
        return super().__new__(cls, tuple(sorted(cards)))

    @property
    def as_int(self):
        hand_int = 0
        num_kings = 0
        hand_iterator = iter(reversed(self))
        try:
            card = next(hand_iterator)
        except StopIteration:
            return 0, 0
        while card[0] is Rank.K:
            num_kings += 1
            try:
                card = next(hand_iterator)
            except StopIteration:
                return hand_int, num_kings
        hand_int |= 1 << ((card[1] * 8) + card[0] - 1)
        for card in hand_iterator:
            hand_int |= 1 << ((card[1] * 8) + card[0] - 1)
        return hand_int, num_kings

    def score(self, king_info=False):
        hand_int, number_of_kings = self.as_int
        if not king_info:
            return _cached_score(hand_int, number_of_kings)
        best_score, king_cards = score_with_king_allocation(hand_int, number_of_kings)
        return best_score, king_cards

    def _repr_html_(self):
        return self.template.render(hand=self)

    @property
    def num_kings(self):
        return len(set(self) & Card.kings())

    @property
    def ranks(self):
        yield from groupby(self, key=itemgetter(0))

    @property
    def suits(self):
        key = itemgetter(1)
        cards_by_suit = sorted(self, key=key)
        yield from groupby(cards_by_suit, key=key)

    @property
    def cards_for_display(self):
        grid_hand = [[None for j in range(8)] for i in range(4)]
        score, king_cards = self.score(king_info=True)
        king_card_iter = iter(king_cards)
        for card in self:
            if card[0] is not Rank.K:
                grid_hand[card[1]][card[0] - 1] = card
            else:
                king_card = next(king_card_iter)
                grid_hand[king_card[1]][king_card[0] - 1] = card
        return grid_hand

    def __add__(self, other):
        return self.__class__(tuple.__add__(self, other))

    def __sub__(self, other):
        return self.__class__(set(self) - set(other))


class Eval(tuple):
    def __new__(cls, multiplicity, w, d, s):
        inst = super().__new__(cls, (w, d, s))
        inst.multiplicity = multiplicity
        return inst

    def __getnewargs__(self):
        # __new__ takes multiplicity separately from (w, d, s); reconstruct it
        # so pickling doesn't drop multiplicity.
        return (self.multiplicity, *self)

    @property
    def eval(self):
        return self.eval_from_wds(*self)

    @staticmethod
    def eval_from_wds(w, d, s):
        return (2 * w + d, w, s)

    @property
    def normed_eval(self):
        ev = self.eval
        multiplicity = self.multiplicity
        return (ev[0] / (2 * multiplicity), *(x / multiplicity for x in ev[1:]))

    def multiplied_wds(self, multiplier):
        return tuple(multiplier * val for val in self)

    def multiplied_eval(self, multiplier):
        return self.eval_from_wds(*self.multiplied_wds(multiplier))

    def __apply_op(self, op, other):
        self_multip = self.multiplicity
        other_multip = other.multiplicity

        if self_multip == other_multip:
            return op(self.eval, other.eval)
        return op(
            self.multiplied_eval(other.multiplicity),
            other.multiplied_eval(self.multiplicity),
        )

    def __lt__(self, other):
        return self.__apply_op(lt, other)

    def __le__(self, other):
        return self.__apply_op(le, other)

    def __gt__(self, other):
        return self.__apply_op(gt, other)

    def __ge__(self, other):
        return self.__apply_op(ge, other)

    def __eq__(self, other):
        return self.__apply_op(eq, other)

    def __neg__(self):
        return self.__class__(
            self.multiplicity, self.multiplicity - self[0] - self[1], self[1], -self[2]
        )

    def decisively_exceeds(self, other):
        # Prune-safe strict comparison on 2w+d alone (the full (2w+d, w, s)
        # ordering isn't negation-safe; see evaluate-alpha-beta-ordering-bug).
        self_multip = self.multiplicity
        other_multip = other.multiplicity
        if self_multip == other_multip:
            return self.eval[0] > other.eval[0]
        return self.eval[0] * other_multip > other.eval[0] * self_multip

    def __repr__(self):
        normed_formatted_str = ", ".join(f"{x:.2f}" for x in self.normed_eval)
        return f"Eval({normed_formatted_str})"


class ProbEval(Counter):
    def __init__(self, multiplicity=1, initial_counts=None):
        super().__init__()
        if initial_counts:
            self.update(initial_counts)
        self.multiplicity = multiplicity

    @property
    def observed(self):
        return sum(v for v in self.values())

    @property
    def wds(self):
        if self.observed != self.multiplicity:
            raise ValueError("Score is not fully evaluated!")
        return self.observed_wds

    @property
    def observed_wds(self):
        w = sum(v for k, v in self.items() if k > 0)
        d = self[0]
        s = sum(k * v for k, v in self.items())
        return w, d, s

    def _bound_evals(self):
        """(lower_bound.eval, upper_bound.eval) in a single pass with no Counter
        copies - equivalent to self.bound(-26).eval / self.bound(+26).eval."""
        w = d = s = observed = 0
        for k, v in self.items():
            observed += v
            if k > 0:
                w += v
            elif k == 0:
                d += v
            s += k * v
        remaining = self.multiplicity - observed
        filled = _SCORE_DIFFERENCE_BOUND * remaining
        return (
            Eval(self.multiplicity, w, d, s - filled),
            Eval(self.multiplicity, w + remaining, d, s + filled),
        )

    def copy(self):
        return ProbEval(multiplicity=self.multiplicity, initial_counts=dict(self))

    def bound(self, fill_value):
        multiplicity = self.multiplicity
        observed = self.observed
        if multiplicity == observed:
            return self
        inst = self.copy()
        inst.update({fill_value: multiplicity - observed})
        return inst

    @property
    def lower_bound(self):
        return self.bound(-_SCORE_DIFFERENCE_BOUND)

    @property
    def upper_bound(self):
        return self.bound(_SCORE_DIFFERENCE_BOUND)

    @property
    def eval(self):
        return Eval(self.multiplicity, *self.wds)

    @property
    def observed_eval(self):
        return Eval(self.multiplicity, *self.observed_wds)

    @classmethod
    def combine(cls, prob_evals):
        inst = cls(multiplicity=sum(prob_eval.multiplicity for prob_eval in prob_evals))
        for prob_eval in prob_evals:
            inst.update(prob_eval)
        return inst

    def __lt__(self, other):
        return self._bound_evals()[1] < other._bound_evals()[0]

    def __le__(self, other):
        return self._bound_evals()[1] <= other._bound_evals()[0]

    def __gt__(self, other):
        return self._bound_evals()[0] > other._bound_evals()[1]

    def __ge__(self, other):
        return self._bound_evals()[0] >= other._bound_evals()[1]

    def __eq__(self, other):
        # Compare on (w, d, s) bounds, consistent with __lt__/__gt__ (not the
        # inherited dict __eq__ on raw Counter contents).
        self_bounds = self._bound_evals()
        other_bounds = other._bound_evals()
        return self_bounds[0] == other_bounds[0] and self_bounds[1] == other_bounds[1]

    def __neg__(self):
        inst = ProbEval(multiplicity=self.multiplicity)
        for k, v in self.items():
            inst[-k] = v
        return inst

    def __repr__(self):
        ordered_dict = dict(sorted(self.items()))
        return f"{self.__class__.__name__}({self.observed}/{self.multiplicity} possibilities, {ordered_dict!r})"


class Game:
    starting_position = (2, 2)
    # The marker may be placed on any of the four central face-down cards
    # (the non-first player chooses). Sorted, so each cell's index is a
    # stable token in the save string.
    _valid_starting_positions = ((2, 2), (2, 3), (3, 2), (3, 3))
    possible_moves = {
        (i, j): (
            ({(i, j2) for j2 in range(6)} | {(i2, j) for i2 in range(6)}) - {(i, j)}
        )
        for i in range(6)
        for j in range(6)
    }
    template = env.get_template("game.html.jinja2")

    def __init__(self, board, moves, start=starting_position):
        self.board = board
        self.moves = moves
        # The cell the marker was placed on, or None if unplaced. Constant
        # across a game's lineage; _child carries it to every descendant.
        self.start = start

    @classmethod
    def deal(cls, marker=starting_position):
        # marker=None deals the board with no marker placed yet; call
        # place_marker to choose a starting cell before play begins.
        if marker is not None and marker not in cls._valid_starting_positions:
            raise ValueError("Invalid starting position")
        return cls(Board.deal(), tuple(), start=marker)

    def place_marker(self, row, col):
        """Place the marker on one of the central face-down cards to start
        the game. Only valid before it has been placed; the card is not
        collected (it's revealed only if a move lands on it later)."""
        if self.start is not None or self.moves:
            raise ValueError("The marker has already been placed")
        if (row, col) not in self._valid_starting_positions:
            raise ValueError("The marker must start on a central face-down card")
        return self.__class__(self.board, self.moves, start=(row, col))

    def clear_marker(self):
        """Undo a marker placement, returning to the unplaced board. Only valid
        before any move (placement changes nothing but `start`)."""
        if self.moves:
            raise ValueError("Cannot clear the marker once play has begun")
        return self.__class__(self.board, self.moves, start=None)

    def _child(self, board, moves):
        # The single place that carries the marker start to descendants.
        return self.__class__(board, moves, start=self.start)

    @property
    def marker(self):
        return self.moves[-1] if self.moves else self.start

    @property
    def _taken_masks(self):
        # Row-major and column-major occupancy bitmasks of the taken cells,
        # built once per game; `move`/`all_moves` extend them incrementally.
        try:
            return self._taken_masks_cache
        except AttributeError:
            rows = cols = 0
            for r, c in self.moves:
                rows |= 1 << (r * 6 + c)
                cols |= 1 << (c * 6 + r)
            self._taken_masks_cache = (rows, cols)
            return self._taken_masks_cache

    @property
    def legal_moves(self):
        # Memoised per game and globally via _LEGAL_MEMO: two shifts and a dict
        # probe replace the set difference over the move list.
        try:
            return self._legal_moves_cache
        except AttributeError:
            pass
        marker = self.moves[-1] if self.moves else self.start
        if marker is None:
            # No marker placed yet - place_marker must be called first.
            self._legal_moves_cache = frozenset()
            return self._legal_moves_cache
        row, col = marker
        rows, cols = self._taken_masks
        # Force the marker's own bits on so it excludes itself like any other.
        row_bits = ((rows >> (row * 6)) | (1 << col)) & 63
        col_bits = ((cols >> (col * 6)) | (1 << row)) & 63
        key = (row * 6 + col, row_bits, col_bits)
        legal = _LEGAL_MEMO.get(key)
        if legal is None:
            legal = frozenset(
                [(row, j) for j in range(6) if not row_bits >> j & 1]
                + [(i, col) for i in range(6) if not col_bits >> i & 1]
            )
            _LEGAL_MEMO[key] = legal
        self._legal_moves_cache = legal
        return legal

    def all_moves(self):
        for row, col in self.legal_moves:
            yield self.move(row, col)

    def move(self, row, col):
        if (row, col) not in self.legal_moves:
            raise ValueError("Illegal move")
        new_moves = self.moves + ((row, col),)
        rows, cols = self._taken_masks
        masks = (rows | 1 << (row * 6 + col), cols | 1 << (col * 6 + row))
        if self.board[row][col].facedown:
            children = tuple(
                self._child(new_board, new_moves)
                for new_board in self.board.resolve(row, col)
            )
        else:
            children = (self._child(self.board, new_moves),)
        for child in children:
            child._taken_masks_cache = masks
        return children

    def random_move(self):
        return choice(choice(list(self.all_moves())))

    def get_hand(self, key):
        return Hand(
            self.board[position[0]][position[1]] for position in self.moves[key]
        )

    @property
    def score(self):
        return self.p1.score() - self.p2.score()

    @property
    def negamax_score(self):
        if len(self.moves) % 2:
            return -self.score
        return self.score

    @property
    def multiplicity(self):
        return _FACTORIAL[len(self.board.facedown_cards)]

    @property
    def p1(self):
        return self.get_hand(slice(None, None, 2))

    @property
    def p2(self):
        return self.get_hand(slice(1, None, 2))

    @property
    def needs_marker(self):
        # Dealt but not yet placed: place_marker must be called before play.
        return self.marker is None

    @property
    def is_p1_turn(self):
        return bool(self.legal_moves and (len(self.moves) % 2 == 0))

    @property
    def is_p2_turn(self):
        # Player 2 also acts during placement, choosing the marker's start.
        return self.needs_marker or bool(self.legal_moves and (len(self.moves) % 2))

    def _repr_html_(self):
        return self.template.render(game=self)

    def undo(self, number_of_moves=1):
        if number_of_moves < 0:
            raise ValueError("Cannot undo a negative number of moves")
        elif number_of_moves == 0:
            return self
        elif number_of_moves > len(self.moves):
            raise ValueError(f"There are only {len(self.moves)} to undo!")

        moves_to_undo = self.moves[-number_of_moves:]
        moves_which_were_fd = set(moves_to_undo) & set(self.board.facedown_positions)
        if moves_which_were_fd:
            fd_card = Card(facedown=True)
            to_add_back_to_fd = []
            new_board = [list(row) for row in self.board]
            for row, col in moves_which_were_fd:
                to_add_back_to_fd.append(self.board[row][col])
                new_board[row][col] = fd_card
            board = Board(
                new_board, self.board.facedown_cards + tuple(to_add_back_to_fd)
            )
        else:
            board = self.board
        return self._child(board, tuple(self.moves[:-number_of_moves]))

    @property
    def taken_card(self):
        if not self.moves:
            raise ValueError("No cards have been taken")
        row, col = self.marker
        return self.board[row][col]

    def _hand_state(self):
        """Both hands as scoring ints, mover first: (mover_int, mover_kings,
        other_int, other_kings), in Hand.as_int's encoding. Threaded through
        evaluate so terminals score from the cache with no Hand construction."""
        hands = [[0, 0], [0, 0]]
        board = self.board
        for i, (row, col) in enumerate(self.moves):
            card = board[row][col]
            if card[0] is Rank.K:
                hands[i % 2][1] += 1
            else:
                hands[i % 2][0] |= 1 << ((card[1] * 8) + card[0] - 1)
        mover = len(self.moves) % 2
        return (*hands[mover], *hands[1 - mover])

    @staticmethod
    def _child_hand_state(state, card):
        """_hand_state after the mover takes `card`: the perspectives swap
        and the card joins what is now the opponent's hand."""
        mover_int, mover_kings, other_int, other_kings = state
        if card[0] is Rank.K:
            return (other_int, other_kings, mover_int, mover_kings + 1)
        return (
            other_int,
            other_kings,
            mover_int | (1 << ((card[1] * 8) + card[0] - 1)),
            mover_kings,
        )

    @staticmethod
    def _get_bounds(branch_multiplicity, move_score, alpha, beta):
        # One-pass, no-allocation fill of the remaining unevaluated mass:
        # at -26 adds (0, 0, -26*rem) to the observed (w, d, s), at +26 (rem, 0, +26*rem).
        w, d, s = move_score.observed_wds
        remaining_unevaled_after_branch = (
            move_score.multiplicity - move_score.observed - branch_multiplicity
        )
        filled = _SCORE_DIFFERENCE_BOUND * remaining_unevaled_after_branch
        # subbeta: how good the branch eval must be for the combined lower bound
        # to exceed beta.
        subbeta = Eval(
            branch_multiplicity, beta[0] - w, beta[1] - d, beta[2] - (s - filled)
        )
        # subalpha: how bad it must be for the combined upper bound to fall below alpha.
        subalpha = Eval(
            branch_multiplicity,
            alpha[0] - (w + remaining_unevaled_after_branch),
            alpha[1] - d,
            alpha[2] - (s + filled),
        )

        branch_bound = _SCORE_DIFFERENCE_BOUND * branch_multiplicity
        lb = Eval(branch_multiplicity, 0, 0, -branch_bound)
        ub = Eval(branch_multiplicity, branch_multiplicity, 0, branch_bound)
        if subalpha < lb:
            subalpha = lb
        if subbeta > ub:
            subbeta = ub
        return subalpha, subbeta

    def evaluate(self):
        """Exact value of this position as a single ProbEval (the score-
        difference distribution under optimal play, in the mover's perspective)
        with the best move's marker (None at a terminal). Native-backed when the
        Rust core is available, else the pure-Python `_evaluate_python` engine.

        Returns (ProbEval, best_marker). The two engines agree exactly on the
        value's (w, d, s) and the best move; they may differ in how the
        histogram distributes weight among tied-optimal lines.
        """
        from .solver_native import NATIVE_AVAILABLE, evaluate_native

        if NATIVE_AVAILABLE:
            return evaluate_native(self)
        result = self._evaluate_python()
        return result["Evaluation"], result["best_marker"]

    def _evaluate_python(self, alpha=None, beta=None, _state=None):
        # Pure-Python reference engine behind Game.evaluate, recursing under
        # negated windows. Returns the full internal dict (Evaluation, the
        # deterministic optimal PV, per-branch scores, best marker); `evaluate`
        # reduces that to (ProbEval, best_marker).
        #
        # Unexplored-branch placeholders use the loose global ±26; tighter
        # per-position bounds break this engine's fail-soft bookkeeping (see
        # position-bounds-finding).
        multiplicity = self.multiplicity
        if not self.legal_moves:
            # _state is the threaded (mover_int, mover_kings, other_int,
            # other_kings), with the scoring DP cached.
            if _state is None:
                _state = self._hand_state()
            diff = _cached_score(_state[0], _state[1]) - _cached_score(
                _state[2], _state[3]
            )
            return {
                "Evaluation": ProbEval(multiplicity, {diff: multiplicity}),
                "Deterministic optimal moves": tuple(),
                "best_marker": None,
            }
        if _state is None:
            _state = self._hand_state()
        if not alpha:
            bound_sum = _SCORE_DIFFERENCE_BOUND * multiplicity
            alpha = Eval(multiplicity, 0, 0, -bound_sum)
            beta = Eval(multiplicity, multiplicity, 0, bound_sum)
        best_score = ProbEval(multiplicity).lower_bound
        best_move_seq = ((-1, -1),)
        best_marker = None
        ordered_moves = sorted(self.all_moves(), key=len)
        detailed_move_scores = {}
        child_state = self._child_hand_state

        for move in ordered_moves:
            move_marker = move[0].marker
            if len(move) == 1:
                move_eval = move[0]._evaluate_python(
                    -beta, -alpha, _state=child_state(_state, move[0].taken_card)
                )
                move_score = -(move_eval["Evaluation"])
                detailed_move_scores[move_marker] = move_score
                if (move_score, move_marker) > (best_score, best_move_seq[0]):
                    best_score = move_score
                    best_marker = move_marker
                    best_move_seq = (move[0].taken_card,) + move_eval[
                        "Deterministic optimal moves"
                    ]
                alpha = max(best_score._bound_evals()[0], alpha)
            else:
                branch_multiplicity = move[0].multiplicity
                detailed_move_scores[move_marker] = {
                    fd_card: ProbEval(branch_multiplicity)
                    for fd_card in self.board.facedown_cards
                }
                move_score = ProbEval(multiplicity)
                possibility_move_seqs = []
                curr_move_best_move = False
                for possibility in move:
                    subalpha, subbeta = self._get_bounds(
                        branch_multiplicity, move_score, alpha, beta
                    )
                    possibility_eval = possibility._evaluate_python(
                        -subbeta,
                        -subalpha,
                        _state=child_state(_state, possibility.taken_card),
                    )
                    possibility_score = -(possibility_eval["Evaluation"])
                    possibility_move_seqs.append(
                        possibility_eval["Deterministic optimal moves"]
                    )
                    detailed_move_scores[move_marker][possibility.taken_card] = (
                        possibility_score
                    )
                    move_score.update(possibility_score)
                    if (move_score, move_marker) > (best_score, best_move_seq[0]):
                        best_score = move_score
                        best_marker = move_marker
                        curr_move_best_move = True
                    move_lower, move_upper = move_score._bound_evals()
                    if move_upper < alpha:
                        break
                    if best_score is move_score:
                        alpha = max(move_lower, alpha)
                    else:
                        alpha = max(best_score._bound_evals()[0], alpha)
                    if alpha.decisively_exceeds(beta):
                        break
                if curr_move_best_move:
                    best_move_seq = (move_marker,) + _common_prefix(
                        possibility_move_seqs
                    )
            if alpha.decisively_exceeds(beta):
                break
        return {
            "Evaluation": best_score,
            "Deterministic optimal moves": best_move_seq,
            "Known info for other branches": detailed_move_scores,
            "best_marker": best_marker,
        }

    def save(self, alnum=False):
        # Format: "<board>//<start><moves>". <start> indexes the marker's
        # starting cell in _valid_starting_positions; each <moves> digit indexes
        # the taken cell in the sorted moves from the previous marker. An
        # unplaced game leaves the trailing section empty.
        board_str = self.board.save(alnum=alnum)
        if self.start is None:
            return f"{board_str}//"
        ordered_poss_moves = {
            marker: sorted(moves) for marker, moves in self.possible_moves.items()
        }
        move_ind = [
            str(ordered_poss_moves[marker].index(move))
            for marker, move in zip((self.start,) + self.moves[:-1], self.moves)
        ]
        start_ind = str(self._valid_starting_positions.index(self.start))
        return f"{board_str}//{start_ind}{''.join(move_ind)}"

    @classmethod
    def load(cls, save):
        board, section = save.rsplit("//", maxsplit=1)
        board = Board.load(board)
        if not section:
            return cls(board, tuple(), start=None)
        ordered_poss_moves = {
            marker: sorted(moves) for marker, moves in cls.possible_moves.items()
        }
        start = cls._valid_starting_positions[int(section[0])]
        position = start
        moves_list = []
        for char in section[1:]:
            position = ordered_poss_moves[position][int(char)]
            moves_list.append(position)
        return cls(board, tuple(moves_list), start=start)
