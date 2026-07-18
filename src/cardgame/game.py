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

# This is the approximate upper bound of the difference between 2 players scores
# Not proven, but here's two disjoint hands that achieve this
# 31 points - [A♣, 2♣, 2♠, 3♥, 3♣, 3♠, 4♣, 4♦, 4♠, 5♣, 5♠, 6♥, 6♣, 6♠, K♥, K♣]
# 5 points -  [A♥, A♠, 2♥, 2♦, 3♦, 4♥, 5♥, 5♦, 6♦, 7♥, 7♠, 8♥, 8♣, 8♦, 8♠]
_SCORE_DIFFERENCE_BOUND = 26

# Scoring dominates the tree search (terminal positions outnumber interior
# ones several-fold and each used to pay two uncached DP solves), and
# transpositions make hands recur constantly, so a big cache pays for
# itself many times over. Hand.score routes through this too.
_cached_score = lru_cache(maxsize=1 << 20)(score_dp)

_FACTORIAL = tuple(factorial(n) for n in range(13))


# Legal-move memo shared by every game: the marker cell plus the occupancy
# of its row and column fully determine the legal set, and the marker's own
# bits are always set in the key, so there are at most 36 * 32 * 32 distinct
# keys process-wide. Values are frozensets, shared rather than rebuilt.
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
        # Same mismatch as Card: __new__ takes the row layout and
        # facedown_cards (an instance attribute, not part of the tuple
        # contents) as separate args, not what tuple's default pickling
        # reduction assumes. Unwrapped to a plain tuple - passing `self`
        # here would recurse back into pickling this same Board forever.
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
        # Reuses the five unchanged row tuples and bypasses __new__'s
        # re-tupling walk - this runs once per possibility of every chance
        # node in the search.
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

    def __lte__(self, other):
        return self.__apply_op(le, other)

    def __gt__(self, other):
        return self.__apply_op(gt, other)

    def __gte__(self, other):
        return self.__apply_op(ge, other)

    def __eq__(self, other):
        return self.__apply_op(eq, other)

    def __neg__(self):
        return self.__class__(
            self.multiplicity, self.multiplicity - self[0] - self[1], self[1], -self[2]
        )

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
        """(lower_bound.eval, upper_bound.eval) in a single pass with no
        Counter copies - equivalent to self.bound(-26).eval /
        self.bound(+26).eval, which the search consults constantly."""
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

    def __lte__(self, other):
        return self._bound_evals()[1] <= other._bound_evals()[0]

    def __gt__(self, other):
        return self._bound_evals()[0] > other._bound_evals()[1]

    def __gte__(self, other):
        return self._bound_evals()[0] >= other._bound_evals()[1]

    def __eq__(self, other):
        return bool(
            (self.multiplicity == other.multiplicity) and (dict(self) == dict(other))
        )

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
    possible_moves = {
        (i, j): (
            ({(i, j2) for j2 in range(6)} | {(i2, j) for i2 in range(6)}) - {(i, j)}
        )
        for i in range(6)
        for j in range(6)
    }
    template = env.get_template("game.html.jinja2")

    def __init__(self, board, moves):
        self.board = board
        self.moves = moves

    @classmethod
    def deal(cls):
        return cls(Board.deal(), tuple())

    @property
    def marker(self):
        return self.moves[-1] if self.moves else self.starting_position

    @property
    def _taken_masks(self):
        # Row-major and column-major occupancy bitmasks of the taken cells.
        # Built once per game from the move list; `move` and `all_moves`
        # extend them incrementally so search descents never rebuild them.
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
        # Memoised twice: per game (games are immutable, and the search
        # consults this several times per node), and globally via
        # _LEGAL_MEMO - two shifts and a dict probe replace the set
        # difference over the whole move list.
        try:
            return self._legal_moves_cache
        except AttributeError:
            pass
        row, col = self.moves[-1] if self.moves else self.starting_position
        rows, cols = self._taken_masks
        # The marker's own bits are forced on so the root marker (whose
        # starting cell was never taken) excludes itself like any other.
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
                self.__class__(new_board, new_moves)
                for new_board in self.board.resolve(row, col)
            )
        else:
            children = (self.__class__(self.board, new_moves),)
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
    def is_p1_turn(self):
        return bool(self.legal_moves and (len(self.moves) % 2 == 0))

    @property
    def is_p2_turn(self):
        return bool(self.legal_moves and (len(self.moves) % 2))

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
        return self.__class__(board, tuple(self.moves[:-number_of_moves]))

    @property
    def taken_card(self):
        if not self.moves:
            raise ValueError("No cards have been taken")
        row, col = self.marker
        return self.board[row][col]

    def _hand_state(self):
        """Both hands as scoring ints, mover first: (mover_int,
        mover_kings, other_int, other_kings), in Hand.as_int's encoding.
        Threaded incrementally through evaluate so terminals score from
        the cache with no Hand construction."""
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

    @property
    def move_evals(self):
        move_evals = {}
        for move in self.all_moves():
            move_evals[move[0].marker] = {
                "resolved_evals": {
                    move_possibility.taken_card: -(move_possibility.score_walk()[0])
                    for move_possibility in move
                }
            }
            if move[0].marker in self.board.facedown_positions:
                move_evals[move[0].marker]["combined_eval"] = ProbEval.combine(
                    list(move_evals[move[0].marker]["resolved_evals"].values())
                )
            else:
                move_evals[move[0].marker]["combined_eval"] = move_evals[
                    move[0].marker
                ]["resolved_evals"][move[0].taken_card]
        return move_evals

    def score_walk(self, _state=None):
        # _state is the same threaded (mover_int, mover_kings, other_int,
        # other_kings) evaluate uses, so terminals score from the cache
        # with no Hand construction.
        if _state is None:
            _state = self._hand_state()
        if not self.legal_moves:
            multiplicity = self.multiplicity
            diff = _cached_score(_state[0], _state[1]) - _cached_score(
                _state[2], _state[3]
            )
            return ProbEval(
                multiplicity=multiplicity,
                initial_counts={diff: multiplicity},
            ), (-1, -1)
        child_state = self._child_hand_state
        best_score = max(
            (
                -ProbEval.combine(
                    [
                        move_possibility.score_walk(
                            child_state(_state, move_possibility.taken_card)
                        )[0]
                        for move_possibility in move
                    ]
                ),
                move[0].marker,
            )
            for move in self.all_moves()
        )
        return best_score

    @staticmethod
    def _get_bounds(branch_multiplicity, move_score, alpha, beta):
        # Same arithmetic as the original Counter-copy formulation, one
        # pass and no allocation: filling the remaining unevaluated mass
        # at -26 adds (0, 0, -26*rem) to the observed (w, d, s); at +26 it
        # adds (rem, 0, +26*rem).
        w, d, s = move_score.observed_wds
        remaining_unevaled_after_branch = (
            move_score.multiplicity - move_score.observed - branch_multiplicity
        )
        filled = _SCORE_DIFFERENCE_BOUND * remaining_unevaled_after_branch
        # How good does the branch evaluation have to be
        # So that the LOWER BOUND of the combined eval with the move score
        # would be ABOVE beta
        subbeta = Eval(
            branch_multiplicity, beta[0] - w, beta[1] - d, beta[2] - (s - filled)
        )
        # How bad does the branch evaluation have to be
        # So that the UPPER BOUND of the combined eval with the move score
        # would be BELOW alpha
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

    def evaluate(self, alpha=None, beta=None, _state=None):
        # Note: unexplored-branch placeholders here deliberately use the
        # loose global ±26. Tighter per-position bounds (from scoring's
        # monotonicity: a hand vs the hand plus everything left on the
        # board) were tried and are provably sound, but this engine's
        # fail-soft bookkeeping folds the observed mass of partially
        # evaluated moves into complete results, which is only safe when a
        # chance move is abandoned under the most optimistic completion
        # possible anywhere - tightening the abandonment check corrupts
        # evaluations (found by counterexample), and tightening the other
        # fill sites measurably prunes nothing. Such bounds suit search
        # schemes with explicit per-node intervals instead.
        multiplicity = self.multiplicity
        if not self.legal_moves:
            # _state is the incrementally threaded (mover_int, mover_kings,
            # other_int, other_kings) - equal to negamax_score, minus the
            # Hand construction, with the scoring DP cached.
            if _state is None:
                _state = self._hand_state()
            diff = _cached_score(_state[0], _state[1]) - _cached_score(
                _state[2], _state[3]
            )
            return {
                "Evaluation": ProbEval(multiplicity, {diff: multiplicity}),
                "Deterministic optimal moves": tuple(),
            }
        if _state is None:
            _state = self._hand_state()
        if not alpha:
            bound_sum = _SCORE_DIFFERENCE_BOUND * multiplicity
            alpha = Eval(multiplicity, 0, 0, -bound_sum)
            beta = Eval(multiplicity, multiplicity, 0, bound_sum)
        best_score = ProbEval(multiplicity).lower_bound
        best_move_seq = ((-1, -1),)
        ordered_moves = sorted(self.all_moves(), key=len)
        detailed_move_scores = {}
        child_state = self._child_hand_state

        for move in ordered_moves:
            move_marker = move[0].marker
            if len(move) == 1:
                move_eval = move[0].evaluate(
                    -beta, -alpha, _state=child_state(_state, move[0].taken_card)
                )
                move_score = -(move_eval["Evaluation"])
                detailed_move_scores[move_marker] = move_score
                if (move_score, move_marker) > (best_score, best_move_seq[0]):
                    best_score = move_score
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
                    possibility_eval = possibility.evaluate(
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
                        curr_move_best_move = True
                    move_lower, move_upper = move_score._bound_evals()
                    if move_upper < alpha:
                        break
                    if best_score is move_score:
                        alpha = max(move_lower, alpha)
                    else:
                        alpha = max(best_score._bound_evals()[0], alpha)
                    if alpha > beta and not (-alpha) > (-beta):
                        break
                if curr_move_best_move:
                    best_move_seq = (move_marker,) + _common_prefix(
                        possibility_move_seqs
                    )
            if alpha > beta and not (-alpha) > (-beta):
                break
        return {
            "Evaluation": best_score,
            "Deterministic optimal moves": best_move_seq,
            "Known info for other branches": detailed_move_scores,
        }

    def save(self, alnum=False):
        board_str = self.board.save(alnum=alnum)
        ordered_poss_moves = {
            marker: sorted(moves) for marker, moves in self.possible_moves.items()
        }
        move_ind = [
            str(ordered_poss_moves[marker].index(move))
            for marker, move in zip(
                (self.starting_position,) + self.moves[:-1], self.moves
            )
        ]
        moves_str = "".join(move_ind)
        return f"{board_str}//{moves_str}"

    @classmethod
    def load(cls, save):
        board, moves = save.rsplit("//", maxsplit=1)
        board = Board.load(board)
        ordered_poss_moves = {
            marker: sorted(moves) for marker, moves in cls.possible_moves.items()
        }
        move_indexes = [int(move) for move in moves]
        position = cls.starting_position
        moves_list = []
        for move_index in move_indexes:
            position = ordered_poss_moves[position][move_index]
            moves_list.append(position)
        return cls(board, tuple(moves_list))
