from enum import IntEnum
from jinja2 import Environment, PackageLoader
from random import shuffle, choice
from operator import itemgetter, or_
from functools import reduce
from itertools import groupby, combinations, product
from math import factorial
from collections import Counter

__all__ = ("Suit", "Rank", "Card", "Board", "Hand", "Game")

env = Environment(loader=PackageLoader(package_name="cardgame", package_path="../../templates"))

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
            suit_scores[(reduce(or_, (1<<i for i, card in enumerate(pmut) if card), 0), reduce(or_, (1<<i for i, card in enumerate(kings) if card), 0))] = score
    rank_scores = {
        (reduce(or_, (1 << (8 * i) for i, card in enumerate(pmut) if card), 0), reduce(or_, (1 << (8 * i) for i, card in enumerate(kings) if card), 0)): (2 * sum(pmut) - 3 + sum(kings) if sum(pmut) + sum(kings) >= 3 else 0)
        for pmut in product([False, True], repeat=4) for kings in product(*(([False] if card else [False, True]) for card in pmut))
    }
    return rank_scores, suit_scores

_rank_scores, _suit_scores = _calculate_score_lookups()

Suit = IntEnum("suit", names=("hearts", "clubs", "diamonds", "spades"), start=0)
Rank = IntEnum("rank", names=list("A2345678K"))

class Card(tuple):

    template = env.get_template("card.html.jinja2")
    _symbols = "♥♣♦♠"

    def __new__(cls, rank=None, suit=None, facedown=False):
        if facedown:
            instance = super().__new__(cls, (None, None))
        else:
            instance = super().__new__(cls, (Rank(rank), Suit(suit)))
        instance.facedown = facedown
        return instance

    @property
    def rank(self):
        return '?' if self.facedown else self[0].name

    @property
    def suit(self):
        return '?' if self.facedown else self._symbols[self[1]]

    @property
    def suit_name(self):
        return None if self.facedown else self[1].name

    def __repr__(self):
        return self.rank + self.suit

    def _repr_html_(self):
        return self.template.render(card=self)

    @classmethod
    @property
    def deck(cls):
        return tuple(cls(rank=rank, suit=suit) for suit in Suit for rank in Rank)

    @classmethod
    @property
    def kings(cls):
        return {cls(rank=Rank.K, suit=suit) for suit in Suit}

    @classmethod
    @property
    def nonkings(cls):
        return set(cls.deck) - cls.kings

class Board(tuple):

    template = env.get_template("board.html.jinja2")
    facedown_indices = set(range(0, 36, 5)) | set(range(0, 36, 7))
    facedown_positions = {(i//6, i%6) for i in facedown_indices}

    def __new__(cls, cards, facedown_cards):
        instance = super().__new__(cls, tuple(tuple(row) for row in cards))
        instance.facedown_cards = tuple(facedown_cards)
        return instance

    @classmethod
    def deal(cls):
        deck = list(Card.deck)
        shuffle(deck)
        facedown_cards = itemgetter(*cls.facedown_indices)(deck)
        facedown_card = Card(facedown=True)
        for index in cls.facedown_indices:
            deck[index] = facedown_card
        return cls([deck[i:i+6] for i in range(0, 36, 6)], facedown_cards)

    def __repr__(self):
        return '\n'.join(
            ' '.join(str(card) for card in row)
            for row in self
        )

    def _repr_html_(self):
        return self.template.render(board=self)

    def resolve(self, row, col):

        current_board = list(self)
        row_to_change = list(current_board[row])
        for i, card in enumerate(self.facedown_cards):
            row_to_change[col] = card
            current_board[row] = tuple(row_to_change)
            yield self.__class__(tuple(current_board), self.facedown_cards[:i] + self.facedown_cards[i+1:])

def _int_to_bits(n):
    components = []
    while n > 0:
        lsb = n & -n 
        components.append(lsb)
        n &= (n - 1)
    return tuple(reversed(components))

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
                score += _suit_scores[(base_suit, king_suit)]
            for rank in list(Rank)[:-1]:
                base_rank = (hand_int >> (rank - 1)) & rank_mask
                king_rank = (king_int >> (rank - 1)) & rank_mask
                score += _rank_scores[(base_rank, king_rank)]
            best_score, best_kings = max((best_score, best_kings), (score, king_int))
        if not king_info:
            return best_score
        king_cards = tuple(
            Card(
                ((king_int.bit_length() - 1) % 8) + 1, 
                (king_int.bit_length()-1) // 8
            ) for king_int in _int_to_bits(best_kings)
        )
        return best_score, king_cards

    def _repr_html_(self):
        return self.template.render(hand=self)

    @property
    def num_kings(self):
        return len(set(self) & Card.kings)
    
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
            if not card[0] is Rank.K:
                grid_hand[card[1]][card[0] - 1] = card
            else:
                king_card = next(king_card_iter)
                grid_hand[king_card[1]][king_card[0] - 1] = card
        return grid_hand
    
    def __add__(self, other):

        return self.__class__(tuple.__add__(self, other))

    def __sub__(self, other):

        return self.__class__(set(self) - set(other))

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
    def wdls(self):
        if self.observed != self.multiplicity:
            raise ValueError('Score is not fully evaluated!')
        w = sum(v for k, v in self.items() if k > 0)
        d = self[0]
        l = self.multiplicity - w - d
        s = sum(k * v for k, v in self.items())
        return w, d, l, s

    def multiplied_wdl(self, multiplier):
        return tuple(multiplier * val for val in self.wdls)
    
    def multiplied_eval(self, multiplier):
        return self.eval_from_wdls(*self.multiplied_wdl(multiplier))

    @staticmethod
    def normalise_eval(evaluation, multiplicity):
        return tuple(val / multiplicity for val in evaluation)
        
    @staticmethod
    def eval_from_wdls(w, d, l, s):
        return (w + d/2, w, w + d, s)
        
    @property
    def eval(self):
        return self.eval_from_wdls(*self.wdls)

    @property
    def normed_eval(self):
        return self.normalise_eval(self.eval, self.multiplicity)
    
    def __gt__(self, other):
        return self.multiplied_eval(other.multiplicity) > other.multiplied_eval(self.multiplicity)

    def __gte__(self, other):
        return self.multiplied_eval(other.multiplicity) >= other.multiplied_eval(self.multiplicity)

    def __eq__(self, other):
        return self.multiplied_wdl(other.multiplicity) == other.multiplied_wdl(self.multiplicity)

    def __neg__(self):
        inst = self.__class__(self.multiplicity)
        for k, v in self.items():
            inst[-k] = v
        return inst

    @classmethod
    def combine(cls, prob_evals):
        inst = cls(multiplicity=sum(prob_eval.multiplicity for prob_eval in prob_evals))
        for prob_eval in prob_evals:
            inst.update(prob_eval)
        return inst

EVAL_LOWER_BOUND = ProbEval({-25: 1})
EVAL_UPPER_BOUND = ProbEval({-25: 1})

class Game:
    starting_position = (2, 2)
    possible_moves = {
        (i, j): (
            (
                {(i, j2) for j2 in range(6)} | {(i2, j) for i2 in range(6)}
            ) - {(i, j)}
        ) for i in range(6) for j in range(6)
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
    def legal_moves(self):
        return self.possible_moves[self.marker] - set(self.moves)

    def all_moves(self):
        for row, col in self.legal_moves:
            new_moves = self.moves + ((row, col),)
            if self.board[row][col].facedown:
                yield tuple(self.__class__(new_board, new_moves) for new_board in self.board.resolve(row, col))
            else:
                yield (self.__class__(self.board, new_moves),)

    def random_move(self):
        return choice(choice(list(self.all_moves())))
    
    def get_hand(self, key):
        return Hand(self.board[position[0]][position[1]] for position in self.moves[key])

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
        return factorial(len(self.board.facedown_cards))
    
    @property
    def p1(self):
        return self.get_hand(slice(None, None, 2))

    @property
    def p2(self):
        return self.get_hand(slice(1, None, 2))

    def _repr_html_(self):
        return self.template.render(game=self)

    def undo(self, number_of_moves=1):
        
        if number_of_moves < 0:
            raise ValueError('Cannot undo a negative number of moves')
        elif number_of_moves == 0:
            return self
        elif number_of_moves > len(self.moves):
            raise ValueError(f'There are only {len(self.moves)} to undo!')

        moves_to_undo = self.moves[-number_of_moves:]
        moves_which_were_fd = set(moves_to_undo) & set(self.board.facedown_positions)
        if moves_which_were_fd:
            fd_card = Card(facedown=True)
            to_add_back_to_fd = []
            new_board = [list(row) for row in self.board]
            for row, col in moves_which_were_fd:
                to_add_back_to_fd.append(self.board[row][col])
                new_board[row][col] = fd_card
            board = Board(new_board, self.board.facedown_cards + tuple(to_add_back_to_fd))
        else:
            board = self.board
        return self.__class__(board, tuple(self.moves[:-number_of_moves]))

    @property
    def move_evals(self):
        return {
            move[0].marker: -ProbEval.combine([move_possibility.score_walk()[0] for move_possibility in move]) 
            for move in self.all_moves()
        }
    
    def score_walk(self):
    
        if not self.legal_moves:
            multiplicity = self.multiplicity
            return ProbEval(multiplicity=multiplicity, initial_counts={self.negamax_score: multiplicity}), (-1, -1)
        best_score = max((-ProbEval.combine([move_possibility.score_walk()[0] for move_possibility in move]), move[0].marker) for move in self.all_moves())
        return best_score