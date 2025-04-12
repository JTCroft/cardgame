from enum import IntEnum
from jinja2 import Environment, PackageLoader
from random import shuffle, choice
from operator import itemgetter
from itertools import groupby, combinations, islice

__all__ = ("Suit", "Rank", "Card", "Board", "Hand", "Game")

env = Environment(loader=PackageLoader(package_name="cardgame", package_path="../../templates"))

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

class Hand(tuple):

    template = env.get_template("hand.html.jinja2")

    def __new__(cls, cards):
        return super().__new__(cls, tuple(sorted(cards)))
        

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

        grid_hand = [[None for j in range(9)] for i in range(4)]
        for card in self:
            grid_hand[card[1]][card[0] - 1] = card
        return grid_hand

    def __add__(self, other):

        return self.__class__(tuple.__add__(self, other))

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
    def p1(self):
        return self.get_hand(slice(None, None, 2))

    @property
    def p2(self):
        return self.get_hand(slice(1, None, 2))

    def _repr_html_(self):
        return self.template.render(game=self)