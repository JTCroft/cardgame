from enum import IntEnum
from jinja2 import Environment, PackageLoader

env = Environment(
    loader=PackageLoader(package_name="cardgame", package_path="../../templates")
)

Suit = IntEnum("suit", names=("hearts", "clubs", "diamonds", "spades"), start=0)
Rank = IntEnum("rank", names=list("A2345678K"))


class Card(tuple):
    template = env.get_template("card.html.jinja2")
    _symbols = "♥♣♦♠"
    _letters = "HCDS"

    def __new__(cls, rank=None, suit=None, facedown=False):
        if facedown:
            instance = super().__new__(cls, (None, None))
        else:
            instance = super().__new__(cls, (Rank(rank), Suit(suit)))
        instance.facedown = facedown
        return instance

    @classmethod
    def from_str(cls, card_str):
        if card_str == "??":
            return cls(facedown=True)
        rank, suit = card_str
        rank = Rank.__members__[rank]
        suit = Suit(
            cls._symbols.index(suit)
            if suit in cls._symbols
            else cls._letters.index(suit)
        )
        return cls(rank, suit)

    @property
    def rank(self):
        return "?" if self.facedown else self[0].name

    @property
    def suit(self):
        return "?" if self.facedown else self._symbols[self[1]]

    @property
    def letter(self):
        return "?" if self.facedown else self._letters[self[1]]

    @property
    def suit_name(self):
        return None if self.facedown else self[1].name

    def __str__(self):
        return self.rank + self.suit

    def __repr__(self):
        return self.rank + self.letter

    def _repr_pretty_(self, pp, cycle):
        pp.text(str(self))

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
