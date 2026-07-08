"""A placeholder computer opponent for Cross Kings.

`choose_move` is a very basic heuristic - not lookahead or optimal play (see
`Game.evaluate`/`Game.score_walk` for that): it just greedily takes the
highest-ranked face-up card on offer, and picks randomly when every legal
move is still face-down. It exists as a simple, fast default so the web UI
can offer a single-player mode; swap it out for something smarter later.
"""

import random

from .cards import Rank

__all__ = ("choose_move",)

_RANK_VALUE = {rank: value for value, rank in enumerate(Rank, start=1)}


def choose_move(game):
    """Pick a legal (row, col) move for `game`'s current player."""
    legal_moves = list(game.legal_moves)
    visible_moves = [
        (row, col) for row, col in legal_moves if not game.board[row][col].facedown
    ]
    if visible_moves:
        return max(
            visible_moves, key=lambda pos: _RANK_VALUE[game.board[pos[0]][pos[1]][0]]
        )
    return random.choice(legal_moves)
