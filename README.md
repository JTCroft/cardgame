# Cross Kings

A basic python implementation of a 2 player card game using a subset of standard playing cards called Cross Kings

## Gameplay

The game uses the cards Ace-8 of all suits plus the Kings. To start the game deal the cards in a 6 by 6 grid, with all cards on the diagonals face down and all other cards face up.

A marker displays the current position, normally the player who is taking the second turn may choose which of the 4 centre-most cards to place the marker on.

Play then proceeds in alternating turns, with each player moving the marker to any card that is in the same row or column as the marker itself, and collecting that card into their hand. The marker must move on the first turn. The marker may pass over any gaps left by previously taken cards.

The game finishes when the marker is left in a position with no moves, ie: when there are no cards in the same row or column as the marker.

## Scoring

The game is scored at the end, each player can calculate their points individually to determine the winner.

A player earns points by having collected cards that form **consecutive runs** in the same suit, or **sets** of the same rank. A card can be in both a run and a set simultaneously.

Each run or set is worth a base number of points based on how long it is according to the table below

| Run or set length | Points |
| ----------------- | ------ |
| 1                 | 0      |
| 2                 | 0      |
| 3                 | 3      |
| 4                 | 5      |
| 5                 | 7      |
| 6                 | 9      |
| 7                 | 11     |
| 8                 | 13     |

A player can use a king to act as a wildcard, choosing which card it will act as to maximise points. However each time a king forms part of a run or set that run or set will be worth one less point than it would be otherwise.

This logic can be summarised as

```python
if run_or_set_length < 3:
    run_or_set_score = 0
else:
    run_or_set_score = 2 * run_or_set_length - 3 - number_of_kings_in_run_or_set
```

The score for a hand is the sum of the scores for every consecutive run of cards and set of cards that can be formed by the player.

## Usage

To get started using this package, import the ``Game`` class. This package makes use of Jupyter's rich output to display HTML representations of the game

```python
from cardgame import Game
game = Game.deal()
# Put the game at the end of a cell in Jupyter to implicity call the display function
game
```

To simulate gameplay, we can use a simple loop

```python
game = Game.deal()
while game.legal_moves:
    game = game.random_move()
print(game.score)
```

This package is mostly focussed on enabling the calculation of optimal strategies in a given position, which has some nuance due to the random elements of the game.

This would calculate the probabilistic score of each move from the position the game was in 5 moves ago with optimal play, from the perspective of the player to move

```python
game.undo(5).move_evals
```

Examples of further functionality to explore the state space are given below, the implementation is far from exhaustive

```python
# Get the players hands
p1_hand, p2_hand = game.p1, game.p2

# See the full board, including cards that have been taken
game.board

# Access the move sequence in the game
game.moves

# Calculate the scores for each hand
p1_hand.score()

# or the score for the game, which is the score for p1 - the score for p2
game.score

# Find the best move from a given position (naive minimax)
best_score, best_move = game.score_walk()

# Display the evalualtions of each move from a position
best_score, best_move = game.move_eval
```