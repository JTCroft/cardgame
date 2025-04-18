# Cross Kings

A python implementation of a 2 player card game using a subset of standard playing cards called Cross Kings

```bash
pip install git+https://github.com/JTCroft/cardgame.git
```

## Contents

- [Gameplay](#gameplay)
- [Scoring](#scoring)
- [Usage](#usage)
- [Position Evaluation](#position-evaluation)
- [Other functionality](#other-functionality)
- [Possible additions](#possible-additions)

## Gameplay

The game uses the cards Ace to 8 of all suits plus the four Kings. To start the game deal the cards in a 6 by 6 grid, with all cards on the diagonals face down and all other cards face up.

A marker displays the current position, the player who is not taking the first move may choose which of the 4 central face down cards of the grid to place the marker on at the start of the game. In this implementation the starting position is fixed.

Play then proceeds in alternating turns. Each player moves the marker to any card that is in the same row or column as the marker and collects that card into their hand, leaving a gap in the grid of cards with the marker in. The marker must move on the first turn. The marker may pass over any gaps left by previously taken cards.

The hands of both players are open, displayed face up. Both players know which cards their opponent has collected, and when either player collects a facedown card into their hand the card is revealed to both players.

The game finishes when the marker is left in a position with no moves, ie: when there are no cards in the same row or column as the marker.

## Scoring

The game is scored at the end, each player can calculate their points individually to determine which player has the most points and is the winner.

A player earns points by having collected cards that form **consecutive runs** of the same suit, or **sets** of the same rank. A card can be in both a run and a set simultaneously.

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

A player can use a king to act as a wildcard, choosing which card it will act as to maximise points. However each time a king forms part of a scoring run or set that run or set will be worth one less point than it would have been otherwise.

This logic can be summarised as

```python
if run_or_set_length < 3:
    run_or_set_score = 0
else:
    run_or_set_score = 2 * run_or_set_length - 3 - number_of_kings_in_run_or_set
```

The score for a hand is the sum of the scores for every run and set that can be formed by the player.

## Usage

To get started using this package, import the ``Game`` class. This package makes use of Jupyter's rich output to display HTML representations of the game

```python
from cardgame import Game
game = Game.deal()
# Put the game at the end of a cell in Jupyter to implicity call the display function
game
```

To simulate gameplay, we can use a loop

```python
game = Game.deal()
while game.legal_moves:
    game = game.random_move()
print(game.score)
```

This package is mostly to enable analysis of gameplay and calculation of optimal strategies at a given position, which is nuanced due to the random elements of the game, implementing expectiminimax with alpha beta pruning.

`Game.evaluate()` returns a summary dictionary giving

* 'Deterministic optimal moves' - The optimal move sequence from that position (to the point it branches into multiple possibilities)
* 'Evaluation' - The static evaluation of the position, as a frequency map of the score difference in each possible outcome
* 'Known info for other branches' - The full or partial evaluations of the other moves from the position. When the move was able to be pruned in the evaluation as definitely non-optimal the evaluation will be partial

Scores in the evaluation are given from the perspective of the player to move, as the score for the players hand minus the score for the opponents hand at the point the game ended (positive score values &rarr; player whose turn it is will win)

```python
game.undo(5).evaluate()
```

## Position Evaluation

To calculate how 'good' an outcome is for a player has some complexity due to how the final score maps onto the win/draw/loss possibilities.

To determine if a position (which may not be fully evaluated) is better for the player than another I:

* Take the lower bound of the possible outcomes for the position, filling in any unevaluated branches with -25
* Take the upper bound of the possible outcomes for the other position, filling in any unevaluated branches with +25
* Calculate the number of wins, draws, and losses for each of those, and the sum of the scores of all possible outcomes
* With the win count, draw count, and cumulative score given by variables `w`, `d`, and `s` I calculate `(w + d/2, w, s)`
* I compare the tuple calculated for each position

The terms in the evaluation `(w + d/2, w, s)` are based on calculating

* Expected value (1 point for a win, half a point for a draw)
* Preferring decisive outcomes over tied outcomes
* How much the player wins or loses by on average

This evaluation criteria is a choice, and the second term means it is not strictly well ordered which can cause issues in pruning branches in alpha beta search. One issue this introduces is best demonstrated with an example. If choosing between 2 moves where one of guarantees a draw, and one which has a 50% chance of a win and a 50% chance of a draw, the second term which enforces a preference for decisive outcomes means that a player would prefer the one that gives them a chance of winning. When doing a tree search, the other player will believe that the outcome of a draw is WORSE for them than a 50/50 win/loss, which means they may think it is BETTER for the original player causing the branch to be pruned unless you check that the difference is well ordered (implemented as only pruning when both `alpha > beta and not -alpha > -beta`)

Another issue is that a preferable metric for the third component, based on how many points you expect to win or lose by, is also a compromise. I would consider a move to be better by considering which move will have a better outcome for the player (end with more points) more than 50% of the time that would be a better ordering, but that comparison is not well ordered either, as shown by the [intransitive dice](https://en.wikipedia.org/wiki/Intransitive_dice) example. Using the cumulative score difference (which is equivalent to the mean score difference for ordering) is used as a compromise, I could also use have chosen to use the median score, or another measure.

Different move evaluations can still be tied after comparing each component of the evaluation in order. As a tie breaker to make sure that `Game.evaluate` is deterministic, it will prefer the move to a lower row, and then further right column all else equal.

## Other functionality

Examples of further methods to explore the state space are given below, the implementation is far from exhaustive

```python
# Dump the game state to a string, and load from a string for portability
saved_game = game.save()
game_str = '5C2D8C3HKH??/AS??8S3S??7S/5S8H4H??5D7C/AH6C3D6S5HAD/6HKCACKD4C2H/8D2S7H4D7D??//2C3C4S6DKS//731630308968466917018937'
game = Game.load(game_str)

# Get the players hands
p1_hand, p2_hand = game.p1, game.p2

# See the full board, including cards that have been taken
game.board

# Access the move sequence in the game
game.moves

# Calculate the scores for each hand
p1_hand.score()

# or the current score for the game, which is the score for p1 - the score for p2
game.score

# Find the best move from a given position using naive minimax
best_score, best_move = game.score_walk()

# evalualtions of each move from a position
game.move_evals
```

## Possible additions

Looking at improving the ability to use this package to generate insights into the optimal strategy in this game

- Improved constraints for alpha/beta pruning game tree search
- Calculation of legal moves using a sequence of lookups & precalculation
- Changing the search order to bound the evaluation of a position quicker than A/B pruning (\*-minimax search and similar extensions, see "The *-minimax search procedure for trees containing chance nodes", Bruce Ballard)
- Speed optimisation
- Heuristic value of a position for improved move ordering & iterative deepening search
- Transposition tables