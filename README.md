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
- [Playing online](#playing-online)
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

* Take the lower bound of the possible outcomes for the position, filling in any unevaluated branches with -26
* Take the upper bound of the possible outcomes for the other position, filling in any unevaluated branches with +26
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

# See what the current facedown cards are
game.board.facedown_cards

# Access the move sequence in the game
game.moves

# Make a move
from random import choice
game = choice(game.move(3, 4)) # always a tuple of resolutions for facedown cards
game = game.random_move()

# Calculate the scores for each hand
p1_hand.score()

# or the current score for the game, which is the score for p1 - the score for p2
game.score

# get the detailed comparison of the different moves available from a position
from cardgame import analyse_moves
analyse_moves(game)
```

### Iterative bounded search with live rankings

`Game.evaluate` is depth-first, so it can say almost nothing until it is
nearly finished. `cardgame.search_alt` runs the *same* evaluation - the
identical &plusmn;26 unknown-outcome fills, sub-window arithmetic, move
abandonment, and fail-high guard, so its results provably match - but
resumably, one quantum at a time, with moves revisitable in any order.
That turns the solver into an anytime search: at every step the root can
report each move's partial outcome distribution, a best-to-worst ranking
(ordered by the average of each move's already-resolved outcomes - nearby
leaves correlate, so the resolved share is representative), the incumbent
best move, and whether that move is already *proven* best (every rival
abandoned or completed worse) - which can happen before its exact value
is known, a stop the all-or-nothing depth-first solver cannot make.
Subtrees small enough for the calibrated exact gate are solved by
`Game.evaluate` directly, under the search's current window. Children are
instantiated lazily and deleted as soon as they are merged, abandoned, or
resolved, so live memory stays at a few dozen nodes even after tens of
thousands of leaf solves. Benchmarked at 12-14 cards it runs at parity
with a bare `Game.evaluate` call.

```python
from cardgame import live_search, format_snapshot

for snapshot in live_search(game, budget=30):
    print(format_snapshot(snapshot))   # live best-to-worst, updating

# or drive the node iterator directly (the spec in search_alt's docstring)
from cardgame import move_search_iterator
root = move_search_iterator(game)
for _ in root:
    ...   # root.best_move, root.proven, per-move partials, at any point
print(root.result)  # exact Evaluation, identical to Game.evaluate's
```

Displayed intervals for rival moves are window-relative, as in any
alpha-beta engine: a rival cut off early shows "no better than" bounds
rather than its exact value. `cardgame.search.BestFirstSearch` is an
earlier variant of the same idea that keeps fully sound two-sided
distribution bounds for every move (no window truncation) at the cost of
substantially slower proving; it remains useful when honest two-sided
bounds on *every* move matter more than speed.

### Bot development and strength testing

The computer opponent (`cardgame.ai`) is an `AlphaBetaBot`: iterative-deepening
expectiminimax with Star1 chance-node cutoffs and a heuristic leaf evaluation
in the midgame, deferring to the exact solver once the endgame is small
enough. Every tunable (evaluation weights, value bound, resolution sampling,
time management) lives in a `SearchParams` dataclass, so differently
configured bots can be built side by side:

```python
from cardgame import AlphaBetaBot, SearchParams
bot = AlphaBetaBot(time_budget=2.0, params=SearchParams(potential_weight=0.5))
move = bot.choose_move(game)   # module-level choose_move(game) uses defaults
```

Any change to the bot (or its parameters) should be validated with a
duplicate-deal match in `cardgame.validation.arena` before it is kept:

```bash
python -m cardgame.validation.arena --old HEAD --new current --deals 50 --budget 0.3 --jobs 4
```

Each deal is played twice with seats swapped on the same board *and* the same
hidden-card placement, so deal luck cancels within the pair; results are
reported per game (W/D/L, score percentage, Elo estimate) and per pair (the
challenger's combined margin over both seatings, with an exact two-sided sign
test). Bots are specified as `current` (working tree), any git rev (that
revision's `ai.py` imported against the current package), or `file:<path>`
for an arbitrary saved variant. Reduced per-move budgets (0.2-0.5s) are the
intended testing regime — relative strength transfers well, and 50 pairs
finish in minutes with `--jobs`.

## Playing online

A basic Flask + Flask-SocketIO web front end is included so people can play
against each other in a browser, in a shared "room", with moves synced to
everyone watching live over a WebSocket as soon as they happen. Install the
extra web dependencies:

```bash
pip install "git+https://github.com/JTCroft/cardgame.git#egg=cardgame[web]"
```

Then start the dev server:

```bash
cardgame-web
```

By default this serves on `http://127.0.0.1:5000`.

- Visit the homepage to create a room (you'll be redirected to a URL like
  `/room/ABCD`), or go directly to a room by its 4 letter code:
  `http://127.0.0.1:5000/room/ABCD`.
- Visiting a room makes you a spectator by default. The first time you try
  to claim a seat, you'll be asked to pick a display name (kept in your
  session, so it carries across rooms) — that name is what everyone else
  sees instead of "Player 1" / "Player 2". You can change it any time from
  the &#9881; button in the top corner.
- Either spectator can claim a vacant "Join as Player 1/2" seat, and a
  seated player can hand their seat back with "Leave seat", right up until
  the first move of that game is made. Once the game is under way, seats
  are locked in for the rest of it.
- The `/rooms` page lists every room currently in memory — who's seated,
  how many people are spectating, and whether it's waiting for players, in
  progress, or finished — for matchmaking or spectating, and updates live
  the same way as a room page does.
- Since the board and both hands are open information in this game, the
  card most recently added to a hand (which isn't always obvious once it's
  sorted into that player's spread) gets a highlighted glow until the next
  move.
- On narrow (phone-width) screens the layout switches to smaller,
  simplified cards — loosely adapted from the ["inText" mode](https://github.com/selfthinker/CSS-Playing-Cards)
  of the referenced CSS-Playing-Cards project — and each hand collapses
  into a bottom drawer you can flick open from a tab, instead of two full
  hands competing with the board for space.
- The game can't start until both seats are filled — a lone player just
  sees a "waiting for another player to join" message, with no clickable
  cells, and a direct move attempt is rejected server-side too even if
  something bypasses the UI.
- When a game ends, a modal pops up once for everyone watching (announcing
  a win/loss/draw and the final score) rather than a small inline message;
  it's dismissible and reopenable ("View result") without losing your
  place. From there, either player can step back and forth through the
  finished game's move history (independently per viewer — one person
  browsing old moves doesn't affect what anyone else sees), and either can
  request a rematch, which the other player has to accept or decline
  before a new game actually deals. Seats free up again once a game is
  over (whether it finished naturally or a rematch was declined), the same
  as before a game starts.
- A second solo mode, **Play vs computer (live eval)** (`/play/live`),
  runs the anytime search (`cardgame.search_alt`) continuously on
  whatever the current position is, streaming a live move-ranking panel
  to the page: each move's average outcome over the lines resolved so
  far, bounds on its final score margin, how much of it has been
  explored, and best/out badges as moves get proven or ruled out. The
  search restarts whenever a move changes the position and pauses at a
  per-position time/work cap (early positions are far too big to finish -
  the panel just shows how far it got), and the background thread stops
  whenever nobody is connected. Rival moves' ranges are window-relative,
  as in any alpha-beta engine.
- While stepping back through a finished game, positions near the end
  also show a **Move comparison** panel — the same analysis as
  `analyse_moves` (see `examples/Move comparison.ipynb`): every move
  available from that position, ranked by how much it swings the average
  final score relative to the best move under optimal play from both
  sides, split into the effect on each player's own total, with the best
  move and the move actually played badged. Computing this means walking
  the entire remaining game tree with no pruning, so the work starts
  *during* the game: each position is fixed the moment its move is made,
  so once the game reaches a sensible starting point a background worker
  (one per room, one position at a time) analyses the already-played
  positions while the players think - and when it has caught up, it
  keeps backtracking one position deeper with no ceiling at all, the
  running game itself being the budget: a long, thoughtful game buys
  itself review depth no fixed cutoff could. An analysis still in flight
  when the game ends is abandoned cooperatively rather than left burning
  CPU. At game end a drain fills the cheap tail immediately (skipping
  anything the live worker finished), walking backwards adaptively
  within a calibrated ceiling and time budget. Positions deeper than the
  work ever reached simply don't show the panel.
  The panel appears in both a room's own post-game history stepping and
  the frozen "Recently finished games" review pages, computed once and
  shared.

Useful environment variables:

- `CARDGAME_HOST` / `CARDGAME_PORT` - interface and port to bind (default `127.0.0.1:5000`)
- `CARDGAME_DEBUG` - set to `1` to run Flask in debug/reload mode
- `CARDGAME_SECRET_KEY` - session signing key; set this to a fixed value if
  you want player sessions to survive server restarts

Note that game state (and the mapping of connected sockets to rooms) is
kept in memory in a single process, which is fine for local play or a
small demo deployment, but means state is lost on restart and won't be
shared across multiple worker processes. The dev server here runs
Flask-SocketIO in "threading" mode, which is the simplest way to get real
WebSocket support without extra dependencies, but it only really scales to
one process. For a more robust deployment: run behind a proper WSGI/ASGI
setup with a single worker (e.g. gunicorn with an eventlet or gevent
worker class, which Flask-SocketIO integrates with directly), or move room
state into a shared store and configure Flask-SocketIO's `message_queue`
option (e.g. backed by Redis) so events can be fanned out across multiple
worker processes. Rooms also currently live forever once created (there's
no pruning of old/abandoned rooms), so the `/rooms` list will grow
unbounded over a long-running server's lifetime.

### Architecture changes behind the above

Supporting names, flexible seating, and a lobby meant reworking a few
things rather than bolting them on:

- **Seating became an explicit, revocable action instead of an automatic
  side effect of visiting a URL.** Previously, the first two browsers to
  open a room were permanently assigned Player 1/2 on arrival. `RoomState`
  now tracks seats as a plain `{1: player_id, 2: player_id}` mapping that's
  only mutated by explicit `claim_seat` / `vacate_seat` actions (locked to
  before the first move), with a room's connected sockets always defaulting
  to spectator. This is what makes "leave your seat" / "spectator claims a
  vacant seat" possible at all, and it's also what let the name prompt
  attach naturally to the moment you try to claim a seat, rather than
  needing its own separate flow.
- **Player identity now has two parts.** `player_id` (an anonymous
  per-browser id) is unchanged, but there's now also a `player_name`, kept
  in the same session cookie so it persists across every room you visit in
  that browser. Each room keeps its own small `player_id -> name` cache
  (`RoomState.player_names`) purely so it can render *other* people's names
  to everyone watching; the session copy is the source of truth for "your"
  name.
- **A lobby subsystem sits alongside the per-room state.** Every room's
  live state is personalised per viewer (your own seat, whether you can
  join a vacant one, etc.), so it's pushed individually to each connected
  socket rather than broadcast. The `/rooms` listing has no such
  personalisation — everyone sees the same table — so it uses a real
  Flask-SocketIO broadcast group ("lobby") instead, refreshed whenever any
  room's seats, name, or game state changes.
- **The shared rendering macros gained a couple of backward-compatible
  hooks** rather than being copied and modified: `draw_card` takes an
  optional `highlight` flag, and `draw_board` takes optional
  `legal_moves`/`move_handler` params for turning cells into buttons -
  existing callers (including the Jupyter notebook `_repr_html_` output)
  are unaffected since the new parameters all default to off.
- **The base template now owns one shared Socket.IO connection and the name
  modal** for every page (home, room, lobby), instead of each page wiring
  up its own; page-specific handlers (board updates, lobby updates) hook
  into that same connection via a Jinja block.

## Possible additions

Looking at improving the ability to use this package to generate insights into the optimal strategy in this game

- Improved constraints for alpha/beta pruning game tree search
- Calculation of legal moves using a sequence of lookups & precalculation
- Changing the search order to bound the evaluation of a position quicker than A/B pruning (\*-minimax search and similar extensions, see "The *-minimax search procedure for trees containing chance nodes", Bruce Ballard)
- Speed optimisation
- Heuristic value of a position for improved move ordering & iterative deepening search
- Transposition tables