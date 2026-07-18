"""Bootstrapped labels for early-game positions (19+ cards left).

Exact labeling is infeasible above ~18 cards left (chance-node
multiplicity is factorial in the unresolved face-down count), so nothing
constrains the leaf eval in the opening - the direct cause of the
2026-07-17 joint-refit arena loss. This module labels opening positions
with the value of a timed iterative-deepening search by the current bot:
search amplifies eval quality (measured depth-1 -> depth-3 oracle
agreement 58% -> 67%), and deep-search leaves from these positions land
in or near the oracle-validated 8-18 card band, so label quality is
anchored from below. The labels are *estimates*, not ground truth: they
inherit the current eval's systematic biases (bootstrap circularity), so
any weights fitted to them still face the arena as the final gate.

Record format mirrors the oracle corpus, with a scalar search value per
child instead of an exact outcome distribution:

    {"save": ..., "cards_left": N, "facedown": N, "budget": secs,
     "moves": [{"marker": [r, c],
                "resolutions": [{"card": "...", "v": value,
                                 "depth": completed_depth}, ...]}, ...]}

`v` is from the *child mover's* perspective (same convention as the
oracle corpus) in the search's value scale (score diff + win-bonus
expectation, the same scale fit_weights targets use). `depth` is the last
fully completed deepening iteration behind the value (-1 for terminal
children, whose `v` is exact). Face-down resolutions are sampled to the
bot's resolution cap (rank-ordered, evenly spaced) - fine for fitting
rows, but these records do NOT support optimal_markers-style move
adjudication.

CLI (append-mode, flushed per line; use a fresh --seed for top-ups):

    python -m cardgame.validation.bootstrap generate \
        --out data/bootstrap_labels.jsonl --positions 1000 \
        --min-cards 19 --max-cards 30 --budget 2.0 --jobs 4
"""
import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..ai import (
    AlphaBetaBot,
    _TOKEN,
    _Timeout,
    _add_card,
    _remaining_cards,
    _without,
)
from ..game import Game
from .oracle import sample_position


def _hands(game):
    if len(game.moves) % 2 == 0:
        return game.p1.as_int, game.p2.as_int
    return game.p2.as_int, game.p1.as_int


def timed_value(bot, game, budget):
    """Root value of `game` (mover's perspective) from iterative deepening
    under `budget` seconds, and the depth of the last completed iteration.
    Depth 1 runs without a deadline so every call returns a real value;
    an interrupted deeper iteration is discarded (its root value is
    partial), unlike move choice where mid-iteration improvements are
    sound - labels need values, not moves."""
    me, opp = _hands(game)
    remaining = tuple(_TOKEN[card] for card in _remaining_cards(game))
    mask = 0
    for row, col in game.moves:
        mask |= 1 << (row * 6 + col)
    bot._tt = {}
    bot._tt_cuts = 0
    bot._exact_cache = {}
    bot._fullpot = {}
    bot._root_tokens = remaining
    bound = bot.params.value_bound
    deadline = time.perf_counter() + budget
    moves = [
        (marker, bot._resolutions(game, marker, facedown))
        for marker, facedown in bot._ordered_markers(game, me, opp, mask)
    ]
    value = None
    completed = 0
    depth = 1
    while depth <= 36 - len(game.moves):
        alpha = -bound
        scores = {}
        iter_deadline = float("inf") if depth == 1 else deadline
        try:
            for marker, resolutions in moves:
                child_mask = mask | (1 << (marker[0] * 6 + marker[1]))
                if len(resolutions) == 1:
                    child = resolutions[0]
                    card = child.taken_card
                    token = _TOKEN[card]
                    v = -bot._search(
                        child, depth - 1, -bound, -alpha, opp,
                        _add_card(me, card), (token,),
                        _without(remaining, token), child_mask, iter_deadline,
                    )
                else:
                    v = bot._chance_value(
                        resolutions, depth, alpha, bound, me, opp, (),
                        remaining, child_mask, iter_deadline,
                    )
                scores[marker] = v
                if v > alpha:
                    alpha = v
        except _Timeout:
            break
        value = alpha
        completed = depth
        moves.sort(key=lambda entry: (-scores[entry[0]], entry[0]))
        depth += 1
        if time.perf_counter() > deadline:
            break
    return value, completed


def label_position(game, budget, bot=None):
    """Bootstrap labels for every legal move of `game`: per (sampled)
    face-down resolution, the child's timed-search value from the child
    mover's perspective."""
    bot = bot or AlphaBetaBot()
    moves = []
    for resolutions in game.all_moves():
        marker = resolutions[0].marker
        facedown = marker in game.board.facedown_positions
        if facedown:
            resolutions = bot._sample_resolutions(resolutions)
        entries = []
        for child in resolutions:
            if child.legal_moves:
                v, depth = timed_value(bot, child, budget)
            else:
                child_me, child_opp = _hands(child)
                v, depth = bot._terminal_value(child_me, child_opp), -1
            entries.append(
                {"card": repr(child.taken_card), "v": v, "depth": depth}
            )
        moves.append({"marker": list(marker), "resolutions": entries})
    return {"save": game.save(alnum=True), "cards_left": 36 - len(game.moves),
            "facedown": len(game.board.facedown_cards), "budget": budget,
            "moves": moves}


def _generate_one(args):
    seed, band, budget = args
    rng = random.Random(seed)
    while True:
        game = sample_position(rng, *band)
        if game is not None:
            return json.dumps(label_position(game, budget))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="sample and bootstrap-label positions")
    gen.add_argument("--out", required=True)
    gen.add_argument("--positions", type=int, default=1000)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--jobs", type=int, default=4)
    gen.add_argument("--min-cards", type=int, default=19)
    gen.add_argument("--max-cards", type=int, default=30)
    gen.add_argument("--max-facedown", type=int, default=12)
    gen.add_argument("--budget", type=float, default=2.0)
    args = parser.parse_args(argv)

    band = (args.min_cards, args.max_cards, args.max_facedown)
    tasks = [
        (f"{args.seed}:{i}", band, args.budget) for i in range(args.positions)
    ]
    out = Path(args.out)
    if out.exists():
        print(f"appending to existing {out}")
    start = time.time()
    with out.open("a") as fh:
        if args.jobs <= 1:
            for i, task in enumerate(tasks):
                fh.write(_generate_one(task) + "\n")
                fh.flush()
                if (i + 1) % 5 == 0:
                    el = time.time() - start
                    print(f"{i + 1}/{args.positions}  {el / (i + 1):.0f}s/pos  "
                          f"eta {el / (i + 1) * (args.positions - i - 1) / 60:.0f}m",
                          flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(_generate_one, t) for t in tasks]
                for i, future in enumerate(as_completed(futures)):
                    fh.write(future.result() + "\n")
                    fh.flush()
                    if (i + 1) % 5 == 0:
                        el = time.time() - start
                        print(
                            f"{i + 1}/{args.positions}  {el / (i + 1):.1f}s/pos  "
                            f"eta {el / (i + 1) * (args.positions - i - 1) / 60:.0f}m",
                            flush=True,
                        )
    print(f"done: {args.positions} positions -> {out}")


if __name__ == "__main__":
    main()
