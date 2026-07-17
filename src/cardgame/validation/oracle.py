"""Oracle labeling and agreement testing for the computer opponent.

The exact solver (`Game.evaluate`) terminates quickly enough on late
positions to act as an oracle there. This module turns that into two
tools for improving the heuristic bot:

* **A labeled dataset**: sample random late positions and record, for
  every legal move and every face-down resolution, the child position's
  exact outcome distribution. One labeled parent yields both the set of
  provably optimal moves (for agreement testing) and a batch of
  (position -> exact value) rows (for fitting evaluation weights).

* **An agreement metric**: the fraction of labeled positions where a bot
  configuration's move is in the optimal set. Run at a small *fixed
  depth* with the exact-endgame gate disabled, it isolates evaluation
  quality from search speed and time management, and it is paired across
  configurations (same positions), so McNemar's test gives far more
  statistical power per CPU-hour than arena matches. The arena stays the
  final validator - agreement gains are a hypothesis until they survive
  full games.

Positions are sampled by random playout into a card band where labeling
is affordable; the working assumption is that evaluation weights fitted
and tested on this band generalize to earlier positions, which the arena
match ultimately checks.

CLI:
    python -m cardgame.validation.oracle generate --positions 800 --out labels.jsonl --jobs 8
    python -m cardgame.validation.oracle agree --labels labels.jsonl --depth 3 \
        --a "" --b "potential_weight=0.5"
"""

import argparse
import json
import random
import signal
import sys
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import comb, sqrt
from pathlib import Path

from ..ai import (
    AlphaBetaBot,
    SearchParams,
    _add_card,
    _remaining_cards,
    _without,
    _TOKEN,
)
from ..cards import Card
from ..game import Board, Eval, Game

__all__ = (
    "sample_position",
    "label_position",
    "optimal_markers",
    "fixed_depth_move",
    "agreement",
    "main",
)


# --------------------------------------------------------------------------
# Sampling and labeling


def sample_position(rng, min_cards=8, max_cards=11, max_facedown=5):
    """Random playout to a labelable position (mover has a real choice),
    or None if this playout missed the band."""
    deck = list(Card.deck())
    rng.shuffle(deck)
    indices = sorted(Board.facedown_indices)
    facedown_cards = tuple(deck[i] for i in indices)
    placeholder = Card(facedown=True)
    for i in indices:
        deck[i] = placeholder
    game = Game(Board([deck[i : i + 6] for i in range(0, 36, 6)], facedown_cards), ())
    target = rng.randint(36 - max_cards, 36 - min_cards)
    for _ in range(target):
        if not game.legal_moves:
            return None
        marker = rng.choice(sorted(game.legal_moves))
        game = rng.choice(game.move(*marker))
    if (
        len(game.legal_moves) > 1
        and len(game.board.facedown_cards) <= max_facedown
    ):
        return game
    return None


def label_position(game):
    """Exact labels for every legal move of `game`: per face-down
    resolution, the child position's outcome distribution (w, d, s, m)
    from the *child mover's* perspective."""
    moves = []
    for resolutions in game.all_moves():
        entries = []
        for child in resolutions:
            ev = child.evaluate()["Evaluation"]
            w, d, s = ev.wds
            entries.append(
                {"card": repr(child.taken_card), "w": w, "d": d, "s": s,
                 "m": ev.multiplicity}
            )
        moves.append({"marker": list(resolutions[0].marker), "resolutions": entries})
    return {"save": game.save(alnum=True), "cards_left": 36 - len(game.moves),
            "facedown": len(game.board.facedown_cards), "moves": moves}


def optimal_markers(record):
    """The set of provably optimal moves in a labeled record, under the
    exact criterion (a negated child distribution is the move's value;
    face-down resolutions combine by summing counts)."""
    move_evals = {}
    for move in record["moves"]:
        w = d = s = m = 0
        for entry in move["resolutions"]:
            # Negate the child-perspective (w, d, s): losses become wins.
            losses = entry["m"] - entry["w"] - entry["d"]
            w += losses
            d += entry["d"]
            s -= entry["s"]
            m += entry["m"]
        move_evals[tuple(move["marker"])] = Eval(m, w, d, s)
    best = max(move_evals.values())
    return {marker for marker, ev in move_evals.items() if not ev < best}


# --------------------------------------------------------------------------
# Fixed-depth agreement


def fixed_depth_move(bot, game, depth):
    """The bot's move from one full-width iteration at exactly `depth`,
    bypassing time management and the exact-endgame gate - a probe of pure
    evaluation quality behind a fixed amount of search."""
    if len(game.moves) % 2 == 0:
        me, opp = game.p1.as_int, game.p2.as_int
    else:
        me, opp = game.p2.as_int, game.p1.as_int
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
    deadline = float("inf")
    alpha = -bound
    best = None
    for marker, facedown in bot._ordered_markers(game, me, opp, mask):
        child_mask = mask | (1 << (marker[0] * 6 + marker[1]))
        resolutions = bot._resolutions(game, marker, facedown)
        if len(resolutions) == 1:
            child = resolutions[0]
            card = child.taken_card
            token = _TOKEN[card]
            value = -bot._search(
                child, depth - 1, -bound, -alpha, opp, _add_card(me, card),
                (token,), _without(remaining, token), child_mask, deadline,
            )
        else:
            value = bot._chance_value(
                resolutions, depth, alpha, bound, me, opp, (), remaining,
                child_mask, deadline,
            )
        if value > alpha:
            alpha = value
            best = marker
    return best


def _params_from(query):
    if not query:
        return SearchParams(exact_endgame=False)
    overrides = {}
    for item in query.split("&"):
        key, _, value = item.partition("=")
        overrides[key] = literal_eval(value)
    overrides.setdefault("exact_endgame", False)
    return SearchParams(**overrides)


def agreement(records, params, depth):
    """Per-record hit list: 1 when the config's fixed-depth move is in the
    oracle-optimal set."""
    bot = AlphaBetaBot(params=params)
    hits = []
    for record in records:
        game = Game.load(record["save"])
        optimal = optimal_markers(record)
        hits.append(1 if fixed_depth_move(bot, game, depth) in optimal else 0)
    return hits


def _mcnemar_p(b, c):
    """Exact two-sided McNemar on discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2.0 * tail)


# --------------------------------------------------------------------------
# CLI


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _generate_one(args):
    seed, band, max_seconds = args
    rng = random.Random(seed)
    if max_seconds:
        signal.signal(signal.SIGALRM, _alarm)
    while True:
        game = sample_position(rng, *band)
        if game is None:
            continue
        if not max_seconds:
            return json.dumps(label_position(game))
        signal.alarm(max_seconds)
        try:
            result = json.dumps(label_position(game))
        except _Timeout:
            continue
        finally:
            signal.alarm(0)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="sample and label positions")
    gen.add_argument("--positions", type=int, default=800)
    gen.add_argument("--out", required=True)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--jobs", type=int, default=1)
    gen.add_argument("--min-cards", type=int, default=8)
    gen.add_argument("--max-cards", type=int, default=11)
    gen.add_argument("--max-facedown", type=int, default=5)
    gen.add_argument("--max-seconds", type=int, default=0,
                      help="skip and resample a position if labeling takes longer than this (0=no limit)")

    agr = sub.add_parser("agree", help="fixed-depth agreement of one or two configs")
    agr.add_argument("--labels", required=True)
    agr.add_argument("--depth", type=int, default=3)
    agr.add_argument("--a", default="", help="SearchParams overrides, k=v&k=v")
    agr.add_argument("--b", default=None, help="optional second config to compare")

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        band = (args.min_cards, args.max_cards, args.max_facedown)
        tasks = [(f"{args.seed}:{i}", band, args.max_seconds) for i in range(args.positions)]
        out = Path(args.out)
        with out.open("w") as fh:
            if args.jobs <= 1:
                for i, task in enumerate(tasks):
                    fh.write(_generate_one(task) + "\n")
                    if (i + 1) % 25 == 0:
                        print(f"{i + 1}/{args.positions}", flush=True)
            else:
                with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                    futures = [pool.submit(_generate_one, t) for t in tasks]
                    for i, future in enumerate(as_completed(futures)):
                        fh.write(future.result() + "\n")
                        if (i + 1) % 25 == 0:
                            print(f"{i + 1}/{args.positions}", flush=True)
        print(f"wrote {args.positions} labeled positions to {out}")
        return

    records = [json.loads(line) for line in Path(args.labels).open()]
    hits_a = agreement(records, _params_from(args.a), args.depth)
    rate_a = sum(hits_a) / len(hits_a)
    se_a = sqrt(rate_a * (1 - rate_a) / len(hits_a))
    print(f"config A: {sum(hits_a)}/{len(hits_a)} optimal ({100 * rate_a:.1f}% ± {100 * se_a:.1f}%)")
    if args.b is not None:
        hits_b = agreement(records, _params_from(args.b), args.depth)
        rate_b = sum(hits_b) / len(hits_b)
        se_b = sqrt(rate_b * (1 - rate_b) / len(hits_b))
        print(f"config B: {sum(hits_b)}/{len(hits_b)} optimal ({100 * rate_b:.1f}% ± {100 * se_b:.1f}%)")
        b = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
        c = sum(1 for x, y in zip(hits_a, hits_b) if y and not x)
        print(f"discordant: A-only {b}, B-only {c}  (McNemar p={_mcnemar_p(b, c):.4f})")


if __name__ == "__main__":
    main()
