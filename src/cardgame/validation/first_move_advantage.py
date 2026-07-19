"""Estimate the first-move (Player 1) advantage under the production time budget.

Plays the *same* bot against itself on many independent random deals at the
6s/move budget used by the web UI (`ai.choose_move` default), and reports the
mean P1-P2 score margin and P1 win rate with confidence intervals.

Unlike `cardgame.validation.arena` (which compares two different bot specs
and uses seat-swapped duplicate deals to cancel deal luck between them),
there is nothing to swap here: both seats run the identical bot, so any
systematic P1-P2 margin *is* the thing being measured, not noise to cancel.
Each deal is simply played once; variance is reduced by playing many deals.

Results are appended to a JSON-lines file after every game, so the run can
be stopped (Ctrl-C, crash, timeout) and resumed without losing progress.

Usage:

    uv run python -m cardgame.validation.first_move_advantage \
        --games 400 --budget 6.0 --jobs 4 --out data/first_move_advantage.jsonl

Resuming: just re-run with the same --out file; already-played seeds are
skipped automatically.
"""

import argparse
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .arena import _bot, deal_with_assignment, play_game
import random

__all__ = ("play_one", "load_results", "summarise", "main")


def play_one(seed, bot_spec, budget):
    """Play one game with `bot_spec` as both P1 and P2. Returns a result dict.

    The game is a rook's-move claiming game, not a fixed-length one: it ends
    as soon as the marker cell has no unclaimed row/column mate, so the total
    number of moves (and therefore who ends up with an extra card) depends on
    the path the players take. `moves` records that path length so P1's
    score advantage can be broken out by whether P1 got an extra card
    (odd total) or the hands ended up equal-sized (even total)."""
    bot = _bot(bot_spec, budget)
    rng = random.Random(seed)
    board, assignment = deal_with_assignment(rng)
    game = play_game(board, assignment, (bot, bot))
    return {
        "seed": seed,
        "score": game.score,
        "moves": len(game.moves),
        "p1_cards": len(game.p1),
        "p2_cards": len(game.p2),
    }


def load_results(path):
    """Load already-played results from a jsonl file (empty list if absent)."""
    if not path.exists():
        return []
    results = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _score_stats(scores):
    n = len(scores)
    if n == 0:
        return None
    w = sum(s > 0 for s in scores)
    d = sum(s == 0 for s in scores)
    l = n - w - d
    win_frac = (w + d / 2) / n
    win_se = math.sqrt(win_frac * (1 - win_frac) / n) if n > 1 else 0.0

    mean = statistics.fmean(scores)
    if n > 1 and any(scores):
        se = statistics.stdev(scores) / math.sqrt(n)
        t = mean / se if se else 0.0
        p_t = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        se, t, p_t = 0.0, 0.0, 1.0
    return {
        "n": n, "w": w, "d": d, "l": l,
        "win_frac": win_frac, "win_se": win_se,
        "mean": mean, "se": se, "t": t, "p_t": p_t,
    }


def _format_stats(stats, label):
    if stats is None:
        return f"{label}: no games"
    s = stats
    return (
        f"{label}: {s['n']} games\n"
        f"  P1 record: {s['w']}W {s['d']}D {s['l']}L\n"
        f"  P1 win rate: {100 * s['win_frac']:.1f}% +/- {100 * 1.96 * s['win_se']:.1f}% (95% CI)\n"
        f"  P1-P2 score margin: {s['mean']:+.2f} +/- {1.96 * s['se']:.2f} points/game (95% CI)"
        f"  (t={s['t']:.2f}, p={s['p_t']:.4f})"
    )


def summarise(results):
    scores = [r["score"] for r in results]
    overall = _score_stats(scores)
    if overall is None:
        return "no games played yet"
    lines = [_format_stats(overall, "overall")]

    # Parity breakdown: only available for games recorded with `moves`
    # (older result files may predate that field).
    with_moves = [r for r in results if "moves" in r]
    if with_moves:
        even = [r["score"] for r in with_moves if r["moves"] % 2 == 0]
        odd = [r["score"] for r in with_moves if r["moves"] % 2 == 1]
        lines.append(
            f"parity breakdown ({len(with_moves)}/{len(results)} games have move counts):"
        )
        lines.append(
            _format_stats(_score_stats(even), "  even total moves (equal-size hands)")
        )
        lines.append(
            _format_stats(_score_stats(odd), "  odd total moves (P1 has an extra card)")
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate first-move advantage via self-play under a fixed time budget."
    )
    parser.add_argument("--bot", default="current", help="bot spec (default: current)")
    parser.add_argument("--games", type=int, default=400, help="total games to play")
    parser.add_argument("--budget", type=float, default=6.0, help="seconds per timed move")
    parser.add_argument("--seed", type=int, default=0, help="base seed (same seed = same deals)")
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes")
    parser.add_argument(
        "--out", default="data/first_move_advantage.jsonl", help="results file (append/resume)"
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_results(out_path)
    done_seeds = {r["seed"] for r in done}
    print(f"resuming: {len(done)} games already recorded in {out_path}")
    print(summarise(done))

    all_seeds = [f"{args.seed}:{i}" for i in range(args.games)]
    todo_seeds = [s for s in all_seeds if s not in done_seeds]
    if not todo_seeds:
        print("nothing left to do")
        return

    print(f"playing {len(todo_seeds)} more games (bot={args.bot}, budget={args.budget}s, jobs={args.jobs})")

    results = list(done)
    with out_path.open("a") as f:
        if args.jobs <= 1:
            for i, seed in enumerate(todo_seeds):
                r = play_one(seed, args.bot, args.budget)
                results.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if (i + 1) % 10 == 0 or i + 1 == len(todo_seeds):
                    print(f"[{i + 1}/{len(todo_seeds)}] {summarise(results)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {
                    pool.submit(play_one, seed, args.bot, args.budget): seed
                    for seed in todo_seeds
                }
                done_count = 0
                for future in as_completed(futures):
                    r = future.result()
                    results.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    done_count += 1
                    if done_count % 10 == 0 or done_count == len(todo_seeds):
                        print(f"[{done_count}/{len(todo_seeds)}] {summarise(results)}", flush=True)

    print()
    print("final:")
    print(summarise(results))


if __name__ == "__main__":
    main()
