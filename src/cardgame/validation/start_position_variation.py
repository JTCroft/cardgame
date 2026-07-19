"""Explore how much the marker's starting cell matters, and whether letting
the non-first-mover choose it (as the real rules do) could offset the
first-move tempo advantage measured in `first_move_advantage`.

The README describes the real rule: the player who does *not* move first
chooses which of the 4 central face-down cells the marker starts on.
`Game.starting_position` is hardcoded to (2, 2) in this implementation, so
that choice is currently unmodelled.

For each deal, the same bot (both seats) plays one full game from each of
the 4 candidate starts, board and facedown assignment held fixed. Because
the marker choice is made before any facedown card is revealed, it can only
be a function of the known face-up board — so taking, for each deal, the
start that is *worst for P1* (equivalently best for P2) among the 4 played
games is a fair ceiling estimate of what a P2 able to evaluate all 4
candidates could achieve. It is a ceiling, not a real move-time-budget
result: a bot implementing this rule for real would need to search all 4
roots (roughly 4x the per-move budget for that one decision) to find it,
rather than getting it for free by replaying the whole game 4 times.

Each record stores the played board itself (`Board.save(alnum=True)`), not
just the deal seed — self-contained, so replaying/backfilling a record
never depends on `deal_with_assignment`'s shuffle logic staying unchanged.
`assignment_from_board` recovers the facedown-cell assignment straight from
the saved board (canonical sorted-index order, the same convention
`Board.resolve` itself relies on), no RNG involved.

Usage:

    uv run python -m cardgame.validation.start_position_variation \
        --deals 30 --budget 6.0 --jobs 4 --out data/start_position_variation.jsonl

Resuming: rerun with the same --out; already-played seeds are skipped.
"""

import argparse
import json
import random
import statistics
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..game import Board, Game
from .arena import _bot, deal_with_assignment

__all__ = (
    "central_starts",
    "assignment_from_board",
    "play_one_deal",
    "load_results",
    "summarise",
    "main",
)


def central_starts():
    """The 4 central face-down cells (rows/cols 2-3), sorted."""
    return sorted(
        pos for pos in Board.facedown_positions if pos[0] in (2, 3) and pos[1] in (2, 3)
    )


def assignment_from_board(board):
    """Recover the {(row, col): card} facedown assignment from a Board,
    using the same fixed sorted-index convention `Board.resolve` and
    `arena.deal_with_assignment` both rely on. No RNG involved — works on
    any Board, including one reloaded from a saved string."""
    indices = sorted(Board.facedown_indices)
    return {(i // 6, i % 6): card for i, card in zip(indices, board.facedown_cards)}


def _game_class(start):
    return type(f"GameAt{start}", (Game,), {"starting_position": start})


def _play_game_at(board, assignment, choosers, game_cls):
    game = game_cls(board, tuple())
    while game.legal_moves:
        move = choosers[len(game.moves) % 2](game)
        children = game.move(*move)
        if len(children) == 1:
            game = children[0]
        else:
            card = assignment[move]
            game = next(g for g in children if g.taken_card == card)
    return game


def start_state(board, assignment, start):
    """Describe what a candidate start commits to, before any bot plays:
    the face-up cards reachable in one move from `start` (known to P2 at
    choice time), how many reachable cells are still face-down (unknown
    quantity of that choice), and the true card buried under `start` itself
    (never enters play if this start is chosen — only visible to us in
    hindsight, not to P2 at decision time). Cards are rendered alnum
    (repr) so the jsonl file stays plain ASCII."""
    reachable = Game.possible_moves[start]
    faceup = []
    facedown_count = 0
    for cell in reachable:
        card = board[cell[0]][cell[1]]
        if card.facedown:
            facedown_count += 1
        else:
            faceup.append(repr(card))
    return {
        "faceup": sorted(faceup),
        "facedown_count": facedown_count,
        "buried_card": repr(assignment[start]),
    }


def play_one_deal(deal_seed, bot_spec, budget, starts=None):
    """Play one deal once per starting position (same bot both seats).
    Returns {"seed": deal_seed, "board": board.save(alnum=True),
    "scores": {start_str: score}, "start_state": {start_str: start_state(...)}}
    — `board` makes the record self-contained (see module docstring);
    `start_state` is cheap (no bot involved) and lets later analysis
    correlate what a start exposes/buries with whether it favors P1 or P2."""
    bot = _bot(bot_spec, budget)
    rng = random.Random(deal_seed)
    board, assignment = deal_with_assignment(rng)
    starts = starts or central_starts()
    scores = {}
    states = {}
    for start in starts:
        GameCls = _game_class(start)
        game = _play_game_at(board, assignment, (bot, bot), GameCls)
        scores[str(start)] = game.score
        states[str(start)] = start_state(board, assignment, start)
    return {
        "seed": deal_seed,
        "board": board.save(alnum=True),
        "scores": scores,
        "start_state": states,
    }


def load_results(path):
    if not path.exists():
        return []
    results = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def backfill(results):
    """Bring pre-`board`-field records up to date: add the saved board
    (and start_state, recomputed in the alnum format) from the stored
    seed. This is the last place in the module allowed to call
    `deal_with_assignment` from a stored seed rather than a stored board —
    it's a one-time bridge for data written before `board` was recorded.
    Returns (results, num_backfilled)."""
    filled = 0
    for r in results:
        if "board" in r:
            continue
        rng = random.Random(r["seed"])
        board, assignment = deal_with_assignment(rng)
        r["board"] = board.save(alnum=True)
        r["start_state"] = {
            s: start_state(board, assignment, literal_eval(s)) for s in r["scores"]
        }
        filled += 1
    return results, filled


def summarise(results, default_start="(2, 2)"):
    n = len(results)
    if n == 0:
        return "no deals played yet"
    starts = sorted(results[0]["scores"].keys())
    lines = [f"{n} deals, starts {starts}"]

    lines.append("per-start (P1-P2 score, all games use that start):")
    for s in starts:
        scores = [r["scores"][s] for r in results]
        mean = statistics.fmean(scores)
        w = sum(sc > 0 for sc in scores)
        lines.append(f"  {s}: mean {mean:+.2f}  (P1 wins {w}/{n})")

    # P2-optimal: for each deal, P2 picks the start minimizing P1-P2 score.
    p2_best = [min(r["scores"].values()) for r in results]
    baseline = [r["scores"].get(default_start) for r in results]
    baseline = [b for b in baseline if b is not None]

    lines.append("")
    lines.append(f"baseline (fixed start {default_start}): mean {statistics.fmean(baseline):+.2f}"
                  f"  (P1 wins {sum(b > 0 for b in baseline)}/{len(baseline)})" if baseline else "")
    lines.append(f"P2-optimal choice (ceiling, min over 4 starts): mean {statistics.fmean(p2_best):+.2f}"
                  f"  (P1 wins {sum(b > 0 for b in p2_best)}/{n})")
    lines.append(f"ceiling reduction in P1 margin: {statistics.fmean(baseline) - statistics.fmean(p2_best):+.2f} pts/deal"
                  if baseline else "")

    return "\n".join(l for l in lines if l)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Vary the marker's starting cell across the 4 central face-down cells."
    )
    parser.add_argument("--bot", default="current", help="bot spec (default: current)")
    parser.add_argument("--deals", type=int, default=30, help="number of deals to try")
    parser.add_argument("--budget", type=float, default=6.0, help="seconds per timed move")
    parser.add_argument("--seed", type=int, default=0, help="base seed (same seed = same deals)")
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes")
    parser.add_argument(
        "--out", default="data/start_position_variation.jsonl", help="results file (append/resume)"
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_results(out_path)
    done, num_backfilled = backfill(done)
    if num_backfilled:
        print(f"backfilled {num_backfilled} pre-existing records with board/start_state")
        with out_path.open("w") as f:
            for r in done:
                f.write(json.dumps(r) + "\n")
    done_seeds = {r["seed"] for r in done}
    print(f"resuming: {len(done)} deals already recorded in {out_path}")
    if done:
        print(summarise(done))

    all_seeds = [f"{args.seed}:{i}" for i in range(args.deals)]
    todo_seeds = [s for s in all_seeds if s not in done_seeds]
    if not todo_seeds:
        print("nothing left to do")
        return

    print(f"playing {len(todo_seeds)} more deals x 4 starts (bot={args.bot}, budget={args.budget}s, jobs={args.jobs})")

    results = list(done)
    with out_path.open("a") as f:
        if args.jobs <= 1:
            for i, seed in enumerate(todo_seeds):
                r = play_one_deal(seed, args.bot, args.budget)
                results.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if (i + 1) % 5 == 0 or i + 1 == len(todo_seeds):
                    print(f"[{i + 1}/{len(todo_seeds)}]\n{summarise(results)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {
                    pool.submit(play_one_deal, seed, args.bot, args.budget): seed
                    for seed in todo_seeds
                }
                done_count = 0
                for future in as_completed(futures):
                    r = future.result()
                    results.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    done_count += 1
                    if done_count % 5 == 0 or done_count == len(todo_seeds):
                        print(f"[{done_count}/{len(todo_seeds)}]\n{summarise(results)}", flush=True)

    print()
    print("final:")
    print(summarise(results))


if __name__ == "__main__":
    main()
