"""Duplicate-deal match harness for validating bot changes.

Plays paired matches between two bot versions with the two biggest noise
sources cancelled:

* **Duplicate deals** — every deal is played twice with seats swapped, on
  the same board *and* the same hidden-card placement (the arena deals its
  own boards so it knows which card sits under every face-down cell, and
  reveals identically in both games of a pair), so deal luck cancels
  within the pair.
* **Paired statistics** — alongside per-game win/draw/loss, results are
  aggregated per pair (the challenger's combined score margin over both
  seatings) with a two-sided sign test, so "is the new bot actually
  stronger?" gets a p-value instead of a vibe.

A bot is specified as:

* ``current``            — the working tree's `cardgame.ai` with default params
* ``file:/path/to/ai.py``— any standalone ai.py exposing `choose_move(game, time_budget)`
* anything else          — a git rev (``HEAD``, ``main~2``, a sha); that
  revision's ``src/cardgame/ai.py`` is imported in-process against the
  *current* rest of the package

Any spec may carry `SearchParams` overrides as a query string, e.g.
``current?exact_leaf_cards=6&tempo_bonus=0.0`` — the module must
expose `AlphaBetaBot`/`SearchParams` (values are parsed as Python literals).

Typical use after making a change (committed baseline vs working tree):

    python -m cardgame.validation.arena --old HEAD --new current --deals 50 --budget 0.3 --jobs 4

or from Python / a notebook::

    from cardgame.validation.arena import run_match, summarise
    print(summarise(run_match("HEAD", "current", deals=50, budget=0.3, jobs=4)))

Notes: matches at reduced budgets (0.2-0.5s/move) are the intended regime -
relative strength transfers well and a full-budget match takes hours. The
exact-endgame phase is shared by both bots (unless the change touches it)
and runs regardless of budget. With ``--jobs`` > 1 pairs run in parallel
processes; both bots of a game live in the same process, so CPU contention
slows them equally and stays fair, but keep jobs at or below physical cores.
Scripts that call `run_match` with jobs > 1 must guard the call with
``if __name__ == "__main__":`` (multiprocessing re-imports the main script
in each spawned worker on macOS/Windows).
"""

import argparse
import importlib.util
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from ..cards import Card
from ..game import Board, Game

__all__ = ("deal_with_assignment", "play_game", "run_match", "summarise", "main")


# --------------------------------------------------------------------------
# Bot loading


def _import_ai_module(path, name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registering under a cardgame.* name makes the module's relative
    # imports (.cards, .scoring) resolve against the current package.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _load_module(spec):
    """Resolve a bot spec (without any query string) to an ai module."""
    if spec == "current":
        from .. import ai

        return ai
    if spec.startswith("file:"):
        path = Path(spec[5:]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"bot spec {spec!r}: no such file")
        name = "cardgame._ai_file_" + re.sub(r"\W", "_", str(path))
        return _import_ai_module(path, name)
    # Anything else is a git rev.
    source = subprocess.check_output(
        ["git", "show", f"{spec}:src/cardgame/ai.py"], cwd=_repo_root()
    )
    safe = re.sub(r"\W", "_", spec)
    path = Path(tempfile.gettempdir()) / f"cardgame_ai_{safe}.py"
    path.write_bytes(source)
    return _import_ai_module(path, f"cardgame._ai_rev_{safe}")


def _load_choose_move(spec, budget):
    """Resolve a bot spec (with optional ?param=value overrides) to a
    `choose_move(game)` callable at the given time budget."""
    base, _, query = spec.partition("?")
    module = _load_module(base)
    if not query:
        return partial(module.choose_move, time_budget=budget)
    from ast import literal_eval

    overrides = {}
    for item in query.split("&"):
        key, _, value = item.partition("=")
        overrides[key] = literal_eval(value)
    params = module.SearchParams(**overrides)
    return module.AlphaBetaBot(time_budget=budget, params=params).choose_move


_BOT_CACHE = {}


def _bot(spec, budget):
    """Per-process cached move chooser for (spec, budget)."""
    key = (spec, budget)
    if key not in _BOT_CACHE:
        _BOT_CACHE[key] = _load_choose_move(spec, budget)
    return _BOT_CACHE[key]


# --------------------------------------------------------------------------
# Playing duplicate deals


def deal_with_assignment(rng):
    """Deal a board the way `Board.deal` does, but with a seeded RNG and
    keeping the {position: card} assignment of the face-down cells, so the
    same deal can be replayed with identical reveals."""
    deck = list(Card.deck())
    rng.shuffle(deck)
    indices = sorted(Board.facedown_indices)
    assignment = {(i // 6, i % 6): deck[i] for i in indices}
    facedown_cards = tuple(deck[i] for i in indices)
    placeholder = Card(facedown=True)
    for i in indices:
        deck[i] = placeholder
    board = Board([deck[i : i + 6] for i in range(0, 36, 6)], facedown_cards)
    return board, assignment


def play_game(board, assignment, choosers):
    """Play one full game; `choosers[0]` moves at even plies (Player 1).
    Face-down reveals follow `assignment` — the card actually dealt there.
    Returns the finished Game (`.score` is P1 minus P2)."""
    game = Game(board, tuple())
    while game.legal_moves:
        move = choosers[len(game.moves) % 2](game)
        children = game.move(*move)
        if len(children) == 1:
            game = children[0]
        else:
            card = assignment[move]
            game = next(g for g in children if g.taken_card == card)
    return game


def _play_pair(deal_seed, old_spec, new_spec, budget):
    """Play both orientations of one deal. Returns the new bot's margins
    (new as P1, new as P2) — positive means the new bot outscored the old."""
    old = _bot(old_spec, budget)
    new = _bot(new_spec, budget)
    rng = random.Random(deal_seed)
    board, assignment = deal_with_assignment(rng)
    as_p1 = play_game(board, assignment, (new, old)).score
    as_p2 = -play_game(board, assignment, (old, new)).score
    return as_p1, as_p2


# --------------------------------------------------------------------------
# Match driving and statistics


def run_match(old, new, deals=50, budget=0.3, seed=0, jobs=1, progress=None):
    """Play `deals` duplicate pairs of old vs new. Returns a list of
    (margin_new_as_p1, margin_new_as_p2) per pair. `progress`, if given,
    is called with (pairs_done, total, latest_pair)."""
    seeds = [f"{seed}:{i}" for i in range(deals)]
    pairs = []
    if jobs <= 1:
        for i, deal_seed in enumerate(seeds):
            pair = _play_pair(deal_seed, old, new, budget)
            pairs.append(pair)
            if progress:
                progress(i + 1, deals, pair)
        return pairs
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_play_pair, s, old, new, budget) for s in seeds]
        for i, future in enumerate(as_completed(futures)):
            pair = future.result()
            pairs.append(pair)
            if progress:
                progress(i + 1, deals, pair)
    return pairs


def _sign_test_p(wins, losses):
    """Two-sided exact sign test, ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def summarise(pairs, new_name="new", old_name="old"):
    """Human-readable summary of `run_match` output, from `new`'s side."""
    games = [margin for pair in pairs for margin in pair]
    w = sum(m > 0 for m in games)
    d = sum(m == 0 for m in games)
    l = len(games) - w - d
    frac = (w + d / 2) / len(games)
    if 0 < frac < 1:
        elo = -400 * math.log10(1 / frac - 1)
        elo_str = f"{elo:+.0f} Elo"
    else:
        elo_str = "+inf Elo" if frac == 1 else "-inf Elo"

    margins = [a + b for a, b in pairs]
    pw = sum(m > 0 for m in margins)
    pd = sum(m == 0 for m in margins)
    pl = len(margins) - pw - pd
    p_sign = _sign_test_p(pw, pl)
    mean = statistics.fmean(margins)
    if len(margins) > 1 and any(margins):
        se = statistics.stdev(margins) / math.sqrt(len(margins))
        t = mean / se if se else 0.0
        # Normal approximation is fine at match sample sizes.
        p_t = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        se, t, p_t = 0.0, 0.0, 1.0

    p = min(p_sign, p_t)
    verdict = (
        "no significant difference"
        if p >= 0.05
        else f"{new_name if mean > 0 else old_name} is significantly stronger"
        + (" (on points; win/loss sign test not yet significant)" if p_sign >= 0.05 else "")
    )
    return (
        f"{new_name} vs {old_name}: {len(pairs)} duplicate deals, {len(games)} games\n"
        f"  games: {w}W {d}D {l}L  ({100 * frac:.1f}% score, {elo_str})\n"
        f"  pairs: {new_name} better in {pw}, {old_name} better in {pl}, tied {pd}"
        f"  (sign test p={p_sign:.3f})\n"
        f"  paired margin: {mean:+.2f} ± {se:.2f} points/pair"
        f"  (t={t:.2f}, p={p_t:.3f})\n"
        f"  -> {verdict}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Duplicate-deal strength match between two bot versions."
    )
    parser.add_argument("--old", default="HEAD", help="baseline bot spec (default HEAD)")
    parser.add_argument("--new", default="current", help="challenger bot spec (default current)")
    parser.add_argument("--deals", type=int, default=50, help="duplicate pairs to play")
    parser.add_argument("--budget", type=float, default=0.3, help="seconds per timed move")
    parser.add_argument("--seed", type=int, default=0, help="deal seed (same seed = same deals)")
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes")
    args = parser.parse_args(argv)

    # Fail fast on a bad spec before burning match time.
    for spec in (args.old, args.new):
        _load_choose_move(spec, args.budget)

    def progress(done, total, pair):
        running = sum(sum(p) for p in pairs_so_far)
        print(
            f"pair {done:3d}/{total}: new margin {pair[0]:+d} / {pair[1]:+d}"
            f"   (cumulative {running:+d})",
            flush=True,
        )

    pairs_so_far = []

    def tracking_progress(done, total, pair):
        pairs_so_far.append(pair)
        progress(done, total, pair)

    def short(spec):
        return Path(spec[5:]).stem if spec.startswith("file:") else spec

    pairs = run_match(
        args.old, args.new, args.deals, args.budget, args.seed, args.jobs,
        progress=tracking_progress,
    )
    print()
    print(summarise(pairs, new_name=short(args.new), old_name=short(args.old)))


if __name__ == "__main__":
    main()
