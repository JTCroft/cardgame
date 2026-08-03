"""Deeper EXACT labels for weight fitting, via the native solver.

The oracle corpus (data/oracle_labels.jsonl) stops at ~18 cards left
because Game.evaluate is too slow deeper; above that only the approximate
bootstrap corpus exists. The native ID solver (cardgame.solve_id_native) is
alpha-beta pruned with an iterative-deepening gate and ~11x faster than
Game.evaluate (a further 2.2-2.8x over plain solve_native), so low-face-down
positions stay exactly solvable up to ~21 cards. This module labels single
positions with their exact value and appends them to data/deep_labels.jsonl.

A record is one position (not a parent with all move resolutions - one
solve per row, which lets us aim the sampler at specific depths):

    {"save": ..., "cards_left": N, "facedown": F,
     "sign_sum": w-l, "score_sum": s, "m": multiplicity}

The fit target is the same scalar a perfect leaf returns, and needs only
these three numbers (solve returns them directly):

    target = score_sum/m + WIN_BONUS * sign_sum/m        (mover perspective)

`sign_sum = 2w+d-m = w-l`, so this equals the oracle corpus's
`s/m + WIN_BONUS*(w-losses)/m` - deep and oracle rows mix in one fit.

Feasibility is bounded per solve: solve_id_native runs under a deadline and
returns None if it cannot finish in the budget, in which case the sampled
position is discarded and another drawn. (The native solve releases the GIL
but polls an Instant deadline internally, so it self-aborts without a signal.)

CLI:
    python -m cardgame.validation.deep generate --n 200 --budget 8
    python -m cardgame.validation.deep fit
"""
import argparse
import json
import multiprocessing as mp
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..game import Game
from ..solver_native import solve_id_native, NATIVE_AVAILABLE
from .fit_weights import (
    WIN_BONUS,
    features,
    ols,
    rmse,
    _DEFAULT_LABELS,
    rows_from,
)
from .oracle import sample_position
from ..ai import SearchParams

_DATA = Path(__file__).resolve().parents[3] / "data"
DEEP_LABELS = _DATA / "deep_labels.jsonl"

# Target depths (cards left) for generation, each with a face-down ceiling
# that keeps the exact solve inside the budget. Round-robined so the deep
# corpus itself is depth-balanced.
GEN_BANDS = [(15, 4), (16, 4), (17, 4), (18, 3), (19, 3), (20, 2), (21, 2)]

# Higher-face-down targets (cards -> feasible fd values), pushed toward the
# realistic mode (~5-7 at these depths) as far as the factorial solve cost
# allows (see experiments/measure_fd: 15-19 reach fd5-6, 20 reaches fd4-5;
# 21's realistic mode is out of exact reach). Sampled to EXACTLY these fd,
# bounded by a per-solve process timeout that discards the occasional
# straggler. The old GEN_BANDS corpus is fd<=4 (mode 3); this section is
# fd 4-6, far closer to real game states.
GEN_HIFD = {15: (5, 6), 16: (5, 6), 17: (5, 6), 18: (5,), 19: (5,), 20: (4, 5)}
_FORK = mp.get_context("fork")

# Fit/report bands (cards_left), 2 wide across the midgame. Rows are
# sampled uniformly across whichever of these are populated.
FIT_BANDS = [(12, 13), (14, 15), (16, 17), (18, 19), (20, 21)]


def label_value(game, deadline=None):
    """Exact label for `game`, or None if `deadline` (seconds) trips first."""
    r = solve_id_native(game, deadline=deadline)
    if r is None:
        return None
    return {
        "save": game.save(alnum=True),
        "cards_left": 36 - len(game.moves),
        "facedown": len(game.board.facedown_cards),
        "sign_sum": r["sign_sum"],
        "score_sum": r["score_sum"],
        "m": r["multiplicity"],
    }


def _gen_one(args):
    """Sample and exactly label one position for a target band, discarding
    any whose exact solve does not finish within `budget`."""
    seed, cards, max_fd, budget = args
    rng = random.Random(seed)
    while True:
        game = sample_position(rng, cards, cards, max_fd)
        if game is None:
            continue
        label = label_value(game, deadline=budget)
        if label is not None:
            return json.dumps(label)


def generate(n, budget, seed, out=DEEP_LABELS, progress=True, jobs=8):
    """Append `n` exact deep labels, round-robin across GEN_BANDS."""
    if not NATIVE_AVAILABLE:
        raise ImportError("deep labelling needs cardgame-native (uv pip install ./native)")
    out.parent.mkdir(exist_ok=True)
    tasks = [
        (f"{seed}:{i}", *GEN_BANDS[i % len(GEN_BANDS)], budget) for i in range(n)
    ]
    added = 0
    with out.open("a") as fh:
        if jobs <= 1:
            for task in tasks:
                fh.write(_gen_one(task) + "\n")
                fh.flush()
                added += 1
                if progress and added % 25 == 0:
                    print(f"  +{added}/{n}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = [pool.submit(_gen_one, t) for t in tasks]
                for fut in as_completed(futures):
                    fh.write(fut.result() + "\n")
                    fh.flush()
                    added += 1
                    if progress and added % 25 == 0:
                        print(f"  +{added}/{n}", flush=True)
    return added


def sample_to_fd(rng, cards, fd, tries=6000):
    """A random position with exactly `cards` left and `fd` still hidden
    (mover has a choice), or None if not hit within `tries`."""
    for _ in range(tries):
        game = sample_position(rng, cards, cards, fd)  # fd <= target
        if game is not None and len(game.board.facedown_cards) == fd:
            return game
    return None


def _hifd_worker(seed, cards, fds, q):
    rng = random.Random(seed)
    while True:
        fd = rng.choice(fds)
        game = sample_to_fd(rng, cards, fd)
        if game is not None:
            q.put(json.dumps(label_value(game)))
            return


def generate_hifd(n, timeout=35.0, jobs=8, out=DEEP_LABELS, seed=0, progress=True):
    """Append `n` exact HIGH-face-down deep labels (GEN_HIFD), round-robin
    over card counts, each solve bounded by a `timeout`-second killable
    process (the native solve releases the GIL, so only terminating the
    process can bound a factorial blow-up)."""
    if not NATIVE_AVAILABLE:
        raise ImportError("deep labelling needs cardgame-native (uv pip install ./native)")
    out.parent.mkdir(exist_ok=True)
    cards_cycle = list(GEN_HIFD.items())
    added = launched = 0
    running = []  # [proc, queue, start_time]

    def launch():
        nonlocal launched
        cards, fds = cards_cycle[launched % len(cards_cycle)]
        q = _FORK.Queue()
        p = _FORK.Process(target=_hifd_worker, args=(f"{seed}:{launched}", cards, fds, q))
        p.start()
        running.append([p, q, time.time()])
        launched += 1

    with out.open("a") as fh:
        for _ in range(min(jobs, n)):
            launch()
        while added < n:
            for entry in running[:]:
                p, q, start = entry
                if not q.empty():
                    fh.write(q.get() + "\n")
                    fh.flush()
                    added += 1
                    p.join()
                    running.remove(entry)
                    if progress and added % 25 == 0:
                        print(f"  +{added}/{n}", flush=True)
                    if added + len(running) < n:
                        launch()
                elif not p.is_alive():
                    p.join()
                    running.remove(entry)
                    if added + len(running) < n:
                        launch()
                elif time.time() - start > timeout:
                    p.terminate()
                    p.join()
                    running.remove(entry)
                    if added + len(running) < n:
                        launch()
            time.sleep(0.03)
        for p, q, _ in running:
            p.terminate()
            p.join()
    return added


def rows_from_deep(record):
    game = Game.load(record["save"])
    m = record["m"]
    target = record["score_sum"] / m + WIN_BONUS * record["sign_sum"] / m
    yield features(game), target, record["cards_left"]


def _deep_fd_rows(deep_path=DEEP_LABELS):
    """Deep rows carrying face-down count, for fd-stratified analysis."""
    if not deep_path.exists():
        return
    for line in deep_path.open():
        rec = json.loads(line)
        m = rec["m"]
        target = rec["score_sum"] / m + WIN_BONUS * rec["sign_sum"] / m
        yield features(Game.load(rec["save"])), target, rec["cards_left"], rec["facedown"]


def _oracle_rows(path):
    for line in path.open():
        rec = json.loads(line)
        for f, t in rows_from(rec):
            yield f, t, int(round(f[9]))  # feature 9 is cards_left


_ORACLE_CACHE = {}


def _cached_oracle_rows(path):
    """Oracle rows are expensive to rebuild (features over ~60k child
    positions); cache per path+mtime so repeated retrains only rebuild the
    growing deep corpus."""
    key = (str(path), path.stat().st_mtime)
    if key not in _ORACLE_CACHE:
        _ORACLE_CACHE.clear()
        _ORACLE_CACHE[key] = list(_oracle_rows(path))
    return _ORACLE_CACHE[key]


def load_rows(oracle_path=_DEFAULT_LABELS, deep_path=DEEP_LABELS):
    rows = []
    if oracle_path.exists():
        rows.extend(_cached_oracle_rows(oracle_path))
    if deep_path.exists():
        for line in deep_path.open():
            rows.extend(rows_from_deep(json.loads(line)))
    return rows


def _band_of(cards):
    for lo, hi in FIT_BANDS:
        if lo <= cards <= hi:
            return (lo, hi)
    return None


def _fit_base(rows):
    """OLS with the diff coefficient fixed at 1 (keeps terminal point scale):
    target - diff = w_pot*pot + w_cent*cent + w_mob*mob + intercept."""
    X = [[f[1], f[2], f[3], 1.0] for f, t in rows]
    y = [t - f[0] for f, t in rows]
    return ols(X, y)


def _champion_pred(f):
    P = SearchParams()
    return (f[0] + P.potential_weight * f[1] + P.centrality_weight * f[2]
            + P.mobility_weight * f[3] + P.tempo_bonus)


def _fit_pred(w, f):
    return f[0] + w[0] * f[1] + w[1] * f[2] + w[2] * f[3] + w[3]


def fit_report(oracle_path=_DEFAULT_LABELS, deep_path=DEEP_LABELS, seed=0):
    rows = load_rows(oracle_path, deep_path)
    by_band = defaultdict(list)
    for f, t, cl in rows:
        b = _band_of(cl)
        if b is not None:
            by_band[b].append((f, t))
    if not by_band:
        print("no rows in fit bands")
        return
    bands = sorted(by_band)
    P = SearchParams()

    # Headline deployable weights: equal counts per band (uniform in depth),
    # split 50/50 per band so the holdout is depth-balanced too.
    per_band = min(len(v) for v in by_band.values())
    rng = random.Random(seed)
    train, test = [], []
    for b in bands:
        v = by_band[b][:]
        rng.shuffle(v)
        pick = v[:per_band]
        h = len(pick) // 2
        train += pick[:h]
        test += pick[h:]
    w = _fit_base(train)
    champ_rmse = rmse([f for f, t in test], [t for f, t in test], _champion_pred)
    fit_rmse = rmse([f for f, t in test], [t for f, t in test], lambda f: _fit_pred(w, f))

    print(f"fit bands (cards_left), uniform at {per_band} rows/band:")
    for b in bands:
        note = "  <- deep (native)" if b[0] >= 18 else ""
        print(f"  {b[0]:2d}-{b[1]:<2d}: {len(by_band[b]):7d} available{note}")
    print(f"\nUNIFORM-DEPTH FIT (diff fixed at 1), {len(train)} train / {len(test)} test:")
    print(f"  champion  pot {P.potential_weight:+.3f} cent {P.centrality_weight:+.3f} "
          f"mob {P.mobility_weight:+.3f} tempo {P.tempo_bonus:+.3f}   test rmse {champ_rmse:.3f}")
    print(f"  fitted    pot {w[0]:+.3f} cent {w[1]:+.3f} mob {w[2]:+.3f} "
          f"tempo {w[3]:+.3f}   test rmse {fit_rmse:.3f}")

    # Per-band fits on ALL rows in each band (precise): how the weights drift
    # with depth is the clearest read on what they mean.
    print("\nper-band fits (all rows, shows weight drift with depth):")
    print(f"  {'band':>7} {'n':>7} {'pot':>7} {'cent':>7} {'mob':>7} {'tempo':>7} {'rmse':>6}")
    for b in bands:
        v = by_band[b]
        wb = _fit_base(v)
        rb = rmse([f for f, t in v], [t for f, t in v], lambda f: _fit_pred(wb, f))
        print(f"  {b[0]:2d}-{b[1]:<2d}  {len(v):7d} {wb[0]:+7.3f} {wb[1]:+7.3f} "
              f"{wb[2]:+7.3f} {wb[3]:+7.3f} {rb:6.3f}")

    _fd_strata_report()
    return w


def _fd_strata_report(deep_path=DEEP_LABELS, min_rows=120):
    """Within each deep band, fit low-face-down (<=3) vs high-face-down (>=4)
    rows separately - does raising fd change what the weights say? This is
    the check on whether the low-fd corpus bias distorts the fit."""
    rows = list(_deep_fd_rows(deep_path))
    if not rows:
        return
    groups = defaultdict(list)  # (band, 'lo'/'hi') -> rows
    for f, t, cl, fd in rows:
        b = _band_of(cl)
        if b is not None:
            groups[(b, "lo" if fd <= 3 else "hi")].append((f, t))
    print("\ndeep corpus face-down split (does higher fd shift the weights?):")
    print(f"  {'band':>7} {'fd':>3} {'n':>6} {'pot':>7} {'cent':>7} {'mob':>7} {'tempo':>7} {'rmse':>6}")
    for b in sorted({b for b, _ in groups}):
        for grp in ("lo", "hi"):
            v = groups.get((b, grp), [])
            if len(v) < min_rows:
                print(f"  {b[0]:2d}-{b[1]:<2d} {grp:>3} {len(v):6d}  (too few to fit)")
                continue
            wb = _fit_base(v)
            rb = rmse([f for f, t in v], [t for f, t in v], lambda f: _fit_pred(wb, f))
            print(f"  {b[0]:2d}-{b[1]:<2d} {grp:>3} {len(v):6d} {wb[0]:+7.3f} {wb[1]:+7.3f} "
                  f"{wb[2]:+7.3f} {wb[3]:+7.3f} {rb:6.3f}")


def _uniform_rows(seed=0):
    """All corpus rows, sampled to equal counts per fit band (uniform in
    depth) so the sparse deep bands drive the phase fit as much as the dense
    shallow ones."""
    by_band = defaultdict(list)
    for f, t, cl in load_rows():
        b = _band_of(cl)
        if b is not None:
            by_band[b].append((f, t))
    per_band = min(len(v) for v in by_band.values())
    rng = random.Random(seed)
    out = []
    for v in by_band.values():
        rng.shuffle(v)
        out += v[:per_band]
    return out, per_band


def fit_phase_slopes(seed=0, pivots=(10, 12, 14)):
    """Refit the deployed phase slopes (potential/mobility/tempo, hinged at a
    pivot) on the EXACT corpus with champion base weights frozen - the
    arena-safe structure (bit-identical at cards_left <= pivot). Replaces the
    approximate-bootstrap-fitted slopes with exact deep labels.

    Model: target - flat_champion = pot_slope*h*pot + mob_slope*h*mob
    + tempo_slope*h,  h = max(cards_left - pivot, 0)."""
    rows, per_band = _uniform_rows(seed)
    rng = random.Random(seed + 1)
    rng.shuffle(rows)
    half = len(rows) // 2
    train, test = rows[:half], rows[half:]
    P = SearchParams()
    flat = lambda f: (f[0] + P.potential_weight * f[1] + P.centrality_weight * f[2]
                      + P.mobility_weight * f[3] + P.tempo_bonus)

    def deployed(f):
        h = max(f[9] - P.phase_pivot, 0.0)
        return (flat(f) + P.potential_slope * h * f[1]
                + P.mobility_slope * h * f[3] + P.tempo_slope * h)

    Xte = [f for f, t in test]
    yte = [t for f, t in test]
    print(f"phase-slope refit on EXACT corpus, uniform {per_band}/band, "
          f"{len(train)} train / {len(test)} test:")
    print(f"  flat champion (no slopes):      test rmse {rmse(Xte, yte, flat):.3f}")
    print(f"  deployed slopes (pivot {P.phase_pivot:.0f}, pot_slope "
          f"{P.potential_slope}): test rmse {rmse(Xte, yte, deployed):.3f}")
    best = None
    for pivot in pivots:
        h = lambda f: max(f[9] - pivot, 0.0)
        X = [[h(f) * f[1], h(f) * f[3], h(f)] for f, t in train]
        y = [t - flat(f) for f, t in train]
        w = ols(X, y)
        pred = lambda f: flat(f) + w[0] * h(f) * f[1] + w[1] * h(f) * f[3] + w[2] * h(f)
        te = rmse(Xte, yte, pred)
        # effective potential weight at a few depths, to read the change
        eff = lambda cl: P.potential_weight + w[0] * max(cl - pivot, 0)
        print(f"  pivot {pivot:2d}: pot_slope {w[0]:+.5f} mob_slope {w[1]:+.5f} "
              f"tempo_slope {w[2]:+.4f}  test rmse {te:.3f}   "
              f"pot_eff@16/20 {eff(16):.3f}/{eff(20):.3f}")
        if best is None or te < best[1]:
            best = (pivot, te, w)
    pivot, te, w = best
    print(f"\nbest pivot {pivot}: potential_slope={w[0]:.5f} "
          f"mobility_slope={w[1]:.5f} tempo_slope={w[2]:.5f}")
    print(f"(deployed: potential_slope={P.potential_slope} "
          f"mobility_slope={P.mobility_slope} tempo_slope={P.tempo_slope})")
    return pivot, w


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="append n exact deep labels")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--budget", type=float, default=8.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--jobs", type=int, default=8)
    h = sub.add_parser("genhifd", help="append n high-face-down deep labels")
    h.add_argument("--n", type=int, default=200)
    h.add_argument("--timeout", type=float, default=35.0)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--jobs", type=int, default=8)
    sub.add_parser("fit", help="uniform-depth fit + report")
    args = parser.parse_args(argv)
    if args.cmd == "generate":
        generate(args.n, args.budget, args.seed, jobs=args.jobs)
    elif args.cmd == "genhifd":
        generate_hifd(args.n, timeout=args.timeout, jobs=args.jobs, seed=args.seed)
    else:
        fit_report()


if __name__ == "__main__":
    main()
