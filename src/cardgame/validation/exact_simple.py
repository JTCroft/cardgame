"""PROTOTYPE: integer-aggregate exact solver for position labeling.

`Game.evaluate` threads full score-difference histograms (`ProbEval`, a
Counter subclass) through every node, but every decision it makes - move
comparison, fail-soft bounds, the final label - consumes only the
aggregates (multiplicity, observed mass, wins, draws, score sum). It also
builds a principal variation and a per-move detail dict at every interior
node that `label_position` never reads. This module is a copy of the
solver's control flow with those two costs removed:

* `_SAgg` replaces `ProbEval`: five integers, no Counter/dict allocation.
  Its `__eq__` is coarser (aggregate equality vs histogram equality),
  which can only change *which* of two value-identical moves is kept as
  best - never the returned aggregates.
* no "Deterministic optimal moves", no "Known info for other branches",
  no per-node result dicts: `evaluate_simple` returns the `_SAgg` alone.

The alpha-beta/fail-soft logic, move ordering (`sorted(all_moves,
key=len)`) and bounds arithmetic are ported unchanged, so labels must be
identical. Verify + benchmark against the labeled corpus (records are
ground truth produced by the current machinery):

    python -m cardgame.validation.exact_simple --count 100
    python -m cardgame.validation.exact_simple --count 40 --min-cards 13 --bench

Validated 2026-07-17: bit-exact labels on 553 corpus children across the
8-16 card bands (0 mismatches), x1.35-1.5 wall-time speedup over
Game.evaluate. Profile of the remainder (13-14 card record): ~42%
score_dp cache misses (multi-king DP, ~230us each - the known native-code
candidate), ~27% Game plumbing (move/legal_moves/resolve), ~23% aggregate
comparisons. A full-width memoised DAG solve (transposition reuse instead
of alpha-beta) was tried and REFUTED: x0.3-1.0 vs this solver, never
faster, 0.1-5.9M memo states at 13-16 cards - cutoffs beat transposition
reuse at these depths, consistent with the ai.py TT finding.
"""
import argparse
import json
import random
import time
from pathlib import Path

from ..game import _SCORE_DIFFERENCE_BOUND, Game, _cached_score

_DEFAULT_LABELS = Path(__file__).resolve().parents[3] / "data" / "oracle_labels.jsonl"
_BOUND = _SCORE_DIFFERENCE_BOUND


class _SEval:
    """Eval replacement: (w, d, s) at a multiplicity, compared by the
    (2w + d, w, s) key, cross-multiplied when multiplicities differ."""

    __slots__ = ("m", "w", "d", "s")

    def __init__(self, m, w, d, s):
        self.m = m
        self.w = w
        self.d = d
        self.s = s

    def _key(self, scale=1):
        w, d, s = scale * self.w, scale * self.d, scale * self.s
        return (2 * w + d, w, s)

    def _keys(self, other):
        if self.m == other.m:
            return self._key(), other._key()
        return self._key(other.m), other._key(self.m)

    def __lt__(self, other):
        a, b = self._keys(other)
        return a < b

    def __gt__(self, other):
        a, b = self._keys(other)
        return a > b

    def __eq__(self, other):
        a, b = self._keys(other)
        return a == b

    def __neg__(self):
        return _SEval(self.m, self.m - self.w - self.d, self.d, -self.s)


class _SAgg:
    """ProbEval replacement: multiplicity, observed mass and (w, d, s)
    aggregates of the observed mass. Bounds fill the unobserved remainder
    at +/-_BOUND exactly like ProbEval._bound_evals."""

    __slots__ = ("m", "obs", "w", "d", "s")

    def __init__(self, m, obs=0, w=0, d=0, s=0):
        self.m = m
        self.obs = obs
        self.w = w
        self.d = d
        self.s = s

    def bounds(self):
        remaining = self.m - self.obs
        filled = _BOUND * remaining
        return (
            _SEval(self.m, self.w, self.d, self.s - filled),
            _SEval(self.m, self.w + remaining, self.d, self.s + filled),
        )

    def update(self, other):
        self.obs += other.obs
        self.w += other.w
        self.d += other.d
        self.s += other.s

    def __lt__(self, other):
        return self.bounds()[1] < other.bounds()[0]

    def __gt__(self, other):
        return self.bounds()[0] > other.bounds()[1]

    def __eq__(self, other):
        return (
            self.m == other.m
            and self.obs == other.obs
            and self.w == other.w
            and self.d == other.d
            and self.s == other.s
        )

    def __neg__(self):
        return _SAgg(self.m, self.obs, self.obs - self.w - self.d, self.d, -self.s)


def _get_bounds(branch_multiplicity, move_score, alpha, beta):
    # Port of Game._get_bounds onto aggregates, arithmetic unchanged.
    w, d, s = move_score.w, move_score.d, move_score.s
    remaining_unevaled_after_branch = (
        move_score.m - move_score.obs - branch_multiplicity
    )
    filled = _BOUND * remaining_unevaled_after_branch
    subbeta = _SEval(
        branch_multiplicity, beta.w - w, beta.d - d, beta.s - (s - filled)
    )
    subalpha = _SEval(
        branch_multiplicity,
        alpha.w - (w + remaining_unevaled_after_branch),
        alpha.d - d,
        alpha.s - (s + filled),
    )
    branch_bound = _BOUND * branch_multiplicity
    lb = _SEval(branch_multiplicity, 0, 0, -branch_bound)
    ub = _SEval(branch_multiplicity, branch_multiplicity, 0, branch_bound)
    if subalpha < lb:
        subalpha = lb
    if subbeta > ub:
        subbeta = ub
    return subalpha, subbeta


def evaluate_simple(game, alpha=None, beta=None, _state=None):
    """Exact mover-perspective outcome aggregates of `game` - the
    `["Evaluation"]` of Game.evaluate as a _SAgg, same fail-soft
    alpha-beta control flow, none of the histogram/PV/detail overhead."""
    multiplicity = game.multiplicity
    if _state is None:
        _state = game._hand_state()
    if not game.legal_moves:
        diff = _cached_score(_state[0], _state[1]) - _cached_score(
            _state[2], _state[3]
        )
        return _SAgg(
            multiplicity,
            obs=multiplicity,
            w=multiplicity if diff > 0 else 0,
            d=multiplicity if diff == 0 else 0,
            s=diff * multiplicity,
        )
    if alpha is None:
        bound_sum = _BOUND * multiplicity
        alpha = _SEval(multiplicity, 0, 0, -bound_sum)
        beta = _SEval(multiplicity, multiplicity, 0, bound_sum)
    # ProbEval(multiplicity).lower_bound == everything filled at -_BOUND
    best_score = _SAgg(multiplicity, obs=multiplicity, s=-_BOUND * multiplicity)
    best_marker = (-1, -1)
    child_state = Game._child_hand_state

    for move in sorted(game.all_moves(), key=len):
        move_marker = move[0].marker
        if len(move) == 1:
            move_score = -evaluate_simple(
                move[0], -beta, -alpha,
                _state=child_state(_state, move[0].taken_card),
            )
            if (move_score, move_marker) > (best_score, best_marker):
                best_score = move_score
                best_marker = move_marker
            new_alpha = best_score.bounds()[0]
            if new_alpha > alpha:
                alpha = new_alpha
        else:
            branch_multiplicity = move[0].multiplicity
            move_score = _SAgg(multiplicity)
            for possibility in move:
                subalpha, subbeta = _get_bounds(
                    branch_multiplicity, move_score, alpha, beta
                )
                possibility_score = -evaluate_simple(
                    possibility, -subbeta, -subalpha,
                    _state=child_state(_state, possibility.taken_card),
                )
                move_score.update(possibility_score)
                if (move_score, move_marker) > (best_score, best_marker):
                    best_score = move_score
                    best_marker = move_marker
                move_lower, move_upper = move_score.bounds()
                if move_upper < alpha:
                    break
                if best_score is move_score:
                    new_alpha = move_lower
                else:
                    new_alpha = best_score.bounds()[0]
                if new_alpha > alpha:
                    alpha = new_alpha
                if alpha > beta and not (-alpha) > (-beta):
                    break
        if alpha > beta and not (-alpha) > (-beta):
            break
    return best_score


def label_position_simple(game):
    """oracle.label_position with evaluate_simple as the backend -
    identical record structure."""
    moves = []
    for resolutions in game.all_moves():
        entries = []
        for child in resolutions:
            ev = evaluate_simple(child)
            if ev.obs != ev.m:
                raise ValueError("Score is not fully evaluated!")
            entries.append(
                {"card": repr(child.taken_card), "w": ev.w, "d": ev.d,
                 "s": ev.s, "m": ev.m}
            )
        moves.append({"marker": list(resolutions[0].marker), "resolutions": entries})
    return {"save": game.save(alnum=True), "cards_left": 36 - len(game.moves),
            "facedown": len(game.board.facedown_cards), "moves": moves}


# --------------------------------------------------------------------------
# Verification / benchmark against the labeled corpus


def _stored_map(record):
    return {
        (tuple(move["marker"]), entry["card"]): (
            entry["w"], entry["d"], entry["s"], entry["m"]
        )
        for move in record["moves"]
        for entry in move["resolutions"]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=str(_DEFAULT_LABELS))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--min-cards", type=int, default=8)
    parser.add_argument("--max-cards", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", action="store_true",
                        help="also time the current Game.evaluate machinery")
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in open(args.labels)]
    records = [r for r in records
               if args.min_cards <= r["cards_left"] <= args.max_cards]
    random.Random(args.seed).shuffle(records)
    records = records[: args.count]
    print(f"{len(records)} records, {args.min_cards}-{args.max_cards} cards left")

    mismatches = children = 0
    t_simple = t_current = 0.0
    for i, record in enumerate(records):
        game = Game.load(record["save"])
        stored = _stored_map(record)

        t0 = time.perf_counter()
        relabeled = label_position_simple(game)
        t_simple += time.perf_counter() - t0

        got = _stored_map(relabeled)
        children += len(stored)
        if got != stored:
            mismatches += sum(
                1 for k in stored if got.get(k) != stored[k]
            ) + len(got.keys() - stored.keys())
            print(f"  MISMATCH record {i} ({record['cards_left']} cards)")

        if args.bench:
            t0 = time.perf_counter()
            for resolutions in game.all_moves():
                for child in resolutions:
                    child.evaluate()["Evaluation"]
            t_current += time.perf_counter() - t0

    print(f"children compared: {children}, mismatching entries: {mismatches}")
    print(f"simple : {t_simple:.2f}s total, {t_simple / len(records) * 1000:.0f}ms/record")
    if args.bench:
        print(f"current: {t_current:.2f}s total, "
              f"{t_current / len(records) * 1000:.0f}ms/record  "
              f"(speedup x{t_current / t_simple:.2f})")


if __name__ == "__main__":
    main()
