"""Cross-check analyse_moves' new win/draw/loss/score fields against the
labeled oracle corpus.

data/oracle_labels.jsonl stores, per legal move, per face-down resolution,
the *child's own* (w, d, s, m) - i.e. from the perspective of whoever is to
move in that resolved position (the opponent of whoever is being analysed).
Negating each resolution back to the analysed position's perspective
(w' = m-w-d, d'=d, s'=-s - the same transform as Eval.__neg__) and summing
across resolutions (multiplicities already match, no cross-scaling needed)
gives an independent, ground-truth (w, d, s) for that move, with no fresh
tree search required - reusing the corpus rather than re-deriving it, the
same way exact_simple.py does.

    python -m cardgame.validation.analysis_check --count 100
"""
import argparse
import json
import random
from pathlib import Path

from ..analysis import analyse_moves
from ..game import Eval, Game

_DEFAULT_LABELS = Path(__file__).resolve().parents[3] / "data" / "oracle_labels.jsonl"


def _oracle_wds(record):
    """{marker: (w, d, s, m)} in the analysed position's own perspective,
    summed across a move's face-down resolutions."""
    out = {}
    for move in record["moves"]:
        marker = tuple(move["marker"])
        w = d = s = m = 0
        for res in move["resolutions"]:
            rw, rd, rs, rm = res["w"], res["d"], res["s"], res["m"]
            w += rm - rw - rd  # opponent's losses are this move's wins
            d += rd
            s += -rs
            m += rm
        out[marker] = (w, d, s, m)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=str(_DEFAULT_LABELS))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-cards", type=int, default=8)
    parser.add_argument("--max-cards", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in open(args.labels)]
    records = [r for r in records if args.min_cards <= r["cards_left"] <= args.max_cards]
    random.Random(args.seed).shuffle(records)
    records = records[: args.count]
    print(f"{len(records)} records, {args.min_cards}-{args.max_cards} cards left")

    positions = mismatches = moves_compared = best_mismatches = 0
    for i, record in enumerate(records):
        game = Game.load(record["save"])
        oracle = _oracle_wds(record)
        got = analyse_moves(game)

        if set(got) != set(oracle):
            print(f"  MARKER SET MISMATCH record {i}: {set(got) ^ set(oracle)}")
            mismatches += 1
            continue

        record_bad = False
        for marker, data in got.items():
            moves_compared += 1
            oracle_w, oracle_d, oracle_s, oracle_m = oracle[marker]
            # w/d/s/n no longer live as separate keys - eval's tuple contents
            # already *are* (w, d, s), and its .multiplicity already *is* n.
            got_w, got_d, got_s = data["eval"]
            got_n = data["eval"].multiplicity
            if (got_w, got_d, got_s, got_n) != (oracle_w, oracle_d, oracle_s, oracle_m):
                print(
                    f"  MISMATCH record {i} marker {marker}: "
                    f"got (w={got_w}, d={got_d}, s={got_s}, n={got_n}) "
                    f"oracle (w={oracle_w}, d={oracle_d}, s={oracle_s}, m={oracle_m})"
                )
                record_bad = True
        if record_bad:
            mismatches += 1

        # Also check the set of "best"-flagged moves agrees with the oracle-
        # derived eval-tuple winner(s) - analyse_moves now flags every move
        # tied for best, not just one, so this compares sets rather than a
        # single marker. Eval's own (multiplicity-normalising) comparison is
        # used on both sides to stay consistent with production semantics.
        oracle_evals = {m: Eval(mm, w, d, s) for m, (w, d, s, mm) in oracle.items()}
        best_key = None
        for m, ev in oracle_evals.items():
            candidate = (ev, m)
            if best_key is None or candidate > best_key:
                best_key = candidate
        oracle_best_eval = best_key[0]
        oracle_best_set = {m for m, ev in oracle_evals.items() if ev == oracle_best_eval}
        analysed_best_set = {m for m, d in got.items() if d["best"]}
        if oracle_best_set != analysed_best_set:
            print(
                f"  BEST-MOVE MISMATCH record {i}: got {analysed_best_set}, "
                f"oracle {oracle_best_set}"
            )
            best_mismatches += 1

        positions += 1

    print(
        f"positions checked: {positions}, moves compared: {moves_compared}, "
        f"mismatching positions: {mismatches}, best-move mismatches: {best_mismatches}"
    )


if __name__ == "__main__":
    main()
