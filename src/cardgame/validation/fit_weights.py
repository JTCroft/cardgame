"""Fit leaf-evaluation weights to oracle labels.

Rows: every non-terminal child position in the labeled dataset.
Features (mover perspective; the first four match _evaluate_leaf):
    diff, potential_diff, centrality_diff * decay, mobility,
    sign(diff) * min(|diff|, 8), min opponent replies,
    ceiling_diff * decay, best accessible marginal
Target: s/m + WIN_BONUS * (w - l)/m  - the same scalar the search uses at
terminals, i.e. what a perfect leaf would return.

The last four columns are the 2026-07-17 candidate round (see
examples/leaf_eval_features.md): they improve held-out RMSE in-band but
were arena-refuted (joint refit) / arena-neutral (residual fit) and are
not deployed - kept here as the harness for future feature rounds.

Records are shuffled (seed 0) before the record-level half split, so the
holdout is in-distribution despite the corpus being written in band order.

CLI:
    python -m cardgame.validation.fit_weights [labels.jsonl]
(defaults to the repo corpus at data/oracle_labels.jsonl)
"""
import json
import random
import sys
from pathlib import Path

from ..ai import (
    SearchParams,
    _REPLY_MASK,
    _TOKEN,
    _centrality_sum,
    _marginal,
    _remaining_cards,
    _score,
)
from ..game import Game

WIN_BONUS = 6.0
_ALL_BITS = 0xFFFFFFFF  # every non-king card's Hand.as_int bit
_DEFAULT_LABELS = Path(__file__).resolve().parents[3] / "data" / "oracle_labels.jsonl"


def hands(game):
    if len(game.moves) % 2 == 0:
        return game.p1.as_int, game.p2.as_int
    return game.p2.as_int, game.p1.as_int


def features(game):
    me, opp = hands(game)
    me_int, me_k = me
    opp_int, opp_k = opp
    my_base = _score(me_int, me_k)
    opp_base = _score(opp_int, opp_k)
    my_pot = opp_pot = 0
    king_mine = king_opp = None
    for token in (_TOKEN[c] for c in _remaining_cards(game)):
        if token < 0:
            if king_mine is None:
                king_mine = _score(me_int, me_k + 1) - my_base
                king_opp = _score(opp_int, opp_k + 1) - opp_base
            my_pot += king_mine
            opp_pot += king_opp
        else:
            my_pot += _score(me_int | token, me_k) - my_base
            opp_pot += _score(opp_int | token, opp_k) - opp_base
    decay = 1.0 - len(game.moves) / 36.0
    cent = decay * (_centrality_sum(me) - _centrality_sum(opp)) if decay > 0 else 0.0

    diff = my_base - opp_base
    sdiff8 = (1 if diff > 0 else -1 if diff < 0 else 0) * min(abs(diff), 8)

    ceiling = (_score(_ALL_BITS & ~opp_int, 4 - opp_k) - my_base) - (
        _score(_ALL_BITS & ~me_int, 4 - me_k) - opp_base
    )

    mask = 0
    for row, col in game.moves:
        mask |= 1 << (row * 6 + col)
    board = game.board
    facedown_positions = board.facedown_positions
    fd_marginal = None
    best_access = 0.0
    min_replies = 36
    for marker in game.legal_moves:
        replies = (_REPLY_MASK[marker] & ~mask).bit_count()
        if replies < min_replies:
            min_replies = replies
        if marker in facedown_positions:
            if fd_marginal is None:
                hidden = board.facedown_cards
                fd_marginal = sum(_marginal(me, c) for c in hidden) / len(hidden)
            access = fd_marginal
        else:
            access = float(_marginal(me, board[marker[0]][marker[1]]))
        if access > best_access:
            best_access = access

    return (
        float(diff),
        float(my_pot - opp_pot),
        cent,
        float(len(game.legal_moves)),
        float(sdiff8),
        float(min_replies),
        ceiling * decay if decay > 0 else 0.0,
        best_access,
    )


def rows_from(record):
    game = Game.load(record["save"])
    for move in record["moves"]:
        marker = tuple(move["marker"])
        children = {repr(c.taken_card): c for c in game.move(*marker)}
        for entry in move["resolutions"]:
            child = children[entry["card"]]
            if not child.legal_moves:
                continue  # terminals never reach the leaf evaluator
            m = entry["m"]
            losses = m - entry["w"] - entry["d"]
            target = entry["s"] / m + WIN_BONUS * (entry["w"] - losses) / m
            yield features(child), target


def solve(ATA, ATy):
    # Gaussian elimination, tiny system
    n = len(ATy)
    M = [row[:] + [ATy[i]] for i, row in enumerate(ATA)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(n):
            if r != col and M[r][col]:
                f = M[r][col] / M[col][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [M[i][-1] / M[i][i] for i in range(n)]


def ols(X, y):
    k = len(X[0])
    ATA = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]
    ATy = [sum(x[i] * t for x, t in zip(X, y)) for i in range(k)]
    return solve(ATA, ATy)


def rmse(X, y, predict):
    return (sum((predict(x) - t) ** 2 for x, t in zip(X, y)) / len(y)) ** 0.5


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    labels = Path(args[0]) if args else _DEFAULT_LABELS
    records = [json.loads(line) for line in labels.open()]
    random.Random(0).shuffle(records)
    half = len(records) // 2
    train, test = records[:half], records[half:]
    Xtr, ytr, Xte, yte = [], [], [], []
    for recs, X, y in ((train, Xtr, ytr), (test, Xte, yte)):
        for rec in recs:
            for f, t in rows_from(rec):
                X.append(f)
                y.append(t)
    print(f"rows: train {len(Xtr)}, test {len(Xte)}")

    mean_y = sum(yte) / len(yte)
    base_rmse = rmse(Xte, yte, lambda x: mean_y)
    print(f"test target sd (predict mean): {base_rmse:.3f}")

    # original hand-picked weights
    cur = lambda x: x[0] + 0.35 * x[1] + 0.6 * x[2] + 0.05 * x[3]
    print(f"current weights (1, .35, .60, .05):        test rmse {rmse(Xte, yte, cur):.3f}")

    # deployed champion (SearchParams defaults, incl. tempo intercept)
    P = SearchParams()
    champ = lambda x: (x[0] + P.potential_weight * x[1] + P.centrality_weight * x[2]
                       + P.mobility_weight * x[3] + P.tempo_bonus)
    print(f"champion weights ({P.potential_weight}/{P.centrality_weight}"
          f"/{P.mobility_weight}/+{P.tempo_bonus}):  test rmse {rmse(Xte, yte, champ):.3f}")

    # fit with diff coefficient fixed at 1 (keeps terminal point scale):
    # target - diff = w1*pot + w2*cent + w3*mob (+ w4 intercept)
    X3 = [[x[1], x[2], x[3], 1.0] for x in Xtr]
    y3 = [t - x[0] for x, t in zip(Xtr, ytr)]
    w = ols(X3, y3)
    fit1 = lambda x: x[0] + w[0] * x[1] + w[1] * x[2] + w[2] * x[3] + w[3]
    print(f"fit base (diff fixed): pot={w[0]:+.3f} cent={w[1]:+.3f} mob={w[2]:+.3f} "
          f"intercept={w[3]:+.3f}  test rmse {rmse(Xte, yte, fit1):.3f}")

    # extended fit: base + windiff/replies/ceiling/access (diff fixed at 1)
    X8 = [[x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1.0] for x in Xtr]
    w8 = ols(X8, y3)
    fit8 = lambda x: x[0] + sum(
        a * b for a, b in zip(w8, [x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1.0])
    )
    print(f"fit extended (diff fixed): pot={w8[0]:+.3f} cent={w8[1]:+.3f} "
          f"mob={w8[2]:+.3f} windiff={w8[3]:+.3f} replies={w8[4]:+.3f} "
          f"ceiling={w8[5]:+.3f} access={w8[6]:+.3f} intercept={w8[7]:+.3f}  "
          f"test rmse {rmse(Xte, yte, fit8):.3f}")

    # residual fit: base weights frozen at the deployed values, only the
    # candidate features (+ intercept) fitted. Arena-neutral over 300 pairs
    # (the free joint refit above was arena-refuted outright: the corpus
    # band, 8-18 cards left, does not constrain the opening eval).
    base = lambda x: (x[0] + P.potential_weight * x[1]
                      + P.centrality_weight * x[2] + P.mobility_weight * x[3])
    Xr = [[x[4], x[5], x[6], x[7], 1.0] for x in Xtr]
    yr = [t - base(x) for x, t in zip(Xtr, ytr)]
    wr = ols(Xr, yr)
    fitr = lambda x: base(x) + sum(
        a * b for a, b in zip(wr, [x[4], x[5], x[6], x[7], 1.0])
    )
    print(f"fit residual (base frozen): windiff={wr[0]:+.3f} replies={wr[1]:+.3f} "
          f"ceiling={wr[2]:+.3f} access={wr[3]:+.3f} intercept={wr[4]:+.3f}  "
          f"test rmse {rmse(Xte, yte, fitr):.3f}")


if __name__ == "__main__":
    main()
