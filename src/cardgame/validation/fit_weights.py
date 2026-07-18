"""Fit leaf-evaluation weights to oracle labels.

Rows: every non-terminal child position in the labeled dataset.
Features (mover perspective; the first four match _evaluate_leaf):
    0 diff, 1 potential_diff, 2 centrality_diff * decay (kings at 1.0),
    3 mobility, 4 sign(diff) * min(|diff|, 8), 5 min opponent replies,
    6 ceiling_diff * decay, 7 best accessible marginal,
    8 king_count_diff * decay (frees SearchParams.king_centrality),
    9 cards_left
Target: s/m + WIN_BONUS * (w - l)/m  - the same scalar the search uses at
terminals, i.e. what a perfect leaf would return.

Columns 4-7 are the 2026-07-17 candidate round (see
examples/leaf_eval_features.md): they improve held-out RMSE in-band but
were arena-refuted (joint refit) / arena-neutral (residual fit) and are
not deployed - kept here as the harness for future feature rounds.

If the bootstrap corpus (data/bootstrap_labels.jsonl, 19-30 cards left,
2s-search targets) is present, two more fits run against the combined
corpora: a freed king-centrality column, and the phase-slope fit behind
SearchParams.{potential,mobility,tempo}_slope / phase_pivot - hinge
slopes are fitted at each pivot on a grid, with a depth-parity nuisance
column absorbing the bootstrap labels' (-1)^depth tempo artifact.

Records are shuffled (seed 0) before the record-level half split, so the
holdout is in-distribution despite the corpus being written in band order.

CLI:
    python -m cardgame.validation.fit_weights [labels.jsonl [bootstrap.jsonl]]
(defaults to data/oracle_labels.jsonl and data/bootstrap_labels.jsonl;
the bootstrap sections are skipped if the file is missing)
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
_DATA = Path(__file__).resolve().parents[3] / "data"
_DEFAULT_LABELS = _DATA / "oracle_labels.jsonl"
_DEFAULT_BOOTSTRAP = _DATA / "bootstrap_labels.jsonl"


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
        decay * (me_k - opp_k) if decay > 0 else 0.0,
        36.0 - len(game.moves),
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


def rows_from_bootstrap(record):
    """(features, target, parity) rows from a bootstrap-labeled record.
    `parity` is +/-1 by the label's completed search depth - a nuisance
    regressor for the (-1)^depth tempo reflection in search values -
    and 0 on exact rows, so the two corpora mix in one design matrix."""
    game = Game.load(record["save"])
    for move in record["moves"]:
        marker = tuple(move["marker"])
        children = {repr(c.taken_card): c for c in game.move(*marker)}
        for entry in move["resolutions"]:
            if entry["depth"] < 0:
                continue  # terminal - never reaches the leaf evaluator
            parity = 1.0 if entry["depth"] % 2 == 0 else -1.0
            yield features(children[entry["card"]]), entry["v"], parity


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
    bootstrap = Path(args[1]) if len(args) > 1 else _DEFAULT_BOOTSTRAP
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

    # ---- freed king centrality (exact corpus; kings well-covered in-band) --
    # The cent column (x[2]) carries kings at 1.0; a free king column
    # (x[8], king count diff * decay) on top implies
    # king_centrality = 1 + w_king / w_cent.
    Xk = [[x[1], x[2], x[3], x[8], 1.0] for x in Xtr]
    wk = ols(Xk, y3)
    fitk = lambda x: x[0] + sum(
        a * b for a, b in zip(wk, [x[1], x[2], x[3], x[8], 1.0])
    )
    print(f"fit king centrality: cent={wk[1]:+.3f} king_extra={wk[3]:+.3f} "
          f"-> implied king_centrality={1 + wk[3] / wk[1]:.3f}  "
          f"test rmse {rmse(Xte, yte, fitk):.3f}")

    # ---- phase slopes (needs the bootstrap corpus for opening rows) -------
    if not bootstrap.exists():
        print(f"\n(bootstrap corpus {bootstrap} missing - phase-slope fit "
              f"skipped)")
        return
    brecords = [json.loads(line) for line in bootstrap.open()]
    random.Random(0).shuffle(brecords)
    bhalf = len(brecords) // 2
    Btr, Bte = [], []
    for recs, out in ((brecords[:bhalf], Btr), (brecords[bhalf:], Bte)):
        for rec in recs:
            out.extend(rows_from_bootstrap(rec))
    Ctr = [(x, y, 0.0) for x, y in zip(Xtr, ytr)] + Btr
    Cte = [(x, y, 0.0) for x, y in zip(Xte, yte)] + Bte
    print(f"\nbootstrap rows: train {len(Btr)}, test {len(Bte)} "
          f"(combined test {len(Cte)})")

    # Hinge slopes on the residual of the phase-flat champion base, one
    # OLS per candidate pivot: y - flat = a*h*pot + b*h*mob + c*h + e*par,
    # h = max(cards_left - pivot, 0). The parity column models the
    # bootstrap labels' artifact and is not part of the deployed eval.
    flat = lambda x: (x[0] + P.potential_weight * x[1]
                      + P.centrality_weight * x[2]
                      + P.mobility_weight * x[3] + P.tempo_bonus)
    print("phase-slope fit per pivot (deployed: pivot=12, "
          f"pot={P.potential_slope} mob={P.mobility_slope} "
          f"tempo={P.tempo_slope}):")
    for pivot in (8, 10, 12, 14, 16, 18):
        h = lambda x: max(x[9] - pivot, 0.0)
        Xp = [[h(x) * x[1], h(x) * x[3], h(x), par] for x, y, par in Ctr]
        yp = [y - flat(x) for x, y, par in Ctr]
        wp = ols(Xp, yp)
        pred = lambda r: (flat(r[0]) + wp[0] * h(r[0]) * r[0][1]
                          + wp[1] * h(r[0]) * r[0][3] + wp[2] * h(r[0])
                          + wp[3] * r[2])
        te = (sum((pred(r) - r[1]) ** 2 for r in Cte) / len(Cte)) ** 0.5
        print(f"  pivot {pivot:2d}: pot_slope={wp[0]:+.5f} "
              f"mob_slope={wp[1]:+.5f} tempo_slope={wp[2]:+.5f} "
              f"parity={wp[3]:+.3f}  combined test rmse {te:.3f}")


if __name__ == "__main__":
    main()
