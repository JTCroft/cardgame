//! Native core for cardgame.solver.
//!
//! Same algorithm as the Python solver, replicated deterministically:
//! identical legal-cell order, root move order, order-preserving unknown
//! removal at chance nodes, and fail-soft Star1 window arithmetic - so
//! solve_native(game) equals solve(game) field-for-field, including the
//! upper-bound values reported for pruned rival moves.
//!
//! Node values are (sign_sum, score_sum) pairs under lexicographic
//! order - the same ordered abelian group the Python solver packs into
//! one int - so negamax alpha-beta and Star1 windows are sound under
//! any move ordering.

use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const SCORE_BOUND: i64 = 26;
const FACT: [i64; 13] = [
    1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600,
];

/// (sign_sum, score_sum): lexicographic order, componentwise addition.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct V(i64, i64);

impl V {
    #[inline]
    fn neg(self) -> V {
        V(-self.0, -self.1)
    }
    #[inline]
    fn add(self, o: V) -> V {
        V(self.0 + o.0, self.1 + o.1)
    }
    #[inline]
    fn sub(self, o: V) -> V {
        V(self.0 - o.0, self.1 - o.1)
    }
}

// Sign component far above any reachable value (max real sign is 12!),
// with enough i64 headroom for every window +/- slack derivation.
const INF: V = V(1 << 45, 0);

// ---------------------------------------------------------------------------
// Scoring DP (port of scoring.score_dp)
// ---------------------------------------------------------------------------

struct Trans {
    kings: u8,
    score: i32,
    delta: u32, // base-5 state increment for the chosen king ranks
}

fn run_score(mask: u32, kings: u32) -> i32 {
    let mut score = 0;
    let mut run = 0;
    let mut kings_in_run = 0;
    for r in 0..8 {
        if mask >> r & 1 == 1 {
            run += 1;
        } else if kings >> r & 1 == 1 {
            run += 1;
            kings_in_run += 1;
        } else {
            if run >= 3 {
                score += 2 * run - 3 - kings_in_run;
            }
            run = 0;
            kings_in_run = 0;
        }
    }
    if run >= 3 {
        score += 2 * run - 3 - kings_in_run;
    }
    score
}

/// transitions[mask * 5 + max_kings]: every way to place <= max_kings
/// kings on the empty ranks of this suit.
fn transitions() -> &'static Vec<Vec<Trans>> {
    static TRANS: OnceLock<Vec<Vec<Trans>>> = OnceLock::new();
    TRANS.get_or_init(|| {
        const POW5: [u32; 8] = [1, 5, 25, 125, 625, 3125, 15625, 78125];
        let mut out = Vec::with_capacity(256 * 5);
        for mask in 0u32..256 {
            let empty = !mask & 0xff;
            for max_kings in 0u32..5 {
                let mut list = Vec::new();
                let mut sub = empty;
                loop {
                    if sub.count_ones() <= max_kings {
                        let delta = (0..8)
                            .filter(|r| sub >> r & 1 == 1)
                            .map(|r| POW5[r as usize])
                            .sum();
                        list.push(Trans {
                            kings: sub.count_ones() as u8,
                            score: run_score(mask, sub),
                            delta,
                        });
                    }
                    if sub == 0 {
                        break;
                    }
                    sub = (sub - 1) & empty;
                }
                out.push(list);
            }
        }
        out
    })
}

fn score_dp(hand: u32, kings: u8) -> i32 {
    let trans = transitions();
    let rank_base: Vec<u32> = (0..8)
        .map(|r| ((hand >> r) & 0x0101_0101).count_ones())
        .collect();
    // state: base-5 count of kings placed at each rank; its digit sum is
    // the total placed, so (state -> best run score) is unambiguous.
    let mut dp: HashMap<u32, (i32, u8)> = HashMap::new();
    dp.insert(0, (0, 0));
    for s in 0..4 {
        let mask = (hand >> (8 * s)) & 0xff;
        let mut ndp: HashMap<u32, (i32, u8)> = HashMap::with_capacity(dp.len() * 2);
        for (&state, &(run_sc, used)) in dp.iter() {
            let left = kings - used;
            for t in &trans[(mask as usize) * 5 + left as usize] {
                let ns = state + t.delta;
                let nsc = run_sc + t.score;
                let e = ndp.entry(ns).or_insert((nsc, used + t.kings));
                if e.0 < nsc {
                    *e = (nsc, used + t.kings);
                }
            }
        }
        dp = ndp;
    }
    let mut best = 0;
    for (&state, &(run_sc, used)) in dp.iter() {
        if used != kings {
            continue;
        }
        let mut tmp = state;
        let mut set_sc = 0;
        for r in 0..8 {
            let b = rank_base[r] as i32;
            let k = (tmp % 5) as i32;
            if b + k >= 3 {
                set_sc += 2 * b + k - 3;
            }
            tmp /= 5;
        }
        if run_sc + set_sc > best {
            best = run_sc + set_sc;
        }
    }
    best
}

thread_local! {
    static SCORE_CACHE: RefCell<HashMap<u64, i32>> = RefCell::new(HashMap::new());
}

#[inline]
fn score(hand: u32, kings: u8) -> i64 {
    let key = (hand as u64) | ((kings as u64) << 32);
    SCORE_CACHE.with(|c| {
        if let Some(&v) = c.borrow().get(&key) {
            return v as i64;
        }
        let v = score_dp(hand, kings);
        c.borrow_mut().insert(key, v);
        v as i64
    })
}

// ---------------------------------------------------------------------------
// Search (port of solver._search / _chance, same deterministic order)
// ---------------------------------------------------------------------------

struct Ctx {
    cells: [i64; 36], // -1 facedown, 0 king, else the card's hand bit
    unknowns: [i64; 12],
}

#[inline]
fn legal(cell: usize, rows: u64, cols: u64, out: &mut [usize; 10]) -> usize {
    let row = cell / 6;
    let col = cell % 6;
    let row_bits = ((rows >> (row * 6)) | (1 << col)) & 63;
    let col_bits = ((cols >> (col * 6)) | (1 << row)) & 63;
    let mut n = 0;
    for j in 0..6 {
        if row_bits >> j & 1 == 0 {
            out[n] = row * 6 + j;
            n += 1;
        }
    }
    for i in 0..6 {
        if col_bits >> i & 1 == 0 {
            out[n] = i * 6 + col;
            n += 1;
        }
    }
    n
}

#[inline]
fn col_bit(cell: usize) -> u64 {
    1 << ((cell % 6) * 6 + cell / 6)
}

impl Ctx {
    #[inline]
    fn leaf(&self, mi: u32, mk: u8, oi: u32, ok: u8, nu: usize) -> V {
        let diff = score(mi, mk) - score(oi, ok);
        let m = FACT[nu];
        V(m * diff.signum(), m * diff)
    }

    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self, cell: usize, rows: u64, cols: u64, mi: u32, mk: u8, oi: u32,
        ok: u8, nu: usize, mut alpha: V, beta: V,
    ) -> V {
        let mut cells_buf = [0usize; 10];
        let n = legal(cell, rows, cols, &mut cells_buf);
        if n == 0 {
            return self.leaf(mi, mk, oi, ok, nu);
        }
        let mut best: Option<V> = None;
        for &target in &cells_buf[..n] {
            let nrows = rows | 1 << target;
            let ncols = cols | col_bit(target);
            let payload = self.cells[target];
            let v = if payload < 0 {
                self.chance(target, nrows, ncols, mi, mk, oi, ok, nu, alpha, beta)
            } else if payload > 0 {
                self.search(
                    target, nrows, ncols, oi, ok, mi | payload as u32, mk, nu,
                    beta.neg(), alpha.neg(),
                )
                .neg()
            } else {
                self.search(
                    target, nrows, ncols, oi, ok, mi, mk + 1, nu,
                    beta.neg(), alpha.neg(),
                )
                .neg()
            };
            if best.is_none() || v > best.unwrap() {
                best = Some(v);
                if v >= beta {
                    return v;
                }
                if v > alpha {
                    alpha = v;
                }
            }
        }
        best.unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn chance(
        &mut self, target: usize, nrows: u64, ncols: u64, mi: u32, mk: u8,
        oi: u32, ok: u8, nu: usize, alpha: V, beta: V,
    ) -> V {
        let mc = FACT[nu - 1];
        let child_bound = V(mc, SCORE_BOUND * mc);
        let mut done = V(0, 0);
        for i in 0..nu {
            let rem = (nu - 1 - i) as i64;
            let slack = V(child_bound.0 * rem, child_bound.1 * rem);
            let a_i = alpha.sub(done).sub(slack);
            let b_i = beta.sub(done).add(slack);
            let card = self.unknowns[i];
            // remove index i preserving order (matches Python's tuple
            // slicing), restore after the recursive call
            for j in i..nu - 1 {
                self.unknowns[j] = self.unknowns[j + 1];
            }
            let r = if card > 0 {
                self.search(
                    target, nrows, ncols, oi, ok, mi | card as u32, mk, nu - 1,
                    b_i.neg(), a_i.neg(),
                )
                .neg()
            } else {
                self.search(
                    target, nrows, ncols, oi, ok, mi, mk + 1, nu - 1,
                    b_i.neg(), a_i.neg(),
                )
                .neg()
            };
            for j in (i..nu - 1).rev() {
                self.unknowns[j + 1] = self.unknowns[j];
            }
            self.unknowns[i] = card;
            if r >= b_i {
                return done.add(r).sub(slack);
            }
            if r <= a_i {
                return done.add(r).add(slack);
            }
            done = done.add(r);
        }
        done
    }
}

// ---------------------------------------------------------------------------
// Move analysis (port of analysis._collect_aggregate / _collect_terminals)
//
// Unlike the solver these walks never prune: analyse_moves needs exact
// per-move win/draw/loss/score aggregates for *every* legal move, and the
// (2w+d, w, s) line-selection tie-break is not an ordered group, so
// alpha-beta on it would be unsound. Each root move is independent (no
// pruning between them), which is what lets a caller solve them one at a
// time and stream the results.
// ---------------------------------------------------------------------------

// Score diffs are in [-26, 26]; the histogram carries generous headroom
// and negates by index reversal (key k lives at index k + HIST_OFFSET).
const HIST_OFFSET: i64 = 32;
const HIST_SIZE: usize = 65;
type Hist = [i64; HIST_SIZE];

fn hist_rev(h: &Hist) -> Hist {
    let mut r = [0i64; HIST_SIZE];
    for i in 0..HIST_SIZE {
        r[HIST_SIZE - 1 - i] = h[i];
    }
    r
}

/// (multiplicity, wins, draws, score_sum, mover's-own-score_sum) - the same
/// aggregate analysis._Agg carries, all weighted by chance multiplicity.
#[derive(Clone, Copy)]
struct Agg {
    m: i64,
    w: i64,
    d: i64,
    s: i64,
    mover_sum: i64,
}

impl Agg {
    #[inline]
    fn zero() -> Agg {
        Agg { m: 0, w: 0, d: 0, s: 0, mover_sum: 0 }
    }
    #[inline]
    fn add(self, o: Agg) -> Agg {
        Agg {
            m: self.m + o.m,
            w: self.w + o.w,
            d: self.d + o.d,
            s: self.s + o.s,
            mover_sum: self.mover_sum + o.mover_sum,
        }
    }
    #[inline]
    fn neg(self) -> Agg {
        Agg {
            m: self.m,
            w: self.m - self.w - self.d,
            d: self.d,
            s: -self.s,
            mover_sum: self.mover_sum - self.s,
        }
    }
    #[inline]
    fn key(&self) -> (i64, i64, i64) {
        (2 * self.w + self.d, self.w, self.s)
    }
}

struct ACtx {
    cells: [i64; 36],
    unknowns: [i64; 12],
    deadline: Option<Instant>,
    nodes: u64,
    aborted: bool,
}

impl ACtx {
    #[inline]
    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes & 0x3ff == 0 {
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl {
                    self.aborted = true;
                }
            }
        }
    }

    #[inline]
    fn leaf(&self, mi: u32, mk: u8, oi: u32, ok: u8, nu: usize) -> Agg {
        let mover = score(mi, mk);
        let diff = mover - score(oi, ok);
        let m = FACT[nu];
        Agg {
            m,
            w: if diff > 0 { m } else { 0 },
            d: if diff == 0 { m } else { 0 },
            s: m * diff,
            mover_sum: m * mover,
        }
    }

    /// Exhaustive expectiminimax returning the optimal line's aggregate in
    /// this node's mover perspective. Selection is by (2w+d, w, s) then
    /// marker - the same tie-break analysis._Agg uses, made order-
    /// independent by including the (distinct) marker in the key.
    #[allow(clippy::too_many_arguments)]
    fn agg_search(
        &mut self, cell: usize, rows: u64, cols: u64, mi: u32, mk: u8, oi: u32,
        ok: u8, nu: usize,
    ) -> Agg {
        if self.aborted {
            return Agg::zero();
        }
        self.tick();
        let mut buf = [0usize; 10];
        let n = legal(cell, rows, cols, &mut buf);
        if n == 0 {
            return self.leaf(mi, mk, oi, ok, nu);
        }
        let mut best: Option<(Agg, (usize, usize))> = None;
        for &target in &buf[..n] {
            let nrows = rows | 1 << target;
            let ncols = cols | col_bit(target);
            let payload = self.cells[target];
            let combined = if payload < 0 {
                self.agg_chance(target, nrows, ncols, mi, mk, oi, ok, nu)
            } else if payload > 0 {
                self.agg_search(target, nrows, ncols, oi, ok, mi | payload as u32, mk, nu)
            } else {
                self.agg_search(target, nrows, ncols, oi, ok, mi, mk + 1, nu)
            };
            let candidate = combined.neg();
            let marker = (target / 6, target % 6);
            let better = match best {
                None => true,
                Some((ba, bm)) => (candidate.key(), marker) > (ba.key(), bm),
            };
            if better {
                best = Some((candidate, marker));
            }
        }
        best.unwrap().0
    }

    /// Sum of the resolutions' child-perspective aggregates (no negation -
    /// the caller negates the sum, exactly as _collect_aggregate does).
    #[allow(clippy::too_many_arguments)]
    fn agg_chance(
        &mut self, target: usize, nrows: u64, ncols: u64, mi: u32, mk: u8,
        oi: u32, ok: u8, nu: usize,
    ) -> Agg {
        let mut combined = Agg::zero();
        for i in 0..nu {
            let card = self.unknowns[i];
            for j in i..nu - 1 {
                self.unknowns[j] = self.unknowns[j + 1];
            }
            let r = if card > 0 {
                self.agg_search(target, nrows, ncols, oi, ok, mi | card as u32, mk, nu - 1)
            } else {
                self.agg_search(target, nrows, ncols, oi, ok, mi, mk + 1, nu - 1)
            };
            for j in (i..nu - 1).rev() {
                self.unknowns[j + 1] = self.unknowns[j];
            }
            self.unknowns[i] = card;
            combined = combined.add(r);
        }
        combined
    }

    /// Like agg_search, additionally carrying the optimal line's score-diff
    /// histogram (key = mover_score - other_score at the leaf, negated by
    /// reversal on the way up). Used for the winner's outcome heatmap only.
    #[allow(clippy::too_many_arguments)]
    fn dist_search(
        &mut self, cell: usize, rows: u64, cols: u64, mi: u32, mk: u8, oi: u32,
        ok: u8, nu: usize,
    ) -> (Agg, Hist) {
        if self.aborted {
            return (Agg::zero(), [0; HIST_SIZE]);
        }
        self.tick();
        let mut buf = [0usize; 10];
        let n = legal(cell, rows, cols, &mut buf);
        if n == 0 {
            let agg = self.leaf(mi, mk, oi, ok, nu);
            let mut h = [0i64; HIST_SIZE];
            let diff = score(mi, mk) - score(oi, ok);
            h[(diff + HIST_OFFSET) as usize] = agg.m;
            return (agg, h);
        }
        let mut best: Option<(Agg, Hist, (usize, usize))> = None;
        for &target in &buf[..n] {
            let nrows = rows | 1 << target;
            let ncols = cols | col_bit(target);
            let payload = self.cells[target];
            let (cagg, chist) = if payload < 0 {
                self.dist_chance(target, nrows, ncols, mi, mk, oi, ok, nu)
            } else if payload > 0 {
                self.dist_search(target, nrows, ncols, oi, ok, mi | payload as u32, mk, nu)
            } else {
                self.dist_search(target, nrows, ncols, oi, ok, mi, mk + 1, nu)
            };
            let candidate = cagg.neg();
            let marker = (target / 6, target % 6);
            let better = match best {
                None => true,
                Some((ba, _, bm)) => (candidate.key(), marker) > (ba.key(), bm),
            };
            if better {
                best = Some((candidate, hist_rev(&chist), marker));
            }
        }
        let b = best.unwrap();
        (b.0, b.1)
    }

    #[allow(clippy::too_many_arguments)]
    fn dist_chance(
        &mut self, target: usize, nrows: u64, ncols: u64, mi: u32, mk: u8,
        oi: u32, ok: u8, nu: usize,
    ) -> (Agg, Hist) {
        let mut agg = Agg::zero();
        let mut hist = [0i64; HIST_SIZE];
        for i in 0..nu {
            let card = self.unknowns[i];
            for j in i..nu - 1 {
                self.unknowns[j] = self.unknowns[j + 1];
            }
            let (r_agg, r_hist) = if card > 0 {
                self.dist_search(target, nrows, ncols, oi, ok, mi | card as u32, mk, nu - 1)
            } else {
                self.dist_search(target, nrows, ncols, oi, ok, mi, mk + 1, nu - 1)
            };
            for j in (i..nu - 1).rev() {
                self.unknowns[j + 1] = self.unknowns[j];
            }
            self.unknowns[i] = card;
            agg = agg.add(r_agg);
            for k in 0..HIST_SIZE {
                hist[k] += r_hist[k];
            }
        }
        (agg, hist)
    }
}

fn new_actx(cells: Vec<i64>, unknowns: &[i64], deadline_secs: Option<f64>) -> PyResult<ACtx> {
    if cells.len() != 36 || unknowns.len() > 12 {
        return Err(pyo3::exceptions::PyValueError::new_err("bad state shape"));
    }
    let mut u = [0i64; 12];
    u[..unknowns.len()].copy_from_slice(unknowns);
    Ok(ACtx {
        cells: cells.try_into().unwrap(),
        unknowns: u,
        deadline: deadline_secs.map(|s| Instant::now() + Duration::from_secs_f64(s)),
        nodes: 0,
        aborted: false,
    })
}

/// One root move's exact aggregate, in the analysed (root) player's
/// perspective: (m, w, d, s, mover_sum), or None if the deadline tripped.
/// `target` is the cell being played; the caller enumerates legal cells.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn analyse_move(
    py: Python<'_>, cells: Vec<i64>, target: usize, rows: u64, cols: u64, mi: u32,
    mk: u8, oi: u32, ok: u8, unknowns: Vec<i64>, deadline_secs: Option<f64>,
) -> PyResult<Option<(i64, i64, i64, i64, i64)>> {
    let mut ctx = new_actx(cells, &unknowns, deadline_secs)?;
    let nu = unknowns.len();
    py.detach(move || {
        let nrows = rows | 1 << target;
        let ncols = cols | col_bit(target);
        let payload = ctx.cells[target];
        let combined = if payload < 0 {
            ctx.agg_chance(target, nrows, ncols, mi, mk, oi, ok, nu)
        } else if payload > 0 {
            ctx.agg_search(target, nrows, ncols, oi, ok, mi | payload as u32, mk, nu)
        } else {
            ctx.agg_search(target, nrows, ncols, oi, ok, mi, mk + 1, nu)
        };
        if ctx.aborted {
            return Ok(None);
        }
        let a = combined.neg();
        Ok(Some((a.m, a.w, a.d, a.s, a.mover_sum)))
    })
}

/// The winner's outcome distribution: (score_diff, weight) pairs in the
/// analysed player's own perspective (self - opponent), matching
/// analyse_moves' `distribution`. None if the deadline tripped.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn distribution(
    py: Python<'_>, cells: Vec<i64>, target: usize, rows: u64, cols: u64, mi: u32,
    mk: u8, oi: u32, ok: u8, unknowns: Vec<i64>, deadline_secs: Option<f64>,
) -> PyResult<Option<Vec<(i64, i64)>>> {
    let mut ctx = new_actx(cells, &unknowns, deadline_secs)?;
    let nu = unknowns.len();
    py.detach(move || {
        let nrows = rows | 1 << target;
        let ncols = cols | col_bit(target);
        let payload = ctx.cells[target];
        let (_, hist) = if payload < 0 {
            ctx.dist_chance(target, nrows, ncols, mi, mk, oi, ok, nu)
        } else if payload > 0 {
            ctx.dist_search(target, nrows, ncols, oi, ok, mi | payload as u32, mk, nu)
        } else {
            ctx.dist_search(target, nrows, ncols, oi, ok, mi, mk + 1, nu)
        };
        if ctx.aborted {
            return Ok(None);
        }
        // Top level applies no flip; distribution key is other - mover =
        // -(mover - other), i.e. the child histogram reversed.
        let final_hist = hist_rev(&hist);
        let out: Vec<(i64, i64)> = (0..HIST_SIZE)
            .filter(|&i| final_hist[i] != 0)
            .map(|i| (i as i64 - HIST_OFFSET, final_hist[i]))
            .collect();
        Ok(Some(out))
    })
}

// ---------------------------------------------------------------------------
// Root (port of solver.solve, ordered=False path)
// ---------------------------------------------------------------------------

type MoveRow = ((usize, usize), (i64, i64), bool);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_root(
    py: Python<'_>, cells: Vec<i64>, cell: usize, rows: u64, cols: u64, mi: u32,
    mk: u8, oi: u32, ok: u8, unknowns: Vec<i64>,
) -> PyResult<(Option<(usize, usize)>, (i64, i64), Vec<MoveRow>)> {
    if cells.len() != 36 || unknowns.len() > 12 {
        return Err(pyo3::exceptions::PyValueError::new_err("bad state shape"));
    }
    let mut ctx = Ctx {
        cells: cells.try_into().unwrap(),
        unknowns: [0; 12],
    };
    let nu = unknowns.len();
    ctx.unknowns[..nu].copy_from_slice(&unknowns);

    py.detach(move || {
        let mut buf = [0usize; 10];
        let n = legal(cell, rows, cols, &mut buf);
        if n == 0 {
            let v = ctx.leaf(mi, mk, oi, ok, nu);
            return Ok((None, (v.0, v.1), Vec::new()));
        }
        // face-up moves first, stable within each group - same order as
        // Python's sorted(legal, key=lambda t: cells[t] is None)
        let mut ordered: Vec<usize> = Vec::with_capacity(n);
        ordered.extend(buf[..n].iter().filter(|&&t| ctx.cells[t] >= 0));
        ordered.extend(buf[..n].iter().filter(|&&t| ctx.cells[t] < 0));

        let mut best: Option<(V, (usize, usize))> = None;
        let mut moves: Vec<MoveRow> = Vec::with_capacity(n);
        for target in ordered {
            let alpha = match best {
                None => INF.neg(),
                Some((bv, _)) => bv.sub(V(0, 1)),
            };
            let nrows = rows | 1 << target;
            let ncols = cols | col_bit(target);
            let payload = ctx.cells[target];
            let v = if payload < 0 {
                ctx.chance(target, nrows, ncols, mi, mk, oi, ok, nu, alpha, INF)
            } else if payload > 0 {
                ctx.search(
                    target, nrows, ncols, oi, ok, mi | payload as u32, mk, nu,
                    INF.neg(), alpha.neg(),
                )
                .neg()
            } else {
                ctx.search(
                    target, nrows, ncols, oi, ok, mi, mk + 1, nu,
                    INF.neg(), alpha.neg(),
                )
                .neg()
            };
            let marker = (target / 6, target % 6);
            moves.push((marker, (v.0, v.1), best.is_none() || v > alpha));
            if best.is_none() || (v, marker) > best.unwrap() {
                best = Some((v, marker));
            }
        }
        let (bv, bm) = best.unwrap();
        Ok((Some(bm), (bv.0, bv.1), moves))
    })
}

// ---------------------------------------------------------------------------
// Heuristic midgame search (port of ai.AlphaBetaBot._search_root and friends)
//
// Same iterative-deepening expectiminimax as the Python bot: negamax
// alpha-beta at decision nodes, Star1 over *sampled* face-down resolutions
// at chance nodes, a fitted heuristic leaf, fitted move ordering and a
// transposition table. Node values are f64 in the bot's value scale
// (score diff + win-bonus expectation), reproducing the Python arithmetic
// operation-for-operation so the same evaluation is searched deeper.
// ---------------------------------------------------------------------------

// Move-ordering weights (ai._ORDER_*), fitted constants - speed only.
const ORDER_ME: f64 = 0.5883;
const ORDER_OPP: f64 = 0.1114;
const ORDER_CENT: f64 = 2.8549;
const ORDER_KING: f64 = 0.2874;
const ORDER_FD: f64 = -0.1927;
const ORDER_REPLIES: f64 = -0.4922;
const KING_CENTRALITY: f64 = 1.0;

/// ai._CENTRALITY: runs of length 3..8 covering each rank, scaled 0..1.
fn centrality_table() -> &'static [f64; 8] {
    static T: OnceLock<[f64; 8]> = OnceLock::new();
    T.get_or_init(|| {
        let mut counts = [0i32; 8];
        for rank in 1..=8i32 {
            let mut c = 0;
            for length in 3..=8i32 {
                for start in 1..(10 - length) {
                    if start <= rank && rank <= start + length - 1 {
                        c += 1;
                    }
                }
            }
            counts[(rank - 1) as usize] = c;
        }
        let lo = *counts.iter().min().unwrap();
        let hi = *counts.iter().max().unwrap();
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = (counts[i] - lo) as f64 / (hi - lo) as f64;
        }
        out
    })
}

/// ai._REPLY_MASK: for each cell, its row+column cells (self excluded),
/// bit t = cell r*6+c - the same indexing as the taken-cell `rows` mask.
fn reply_masks() -> &'static [u64; 36] {
    static T: OnceLock<[u64; 36]> = OnceLock::new();
    T.get_or_init(|| {
        let mut out = [0u64; 36];
        for i in 0..6usize {
            for j in 0..6usize {
                let cell = i * 6 + j;
                let mut m = 0u64;
                for j2 in 0..6 {
                    m |= 1 << (i * 6 + j2);
                }
                for i2 in 0..6 {
                    m |= 1 << (i2 * 6 + j);
                }
                out[cell] = m & !(1u64 << cell);
            }
        }
        out
    })
}

/// A hidden-card code (rank*4 + suit, K=rank 9) -> (is_king, hand bit).
#[inline]
fn code_bit(code: i64) -> (bool, u32) {
    let rank = code / 4;
    let suit = (code % 4) as u32;
    if rank == 9 {
        (true, 0)
    } else {
        (false, 1u32 << (suit * 8 + (rank as u32 - 1)))
    }
}

/// ai._marginal: score gained by adding this card to (hi, hk). No guards -
/// an already-held bit ORs to a no-op (marginal 0), matching Python.
#[inline]
fn h_marginal(hi: u32, hk: u8, king: bool, bit: u32) -> f64 {
    if king {
        (score(hi, hk + 1) - score(hi, hk)) as f64
    } else {
        (score(hi | bit, hk) - score(hi, hk)) as f64
    }
}

/// Python round(): round-half-to-even. Inputs here are exact k*(nu-1)/(cap-1).
#[inline]
fn py_round(x: f64) -> usize {
    let f = x.floor();
    let diff = x - f;
    let fi = f as i64;
    let r = if diff < 0.5 {
        fi
    } else if diff > 0.5 {
        fi + 1
    } else if fi % 2 == 0 {
        fi
    } else {
        fi + 1
    };
    r as usize
}

/// ai._sample_resolutions indices into the sorted hidden multiset.
fn sample_indices(nu: usize, cap: usize) -> Vec<usize> {
    if nu <= cap {
        (0..nu).collect()
    } else {
        let last = (nu - 1) as f64;
        (0..cap)
            .map(|k| py_round(k as f64 * last / (cap - 1) as f64))
            .collect()
    }
}

#[derive(Clone, Copy)]
struct HParams {
    value_bound: f64,
    win_bonus: f64,
    potential_weight: f64,
    potential_slope: f64,
    centrality_weight: f64,
    centrality_base: f64,
    king_centrality: f64,
    mobility_weight: f64,
    mobility_slope: f64,
    tempo_bonus: f64,
    tempo_slope: f64,
    phase_pivot: f64,
    resolution_cap: usize,
    deepen_fraction: f64,
}

// (rows, marker cell, mover int/kings, other int/kings) -> the subgame.
type TTKey = (u64, u8, u32, u8, u32, u8);
// (depth, flag: 0 exact / 1 lower / -1 upper, value, best target or 36=none)
type TTEntry = (i64, i8, f64, usize);

struct HCtx {
    cells: [i64; 36],
    unknowns: [i64; 12], // sorted hidden-card codes; chance shifts+restores
    p: HParams,
    deadline: Option<Instant>,
    nodes: u64,
    aborted: bool,
    tt: HashMap<TTKey, TTEntry>,
}

impl HCtx {
    #[inline]
    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes & 0x3ff == 0 {
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl {
                    self.aborted = true;
                }
            }
        }
    }

    /// ai._terminal_value.
    #[inline]
    fn terminal(&self, mi: u32, mk: u8, oi: u32, ok: u8) -> f64 {
        let diff = (score(mi, mk) - score(oi, ok)) as f64;
        let b = self.p.value_bound;
        if diff > 0.0 {
            (diff + self.p.win_bonus).min(b)
        } else if diff < 0.0 {
            (diff - self.p.win_bonus).max(-b)
        } else {
            0.0
        }
    }

    /// Sum of a hand's marginals over every card still on the board:
    /// untaken face-up cells + the unresolved hidden multiset. Integer, so
    /// order-independent (matches ai._evaluate_leaf's potential exactly).
    fn potential(&self, hi: u32, hk: u8, rows: u64, nu: usize) -> i64 {
        let base = score(hi, hk);
        let mut total = 0i64;
        for t in 0..36 {
            if (rows >> t) & 1 == 1 {
                continue;
            }
            let payload = self.cells[t];
            if payload < 0 {
                continue; // face-down: counted via unknowns
            } else if payload == 0 {
                if hk < 4 {
                    total += score(hi, hk + 1) - base;
                }
            } else {
                let bit = payload as u32;
                if hi & bit == 0 {
                    total += score(hi | bit, hk) - base;
                }
            }
        }
        for i in 0..nu {
            let (king, bit) = code_bit(self.unknowns[i]);
            if king {
                if hk < 4 {
                    total += score(hi, hk + 1) - base;
                }
            } else if hi & bit == 0 {
                total += score(hi | bit, hk) - base;
            }
        }
        total
    }

    /// ai._centrality_sum.
    #[inline]
    fn centrality(&self, hi: u32, hk: u8) -> f64 {
        let mut total = hk as f64 * self.p.king_centrality;
        let cent = centrality_table();
        for rank in 0..8u32 {
            total += cent[rank as usize] * ((hi >> rank) & 0x0101_0101).count_ones() as f64;
        }
        total
    }

    /// ai._evaluate_leaf, operation-for-operation.
    fn eval_leaf(&self, mi: u32, mk: u8, oi: u32, ok: u8, rows: u64, nu: usize, n_legal: usize) -> f64 {
        let p = &self.p;
        let my_base = score(mi, mk);
        let opp_base = score(oi, ok);
        let my_pot = self.potential(mi, mk, rows, nu);
        let opp_pot = self.potential(oi, ok, rows, nu);
        let cards_left = 36.0 - rows.count_ones() as f64;
        let phase = if cards_left > p.phase_pivot {
            cards_left - p.phase_pivot
        } else {
            0.0
        };
        let mut value = (my_base - opp_base) as f64 + p.tempo_bonus;
        value += p.tempo_slope * phase;
        value += (p.potential_weight + p.potential_slope * phase) * (my_pot - opp_pot) as f64;
        let decay = cards_left / 36.0;
        if decay > 0.0 {
            value += (p.centrality_base + p.centrality_weight * decay)
                * (self.centrality(mi, mk) - self.centrality(oi, ok));
        }
        value += (p.mobility_weight + p.mobility_slope * phase) * n_legal as f64;
        value.min(p.value_bound).max(-p.value_bound)
    }

    /// ai._ordered_markers: (target, is_facedown) best-first for the mover.
    fn ordered(
        &self, cell: usize, rows: u64, cols: u64, mi: u32, mk: u8, oi: u32, ok: u8, nu: usize,
    ) -> Vec<(usize, bool)> {
        let mut buf = [0usize; 10];
        let n = legal(cell, rows, cols, &mut buf);
        let cent = centrality_table();
        let masks = reply_masks();
        let mut facedown_key: Option<f64> = None;
        let mut items: Vec<(f64, usize, bool)> = Vec::with_capacity(n);
        for &target in &buf[..n] {
            let replies = (masks[target] & !rows).count_ones() as f64;
            let payload = self.cells[target];
            let (key, fd) = if payload < 0 {
                let fk = *facedown_key.get_or_insert_with(|| {
                    let mut total = 0.0;
                    for i in 0..nu {
                        let (king, bit) = code_bit(self.unknowns[i]);
                        total += ORDER_ME * h_marginal(mi, mk, king, bit)
                            + ORDER_OPP * h_marginal(oi, ok, king, bit);
                        if king {
                            total += ORDER_CENT * KING_CENTRALITY + ORDER_KING;
                        } else {
                            total += ORDER_CENT * cent[(bit.trailing_zeros() % 8) as usize];
                        }
                    }
                    total / nu as f64 + ORDER_FD
                });
                (fk, true)
            } else {
                let (king, bit) = if payload == 0 { (true, 0u32) } else { (false, payload as u32) };
                let mut key = ORDER_ME * h_marginal(mi, mk, king, bit)
                    + ORDER_OPP * h_marginal(oi, ok, king, bit);
                if king {
                    key += ORDER_CENT * KING_CENTRALITY + ORDER_KING;
                } else {
                    key += ORDER_CENT * cent[(bit.trailing_zeros() % 8) as usize];
                }
                (key, false)
            };
            items.push((key + ORDER_REPLIES * replies, target, fd));
        }
        // Python sort key (-score, marker): score desc, then target asc.
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        items.into_iter().map(|(_, t, fd)| (t, fd)).collect()
    }

    /// The single decision node's child value for a chosen (target, fd),
    /// negamax perspective (already negated for the caller). `nu` is the
    /// hidden count at this node.
    #[allow(clippy::too_many_arguments)]
    fn child_value(
        &mut self, target: usize, fd: bool, rows: u64, cols: u64, mi: u32, mk: u8, oi: u32,
        ok: u8, nu: usize, depth: i64, alpha: f64, beta: f64,
    ) -> f64 {
        let nrows = rows | (1 << target);
        let ncols = cols | col_bit(target);
        if fd {
            if nu == 1 {
                let (king, bit) = code_bit(self.unknowns[0]);
                let (nmi, nmk) = if king { (mi, mk + 1) } else { (mi | bit, mk) };
                -self.search(target, nrows, ncols, oi, ok, nmi, nmk, 0, depth - 1, -beta, -alpha)
            } else {
                self.chance(target, nrows, ncols, mi, mk, oi, ok, nu, depth, alpha, beta)
            }
        } else {
            let payload = self.cells[target];
            let (nmi, nmk) = if payload == 0 { (mi, mk + 1) } else { (mi | payload as u32, mk) };
            -self.search(target, nrows, ncols, oi, ok, nmi, nmk, nu, depth - 1, -beta, -alpha)
        }
    }

    /// ai._search: fail-soft negamax with a transposition table.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self, cell: usize, rows: u64, cols: u64, mi: u32, mk: u8, oi: u32, ok: u8, nu: usize,
        depth: i64, mut alpha: f64, beta: f64,
    ) -> f64 {
        self.tick();
        if self.aborted {
            return 0.0;
        }
        let mut buf = [0usize; 10];
        let n = legal(cell, rows, cols, &mut buf);
        if n == 0 {
            return self.terminal(mi, mk, oi, ok);
        }
        let key: TTKey = (rows, cell as u8, mi, mk, oi, ok);
        let mut tt_move: Option<usize> = None;
        if let Some(&(td, flag, val, bm)) = self.tt.get(&key) {
            if bm < 36 {
                tt_move = Some(bm);
            }
            if td >= depth
                && (flag == 0 || (flag == 1 && val >= beta) || (flag == -1 && val <= alpha))
            {
                return val;
            }
        }
        if depth == 0 {
            return self.eval_leaf(mi, mk, oi, ok, rows, nu, n);
        }
        let mut markers = self.ordered(cell, rows, cols, mi, mk, oi, ok, nu);
        if let Some(tm) = tt_move {
            if markers[0].0 != tm {
                if let Some(pos) = markers.iter().position(|&(t, _)| t == tm) {
                    let it = markers.remove(pos);
                    markers.insert(0, it);
                }
            }
        }
        let alpha0 = alpha;
        let mut best = -self.p.value_bound;
        let mut best_marker = 36usize;
        for (target, fd) in markers {
            let value =
                self.child_value(target, fd, rows, cols, mi, mk, oi, ok, nu, depth, alpha, beta);
            if self.aborted {
                return 0.0;
            }
            if value > best {
                best = value;
                best_marker = target;
                if value > alpha {
                    alpha = value;
                }
                if alpha >= beta {
                    break;
                }
            }
        }
        let store = match self.tt.get(&key) {
            None => true,
            Some(&(td, _, _, _)) => td <= depth,
        };
        if store {
            let flag = if best <= alpha0 {
                -1
            } else if best >= beta {
                1
            } else {
                0
            };
            self.tt.insert(key, (depth, flag, best, best_marker));
        }
        best
    }

    /// ai._chance_value: Star1 expectation over sampled face-down
    /// resolutions, each searched in the window that could still move the
    /// running mean into (alpha, beta).
    #[allow(clippy::too_many_arguments)]
    fn chance(
        &mut self, target: usize, nrows: u64, ncols: u64, mi: u32, mk: u8, oi: u32, ok: u8,
        nu: usize, depth: i64, alpha: f64, beta: f64,
    ) -> f64 {
        let bound = self.p.value_bound;
        let indices = sample_indices(nu, self.p.resolution_cap);
        let n = indices.len();
        let nf = n as f64;
        let mut total = 0.0f64;
        for (k, &idx) in indices.iter().enumerate() {
            let spread = (n - k - 1) as f64 * bound;
            let lo = (nf * alpha - total - spread).max(-bound);
            let hi = (nf * beta - total + spread).min(bound);
            let code = self.unknowns[idx];
            let (king, bit) = code_bit(code);
            let (nmi, nmk) = if king { (mi, mk + 1) } else { (mi | bit, mk) };
            for j in idx..nu - 1 {
                self.unknowns[j] = self.unknowns[j + 1];
            }
            let child =
                self.search(target, nrows, ncols, oi, ok, nmi, nmk, nu - 1, depth - 1, -hi, -lo);
            for j in (idx..nu - 1).rev() {
                self.unknowns[j + 1] = self.unknowns[j];
            }
            self.unknowns[idx] = code;
            if self.aborted {
                return 0.0;
            }
            total -= child;
            let upper = (total + spread) / nf;
            if upper <= alpha {
                return upper;
            }
            let lower = (total - spread) / nf;
            if lower >= beta {
                return lower;
            }
        }
        total / nf
    }
}

fn new_hctx(cells: Vec<i64>, unknowns: Vec<i64>, params: &[f64], budget: Option<f64>) -> PyResult<(HCtx, usize)> {
    if cells.len() != 36 || unknowns.len() > 12 || params.len() != 12 {
        return Err(pyo3::exceptions::PyValueError::new_err("bad state shape"));
    }
    let mut u = [0i64; 12];
    let nu = unknowns.len();
    u[..nu].copy_from_slice(&unknowns);
    u[..nu].sort_unstable(); // sort hidden codes = sort by (rank, suit)
    let p = HParams {
        value_bound: params[0],
        win_bonus: params[1],
        potential_weight: params[2],
        potential_slope: params[3],
        centrality_weight: params[4],
        centrality_base: params[5],
        king_centrality: params[6],
        mobility_weight: params[7],
        mobility_slope: params[8],
        tempo_bonus: params[9],
        tempo_slope: params[10],
        phase_pivot: params[11],
        resolution_cap: 0, // filled by caller
        deepen_fraction: 1.0,
    };
    Ok((
        HCtx {
            cells: cells.try_into().unwrap(),
            unknowns: u,
            p,
            deadline: budget.map(|s| Instant::now() + Duration::from_secs_f64(s)),
            nodes: 0,
            aborted: false,
            tt: HashMap::new(),
        },
        nu,
    ))
}

/// Iterative-deepening heuristic search from the root (ai._search_root).
/// `mi,mk / oi,ok` are the mover / opponent hands. Returns
/// (best_marker, completed_depth, elapsed_secs).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn heuristic_root(
    py: Python<'_>, cells: Vec<i64>, cell: usize, rows: u64, cols: u64, mi: u32, mk: u8, oi: u32,
    ok: u8, unknowns: Vec<i64>, params: Vec<f64>, resolution_cap: usize, deepen_fraction: f64,
    time_budget: f64,
) -> PyResult<((usize, usize), i64, f64)> {
    let (mut ctx, nu) = new_hctx(cells, unknowns, &params, Some(time_budget))?;
    ctx.p.resolution_cap = resolution_cap;
    ctx.p.deepen_fraction = deepen_fraction;
    py.detach(move || {
        let start = Instant::now();
        let bound = ctx.p.value_bound;
        let mut move_list = ctx.ordered(cell, rows, cols, mi, mk, oi, ok, nu);
        let mut best_marker = move_list[0].0;
        let max_depth = 36 - rows.count_ones() as i64;
        let mut completed = 0i64;
        let mut depth = 1i64;
        while depth <= max_depth {
            let mut alpha = -bound;
            let mut iteration_best: Option<usize> = None;
            let mut scores: HashMap<usize, f64> = HashMap::with_capacity(move_list.len());
            for &(target, fd) in &move_list {
                let value =
                    ctx.child_value(target, fd, rows, cols, mi, mk, oi, ok, nu, depth, alpha, bound);
                if ctx.aborted {
                    break;
                }
                scores.insert(target, value);
                if value > alpha {
                    alpha = value;
                    iteration_best = Some(target);
                }
            }
            if ctx.aborted {
                if let Some(ib) = iteration_best {
                    best_marker = ib;
                }
                break;
            }
            if let Some(ib) = iteration_best {
                best_marker = ib;
            }
            completed = depth;
            move_list.sort_by(|a, b| {
                let sa = scores.get(&a.0).copied().unwrap_or(f64::NEG_INFINITY);
                let sb = scores.get(&b.0).copied().unwrap_or(f64::NEG_INFINITY);
                sb.partial_cmp(&sa).unwrap().then(a.0.cmp(&b.0))
            });
            depth += 1;
            if start.elapsed().as_secs_f64() > ctx.p.deepen_fraction * time_budget {
                break;
            }
        }
        Ok((
            (best_marker / 6, best_marker % 6),
            completed,
            start.elapsed().as_secs_f64(),
        ))
    })
}

/// Validation probe: every root move's exact full-window value at a fixed
/// depth (no deepening, no time limit). Used to check the port against the
/// Python search. Returns (marker, value) per legal move.
/// Choose the marker's starting cell for the placer (the player NOT moving
/// first). `starts` are the candidate cell indices; the placer wants the one
/// MINIMISING the value of the mover's best reply. Mirrors `heuristic_root`'s
/// budgeted iterative deepening, but over all placements jointly: the position
/// after the mover's first move is independent of which cell the marker began
/// on, so the shared transposition table caches every subtree common to two
/// placements, searched once. All placements advance at equal depth; children
/// use the full window so their values are exact and cross-comparable.
/// Returns (best cell (row, col), completed depth, elapsed secs, per-start
/// values in `starts` order).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn heuristic_placement(
    py: Python<'_>, cells: Vec<i64>, starts: Vec<usize>, rows: u64, cols: u64, mi: u32, mk: u8,
    oi: u32, ok: u8, unknowns: Vec<i64>, params: Vec<f64>, resolution_cap: usize,
    deepen_fraction: f64, time_budget: f64,
) -> PyResult<((usize, usize), i64, f64, Vec<f64>)> {
    let (mut ctx, nu) = new_hctx(cells, unknowns, &params, Some(time_budget))?;
    ctx.p.resolution_cap = resolution_cap;
    ctx.p.deepen_fraction = deepen_fraction;
    py.detach(move || {
        let start = Instant::now();
        let bound = ctx.p.value_bound;
        // Each placement's first-move children, ordered once.
        let child_lists: Vec<Vec<(usize, bool)>> = starts
            .iter()
            .map(|&cell| ctx.ordered(cell, rows, cols, mi, mk, oi, ok, nu))
            .collect();
        let mut best_values = vec![-bound; starts.len()];
        let max_depth = 36 - rows.count_ones() as i64;
        let mut completed = 0i64;
        let mut depth = 1i64;
        while depth <= max_depth {
            let mut values = vec![-bound; starts.len()];
            let mut aborted = false;
            for (pi, children) in child_lists.iter().enumerate() {
                let mut v = -bound;
                for &(target, fd) in children {
                    let cv = ctx.child_value(
                        target, fd, rows, cols, mi, mk, oi, ok, nu, depth, -bound, bound,
                    );
                    if ctx.aborted {
                        aborted = true;
                        break;
                    }
                    if cv > v {
                        v = cv;
                    }
                }
                if aborted {
                    break;
                }
                values[pi] = v;
            }
            if aborted {
                break;
            }
            best_values = values;
            completed = depth;
            depth += 1;
            if start.elapsed().as_secs_f64() > ctx.p.deepen_fraction * time_budget {
                break;
            }
        }
        // Placer minimises the mover's value; strict `<` keeps the lowest start
        // index on exact ties, matching Python's min over (value, cell).
        let mut best_i = 0usize;
        for i in 1..starts.len() {
            if best_values[i] < best_values[best_i] {
                best_i = i;
            }
        }
        let cell = starts[best_i];
        Ok((
            (cell / 6, cell % 6),
            completed,
            start.elapsed().as_secs_f64(),
            best_values,
        ))
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn heuristic_probe(
    py: Python<'_>, cells: Vec<i64>, cell: usize, rows: u64, cols: u64, mi: u32, mk: u8, oi: u32,
    ok: u8, unknowns: Vec<i64>, params: Vec<f64>, resolution_cap: usize, depth: i64,
) -> PyResult<Vec<((usize, usize), f64)>> {
    let (mut ctx, nu) = new_hctx(cells, unknowns, &params, None)?;
    ctx.p.resolution_cap = resolution_cap;
    py.detach(move || {
        let bound = ctx.p.value_bound;
        let mut out = Vec::new();
        for (target, fd) in ctx.ordered(cell, rows, cols, mi, mk, oi, ok, nu) {
            let value =
                ctx.child_value(target, fd, rows, cols, mi, mk, oi, ok, nu, depth, -bound, bound);
            out.push(((target / 6, target % 6), value));
        }
        Ok(out)
    })
}

#[pymodule]
fn cardgame_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_root, m)?)?;
    m.add_function(wrap_pyfunction!(analyse_move, m)?)?;
    m.add_function(wrap_pyfunction!(distribution, m)?)?;
    m.add_function(wrap_pyfunction!(heuristic_root, m)?)?;
    m.add_function(wrap_pyfunction!(heuristic_placement, m)?)?;
    m.add_function(wrap_pyfunction!(heuristic_probe, m)?)?;
    Ok(())
}
