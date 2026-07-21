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

#[pymodule]
fn cardgame_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_root, m)?)?;
    m.add_function(wrap_pyfunction!(analyse_move, m)?)?;
    m.add_function(wrap_pyfunction!(distribution, m)?)?;
    Ok(())
}
