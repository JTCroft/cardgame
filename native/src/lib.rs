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
    Ok(())
}
