// SPDX-License-Identifier: GPL-3.0-or-later
//! Linear algebra helpers (nalgebra-backed): SVD, inverse, solve, quantiles.

use nalgebra::DMatrix;
use ndarray::{Array2, Axis};

pub fn to_nalgebra(a: &Array2<f64>) -> DMatrix<f64> {
    let (r, c) = (a.nrows(), a.ncols());
    DMatrix::from_fn(r, c, |i, j| a[[i, j]])
}

pub fn from_nalgebra(m: &DMatrix<f64>) -> Array2<f64> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

/// Full SVD with singular values sorted descending (numpy-compatible ordering).
/// Returns `(U [m,k], s [k], Vt [k,n])` with `k = min(m, n)`.
pub fn svd_full(a: &Array2<f64>) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    let m = to_nalgebra(a);
    let svd = m.svd(true, true);
    let u = svd.u.expect("SVD U");
    let vt = svd.v_t.expect("SVD Vt");
    let s = &svd.singular_values;
    let k = s.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&x, &y| s[y].partial_cmp(&s[x]).unwrap_or(std::cmp::Ordering::Equal));
    let s_sorted: Vec<f64> = order.iter().map(|&i| s[i]).collect();
    let u_sorted = DMatrix::from_fn(u.nrows(), k, |i, j| u[(i, order[j])]);
    let vt_sorted = DMatrix::from_fn(k, vt.ncols(), |i, j| vt[(order[i], j)]);
    (from_nalgebra(&u_sorted), s_sorted, from_nalgebra(&vt_sorted))
}

/// Singular values only. Uses the Gram-matrix eigenvalue route for large matrices
/// (mirrors the Uni-Layer profiler optimization).
pub fn singular_values(a: &Array2<f64>) -> Vec<f64> {
    let (r, c) = (a.nrows(), a.ncols());
    if r.min(c) <= 256 {
        let m = to_nalgebra(a);
        let svd = m.svd(false, false);
        let mut sv: Vec<f64> = svd.singular_values.iter().copied().collect();
        sv.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        return sv;
    }
    let gram = if c < r { a.t().dot(a) } else { a.dot(&a.t()) };
    let g = to_nalgebra(&gram);
    let eig = g.symmetric_eigen();
    let mut ev: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    ev.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    ev.into_iter().map(|v| v.max(0.0).sqrt()).collect()
}

pub fn inv(a: &Array2<f64>) -> Option<Array2<f64>> {
    to_nalgebra(a).try_inverse().map(|m| from_nalgebra(&m))
}

/// Solve `A x = b` (A square). Falls back to inverse when LU fails.
pub fn solve(a: &Array2<f64>, b: &Array2<f64>) -> Option<Array2<f64>> {
    let m = to_nalgebra(a);
    let rhs = to_nalgebra(b);
    if let Some(x) = m.clone().lu().solve(&rhs) {
        return Some(from_nalgebra(&x));
    }
    m.try_inverse().map(|inv| from_nalgebra(&(inv * rhs)))
}

pub fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

pub fn transpose(a: &Array2<f64>) -> Array2<f64> {
    a.t().to_owned()
}

pub fn eye(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| if i == j { 1.0 } else { 0.0 })
}

/// Median of a slice (numpy-compatible, sorts internally).
pub fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// numpy-style linear-interpolation quantile for q in [0, 1].
pub fn quantile(vals: &[f64], q: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
}

pub fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

pub fn std_pop(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mu = mean(vals);
    (vals.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / vals.len() as f64).sqrt()
}

/// Row L2 norms of a 2D matrix.
pub fn row_norms(a: &Array2<f64>) -> Vec<f64> {
    a.rows()
        .into_iter()
        .map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect()
}

pub fn col_norms(a: &Array2<f64>) -> Vec<f64> {
    a.columns()
        .into_iter()
        .map(|c| c.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect()
}

pub fn l2_norm_flat(a: &Array2<f64>) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// argsort descending, stable.
pub fn argsort_desc(vals: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// argsort ascending, stable.
pub fn argsort_asc(vals: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

pub fn to_f64_2d(a: &ndarray::ArrayD<f32>) -> Array2<f64> {
    let shp = a.shape();
    let (r, c) = if shp.len() == 1 {
        (1, shp[0])
    } else {
        (shp[0], shp[1..].iter().product())
    };
    Array2::from_shape_fn((r, c), |(i, j)| a[[i, j]] as f64)
}

pub fn sum_axis0(a: &Array2<f64>) -> Vec<f64> {
    a.sum_axis(Axis(0)).to_vec()
}