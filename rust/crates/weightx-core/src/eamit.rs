// SPDX-License-Identifier: GPL-3.0-or-later
//! L3 batch associative memory editing via EAMIT (EAMET — Embedding Alignment
//! Model Editing in Transformers, arXiv:2505.11876) & closed-form ridge solvers.

use std::collections::HashMap;

use ndarray::Array2;
use serde_json::{json, Value};

use crate::error::{Result, WeightXError};
use crate::linalg::{self, eye, inv, matmul, transpose};
use crate::tensor::Tensor;
use crate::workspace::WeightWorkspace;

#[derive(Debug, Clone)]
pub struct EamitAlignConfig {
    pub lambda_kl: f64,
    pub lambda_mse: f64,
    pub top_m: usize,
    pub tau: f64,
    pub align_steps: usize,
    pub align_lr: f64,
    pub max_block_size: usize,
}

impl Default for EamitAlignConfig {
    fn default() -> Self {
        Self {
            lambda_kl: 0.05,
            lambda_mse: 0.25,
            top_m: 8,
            tau: 0.1,
            align_steps: 20,
            align_lr: 0.03,
            max_block_size: 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EamitAlignStats {
    pub pre_kl: f64,
    pub post_kl: f64,
    pub pre_top_m_mse: f64,
    pub post_top_m_mse: f64,
    pub steps_executed: usize,
}

impl EamitAlignStats {
    pub fn to_value(&self) -> Value {
        let r6 = |v: f64| (v * 1e6).round() / 1e6;
        json!({
            "pre_kl": r6(self.pre_kl),
            "post_kl": r6(self.post_kl),
            "pre_top_m_mse": r6(self.pre_top_m_mse),
            "post_top_m_mse": r6(self.post_top_m_mse),
            "steps_executed": self.steps_executed,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BatchFactSpec {
    pub fact_id: String,
    pub key_vector: Vec<f64>,
    pub target_value_vector: Vec<f64>,
    pub layer_inputs: HashMap<String, Vec<f64>>,
}

impl BatchFactSpec {
    pub fn new(id: impl Into<String>, key: Vec<f64>, value: Vec<f64>) -> Self {
        Self {
            fact_id: id.into(),
            key_vector: key,
            target_value_vector: value,
            layer_inputs: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EamitResult {
    pub num_facts: usize,
    pub target_tensors: Vec<String>,
    pub pre_edit_mse: f64,
    pub post_edit_mse: f64,
    pub relative_error_reduction: f64,
    pub pre_alignment_kl: f64,
    pub post_alignment_kl: f64,
    pub pre_alignment_top_m_mse: f64,
    pub post_alignment_top_m_mse: f64,
    pub per_tensor_l2_deltas: HashMap<String, f64>,
    pub fisher_protected: bool,
}

impl EamitResult {
    pub fn to_value(&self) -> Value {
        let r6 = |v: f64| (v * 1e6).round() / 1e6;
        json!({
            "num_facts": self.num_facts,
            "target_tensors": self.target_tensors,
            "pre_edit_mse": r6(self.pre_edit_mse),
            "post_edit_mse": r6(self.post_edit_mse),
            "relative_error_reduction_pct": (self.relative_error_reduction * 10000.0).round() / 100.0,
            "pre_alignment_kl": r6(self.pre_alignment_kl),
            "post_alignment_kl": r6(self.post_alignment_kl),
            "pre_alignment_top_m_mse": r6(self.pre_alignment_top_m_mse),
            "post_alignment_top_m_mse": r6(self.post_alignment_top_m_mse),
            "per_tensor_l2_deltas": self.per_tensor_l2_deltas.iter().map(|(k, v)| (k.clone(), json!(r6(*v)))).collect::<serde_json::Map<String, Value>>(),
            "fisher_protected": self.fisher_protected,
        })
    }
}

fn as_f64_2d(t: &Tensor) -> Array2<f64> {
    linalg::to_f64_2d(t)
}

fn offdiag_softmax(sim: &Array2<f64>, tau: f64) -> Array2<f64> {
    let m = sim.nrows();
    let t = tau.max(1e-6);
    let mut probs = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        let mut mx = f64::NEG_INFINITY;
        for j in 0..m {
            if j != i {
                let s = sim[[i, j]] / t;
                if s > mx {
                    mx = s;
                }
            }
        }
        let mut denom = 0.0f64;
        for j in 0..m {
            if j != i {
                let e = (sim[[i, j]] / t - mx).exp();
                probs[[i, j]] = e;
                denom += e;
            }
        }
        if denom > 0.0 {
            for j in 0..m {
                if j != i {
                    probs[[i, j]] /= denom;
                }
            }
        } else {
            let uniform = 1.0 / (m.saturating_sub(1).max(1) as f64);
            for j in 0..m {
                if j != i {
                    probs[[i, j]] = uniform;
                }
            }
        }
    }
    probs
}

fn align_block(
    k_blk: &Array2<f64>,
    r_blk: &Array2<f64>,
    cfg: &EamitAlignConfig,
) -> (Array2<f64>, EamitAlignStats) {
    let n = k_blk.ncols();
    let d_in = k_blk.nrows();
    let d_out = r_blk.nrows();
    if n < 2 || d_out == 0 {
        return (
            r_blk.clone(),
            EamitAlignStats {
                pre_kl: 0.0,
                post_kl: 0.0,
                pre_top_m_mse: 0.0,
                post_top_m_mse: 0.0,
                steps_executed: 0,
            },
        );
    }

    let k_norms = linalg::col_norms(k_blk);
    let r_norms = linalg::col_norms(r_blk);
    let active: Vec<usize> = (0..n)
        .filter(|&j| k_norms[j] > 1e-12 && r_norms[j] > 1e-12)
        .collect();
    let m = active.len();
    if m < 2 {
        return (
            r_blk.clone(),
            EamitAlignStats {
                pre_kl: 0.0,
                post_kl: 0.0,
                pre_top_m_mse: 0.0,
                post_top_m_mse: 0.0,
                steps_executed: 0,
            },
        );
    }

    let mut k_hat = Array2::<f64>::zeros((d_in, m));
    let mut r_act = Array2::<f64>::zeros((d_out, m));
    let mut scales = vec![1.0f64; m];
    for (col_idx, &orig_idx) in active.iter().enumerate() {
        let kn = k_norms[orig_idx].max(1e-12);
        for row in 0..d_in {
            k_hat[[row, col_idx]] = k_blk[[row, orig_idx]] / kn;
        }
        let s = (r_norms[orig_idx] / (d_out as f64).sqrt()).max(1e-12);
        scales[col_idx] = s;
        for row in 0..d_out {
            r_act[[row, col_idx]] = r_blk[[row, orig_idx]];
        }
    }

    let mut s_k = matmul(&transpose(&k_hat), &k_hat);
    for v in s_k.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    let p_k = offdiag_softmax(&s_k, cfg.tau);
    let log_p_k = p_k.mapv(|v| (v + 1e-12).ln());

    let top_m = cfg.top_m.min(m - 1).max(1);
    let mut top_mask = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        let mut others: Vec<usize> = (0..m).filter(|&j| j != i).collect();
        others.sort_by(|&a, &b| {
            s_k[[i, b]]
                .partial_cmp(&s_k[[i, a]])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        for &j in others.iter().take(top_m) {
            top_mask[[i, j]] = 1.0;
        }
    }

    let mut r_tilde_raw = Array2::<f64>::zeros((d_out, m));
    for j in 0..m {
        for i in 0..d_out {
            r_tilde_raw[[i, j]] = r_act[[i, j]] / scales[j];
        }
    }
    let mut r_tilde = r_tilde_raw.clone();
    let tau = cfg.tau.max(1e-6);
    let eps_norm = 1e-12f64;

    let eval = |rt: &Array2<f64>| -> (f64, Array2<f64>, f64, f64) {
        let mut r_n = vec![0.0f64; m];
        let mut r_hat = Array2::<f64>::zeros((d_out, m));
        for j in 0..m {
            let mut sq = eps_norm;
            for i in 0..d_out {
                let v = rt[[i, j]];
                sq += v * v;
            }
            let nrm = sq.sqrt();
            r_n[j] = nrm;
            for i in 0..d_out {
                r_hat[[i, j]] = rt[[i, j]] / nrm;
            }
        }
        let s_r = matmul(&transpose(&r_hat), &r_hat);
        let p_r = offdiag_softmax(&s_r, tau);

        let mut log_ratio = Array2::<f64>::zeros((m, m));
        let mut kl_rows_raw = vec![0.0f64; m];
        let mut kl_sum = 0.0f64;
        for i in 0..m {
            let mut row_kl = 0.0f64;
            for j in 0..m {
                if i != j && p_r[[i, j]] > 0.0 {
                    let lr = (p_r[[i, j]] + 1e-12).ln() - log_p_k[[i, j]];
                    log_ratio[[i, j]] = lr;
                    row_kl += p_r[[i, j]] * lr;
                }
            }
            kl_rows_raw[i] = row_kl;
            kl_sum += row_kl.max(0.0);
        }
        let kl_mean = kl_sum / m as f64;

        let mut diff_top = Array2::<f64>::zeros((m, m));
        let mut top_sq = 0.0f64;
        for i in 0..m {
            for j in 0..m {
                let d = (s_r[[i, j]] - s_k[[i, j]]) * top_mask[[i, j]];
                diff_top[[i, j]] = d;
                top_sq += d * d;
            }
        }
        let top_m_mse = top_sq / (m * top_m) as f64;

        let diff_fid = rt - &r_tilde_raw;
        let loss_fid = diff_fid.iter().map(|v| v * v).sum::<f64>() / (m * d_out) as f64;
        let total_loss = loss_fid + cfg.lambda_kl * kl_mean + cfg.lambda_mse * top_m_mse;

        let mut g_mat = Array2::<f64>::zeros((m, m));
        let kl_coef = cfg.lambda_kl / (m as f64 * tau);
        let mse_coef = 2.0 * cfg.lambda_mse / (m * top_m) as f64;
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    let g_kl = kl_coef * p_r[[i, j]] * (log_ratio[[i, j]] - kl_rows_raw[i]);
                    let g_mse = mse_coef * diff_top[[i, j]];
                    g_mat[[i, j]] = g_kl + g_mse;
                }
            }
        }
        let g_sym = &g_mat + &transpose(&g_mat);
        let u = matmul(&r_hat, &g_sym);

        let mut grad = Array2::<f64>::zeros((d_out, m));
        let fid_coef = 2.0 / (m * d_out) as f64;
        for j in 0..m {
            let mut dot = 0.0f64;
            for i in 0..d_out {
                dot += r_hat[[i, j]] * u[[i, j]];
            }
            let inv_rn = 1.0 / r_n[j];
            for i in 0..d_out {
                let g_align = (u[[i, j]] - r_hat[[i, j]] * dot) * inv_rn;
                let g_fid = fid_coef * diff_fid[[i, j]];
                grad[[i, j]] = g_fid + g_align;
            }
        }
        (total_loss, grad, kl_mean, top_m_mse)
    };

    let (loss0, mut grad, pre_kl, pre_mse) = eval(&r_tilde);
    if cfg.align_steps == 0 || (cfg.lambda_kl <= 0.0 && cfg.lambda_mse <= 0.0) {
        return (
            r_blk.clone(),
            EamitAlignStats {
                pre_kl,
                post_kl: pre_kl,
                pre_top_m_mse: pre_mse,
                post_top_m_mse: pre_mse,
                steps_executed: 0,
            },
        );
    }

    let mut best_loss = loss0;
    let mut best_rt = r_tilde.clone();
    let mut best_kl = pre_kl;
    let mut best_mse = pre_mse;

    let mut m_adam = Array2::<f64>::zeros((d_out, m));
    let mut v_adam = Array2::<f64>::zeros((d_out, m));
    let (beta1, beta2, adam_eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut steps_done = 0usize;

    for step in 1..=cfg.align_steps {
        steps_done = step;
        let b1_pow = 1.0 - beta1.powi(step as i32);
        let b2_pow = 1.0 - beta2.powi(step as i32);
        for i in 0..d_out {
            for j in 0..m {
                let g = grad[[i, j]];
                let m_val = beta1 * m_adam[[i, j]] + (1.0 - beta1) * g;
                let v_val = beta2 * v_adam[[i, j]] + (1.0 - beta2) * g * g;
                m_adam[[i, j]] = m_val;
                v_adam[[i, j]] = v_val;
                let m_hat = m_val / b1_pow;
                let v_hat = v_val / b2_pow;
                r_tilde[[i, j]] -= cfg.align_lr * m_hat / (v_hat.sqrt() + adam_eps);
            }
        }
        let (loss_t, next_grad, kl_t, mse_t) = eval(&r_tilde);
        grad = next_grad;
        if loss_t < best_loss {
            best_loss = loss_t;
            best_rt = r_tilde.clone();
            best_kl = kl_t;
            best_mse = mse_t;
        }
    }

    let mut r_out = r_blk.clone();
    for (col_idx, &orig_idx) in active.iter().enumerate() {
        for row in 0..d_out {
            r_out[[row, orig_idx]] = best_rt[[row, col_idx]] * scales[col_idx];
        }
    }
    (
        r_out,
        EamitAlignStats {
            pre_kl,
            post_kl: best_kl,
            pre_top_m_mse: pre_mse,
            post_top_m_mse: best_mse,
            steps_executed: steps_done,
        },
    )
}

/// Run EAMIT key-residual embedding alignment optimization (`K: [d_in, N]`, `R: [d_out, N]`).
pub fn align_residuals_to_keys(
    keys_matrix: &Array2<f64>,
    residuals_matrix: &Array2<f64>,
    cfg: &EamitAlignConfig,
) -> Result<(Array2<f64>, EamitAlignStats)> {
    if keys_matrix.ncols() != residuals_matrix.ncols() {
        return Err(WeightXError::Value(format!(
            "align_residuals_to_keys column mismatch: K={:?}, R={:?}",
            keys_matrix.dim(),
            residuals_matrix.dim()
        )));
    }
    let n = keys_matrix.ncols();
    if n < 2 {
        return Ok((
            residuals_matrix.clone(),
            EamitAlignStats {
                pre_kl: 0.0,
                post_kl: 0.0,
                pre_top_m_mse: 0.0,
                post_top_m_mse: 0.0,
                steps_executed: 0,
            },
        ));
    }

    let bs = cfg.max_block_size.max(2);
    if n <= bs {
        return Ok(align_block(keys_matrix, residuals_matrix, cfg));
    }

    let mut out = residuals_matrix.clone();
    let mut starts: Vec<usize> = (0..n).step_by(bs).collect();
    if starts.len() >= 2 && n - *starts.last().unwrap() < 2 {
        starts.pop();
    }

    let (mut pre_kl_sum, mut post_kl_sum, mut pre_mse_sum, mut post_mse_sum) = (0.0, 0.0, 0.0, 0.0);
    let mut total_w = 0usize;
    let mut steps_max = 0usize;
    for (idx, &s) in starts.iter().enumerate() {
        let e = if idx + 1 == starts.len() { n } else { (s + bs).min(n) };
        let k_sub = keys_matrix.slice(ndarray::s![.., s..e]).to_owned();
        let r_sub = residuals_matrix.slice(ndarray::s![.., s..e]).to_owned();
        let (blk_out, st) = align_block(&k_sub, &r_sub, cfg);
        out.slice_mut(ndarray::s![.., s..e]).assign(&blk_out);
        let w = e - s;
        pre_kl_sum += st.pre_kl * w as f64;
        post_kl_sum += st.post_kl * w as f64;
        pre_mse_sum += st.pre_top_m_mse * w as f64;
        post_mse_sum += st.post_top_m_mse * w as f64;
        total_w += w;
        steps_max = steps_max.max(st.steps_executed);
    }
    let inv_w = 1.0 / total_w.max(1) as f64;
    Ok((
        out,
        EamitAlignStats {
            pre_kl: pre_kl_sum * inv_w,
            post_kl: post_kl_sum * inv_w,
            pre_top_m_mse: pre_mse_sum * inv_w,
            post_top_m_mse: post_mse_sum * inv_w,
            steps_executed: steps_max,
        },
    ))
}

/// Minimum-norm regularized update so `(W + dW) @ K ~= V`.
pub fn apply_closed_form_ridge(
    ws: &mut WeightWorkspace,
    tensor_name: &str,
    keys_matrix: &Array2<f64>,
    values_matrix: &Array2<f64>,
    lambda_reg: f64,
    apply_fisher_guard: bool,
) -> Result<Value> {
    let w_t = ws.get_readonly(tensor_name)?.clone();
    if w_t.ndim() != 2 {
        return Err(WeightXError::Value(format!(
            "closed_form_ridge requires a 2D tensor, got shape {:?}",
            w_t.shape()
        )));
    }
    let w = as_f64_2d(&w_t);
    let (d_out, d_in) = (w.nrows(), w.ncols());

    let mut k = keys_matrix.clone();
    let mut v = values_matrix.clone();
    if k.nrows() != d_in && k.ncols() == d_in {
        k = transpose(&k);
    }
    if v.nrows() != d_out && v.ncols() == d_out {
        v = transpose(&v);
    }
    if k.nrows() != d_in || v.nrows() != d_out || k.ncols() != v.ncols() {
        return Err(WeightXError::Value(format!(
            "Dimension mismatch for {tensor_name} ({d_out},{d_in}): K={:?}, V={:?}",
            (k.nrows(), k.ncols()),
            (v.nrows(), v.ncols())
        )));
    }

    let residual = &v - matmul(&w, &k);
    let pre_mse = residual.iter().map(|x| x * x).sum::<f64>() / residual.len() as f64;

    let mut fisher_diag = vec![0.0f64; d_in];
    if apply_fisher_guard {
        if let Some(guard) = ws.fisher_guard.as_mut() {
            let f_mat = guard.get_fisher(tensor_name, &w_t);
            let f2d = as_f64_2d(&f_mat);
            let col_f: Vec<f64> = (0..d_in)
                .map(|j| f2d.column(j).iter().sum::<f64>() / d_out as f64)
                .collect();
            let mean_cf = linalg::mean(&col_f);
            if mean_cf > 1e-20 {
                for j in 0..d_in {
                    fisher_diag[j] = 0.05 * lambda_reg * (col_f[j] / mean_cf);
                }
            }
        }
    }

    let mut gram = matmul(&k, &transpose(&k)) + eye(d_in).mapv(|x| x * lambda_reg);
    for j in 0..d_in {
        gram[[j, j]] += fisher_diag[j];
    }
    let gram_inv = inv(&gram)
        .ok_or_else(|| WeightXError::Value("Singular gram matrix in closed_form_ridge".into()))?;
    let delta_w = matmul(&matmul(&residual, &transpose(&k)), &gram_inv);
    let w_new = (&w + &delta_w).mapv(|x| x as f32);
    let w_new_t = w_new.into_dyn().into_shape_with_order(w_t.shape()).unwrap().to_owned();

    let f_rep = ws.set_tensor(tensor_name, &w_new_t, apply_fisher_guard, false)?;
    let w_final = as_f64_2d(ws.get_readonly(tensor_name)?);
    let post_res = &v - matmul(&w_final, &k);
    let post_mse = post_res.iter().map(|x| x * x).sum::<f64>() / post_res.len() as f64;
    let l2_delta = linalg::l2_norm_flat(&(&w_final - &w));

    let r6 = |x: f64| (x * 1e6).round() / 1e6;
    Ok(json!({
        "tensor": tensor_name,
        "num_keys": k.ncols(),
        "pre_mse": r6(pre_mse),
        "post_mse": r6(post_mse),
        "l2_delta": r6(l2_delta),
        "fisher_report": f_rep.map(|r| r.to_value()).unwrap_or(Value::Null),
    }))
}

/// EAMIT simultaneous multi-layer batch editing with key-residual embedding alignment.
pub fn apply_eamit_batch(
    ws: &mut WeightWorkspace,
    target_tensors: &[String],
    facts: &[BatchFactSpec],
    lambda_cov: f64,
    apply_fisher_guard: bool,
) -> Result<EamitResult> {
    apply_eamit_batch_with_config(
        ws,
        target_tensors,
        facts,
        lambda_cov,
        apply_fisher_guard,
        &EamitAlignConfig::default(),
    )
}

/// EAMIT simultaneous multi-layer batch editing with custom alignment hyperparameters.
pub fn apply_eamit_batch_with_config(
    ws: &mut WeightWorkspace,
    target_tensors: &[String],
    facts: &[BatchFactSpec],
    lambda_cov: f64,
    apply_fisher_guard: bool,
    align_cfg: &EamitAlignConfig,
) -> Result<EamitResult> {
    if target_tensors.is_empty() {
        return Err(WeightXError::Value("target_tensors must contain at least one tensor name".into()));
    }
    if facts.is_empty() {
        return Err(WeightXError::Value("facts list must not be empty".into()));
    }
    if let Some(arch) = ws.arch.clone() {
        if let Some(res) = arch.eamit_override(ws, target_tensors, facts, lambda_cov, apply_fisher_guard, align_cfg) {
            return res;
        }
    }
    let num_layers = target_tensors.len();
    let num_facts = facts.len();

    let mut shapes = Vec::new();
    for t in target_tensors {
        let shp = ws.shape(t)?;
        if shp.len() != 2 {
            return Err(WeightXError::Value(format!(
                "EAMIT target {t:?} must be 2D, got {shp:?}"
            )));
        }
        shapes.push(shp);
    }
    let (d_out, d_in) = (shapes[shapes.len() - 1][0], shapes[shapes.len() - 1][1]);

    let mut k_base = Array2::<f64>::zeros((d_in, num_facts));
    let mut v_target = Array2::<f64>::zeros((d_out, num_facts));
    for (m, f) in facts.iter().enumerate() {
        if f.key_vector.len() != d_in || f.target_value_vector.len() != d_out {
            return Err(WeightXError::Value(format!(
                "Fact {} dims mismatch: key len {} (need {d_in}), value len {} (need {d_out})",
                f.fact_id,
                f.key_vector.len(),
                f.target_value_vector.len()
            )));
        }
        for i in 0..d_in {
            k_base[[i, m]] = f.key_vector[i];
        }
        for i in 0..d_out {
            v_target[[i, m]] = f.target_value_vector[i];
        }
    }

    let cumulative_output = |ws: &WeightWorkspace| -> Result<Array2<f64>> {
        let mut total = Array2::<f64>::zeros((d_out, num_facts));
        for tname in target_tensors {
            let w_l = as_f64_2d(ws.get_readonly(tname)?);
            if w_l.nrows() == d_out && w_l.ncols() == d_in {
                total = total + matmul(&w_l, &k_base);
            }
        }
        Ok(total)
    };

    let init_out = cumulative_output(ws)?;
    let raw_residual = &v_target - &init_out;
    let pre_mse = raw_residual.iter().map(|x| x * x).sum::<f64>() / raw_residual.len() as f64;

    let (mut remaining, align_stats) = align_residuals_to_keys(&k_base, &raw_residual, align_cfg)?;
    let mut per_tensor_deltas = HashMap::new();

    for (idx, tname) in target_tensors.iter().enumerate() {
        let w_l_t = ws.get_readonly(tname)?.clone();
        let w_l = as_f64_2d(&w_l_t);
        let (l_out, l_in) = (w_l.nrows(), w_l.ncols());
        if l_out != d_out || l_in != d_in {
            continue;
        }
        let mut k_l = Array2::<f64>::zeros((l_in, num_facts));
        for (m, f) in facts.iter().enumerate() {
            let kv = f.layer_inputs.get(tname).unwrap_or(&f.key_vector);
            if kv.len() != l_in {
                return Err(WeightXError::Value(format!(
                    "layer_inputs for {tname} fact {} has wrong length",
                    f.fact_id
                )));
            }
            for i in 0..l_in {
                k_l[[i, m]] = kv[i];
            }
        }

        let layers_left = (num_layers - idx) as f64;
        let layer_target = remaining.mapv(|x| x / layers_left);

        let mut fisher_diag = vec![0.0f64; l_in];
        if apply_fisher_guard {
            if let Some(guard) = ws.fisher_guard.as_mut() {
                let f_mat = guard.get_fisher(tname, &w_l_t);
                let f2d = as_f64_2d(&f_mat);
                let col_f: Vec<f64> = (0..l_in)
                    .map(|j| f2d.column(j).iter().sum::<f64>() / l_out as f64)
                    .collect();
                let mean_cf = linalg::mean(&col_f);
                if mean_cf > 1e-20 {
                    for j in 0..l_in {
                        fisher_diag[j] = 0.05 * lambda_cov * (col_f[j] / mean_cf);
                    }
                }
            }
        }

        let mut gram = matmul(&k_l, &transpose(&k_l)) + eye(l_in).mapv(|x| x * lambda_cov);
        for j in 0..l_in {
            gram[[j, j]] += fisher_diag[j];
        }
        let gram_inv = inv(&gram).ok_or_else(|| {
            WeightXError::Value(format!("Singular gram matrix in EAMIT for {tname}"))
        })?;
        let delta_w = matmul(&matmul(&layer_target, &transpose(&k_l)), &gram_inv);

        let w_proposed = (&w_l + &delta_w).mapv(|x| x as f32);
        let w_proposed_t = w_proposed
            .into_dyn()
            .into_shape_with_order(w_l_t.shape())
            .unwrap()
            .to_owned();
        ws.set_tensor(tname, &w_proposed_t, apply_fisher_guard, false)?;
        let w_actual = as_f64_2d(ws.get_readonly(tname)?);
        let actual_delta = &w_actual - &w_l;
        per_tensor_deltas.insert(tname.clone(), linalg::l2_norm_flat(&actual_delta));

        remaining = &remaining - matmul(&actual_delta, &k_l);
    }

    let final_out = cumulative_output(ws)?;
    let post_res = &v_target - &final_out;
    let post_mse = post_res.iter().map(|x| x * x).sum::<f64>() / post_res.len() as f64;
    let rel_red = (pre_mse - post_mse) / pre_mse.max(1e-12);

    Ok(EamitResult {
        num_facts,
        target_tensors: target_tensors.to_vec(),
        pre_edit_mse: pre_mse,
        post_edit_mse: post_mse,
        relative_error_reduction: rel_red,
        pre_alignment_kl: align_stats.pre_kl,
        post_alignment_kl: align_stats.post_kl,
        pre_alignment_top_m_mse: align_stats.pre_top_m_mse,
        post_alignment_top_m_mse: align_stats.post_top_m_mse,
        per_tensor_l2_deltas: per_tensor_deltas,
        fisher_protected: apply_fisher_guard && ws.fisher_guard.is_some(),
    })
}

/// Helper: build facts from JSON `[{id, key, value}, ...]`.
pub fn facts_from_json(facts: &[Value]) -> Result<Vec<BatchFactSpec>> {
    facts
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let id = f
                .get("id")
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_else(|| idx.to_string());
            let key = json_f64_vec(f.get("key"))?;
            let value = json_f64_vec(f.get("value"))?;
            Ok(BatchFactSpec::new(id, key, value))
        })
        .collect()
}

pub fn json_f64_vec(v: Option<&Value>) -> Result<Vec<f64>> {
    fn flatten(v: &Value, out: &mut Vec<f64>) -> Result<()> {
        match v {
            Value::Array(a) => {
                for x in a {
                    flatten(x, out)?;
                }
                Ok(())
            }
            Value::Number(n) => {
                out.push(n.as_f64().ok_or_else(|| WeightXError::Value("Expected numeric array element".into()))?);
                Ok(())
            }
            Value::String(s) => {
                out.push(s.trim().parse::<f64>().map_err(|_| WeightXError::Value("Expected numeric array element".into()))?);
                Ok(())
            }
            _ => Err(WeightXError::Value("Expected numeric array".into())),
        }
    }
    let mut out = Vec::new();
    match v {
        Some(v) => flatten(v, &mut out)?,
        None => return Err(WeightXError::Value("Expected numeric array".into())),
    }
    Ok(out)
}

pub fn json_f64_matrix(v: &Value) -> Result<Array2<f64>> {
    let rows = v
        .as_array()
        .ok_or_else(|| WeightXError::Value("Expected matrix (array of arrays)".into()))?;
    let mut data = Vec::new();
    let mut ncols = 0;
    for row in rows {
        let r = json_f64_vec(Some(row))?;
        ncols = r.len();
        data.extend(r);
    }
    Ok(Array2::from_shape_vec((rows.len(), ncols), data)
        .map_err(|e| WeightXError::Value(e.to_string()))?)
}
