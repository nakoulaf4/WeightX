// SPDX-License-Identifier: GPL-3.0-or-later
//! Closed-form EAMIT domain calibration on real text.
//! Rust port of `weightx.l3_weights.domain_eamit`.
//!
//! Aligns token residuals `R_l = -dL/dO_l` to token keys `K_l` via EAMIT
//! (KL divergence over cosine similarity distributions + top-M MSE), then solves
//! `dW_l = R_l^T K_l (K_l^T K_l + lambda * N * C0_l + eps*I)^-1` with a line
//! search on the practice split under a control-NLL budget; Vespasian then
//! verifies on the hidden held-out split.

use std::collections::HashMap;

use ndarray::Array2;
use serde_json::{json, Value};

use crate::eamit::{self, EamitAlignConfig};
use crate::error::Result;
use crate::linalg::{self, eye, matmul, solve, transpose};
use crate::model::{cross_entropy_rows, ModelArch};
use crate::workspace::WeightWorkspace;

#[derive(Debug, Clone)]
pub struct DomainCalibrationStep {
    pub iteration: usize,
    pub layers: Vec<String>,
    pub chosen_rel_step: f64,
    pub practice_nll_before: f64,
    pub practice_nll_after: f64,
    pub control_nll_before: f64,
    pub control_nll_after: f64,
    pub alignment_kl_before: f64,
    pub alignment_kl_after: f64,
    pub alignment_top_m_mse_before: f64,
    pub alignment_top_m_mse_after: f64,
}

impl DomainCalibrationStep {
    pub fn to_value(&self) -> Value {
        let r5 = |v: f64| (v * 1e5).round() / 1e5;
        let r6 = |v: f64| (v * 1e6).round() / 1e6;
        json!({
            "iteration": self.iteration,
            "layers": self.layers,
            "chosen_rel_step": self.chosen_rel_step,
            "practice_nll_before": r5(self.practice_nll_before),
            "practice_nll_after": r5(self.practice_nll_after),
            "control_nll_before": r5(self.control_nll_before),
            "control_nll_after": r5(self.control_nll_after),
            "alignment_kl_before": r6(self.alignment_kl_before),
            "alignment_kl_after": r6(self.alignment_kl_after),
            "alignment_top_m_mse_before": r6(self.alignment_top_m_mse_before),
            "alignment_top_m_mse_after": r6(self.alignment_top_m_mse_after),
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct DomainCalibrationResult {
    pub target_tensors: Vec<String>,
    pub steps: Vec<DomainCalibrationStep>,
}

impl DomainCalibrationResult {
    pub fn to_value(&self) -> Value {
        json!({
            "target_tensors": self.target_tensors,
            "steps": self.steps.iter().map(|s| s.to_value()).collect::<Vec<_>>(),
        })
    }
}

/// Mean per-token NLL over all next-token positions + per-sequence NLLs.
pub fn token_nll(
    ws: &WeightWorkspace,
    arch: &dyn ModelArch,
    token_seqs: &[Vec<i64>],
) -> Result<(f64, Vec<f64>)> {
    let mut total = 0.0f64;
    let mut count = 0usize;
    let mut per_seq = Vec::new();
    for seq in token_seqs {
        if seq.len() < 2 {
            per_seq.push(0.0);
            continue;
        }
        let ids = Array2::from_shape_vec((1, seq.len()), seq.clone()).unwrap();
        let out = arch.forward(ws, &ids, false)?;
        let s = seq.len();
        let logits = out.logits.slice(ndarray::s![0, ..s - 1, ..]).to_owned();
        let targets: Vec<i64> = seq[1..].to_vec();
        let losses = cross_entropy_rows(&logits, &targets);
        let loss: f64 = losses.iter().map(|v| *v as f64).sum();
        let n = s - 1;
        total += loss;
        count += n;
        per_seq.push(loss / n.max(1) as f64);
    }
    Ok((total / count.max(1) as f64, per_seq))
}

/// Top-1 next-token accuracy over sequences.
pub fn top1_accuracy(ws: &WeightWorkspace, arch: &dyn ModelArch, seqs: &[Vec<i64>]) -> Result<f64> {
    let mut hit = 0usize;
    let mut tot = 0usize;
    for seq in seqs {
        if seq.len() < 2 {
            continue;
        }
        let ids = Array2::from_shape_vec((1, seq.len()), seq.clone()).unwrap();
        let out = arch.forward(ws, &ids, false)?;
        let s = seq.len();
        let logits = out.logits.slice(ndarray::s![0, ..s - 1, ..]).to_owned();
        for (i, &tgt) in seq[1..].iter().enumerate() {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (j, &v) in logits.row(i).iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = j;
                }
            }
            if best as i64 == tgt {
                hit += 1;
            }
            tot += 1;
        }
    }
    Ok(hit as f64 / tot.max(1) as f64)
}

/// Pick `n_layers` MLP output projections spread over the upper-middle depth band.
pub fn select_edit_layers(arch: &dyn ModelArch, n_layers: usize, lo: f64, hi: f64) -> Vec<String> {
    let mods = arch.mlp_out_names();
    if mods.is_empty() {
        return vec![];
    }
    let total = mods.len();
    let start = (lo * (total as f64 - 1.0)) as usize;
    let end = ((hi * (total as f64 - 1.0)) as usize).max(start);
    let count = n_layers.min(end - start + 1);
    let mut idxs: Vec<usize> = (0..count)
        .map(|i| {
            let t = if count == 1 { 0.0 } else { i as f64 / (count - 1) as f64 };
            (start as f64 + t * (end - start) as f64).round() as usize
        })
        .collect();
    idxs.sort_unstable();
    idxs.dedup();
    idxs.into_iter().map(|i| mods[i].clone()).collect()
}

pub struct DomainCalibOptions {
    pub layer_names: Option<Vec<String>>,
    pub n_layers: usize,
    pub lambda_preserve: f64,
    pub rel_steps: Vec<f64>,
    pub max_control_nll_increase: f64,
    pub iterations: usize,
    pub apply_fisher_guard: bool,
    pub align_config: EamitAlignConfig,
}

impl Default for DomainCalibOptions {
    fn default() -> Self {
        Self {
            layer_names: None,
            n_layers: 4,
            lambda_preserve: 1.0,
            rel_steps: vec![0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075],
            max_control_nll_increase: 0.01,
            iterations: 1,
            apply_fisher_guard: true,
            align_config: EamitAlignConfig {
                align_steps: 16,
                ..EamitAlignConfig::default()
            },
        }
    }
}

/// Run one or more closed-form EAMIT domain calibration iterations.
pub fn eamit_domain_calibration(
    ws: &mut WeightWorkspace,
    arch: &dyn ModelArch,
    practice_seqs: &[Vec<i64>],
    control_seqs: &[Vec<i64>],
    opts: &DomainCalibOptions,
) -> Result<DomainCalibrationResult> {
    let mod_names = opts
        .layer_names
        .clone()
        .unwrap_or_else(|| select_edit_layers(arch, opts.n_layers, 0.45, 0.9));
    let mut result = DomainCalibrationResult { target_tensors: mod_names.clone(), steps: vec![] };
    if mod_names.is_empty() {
        return Ok(result);
    }

    let (ctrl_keys, _) = arch.mlp_keys_grads(ws, control_seqs, false)?;

    for it in 0..opts.iterations {
        let (prac_nll0, _) = token_nll(ws, arch, practice_seqs)?;
        let (ctrl_nll0, _) = token_nll(ws, arch, control_seqs)?;
        let (k_prac, g_prac) = arch.mlp_keys_grads(ws, practice_seqs, true)?;

        let mut directions: HashMap<String, Array2<f64>> = HashMap::new();
        let mut base_weights: HashMap<String, Array2<f64>> = HashMap::new();
        let mut kl_before = Vec::new();
        let mut kl_after = Vec::new();
        let mut tmse_before = Vec::new();
        let mut tmse_after = Vec::new();

        for tn in &mod_names {
            let (Some(k_mat), Some(g)) = (k_prac.get(tn), g_prac.get(tn)) else { continue };
            let Some(kc) = ctrl_keys.get(tn) else { continue };
            let w_t = ws.get_readonly(tn)?;
            if w_t.ndim() != 2 {
                continue;
            }
            let w0 = linalg::to_f64_2d(w_t);
            let r_raw = g.mapv(|v| -v); // R_raw = -dL/dO
            let n_p = k_mat.nrows();
            let d_in = k_mat.ncols();
            if d_in == 0 || n_p == 0 {
                continue;
            }

            // EAMIT key-residual embedding alignment
            let (r_aligned_t, a_stats) = eamit::align_residuals_to_keys(
                &transpose(k_mat),
                &transpose(&r_raw),
                &opts.align_config,
            )?;
            let r_mat = transpose(&r_aligned_t);

            let c0 = matmul(&transpose(kc), kc).mapv(|v| v / kc.nrows().max(1) as f64);
            let mut a_mat = matmul(&transpose(k_mat), k_mat)
                + c0.mapv(|v| v * opts.lambda_preserve * n_p as f64);
            let ridge = 1e-3 * a_mat.diag().iter().sum::<f64>() / d_in as f64 + 1e-8;
            a_mat = a_mat + eye(d_in).mapv(|v| v * ridge);
            // D_lin = R^T K A^-1  (A symmetric)
            let rhs = transpose(&matmul(&transpose(&r_mat), k_mat)); // [d_in, d_out]
            let Some(sol) = solve(&a_mat, &rhs) else { continue };
            let d_lin = transpose(&sol); // [d_out, d_in]
            // Conv1D (GPT-2) stores weights as [d_in, d_out]: transpose the direction.
            let d_lin = if d_lin.shape() != w0.shape() && transpose(&d_lin).shape() == w0.shape() {
                transpose(&d_lin)
            } else {
                d_lin
            };
            if d_lin.shape() != w0.shape() {
                continue;
            }
            let dn = linalg::l2_norm_flat(&d_lin);
            if dn < 1e-12 {
                continue;
            }
            let scale = linalg::l2_norm_flat(&w0) / dn;
            directions.insert(tn.clone(), d_lin.mapv(|v| v * scale));
            base_weights.insert(tn.clone(), w0);
            kl_before.push(a_stats.pre_kl);
            kl_after.push(a_stats.post_kl);
            tmse_before.push(a_stats.pre_top_m_mse);
            tmse_after.push(a_stats.post_top_m_mse);
        }

        if directions.is_empty() {
            break;
        }

        ws.checkpoint("__domain_calib_pre__");
        let mut best = (0.0f64, prac_nll0, ctrl_nll0);
        for &step in &opts.rel_steps {
            for (tn, dir) in &directions {
                let w0 = &base_weights[tn];
                let new_w = (w0 + dir.mapv(|v| v * step)).mapv(|v| v as f32);
                let shp = ws.shape(tn)?;
                let t = new_w.into_dyn().into_shape_with_order(shp).unwrap().to_owned();
                ws.set_tensor(tn, &t, opts.apply_fisher_guard, true)?;
            }
            let (p_nll, _) = token_nll(ws, arch, practice_seqs)?;
            let (c_nll, _) = token_nll(ws, arch, control_seqs)?;
            if c_nll <= ctrl_nll0 * (1.0 + opts.max_control_nll_increase) && p_nll < best.1 {
                best = (step, p_nll, c_nll);
            }
            ws.restore("__domain_calib_pre__")?;
        }

        if best.0 > 0.0 {
            for (tn, dir) in &directions {
                let w0 = &base_weights[tn];
                let new_w = (w0 + dir.mapv(|v| v * best.0)).mapv(|v| v as f32);
                let shp = ws.shape(tn)?;
                let t = new_w.into_dyn().into_shape_with_order(shp).unwrap().to_owned();
                ws.set_tensor(tn, &t, opts.apply_fisher_guard, true)?;
            }
        }
        result.steps.append(&mut vec![DomainCalibrationStep {
            iteration: it + 1,
            layers: mod_names.clone(),
            chosen_rel_step: best.0,
            practice_nll_before: prac_nll0,
            practice_nll_after: best.1,
            control_nll_before: ctrl_nll0,
            control_nll_after: best.2,
            alignment_kl_before: linalg::mean(&kl_before),
            alignment_kl_after: linalg::mean(&kl_after),
            alignment_top_m_mse_before: linalg::mean(&tmse_before),
            alignment_top_m_mse_after: linalg::mean(&tmse_after),
        }]);
        if best.0 == 0.0 {
            break;
        }
    }
    Ok(result)
}
