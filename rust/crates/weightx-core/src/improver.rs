// SPDX-License-Identifier: GPL-3.0-or-later
//! WeightX Small Model Improver & Benchmark Suite — Rust port of `weightx.improver`.
//!
//! Gradient-free multi-stage pipeline: baseline evaluation -> L1/L2 diagnosis &
//! YAML outlier/spectral repair -> closed-form subspace ridge calibration ->
//! Vespasian held-out verification with automatic rollback.

use std::path::Path;
use std::sync::Arc;

use ndarray::Array2;
use serde_json::{json, Value};

use crate::catalog::{self, ModelCatalog};
use crate::corpora;
use crate::domain::{self, DomainCalibOptions};
use crate::error::{Result, WeightXError};
use crate::fisher::FisherGuard;
use crate::l1;
use crate::l2;
use crate::linalg;
use crate::model::{self, minillama, ModelArch};
use crate::plan;
use crate::studio::{stack_min_len, synthetic_splits};
use crate::tensor::{self, Tensor};
use crate::toon;
use crate::vespasian::{EvalKind, EvaluationSplit, VespasianEngine};
use crate::workspace::WeightWorkspace;

#[derive(Debug, Clone, Copy, Default)]
pub struct LMQualityMetrics {
    pub perplexity: f64,
    pub mean_nll: f64,
    pub top1_accuracy: f64,
    pub top5_accuracy: f64,
    pub vespasian_score: f64,
    pub logit_kurtosis: f64,
}

impl LMQualityMetrics {
    pub fn to_value(&self) -> Value {
        json!({
            "perplexity": (self.perplexity * 1e4).round() / 1e4,
            "mean_nll": (self.mean_nll * 1e5).round() / 1e5,
            "top1_accuracy": (self.top1_accuracy * 1e4).round() / 1e4,
            "top5_accuracy": (self.top5_accuracy * 1e4).round() / 1e4,
            "vespasian_score": (self.vespasian_score * 1e5).round() / 1e5,
            "logit_kurtosis": (self.logit_kurtosis * 1e4).round() / 1e4,
        })
    }
}

#[derive(Debug, Clone)]
pub struct StageImprovementRecord {
    pub stage_name: String,
    pub description: String,
    pub committed: bool,
    pub pre_perplexity: f64,
    pub post_perplexity: f64,
    pub pre_top1_acc: f64,
    pub post_top1_acc: f64,
    pub vespasian_verdict: String,
    pub yaml_plan_used: Option<String>,
}

impl StageImprovementRecord {
    pub fn to_value(&self) -> Value {
        json!({
            "stage_name": self.stage_name,
            "description": self.description,
            "committed": self.committed,
            "pre_perplexity": self.pre_perplexity,
            "post_perplexity": self.post_perplexity,
            "pre_top1_acc": self.pre_top1_acc,
            "post_top1_acc": self.post_top1_acc,
            "vespasian_verdict": self.vespasian_verdict,
            "yaml_plan_used": self.yaml_plan_used,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ModelImprovementReport {
    pub model_id: String,
    pub hf_repo_id: String,
    pub family: String,
    pub total_params: usize,
    pub changed_params: usize,
    pub initial_metrics: LMQualityMetrics,
    pub final_metrics: LMQualityMetrics,
    pub ppl_reduction_pct: f64,
    pub top1_accuracy_gain_pct: f64,
    pub vespasian_gain: f64,
    pub kci_gini: f64,
    pub bottleneck_layers: Vec<String>,
    pub stages: Vec<StageImprovementRecord>,
    pub saved_checkpoint_path: Option<String>,
}

impl ModelImprovementReport {
    pub fn to_value(&self) -> Value {
        json!({
            "model_id": self.model_id,
            "hf_repo_id": self.hf_repo_id,
            "family": self.family,
            "total_params": self.total_params,
            "changed_params": self.changed_params,
            "changed_fraction": ((self.changed_params as f64 / self.total_params.max(1) as f64) * 1e6).round() / 1e6,
            "initial_metrics": self.initial_metrics.to_value(),
            "final_metrics": self.final_metrics.to_value(),
            "ppl_reduction_pct": (self.ppl_reduction_pct * 100.0).round() / 100.0,
            "top1_accuracy_gain_pct": (self.top1_accuracy_gain_pct * 100.0).round() / 100.0,
            "vespasian_gain": (self.vespasian_gain * 1e5).round() / 1e5,
            "kci_gini": (self.kci_gini * 1e4).round() / 1e4,
            "bottleneck_layers": self.bottleneck_layers,
            "stages": self.stages.iter().map(|s| s.to_value()).collect::<Vec<_>>(),
            "saved_checkpoint_path": self.saved_checkpoint_path,
        })
    }
    pub fn to_toon(&self) -> String {
        toon::dumps_toon(&self.to_value())
    }
}

/// Perplexity / NLL / Top-1 / Top-5 / logit kurtosis on a `Targets` split.
pub fn evaluate_lm_quality(
    ws: &WeightWorkspace,
    arch: &dyn ModelArch,
    split: &EvaluationSplit,
) -> Result<LMQualityMetrics> {
    let (input_ids, target_ids) = match &split.kind {
        EvalKind::Targets { input_ids, target_ids } => (input_ids, target_ids),
        _ => return Err(WeightXError::Value("evaluate_lm_quality requires a Targets split".into())),
    };
    let out = arch.forward(ws, input_ids, false)?;
    let s = out.logits.shape()[1];
    let last = out.logits.slice(ndarray::s![.., s - 1, ..]).to_owned();
    let losses = model::cross_entropy_rows(&last, target_ids);
    let nll = losses.iter().map(|v| *v as f64).sum::<f64>() / losses.len().max(1) as f64;
    let ppl = nll.min(20.0).exp();

    let probs = model::softmax_rows(&last);
    let n = target_ids.len();
    let mut top1 = 0usize;
    let mut top5 = 0usize;
    let mut kurt_sum = 0.0f64;
    let k = 5.min(probs.ncols());
    for i in 0..n {
        let row = probs.row(i);
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
        let t = target_ids[i].max(0) as usize;
        if idx[0] == t {
            top1 += 1;
        }
        if idx[..k].contains(&t) {
            top5 += 1;
        }
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / row.len() as f64;
        let var = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / row.len() as f64 + 1e-8;
        let m4 = row.iter().map(|v| (*v as f64 - mean).powi(4)).sum::<f64>() / row.len() as f64;
        kurt_sum += m4 / (var * var) - 3.0;
    }
    let top1_acc = top1 as f64 / n.max(1) as f64;
    let top5_acc = top5 as f64 / n.max(1) as f64;
    Ok(LMQualityMetrics {
        perplexity: ppl,
        mean_nll: nll,
        top1_accuracy: top1_acc,
        top5_accuracy: top5_acc,
        vespasian_score: 0.70 * top1_acc + 0.30 * (-0.25 * nll).exp(),
        logit_kurtosis: kurt_sum / n.max(1) as f64,
    })
}

/// Build next-token prediction windows from real text corpora.
pub fn build_token_splits_from_texts(
    tokenizer: &tokenizers::Tokenizer,
    practice_texts: &[&str],
    held_out_texts: &[&str],
    seq_len: usize,
) -> Result<(EvaluationSplit, EvaluationSplit)> {
    let encode_windows = |texts: &[&str], name: &str| -> Result<EvaluationSplit> {
        let mut inputs: Vec<Vec<i64>> = Vec::new();
        let mut targets: Vec<i64> = Vec::new();
        for txt in texts {
            let enc = tokenizer
                .encode(*txt, false)
                .map_err(|e| WeightXError::Value(format!("tokenizer error: {e}")))?;
            let ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
            if ids.len() <= seq_len {
                let mut padded = vec![0i64; seq_len + 1 - ids.len()];
                padded.extend_from_slice(&ids);
                inputs.push(padded[..seq_len].to_vec());
                targets.push(padded[seq_len]);
            } else {
                let mut start = 0;
                while start + seq_len < ids.len() {
                    let window = &ids[start..start + seq_len + 1];
                    inputs.push(window[..seq_len].to_vec());
                    targets.push(window[seq_len]);
                    start += 2;
                }
            }
        }
        if inputs.is_empty() {
            return Err(WeightXError::Value(format!("no evaluation windows for split {name}")));
        }
        let rows = inputs.len();
        let flat: Vec<i64> = inputs.iter().flatten().copied().collect();
        let x = Array2::from_shape_vec((rows, seq_len), flat)
            .map_err(|e| WeightXError::Value(format!("split shape error: {e}")))?;
        let reference = x.slice(ndarray::s![..8.min(rows), ..]).to_owned();
        Ok(EvaluationSplit::targets(name, x, targets, Some(reference)))
    };
    Ok((
        encode_windows(practice_texts, "practice_text")?,
        encode_windows(held_out_texts, "held_out_text")?,
    ))
}

/// Closed-form ridge subspace alignment on the LM head / final projection.
fn closed_form_lm_head_calibration(
    ws: &mut WeightWorkspace,
    arch: &dyn ModelArch,
    practice_split: &EvaluationSplit,
    ridge_lambda: f64,
    step_size: f64,
) -> Result<(String, usize)> {
    let (input_ids, target_ids) = match &practice_split.kind {
        EvalKind::Targets { input_ids, target_ids } => (input_ids.clone(), target_ids.clone()),
        _ => return Err(WeightXError::Value("calibration requires a Targets split".into())),
    };
    let names = ws.all_names();
    let mut candidates: Vec<String> = names.iter()
        .filter(|n| (n.contains("lm_head") || n.contains("embed_out"))
            && ws.shape(n).map(|s| s.len() == 2).unwrap_or(false))
        .cloned().collect();
    if candidates.is_empty() {
        candidates = names.iter()
            .filter(|n| ws.shape(n).map(|s| s.len() == 2).unwrap_or(false))
            .cloned().collect();
    }
    let target_name = candidates.last()
        .ok_or_else(|| WeightXError::Value("no 2D output projection found".into()))?
        .clone();

    let out = arch.forward(ws, &input_ids, true)?;
    let s = out.logits.shape()[1];
    let last_logits = out.logits.slice(ndarray::s![.., s - 1, ..]).to_owned();
    let Some(hidden) = out.hidden_states.last() else {
        return Ok((target_name, 0));
    };
    let hs = hidden.shape()[1];
    let mut h_last = hidden.slice(ndarray::s![.., hs - 1, ..]).to_owned();
    // Apply the final norm when the target is the vocabulary projection.
    if target_name.contains("lm_head") || target_name.contains("embed_out") {
        if let Ok(norm_w) = model::weight_1d(ws, "model.norm.weight") {
            let eps = arch.config().rms_norm_eps as f32;
            h_last = model::rmsnorm(&h_last, &norm_w, eps);
        } else if let Ok(ln_w) = model::weight_1d(ws, "transformer.ln_f.weight") {
            let ln_b = model::weight_1d(ws, "transformer.ln_f.bias").unwrap_or_else(|_| vec![0.0; ln_w.len()]);
            let eps = arch.config().layer_norm_eps as f32;
            h_last = model::layernorm(&h_last, &ln_w, &ln_b, eps);
        }
    }

    let probs = model::softmax_rows(&last_logits);
    let b = input_ids.nrows();
    let v = probs.ncols();
    let mut residual = Array2::<f64>::zeros((b, v));
    for i in 0..b {
        for j in 0..v {
            residual[[i, j]] = -probs[[i, j]] as f64;
        }
        let t = target_ids[i].max(0) as usize;
        if t < v {
            residual[[i, t]] += 1.0;
        }
    }
    let h_mat: Array2<f64> = h_last.mapv(|x| x as f64);
    let w_t = ws.get_readonly(&target_name)?.clone();
    let w = linalg::to_f64_2d(&w_t);
    let d = h_mat.ncols();
    let cov = linalg::matmul(&linalg::transpose(&h_mat), &h_mat) + linalg::eye(d) * ridge_lambda;
    let cov_inv = match linalg::inv(&cov) {
        Some(m) => m,
        None => return Ok((target_name, 0)),
    };
    let delta_w = if w.nrows() == v && w.ncols() == d {
        linalg::matmul(&linalg::matmul(&linalg::transpose(&residual), &h_mat), &cov_inv) * step_size
    } else if w.nrows() == d && w.ncols() == v {
        linalg::matmul(&cov_inv, &linalg::matmul(&linalg::transpose(&h_mat), &residual)) * step_size
    } else {
        return Ok((target_name, 0));
    };
    let updated = &w + &delta_w;
    let new_t = tensor::from_vec(w_t.shape(), updated.iter().map(|&x| x as f32).collect());
    ws.set_tensor(&target_name, &new_t, true, true)?;
    Ok((target_name, delta_w.len()))
}

/// End-to-end pipeline: benchmark, diagnose, repair and improve a small LLM.
pub fn improve_small_model(
    model_id_or_repo: &str,
    inject_realistic_corruption: bool,
    save_dir: Option<&Path>,
    seed: u64,
) -> Result<ModelImprovementReport> {
    let mut catalog = ModelCatalog::new();
    let mut practice_split: EvaluationSplit;
    let held_out_split: EvaluationSplit;
    let calib_ids: Array2<i64>;
    let arch: Option<Arc<dyn ModelArch>>;
    let mut tensors: Vec<(String, Tensor)>;
    let entry;
    let tokenizer: Option<tokenizers::Tokenizer>;
    let bos_token_id: Option<i64>;
    let dtypes: std::collections::HashMap<String, String>;

    if model_id_or_repo == "minillama-demo" {
        let bundle = minillama::build_corrupted_model_and_splits(seed)?;
        entry = catalog.get("minillama-demo")
            .ok_or_else(|| WeightXError::Value("minillama-demo entry missing".into()))?;
        practice_split = bundle.practice;
        held_out_split = bundle.held_out;
        calib_ids = bundle.calib_ids;
        arch = Some(Arc::new(bundle.model));
        tensors = bundle.tensors;
        tokenizer = None;
        bos_token_id = None;
        dtypes = std::collections::HashMap::new();
    } else {
        let lm = catalog.load_model(model_id_or_repo)?;
        entry = lm.entry.clone();
        arch = lm.arch.clone();
        tensors = lm.tensors;
        tokenizer = lm.tokenizer.clone();
        bos_token_id = lm.bos_token_id;
        dtypes = lm.dtypes.clone();
        if let Some(tok) = &lm.tokenizer {
            let (p, h) = build_token_splits_from_texts(
                tok,
                corpora::DEFAULT_PRACTICE_TEXTS,
                corpora::DEFAULT_HELD_OUT_TEXTS,
                8,
            )?;
            practice_split = p;
            held_out_split = h;
            if inject_realistic_corruption {
                for (name, t) in tensors.iter_mut() {
                    if t.ndim() == 2 && (name.contains("mlp") || name.contains("c_proj") || name.contains("fc")) {
                        let rows = t.shape()[0];
                        let cols = t.shape()[1];
                        let mut m2 = tensor::as_2d(t);
                        for j in 0..cols {
                            m2[[0, j]] *= 8.5;
                            if rows > 1 {
                                m2[[rows - 1, j]] *= 7.5;
                            }
                        }
                        *t = m2.into_dyn();
                        break;
                    }
                }
            }
            calib_ids = match &practice_split.kind {
                EvalKind::Targets { input_ids, .. } => input_ids.slice(ndarray::s![..6.min(input_ids.nrows()), ..]).to_owned(),
                _ => Array2::zeros((0, 0)),
            };
        } else {
            let (p, h, c) = synthetic_splits(arch.as_deref(), &tensors);
            practice_split = p;
            held_out_split = h;
            calib_ids = c;
        }
    }

    let Some(arch) = arch else {
        return Err(WeightXError::Value("improve_small_model requires an inferrable model architecture".into()));
    };

    let guard = FisherGuard::new(Default::default(), 0.15, 0.05, 2.5, true);
    let mut workspace = WeightWorkspace::from_tensors_with_dtypes(tensors, dtypes, Some(guard));
    if calib_ids.ncols() >= 2 {
        if let Ok(fmap) = crate::fisher::compute_empirical_fisher(&workspace, arch.as_ref(), &[calib_ids.clone()]) {
            if let Some(g) = workspace.fisher_guard.as_mut() {
                g.fisher_map = fmap;
            }
        }
    }
    let mut vespasian = VespasianEngine::new(
        &mut workspace,
        Some(arch.as_ref()),
        Some(practice_split.try_clone().expect("data-backed split")),
        Some(held_out_split.try_clone().expect("data-backed split")),
        0.0005,
        5.0,
        0.25,
        8,
    );

    let initial_metrics = evaluate_lm_quality(&workspace, arch.as_ref(), &held_out_split)?;
    let arch_report = l1::inspect_workspace(&workspace);
    let uni_report = l2::profile_model_layers(&workspace, Some(arch.as_ref()), Some(&calib_ids), "target_model", 2.5);
    let kmap_probes = tokenizer.as_ref().map(|t| l2::knowledge_probes_from_tokenizer(t, bos_token_id));
    let kmap_report = l2::run_knowledge_map(&workspace, Some(arch.as_ref()), kmap_probes.as_deref(), None);

    let mut stages: Vec<StageImprovementRecord> = Vec::new();

    // Stage 1: declarative YAML outlier & spectral spike repair.
    let pre_s1 = evaluate_lm_quality(&workspace, arch.as_ref(), &held_out_split)?;
    let mut outlier_tensors: Vec<String> = Vec::new();
    for on in &uni_report.outlier_neurons {
        if !on.tensor_name.is_empty() && !outlier_tensors.contains(&on.tensor_name) {
            outlier_tensors.push(on.tensor_name.clone());
        }
    }
    if outlier_tensors.is_empty() {
        outlier_tensors = workspace.all_names().into_iter()
            .filter(|n| workspace.shape(n).map(|s| s.len() == 2).unwrap_or(false)
                && (n.contains("mlp") || n.contains("c_proj") || n.contains("down_proj")))
            .take(2)
            .collect();
    }
    if let Some(target_t) = outlier_tensors.first().cloned() {
        let t_shape = workspace.shape(&target_t)?;
        let t_dtype = workspace.dtype(&target_t)?;
        let yaml_plan = format!(
            "transforms:\n  - assert:\n      all:\n        - exists: {{ of: '{target_t}' }}\n        - shape: {{ of: '{target_t}', is: {t_shape:?} }}\n        - dtype: {{ of: '{target_t}', is: '{t_dtype}' }}\n        - finite: {{ of: '{target_t}' }}\n  - rescale_outlier_rows_:\n      target: '{target_t}'\n      min_ratio: 2.2\n  - spectral_norm_cap_:\n      target: '{target_t}'\n      max_median_ratio: 3.5\n  - assert:\n      finite: {{ of: '{target_t}' }}\n"
        );
        plan::execute_yaml_plan(&mut workspace, &yaml_plan, false, true, true);
        let v1 = vespasian.verify_and_decide(
            &mut workspace,
            Some(arch.as_ref()),
            &format!("Stage 1: Outlier & Spectral Repair on {target_t}"),
        );
        let post_s1 = evaluate_lm_quality(&workspace, arch.as_ref(), &held_out_split)?;
        stages.push(StageImprovementRecord {
            stage_name: "Stage 1: BrainSurgery-WX Outlier & Spectral Repair".into(),
            description: format!("Rescaled outlier rows and capped spectral spikes in {target_t}"),
            committed: v1.accepted,
            pre_perplexity: pre_s1.perplexity,
            post_perplexity: post_s1.perplexity,
            pre_top1_acc: pre_s1.top1_accuracy,
            post_top1_acc: post_s1.top1_accuracy,
            vespasian_verdict: v1.verdict_reason,
            yaml_plan_used: Some(yaml_plan),
        });
    }

    // Stage 2: closed-form subspace ridge calibration with Fisher protection.
    let pre_s2 = evaluate_lm_quality(&workspace, arch.as_ref(), &held_out_split)?;
    let (calib_target, _n) = closed_form_lm_head_calibration(
        &mut workspace, arch.as_ref(), &practice_split, 2.5, 0.30,
    )?;
    let v2 = vespasian.verify_and_decide(
        &mut workspace,
        Some(arch.as_ref()),
        &format!("Stage 2: Closed-Form Subspace Calibration on {calib_target}"),
    );
    let post_s2 = evaluate_lm_quality(&workspace, arch.as_ref(), &held_out_split)?;
    stages.push(StageImprovementRecord {
        stage_name: "Stage 2: Closed-Form Subspace Ridge Calibration".into(),
        description: format!("Aligned output subspace on {calib_target} with Fisher guard"),
        committed: v2.accepted,
        pre_perplexity: pre_s2.perplexity,
        post_perplexity: post_s2.perplexity,
        pre_top1_acc: pre_s2.top1_accuracy,
        post_top1_acc: post_s2.top1_accuracy,
        vespasian_verdict: v2.verdict_reason,
        yaml_plan_used: None,
    });

    let final_metrics = evaluate_lm_quality(&workspace, arch.as_ref(), &held_out_split)?;
    let diff = workspace.diff();
    let ppl_red = (initial_metrics.perplexity - final_metrics.perplexity)
        / initial_metrics.perplexity.max(1e-9) * 100.0;
    let acc_gain = (final_metrics.top1_accuracy - initial_metrics.top1_accuracy) * 100.0;
    let vesp_gain = final_metrics.vespasian_score - initial_metrics.vespasian_score;

    let mut saved_path_str = None;
    if let Some(dir) = save_dir {
        std::fs::create_dir_all(dir)?;
        let safe_id = entry.model_id.replace(':', "_").replace('/', "_");
        let out_file = dir.join(format!("{safe_id}_improved.safetensors"));
        workspace.export_safetensors(&out_file, false)?;
        saved_path_str = Some(out_file.display().to_string());
    }

    Ok(ModelImprovementReport {
        model_id: entry.model_id,
        hf_repo_id: entry.hf_repo_id,
        family: arch_report.family,
        total_params: diff.total_params,
        changed_params: diff.changed_params,
        initial_metrics,
        final_metrics,
        ppl_reduction_pct: ppl_red,
        top1_accuracy_gain_pct: acc_gain,
        vespasian_gain: vesp_gain,
        kci_gini: kmap_report.kci_overall,
        bottleneck_layers: uni_report.bottleneck_layers,
        stages,
        saved_checkpoint_path: saved_path_str,
    })
}

// ===========================================================================
// Real pretrained LM improvement (HuggingFace models with a tokenizer)
// ===========================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct TextQualityMetrics {
    pub held_out_ppl: f64,
    pub held_out_top1: f64,
    pub control_ppl: f64,
    pub control_top1: f64,
}

impl TextQualityMetrics {
    pub fn to_value(&self) -> Value {
        json!({
            "held_out_ppl": (self.held_out_ppl * 1e4).round() / 1e4,
            "held_out_top1": (self.held_out_top1 * 1e4).round() / 1e4,
            "control_ppl": (self.control_ppl * 1e4).round() / 1e4,
            "control_top1": (self.control_top1 * 1e4).round() / 1e4,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PretrainedImprovementReport {
    pub model_id: String,
    pub hf_repo_id: String,
    pub total_params: usize,
    pub changed_params: usize,
    pub edited_tensors: Vec<String>,
    pub before: TextQualityMetrics,
    pub after: TextQualityMetrics,
    pub held_out_ppl_reduction_pct: f64,
    pub control_ppl_change_pct: f64,
    pub vespasian_verdicts: Vec<String>,
    pub calibration_steps: Vec<Value>,
    pub saved_checkpoint_path: Option<String>,
}

impl PretrainedImprovementReport {
    pub fn to_value(&self) -> Value {
        json!({
            "model_id": self.model_id,
            "hf_repo_id": self.hf_repo_id,
            "total_params": self.total_params,
            "changed_params": self.changed_params,
            "changed_fraction": ((self.changed_params as f64 / self.total_params.max(1) as f64) * 1e6).round() / 1e6,
            "edited_tensors": self.edited_tensors,
            "before": self.before.to_value(),
            "after": self.after.to_value(),
            "held_out_ppl_reduction_pct": (self.held_out_ppl_reduction_pct * 100.0).round() / 100.0,
            "control_ppl_change_pct": (self.control_ppl_change_pct * 100.0).round() / 100.0,
            "vespasian_verdicts": self.vespasian_verdicts,
            "calibration": {"steps": self.calibration_steps},
            "saved_checkpoint_path": self.saved_checkpoint_path,
        })
    }
    pub fn to_toon(&self) -> String {
        toon::dumps_toon(&self.to_value())
    }
}

fn text_metrics(
    ws: &WeightWorkspace,
    arch: &dyn ModelArch,
    held: &[Vec<i64>],
    ctrl: &[Vec<i64>],
) -> Result<TextQualityMetrics> {
    let (h_nll, _) = domain::token_nll(ws, arch, held)?;
    let (c_nll, _) = domain::token_nll(ws, arch, ctrl)?;
    Ok(TextQualityMetrics {
        held_out_ppl: h_nll.exp(),
        held_out_top1: domain::top1_accuracy(ws, arch, held)?,
        control_ppl: c_nll.exp(),
        control_top1: domain::top1_accuracy(ws, arch, ctrl)?,
    })
}

/// Measurably improve a real pretrained causal LM on a target domain without SGD.
pub fn improve_pretrained_lm(
    model_id_or_repo: &str,
    iterations: usize,
    n_layers: usize,
    max_control_ppl_increase: f64,
    save_dir: Option<&Path>,
    log: &dyn Fn(&str),
) -> Result<PretrainedImprovementReport> {
    let mut catalog = ModelCatalog::new();
    log(&format!("[load] {model_id_or_repo}"));
    let lm = catalog.load_model(model_id_or_repo)?;
    let Some(tok) = &lm.tokenizer else {
        return Err(WeightXError::Value("improve_pretrained_lm requires a HuggingFace model with a tokenizer".into()));
    };
    let Some(arch) = lm.arch.clone() else {
        return Err(WeightXError::Value("cannot infer model architecture".into()));
    };
    let prac = catalog::tokenize_texts(tok, lm.bos_token_id, corpora::WSP_PRACTICE)?;
    let held = catalog::tokenize_texts(tok, lm.bos_token_id, corpora::WSP_HELD_OUT)?;
    let ctrl = catalog::tokenize_texts(tok, lm.bos_token_id, corpora::GENERAL_CONTROL)?;

    let mut workspace = WeightWorkspace::from_tensors(lm.tensors, None);
    let before = text_metrics(&workspace, arch.as_ref(), &held, &ctrl)?;
    log(&format!("[baseline] {}", toon::dumps_toon(&before.to_value())));

    let ctrl_ref = stack_min_len(&ctrl);
    let held_split = EvaluationSplit::seq_nll("held_out_hidden", held.clone(), ctrl_ref);
    let mut vespasian = VespasianEngine::new(
        &mut workspace,
        Some(arch.as_ref()),
        None,
        Some(held_split),
        1e-4,
        0.05,
        0.10,
        8,
    );

    let mut verdicts = Vec::new();
    let mut calib_log: Vec<Value> = Vec::new();
    let mut edited: Vec<String> = Vec::new();
    for it in 0..iterations {
        let opts = DomainCalibOptions {
            n_layers,
            iterations: 1,
            max_control_nll_increase: max_control_ppl_increase / 2.0,
            apply_fisher_guard: false,
            ..Default::default()
        };
        let res = domain::eamit_domain_calibration(&mut workspace, arch.as_ref(), &prac, &ctrl, &opts)?;
        edited = res.target_tensors.clone();
        for s in &res.steps {
            let mut v = s.to_value();
            v["iteration"] = json!(it + 1);
            calib_log.push(v);
        }
        let last = res.steps.last();
        if last.map(|s| s.chosen_rel_step == 0.0).unwrap_or(true) {
            log(&format!("[iter {}] line search found no safe improving step; stopping", it + 1));
            break;
        }
        let st = last.unwrap();
        log(&format!(
            "[iter {}] step={} practice NLL {:.4}->{:.4}, control NLL {:.4}->{:.4}",
            it + 1, st.chosen_rel_step, st.practice_nll_before, st.practice_nll_after,
            st.control_nll_before, st.control_nll_after
        ));
        let v = vespasian.verify_and_decide(
            &mut workspace,
            Some(arch.as_ref()),
            &format!("EAMIT domain calibration iteration {}", it + 1),
        );
        verdicts.push(v.verdict_reason.clone());
        log(&format!("[vespasian] {}", v.verdict_reason));
        if !v.accepted {
            break;
        }
    }

    let after = text_metrics(&workspace, arch.as_ref(), &held, &ctrl)?;
    log(&format!("[final] {}", toon::dumps_toon(&after.to_value())));
    let diff = workspace.diff();

    let mut saved = None;
    if let Some(dir) = save_dir {
        if diff.changed_params > 0 {
            std::fs::create_dir_all(dir)?;
            let safe = lm.entry.model_id.replace('/', "_").replace(':', "_");
            let out = dir.join(format!("{safe}_weightx_improved.safetensors"));
            workspace.export_safetensors(&out, true)?;
            saved = Some(out.display().to_string());
        }
    }

    Ok(PretrainedImprovementReport {
        model_id: lm.entry.model_id,
        hf_repo_id: lm.entry.hf_repo_id,
        total_params: diff.total_params,
        changed_params: diff.changed_params,
        edited_tensors: edited,
        before,
        after,
        held_out_ppl_reduction_pct: (before.held_out_ppl - after.held_out_ppl) / before.held_out_ppl * 100.0,
        control_ppl_change_pct: (after.control_ppl - before.control_ppl) / before.control_ppl * 100.0,
        vespasian_verdicts: verdicts,
        calibration_steps: calib_log,
        saved_checkpoint_path: saved,
    })
}