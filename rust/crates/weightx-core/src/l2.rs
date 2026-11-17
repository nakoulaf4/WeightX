// SPDX-License-Identifier: GPL-3.0-or-later
//! L2 profiling: Uni-Layer 13-metric profiler + Knowledge Map / Gini KCI.
//! Rust port of `weightx.l2_profile.unilayer` and `weightx.l2_profile.knowledge_map`.

use std::collections::HashMap;

use ndarray::{Array2, Array3};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use regex::Regex;
use serde_json::{json, Value};

use crate::linalg;
use crate::model::{softmax_rows, ModelArch};
use crate::tensor::{self, Tensor};
use crate::workspace::WeightWorkspace;

pub const METRIC_NAMES_13: [&str; 13] = [
    "effective_rank",
    "stable_rank",
    "weight_l2_norm",
    "weight_sparsity",
    "weight_outlier_ratio",
    "weight_kurtosis",
    "condition_number",
    "block_influence",
    "activation_entropy",
    "neuron_outlier_ratio",
    "fisher_information",
    "gradient_norm",
    "wanda_score",
];

#[derive(Debug, Clone)]
pub struct OutlierNeuronRecord {
    pub layer: String,
    pub tensor_name: String,
    pub row_indices: Vec<usize>,
    pub max_ratio_to_median: f64,
    pub median_row_norm: f64,
}

impl OutlierNeuronRecord {
    pub fn to_value(&self) -> Value {
        json!({
            "layer": self.layer,
            "tensor_name": self.tensor_name,
            "row_indices": self.row_indices,
            "max_ratio_to_median": (self.max_ratio_to_median * 1e3).round() / 1e3,
            "median_row_norm": (self.median_row_norm * 1e5).round() / 1e5,
        })
    }
}

#[derive(Debug, Clone)]
pub struct UniLayerProfileReport {
    pub model_name: String,
    pub num_layers: usize,
    pub summary: String,
    pub redundant_layers: Vec<String>,
    pub bottleneck_layers: Vec<String>,
    pub consensus_ranking: Vec<(String, usize, f64)>,
    pub outlier_neurons: Vec<OutlierNeuronRecord>,
    pub per_layer_metrics: Vec<Value>,
}

impl UniLayerProfileReport {
    pub fn to_value(&self, include_full_table: bool) -> Value {
        let ranking: Vec<Value> = self
            .consensus_ranking
            .iter()
            .map(|(l, r, s)| json!({"layer": l, "rank": r, "consensus_score": s}))
            .collect();
        let mut out = json!({
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "metrics_count": METRIC_NAMES_13.len(),
            "summary": self.summary,
            "redundant_layers": self.redundant_layers,
            "bottleneck_layers": self.bottleneck_layers,
            "outlier_neurons": self.outlier_neurons.iter().map(|o| o.to_value()).collect::<Vec<_>>(),
            "anomalies": {},
            "depth_trends": {},
            "consensus_ranking": ranking,
        });
        if include_full_table {
            out["per_layer_metrics"] = Value::Array(self.per_layer_metrics.clone());
        }
        out
    }

    pub fn to_toon(&self, include_full_table: bool) -> String {
        crate::toon::dumps_toon(&self.to_value(include_full_table))
    }
}

/// Group tensor names into canonical layer buckets (`layer_0`, ...).
pub fn group_parameters_by_layer(ws: &WeightWorkspace) -> Vec<(String, Vec<String>)> {
    let re1 = Regex::new(r"(?:layers|h|blocks|layer)\.(\d+)\.").unwrap();
    let re2 = Regex::new(r"(?:^|\.)(\d+)\.").unwrap();
    let mut buckets: HashMap<usize, Vec<String>> = HashMap::new();
    let mut fallback: HashMap<String, Vec<String>> = HashMap::new();
    for name in ws.all_names() {
        let cap = re1.captures(&name).or_else(|| re2.captures(&name));
        if let Some(c) = cap {
            if let Ok(idx) = c.get(1).unwrap().as_str().parse::<usize>() {
                buckets.entry(idx).or_default().push(name);
                continue;
            }
        }
        if ws.shape(&name).map(|s| s.len() >= 2).unwrap_or(false) {
            let prefix = name.rsplit_once('.').map(|(p, _)| p.to_string()).unwrap_or_default();
            fallback.entry(prefix).or_default().push(name);
        }
    }
    if !buckets.is_empty() {
        let mut idxs: Vec<usize> = buckets.keys().copied().collect();
        idxs.sort_unstable();
        return idxs
            .into_iter()
            .map(|i| (format!("layer_{i}"), buckets.remove(&i).unwrap()))
            .collect();
    }
    let mut keys: Vec<String> = fallback.keys().cloned().collect();
    keys.sort();
    keys.into_iter()
        .map(|k| (k.clone(), fallback.remove(&k).unwrap()))
        .collect()
}

/// (effective_rank, stable_rank, condition_number) via singular values.
fn matrix_spectral_stats(w: &Array2<f64>) -> (f64, f64, f64) {
    let sv: Vec<f64> = linalg::singular_values(w)
        .into_iter()
        .filter(|&v| v > 1e-12)
        .collect();
    if sv.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    let sum: f64 = sv.iter().sum();
    let entropy: f64 = sv
        .iter()
        .map(|&s| {
            let p = (s / sum).max(1e-15);
            -p * p.ln()
        })
        .sum();
    let eff_rank = entropy.exp();
    let stable_rank = sv.iter().map(|s| s * s).sum::<f64>() / (sv[0] * sv[0]).max(1e-15);
    let cond = sv[0] / sv[sv.len() - 1].max(1e-8);
    (eff_rank, stable_rank, cond)
}

fn layer_idx_of(name: &str) -> Option<usize> {
    let re = Regex::new(r"(?:layers|h|blocks|layer)\.(\d+)\.").unwrap();
    let cap = re.captures(name).or_else(|| {
        Regex::new(r"(?:^|\.)(\d+)\.").unwrap().captures(name)
    });
    cap.and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<usize>().ok())
}

/// Compute all 13 Uni-Layer metrics per layer. When `arch` + `calib` are given,
/// activation/gradient/Fisher metrics come from a real forward/backward pass;
/// otherwise weights-only spectral approximations are used.
pub fn profile_model_layers(
    ws: &WeightWorkspace,
    arch: Option<&dyn ModelArch>,
    calib: Option<&Array2<i64>>,
    model_name: &str,
    outlier_row_threshold: f64,
) -> UniLayerProfileReport {
    let groups = group_parameters_by_layer(ws);
    let layer_names: Vec<String> = groups.iter().map(|(l, _)| l.clone()).collect();

    // Optional calibration forward/backward.
    let mut hidden: Vec<Array3<f32>> = Vec::new();
    let mut grad_norms: HashMap<String, f64> = HashMap::new();
    let mut fisher_traces: HashMap<String, f64> = HashMap::new();
    if let (Some(arch), Some(ids)) = (arch, calib) {
        if let Ok(out) = arch.forward(ws, ids, true) {
            hidden = out.hidden_states;
        }
        if let Ok(grads) = arch.weight_grads(ws, ids) {
            for (lname, params) in &groups {
                let mut sq = 0.0f64;
                let mut n = 0usize;
                for p in params {
                    if let Some(g) = grads.get(p) {
                        for v in g.iter() {
                            let v = *v as f64;
                            sq += v * v;
                        }
                        n += g.len();
                    }
                }
                grad_norms.insert(lname.clone(), sq.sqrt());
                fisher_traces.insert(lname.clone(), sq / n.max(1) as f64);
            }
        }
    }

    let mut outlier_records: Vec<OutlierNeuronRecord> = Vec::new();
    let mut per_layer_rows: Vec<Value> = Vec::new();
    let mut contributions: Vec<(String, Value)> = Vec::new();

    for (idx, (lname, params)) in groups.iter().enumerate() {
        let mats: Vec<(String, Array2<f64>)> = params
            .iter()
            .filter_map(|n| {
                let t = ws.get_readonly(n).ok()?;
                if t.ndim() >= 2 {
                    Some((n.clone(), linalg::to_f64_2d(t)))
                } else {
                    let flat: Vec<f64> = t.iter().map(|v| *v as f64).collect();
                    let cols = flat.len().max(1);
                    Some((n.clone(), Array2::from_shape_vec((1, cols), flat).ok()?))
                }
            })
            .collect();
        if mats.is_empty() {
            continue;
        }

        let mut eff_ranks = Vec::new();
        let mut stable_ranks = Vec::new();
        let mut conds = Vec::new();
        let mut l2_sq = 0.0f64;
        let mut total_elems = 0usize;
        let mut near_zeros = 0usize;
        let mut weight_outliers = 0usize;
        let mut kurtosis_vals = Vec::new();
        let mut outlier_row_fracs = Vec::new();
        let mut wanda_vals = Vec::new();
        let mut last_cols = 1usize;
        let mut last_rows = 1usize;

        // Activation column norms for this layer (input hidden states).
        let act_norms: Option<Vec<f64>> = if idx > 0 && hidden.len() >= idx {
            let h_in = &hidden[idx - 1];
            let shp = h_in.shape();
            let d = shp[2];
            let flat = h_in
                .clone()
                .into_shape_with_order((shp[0] * shp[1], d))
                .unwrap();
            Some(linalg::col_norms(&flat.mapv(|v| v as f64)))
        } else {
            None
        };

        for (tname, m2d) in &mats {
            last_rows = m2d.nrows();
            last_cols = m2d.ncols();
            let (er, sr, cn) = matrix_spectral_stats(m2d);
            eff_ranks.push(er);
            stable_ranks.push(sr);
            conds.push(cn);

            let flat: Vec<f64> = m2d.iter().copied().collect();
            for v in &flat {
                l2_sq += v * v;
            }
            total_elems += flat.len();
            let mu = linalg::mean(&flat);
            let sd = linalg::std_pop(&flat) + 1e-12;
            let mut kurt_sum = 0.0f64;
            for v in &flat {
                if v.abs() < 1e-4 * sd {
                    near_zeros += 1;
                }
                let z = (v - mu).abs() / sd;
                if z > 3.0 {
                    weight_outliers += 1;
                }
                kurt_sum += z.powi(4);
            }
            kurtosis_vals.push(kurt_sum / flat.len().max(1) as f64 - 3.0);

            let row_norms = linalg::row_norms(m2d);
            let med_norm = linalg::median(&row_norms);
            if med_norm > 1e-12 && row_norms.len() >= 4 {
                let ratios: Vec<f64> = row_norms.iter().map(|n| n / med_norm).collect();
                let mut bad: Vec<usize> = (0..ratios.len())
                    .filter(|&i| ratios[i] >= outlier_row_threshold)
                    .collect();
                outlier_row_fracs.push(bad.len() as f64 / row_norms.len() as f64);
                if !bad.is_empty() {
                    bad.sort_by(|&a, &b| ratios[b].partial_cmp(&ratios[a]).unwrap());
                    outlier_records.push(OutlierNeuronRecord {
                        layer: lname.clone(),
                        tensor_name: tname.clone(),
                        row_indices: bad.iter().take(16).copied().collect(),
                        max_ratio_to_median: ratios[bad[0]],
                        median_row_norm: med_norm,
                    });
                }
            } else {
                outlier_row_fracs.push(0.0);
            }

            // Wanda score
            match &act_norms {
                Some(an) if an.len() == m2d.ncols() => {
                    let mut s = 0.0f64;
                    for (i, row) in m2d.rows().into_iter().enumerate() {
                        let _ = i;
                        for (j, v) in row.iter().enumerate() {
                            s += v.abs() * an[j];
                        }
                    }
                    wanda_vals.push(s / m2d.len().max(1) as f64);
                }
                _ => {
                    let abs_mean: f64 =
                        m2d.iter().map(|v| v.abs()).sum::<f64>() / m2d.len().max(1) as f64;
                    wanda_vals.push(abs_mean * (m2d.ncols() as f64).sqrt());
                }
            }
        }

        // Block influence & activation entropy
        let (block_inf, act_ent) = if idx > 0 && hidden.len() > idx {
            let h_in = &hidden[idx - 1];
            let h_out = &hidden[idx];
            if h_in.shape() == h_out.shape() {
                let shp = h_in.shape();
                let d = shp[2];
                let n = shp[0] * shp[1];
                let vin = h_in.clone().into_shape_with_order((n, d)).unwrap();
                let vout = h_out.clone().into_shape_with_order((n, d)).unwrap();
                let mut cos_sum = 0.0f64;
                for i in 0..n {
                    let a = vin.row(i);
                    let b = vout.row(i);
                    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                    cos_sum += (dot / (na * nb).max(1e-12)) as f64;
                }
                let cos_sim = cos_sum / n.max(1) as f64;
                let bi = (1.0 - cos_sim).max(0.0);
                let mut ch_energy = vec![0.0f64; d];
                for row in vout.rows() {
                    for (j, v) in row.iter().enumerate() {
                        let v = *v as f64;
                        ch_energy[j] += v * v;
                    }
                }
                for e in ch_energy.iter_mut() {
                    *e /= n.max(1) as f64;
                }
                let tot: f64 = ch_energy.iter().sum::<f64>() + 1e-12;
                let ent: f64 = ch_energy
                    .iter()
                    .map(|e| {
                        let p = (e / tot).max(1e-12);
                        -p * p.ln()
                    })
                    .sum();
                (bi, ent / (d as f64).ln().max(1.0))
            } else {
                spectral_approx(&eff_ranks, &stable_ranks, last_rows)
            }
        } else {
            spectral_approx(&eff_ranks, &stable_ranks, last_rows)
        };

        let w_l2 = l2_sq.sqrt();
        let g_norm = *grad_norms.get(lname).unwrap_or(&(w_l2 / (total_elems.max(1) as f64).sqrt()));
        let f_info = *fisher_traces
            .get(lname)
            .unwrap_or(&(g_norm * g_norm / total_elems.max(1) as f64));

        let r4 = |v: f64| (v * 1e4).round() / 1e4;
        let r5 = |v: f64| (v * 1e5).round() / 1e5;
        let r6 = |v: f64| (v * 1e6).round() / 1e6;
        let r8 = |v: f64| (v * 1e8).round() / 1e8;
        let m_dict = json!({
            "layer_idx": idx,
            "layer_type": "transformer_block",
            "effective_rank": r4(linalg::mean(&eff_ranks)),
            "stable_rank": r4(linalg::mean(&stable_ranks)),
            "weight_l2_norm": r4(w_l2),
            "weight_sparsity": r6(near_zeros as f64 / total_elems.max(1) as f64),
            "weight_outlier_ratio": r6(weight_outliers as f64 / total_elems.max(1) as f64),
            "weight_kurtosis": r4(linalg::mean(&kurtosis_vals)),
            "condition_number": r4(linalg::mean(&conds)),
            "block_influence": r5(block_inf),
            "activation_entropy": r5(act_ent),
            "neuron_outlier_ratio": r6(outlier_row_fracs.iter().cloned().fold(0.0f64, f64::max)),
            "fisher_information": r8(f_info),
            "gradient_norm": r5(g_norm),
            "wanda_score": r5(linalg::mean(&wanda_vals)),
        });
        let mut row = json!({"layer": lname});
        if let (Some(a), Some(b)) = (row.as_object_mut(), m_dict.as_object()) {
            for (k, v) in b {
                a.insert(k.clone(), v.clone());
            }
        }
        contributions.push((lname.clone(), m_dict));
        per_layer_rows.push(row);
    }

    // Fallback consensus analysis (port of `_fallback_profile_analysis`).
    let n = contributions.len();
    let bi: Vec<f64> = contributions
        .iter()
        .map(|(_, m)| m["block_influence"].as_f64().unwrap_or(0.0))
        .collect();
    let er: Vec<f64> = contributions
        .iter()
        .map(|(_, m)| m["effective_rank"].as_f64().unwrap_or(0.0))
        .collect();
    let redundant: Vec<String> = layer_names
        .iter()
        .enumerate()
        .filter(|(i, _)| bi[*i] < 0.02)
        .map(|(_, l)| l.clone())
        .collect();
    let mut bottlenecks = Vec::new();
    for i in 1..n.saturating_sub(1) {
        let neigh = er[i - 1] + er[i + 1];
        if neigh > 0.0 && er[i] / (neigh / 2.0) < 0.7 {
            bottlenecks.push(layer_names[i].clone());
        }
    }
    let ranking: Vec<(String, usize, f64)> = layer_names
        .iter()
        .enumerate()
        .map(|(i, l)| {
            (
                l.clone(),
                i + 1,
                ((1.0 - i as f64 / (n.saturating_sub(1)).max(1) as f64) * 1e4).round() / 1e4,
            )
        })
        .collect();
    let mut summary = format!(
        "{model_name} ({n}-layer model). analyzed with 13 metrics. redundant={}, bottlenecks={}.",
        redundant.len(),
        bottlenecks.len()
    );
    if !outlier_records.is_empty() {
        let desc: Vec<String> = outlier_records
            .iter()
            .take(3)
            .map(|o| {
                format!(
                    "{} rows {:?} ({:.1}x median)",
                    o.tensor_name,
                    &o.row_indices[..o.row_indices.len().min(4)],
                    o.max_ratio_to_median
                )
            })
            .collect();
        summary += &format!(
            " Detected {} tensor(s) with outlier neurons: {}.",
            outlier_records.len(),
            desc.join(", ")
        );
    }

    UniLayerProfileReport {
        model_name: model_name.to_string(),
        num_layers: layer_names.len(),
        summary,
        redundant_layers: redundant,
        bottleneck_layers: bottlenecks,
        consensus_ranking: ranking,
        outlier_neurons: outlier_records,
        per_layer_metrics: per_layer_rows,
    }
}

fn spectral_approx(eff_ranks: &[f64], stable_ranks: &[f64], rows: usize) -> (f64, f64) {
    let bi = linalg::mean(stable_ranks) / linalg::mean(eff_ranks).max(1.0);
    let ent = (linalg::mean(eff_ranks).max(1.001)).ln() / (rows.max(2) as f64).ln();
    (bi, ent)
}

// ---------------------------------------------------------------------------
// Knowledge Map: normalized Logit Lens + per-head domain probing + Gini KCI
// ---------------------------------------------------------------------------

/// A pre-tokenized probe: `(input_ids, target_token_id, domain)`.
#[derive(Debug, Clone)]
pub struct TokenProbe {
    pub ids: Vec<i64>,
    pub target: i64,
    pub domain: String,
}

#[derive(Debug, Clone)]
pub struct LogitLensLayerStat {
    pub layer_idx: usize,
    pub top1_convergence_rate: f64,
    pub target_prob_mean: f64,
    pub target_reciprocal_rank: f64,
    pub predictive_entropy: f64,
}

impl LogitLensLayerStat {
    pub fn to_value(&self) -> Value {
        let r4 = |v: f64| (v * 1e4).round() / 1e4;
        let r5 = |v: f64| (v * 1e5).round() / 1e5;
        json!({
            "layer_idx": self.layer_idx,
            "top1_convergence_rate": r4(self.top1_convergence_rate),
            "target_prob_mean": r5(self.target_prob_mean),
            "target_reciprocal_rank": r4(self.target_reciprocal_rank),
            "predictive_entropy": r4(self.predictive_entropy),
        })
    }
}

#[derive(Debug, Clone)]
pub struct KnowledgeMapReport {
    pub n_layers: usize,
    pub n_heads: usize,
    pub kci_overall: f64,
    pub kci_per_domain: Vec<(String, f64)>,
    pub concentration_regime: String,
    pub avg_crystallization_layer: f64,
    pub crystallization_fraction: f64,
    pub dominant_layers_by_domain: Vec<(String, Vec<usize>)>,
    pub logit_lens_layers: Vec<LogitLensLayerStat>,
    pub head_summary: Vec<(String, usize)>,
}

impl KnowledgeMapReport {
    pub fn to_value(&self) -> Value {
        let r4 = |v: f64| (v * 1e4).round() / 1e4;
        let r3 = |v: f64| (v * 1e3).round() / 1e3;
        let kci: serde_json::Map<String, Value> = self
            .kci_per_domain
            .iter()
            .map(|(k, v)| (k.clone(), json!(r4(*v))))
            .collect();
        let dom: serde_json::Map<String, Value> = self
            .dominant_layers_by_domain
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect();
        let heads: serde_json::Map<String, Value> = self
            .head_summary
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect();
        json!({
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "kci_overall": r4(self.kci_overall),
            "kci_per_domain": Value::Object(kci),
            "concentration_regime": self.concentration_regime,
            "avg_crystallization_layer": r3(self.avg_crystallization_layer),
            "crystallization_fraction": r3(self.crystallization_fraction),
            "dominant_layers_by_domain": Value::Object(dom),
            "head_summary": Value::Object(heads),
            "logit_lens_layers": self.logit_lens_layers.iter().map(|s| s.to_value()).collect::<Vec<_>>(),
        })
    }

    pub fn to_toon(&self) -> String {
        crate::toon::dumps_toon(&self.to_value())
    }
}

/// Gini coefficient over non-negative contributions, clipped to [0, 1].
pub fn compute_gini_kci(contributions: &[f64]) -> f64 {
    let mut arr: Vec<f64> = contributions.iter().map(|v| v.max(0.0)).collect();
    let sum: f64 = arr.iter().sum();
    if arr.is_empty() || sum <= 1e-15 {
        return 0.0;
    }
    arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = arr.len() as f64;
    let weighted: f64 = arr.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v).sum();
    let gini = (2.0 * weighted) / (n * sum) - (n + 1.0) / n;
    gini.clamp(0.0, 1.0)
}

/// Synthetic deterministic probes across 3 domains (tokenizer-free models).
pub fn synthetic_probes(vocab_size: usize) -> Vec<TokenProbe> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let hi = vocab_size.max(4) as i64;
    let mut probes = Vec::new();
    for domain in ["factual", "syntactic", "reasoning"] {
        for _ in 0..4 {
            let seq: Vec<i64> = (0..8).map(|_| rng.gen_range(1..hi)).collect();
            let tgt = rng.gen_range(1..hi);
            probes.push(TokenProbe { ids: seq, target: tgt, domain: domain.into() });
        }
    }
    probes
}

/// Real-text domain probes tokenized with the model tokenizer
/// (port of the `DEFAULT_KNOWLEDGE_PROBES` path in `run_knowledge_map`).
pub fn knowledge_probes_from_tokenizer(
    tokenizer: &tokenizers::Tokenizer,
    bos_token_id: Option<i64>,
) -> Vec<TokenProbe> {
    let mut probes = Vec::new();
    for (domain, prompt, answer) in crate::corpora::DEFAULT_KNOWLEDGE_PROBES {
        let Ok(enc_p) = tokenizer.encode(*prompt, true) else { continue };
        let mut ids: Vec<i64> = enc_p.get_ids().iter().map(|&v| v as i64).collect();
        if let Some(bos) = bos_token_id {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        let Ok(enc_a) = tokenizer.encode(format!(" {answer}"), false) else { continue };
        let Some(&first_ans) = enc_a.get_ids().first() else { continue };
        if ids.len() < 2 {
            continue;
        }
        probes.push(TokenProbe {
            ids,
            target: first_ans as i64,
            domain: domain.to_string(),
        });
    }
    probes
}

fn argmax_row(logits: &Array2<f32>, i: usize) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (j, &v) in logits.row(i).iter().enumerate() {
        if v > bv {
            bv = v;
            best = j;
        }
    }
    best
}

/// Full L2 Knowledge Map. Without a runtime (`arch == None`) returns an empty
/// report; with a runtime runs normalized Logit Lens + per-head residual probing.
pub fn run_knowledge_map(
    ws: &WeightWorkspace,
    arch: Option<&dyn ModelArch>,
    token_probes: Option<&[TokenProbe]>,
    n_heads_hint: Option<usize>,
) -> KnowledgeMapReport {
    let empty = KnowledgeMapReport {
        n_layers: 0,
        n_heads: n_heads_hint.unwrap_or(4),
        kci_overall: 0.0,
        kci_per_domain: vec![],
        concentration_regime: "distributed_robust".into(),
        avg_crystallization_layer: 0.0,
        crystallization_fraction: 0.0,
        dominant_layers_by_domain: vec![],
        logit_lens_layers: vec![],
        head_summary: vec![],
    };
    let Some(arch) = arch else { return empty };
    let cfg = arch.config();
    let n_heads = n_heads_hint.unwrap_or(cfg.num_attention_heads);

    let probes: Vec<TokenProbe> = match token_probes {
        Some(p) => p.to_vec(),
        None => synthetic_probes(cfg.vocab_size),
    };

    // Per-layer lens statistics accumulators.
    let mut conv: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut prob: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut rr: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut ent: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut first_crystal: Vec<usize> = Vec::new();

    // Per-domain head scores & hidden states for residual probing.
    let mut domain_names: Vec<String> = probes.iter().map(|p| p.domain.clone()).collect();
    domain_names.sort();
    domain_names.dedup();
    let mut n_layers = 0usize;
    let mut domain_head_scores: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    let mut domain_counts: HashMap<String, usize> = HashMap::new();

    for probe in &probes {
        if probe.ids.len() < 2 {
            continue;
        }
        let ids = Array2::from_shape_vec((1, probe.ids.len()), probe.ids.clone()).unwrap();
        let Ok(out) = arch.forward(ws, &ids, true) else { continue };
        if out.hidden_states.is_empty() {
            continue;
        }
        n_layers = out.hidden_states.len();
        let Ok(lens) = arch.lens_logits(ws, &out.hidden_states) else { continue };
        let seq_len = out.logits.shape()[1];
        let last = seq_len - 1;
        let final_logits = out.logits.slice(ndarray::s![0, last, ..]).to_owned();
        let final_top1 = argmax_row(&final_logits.insert_axis(ndarray::Axis(0)), 0) as i64;

        let mut found_first: Option<usize> = None;
        for (l_idx, layer_logits) in lens.iter().enumerate() {
            let probs = softmax_rows(layer_logits);
            let pred_id = argmax_row(&probs, 0) as i64;
            let converged = if pred_id == probe.target || pred_id == final_top1 { 1.0 } else { 0.0 };
            if converged > 0.5 && found_first.is_none() {
                found_first = Some(l_idx);
            }
            let tgt_idx = (probe.target.max(0) as usize).min(probs.ncols() - 1);
            let tgt_prob = probs[[0, tgt_idx]] as f64;
            let tgt_logit = layer_logits[[0, tgt_idx]];
            let higher = layer_logits.row(0).iter().filter(|&&v| v > tgt_logit).count();
            let reciprocal_rank = 1.0 / (higher as f64 + 1.0);
            let entropy: f64 = probs
                .row(0)
                .iter()
                .map(|&p| {
                    let p = (p as f64).max(1e-12);
                    -p * p.ln()
                })
                .sum();
            conv.entry(l_idx).or_default().push(converged);
            prob.entry(l_idx).or_default().push(tgt_prob);
            rr.entry(l_idx).or_default().push(reciprocal_rank);
            ent.entry(l_idx).or_default().push(entropy);
        }
        if let Some(f) = found_first {
            first_crystal.push(f);
        }

        // Per-head residual update norms at the last position.
        let scores = domain_head_scores
            .entry(probe.domain.clone())
            .or_insert_with(|| vec![vec![0.0; n_heads]; n_layers]);
        let mut prev_h: Option<Vec<f32>> = None;
        for (l_idx, hs) in out.hidden_states.iter().enumerate().take(n_layers) {
            let seq_len = hs.shape()[1];
            let last = seq_len - 1;
            let cur: Vec<f32> = hs.slice(ndarray::s![0, last, ..]).to_vec();
            let delta: Vec<f32> = match &prev_h {
                Some(p) if l_idx > 0 => cur.iter().zip(p.iter()).map(|(a, b)| a - b).collect(),
                _ => cur.clone(),
            };
            prev_h = Some(cur);
            let head_dim = (delta.len() / n_heads).max(1);
            for h_i in 0..n_heads {
                let chunk = &delta[h_i * head_dim..((h_i + 1) * head_dim).min(delta.len())];
                let norm: f64 = chunk.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
                if l_idx < scores.len() && h_i < scores[l_idx].len() {
                    scores[l_idx][h_i] += norm;
                }
            }
        }
        *domain_counts.entry(probe.domain.clone()).or_insert(0) += 1;
    }

    let mut lens_stats: Vec<LogitLensLayerStat> = Vec::new();
    let mut l_idxs: Vec<usize> = conv.keys().copied().collect();
    l_idxs.sort_unstable();
    for l_idx in l_idxs {
        lens_stats.push(LogitLensLayerStat {
            layer_idx: l_idx,
            top1_convergence_rate: linalg::mean(&conv[&l_idx]),
            target_prob_mean: linalg::mean(&prob[&l_idx]),
            target_reciprocal_rank: linalg::mean(&rr[&l_idx]),
            predictive_entropy: linalg::mean(&ent[&l_idx]),
        });
    }
    let n_layers = n_layers.max(1);
    let avg_crystal = if first_crystal.is_empty() {
        0.0
    } else {
        first_crystal.iter().sum::<usize>() as f64 / first_crystal.len() as f64
    };

    let mut kci_per_domain = Vec::new();
    let mut dominant_layers = Vec::new();
    for d in &domain_names {
        let scores = domain_head_scores.get_mut(d).unwrap();
        let count = domain_counts.get(d).copied().unwrap_or(0).max(1);
        for row in scores.iter_mut() {
            for v in row.iter_mut() {
                *v /= count as f64;
            }
        }
        let flat: Vec<f64> = scores.iter().flatten().copied().collect();
        kci_per_domain.push((d.clone(), compute_gini_kci(&flat)));
        let mut layer_totals: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, row)| (i, row.iter().sum()))
            .collect();
        layer_totals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        dominant_layers.push((
            d.clone(),
            layer_totals.iter().take(3.min(n_layers)).map(|(i, _)| *i).collect(),
        ));
    }

    let mut head_summary: Vec<(String, usize)> = domain_names
        .iter()
        .map(|d| (format!("n_{d}_heads"), 0))
        .collect();
    for l_idx in 0..n_layers {
        for h_i in 0..n_heads {
            let mut best_d = &domain_names[0];
            let mut best_v = f64::NEG_INFINITY;
            for d in &domain_names {
                let v = domain_head_scores
                    .get(d)
                    .and_then(|s| s.get(l_idx))
                    .and_then(|r| r.get(h_i))
                    .copied()
                    .unwrap_or(0.0);
                if v > best_v {
                    best_v = v;
                    best_d = d;
                }
            }
            if let Some(entry) = head_summary.iter_mut().find(|(k, _)| k == &format!("n_{best_d}_heads")) {
                entry.1 += 1;
            }
        }
    }

    let kci_overall = if kci_per_domain.is_empty() {
        0.0
    } else {
        kci_per_domain.iter().map(|(_, v)| v).sum::<f64>() / kci_per_domain.len() as f64
    };
    let regime = if kci_overall >= 0.45 {
        "concentrated_fragile"
    } else if kci_overall >= 0.20 {
        "moderate_localized"
    } else {
        "distributed_robust"
    };

    KnowledgeMapReport {
        n_layers,
        n_heads,
        kci_overall,
        kci_per_domain,
        concentration_regime: regime.into(),
        avg_crystallization_layer: avg_crystal,
        crystallization_fraction: avg_crystal / n_layers.max(1) as f64,
        dominant_layers_by_domain: dominant_layers,
        logit_lens_layers: lens_stats,
        head_summary,
    }
}

#[allow(dead_code)]
fn _tensor_helper(t: &Tensor) -> usize {
    tensor::numel(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gini() {
        assert_eq!(compute_gini_kci(&[1.0, 1.0, 1.0, 1.0]), 0.0);
        assert!(compute_gini_kci(&[0.0, 0.0, 0.0, 10.0]) > 0.7);
        assert_eq!(compute_gini_kci(&[]), 0.0);
    }
}