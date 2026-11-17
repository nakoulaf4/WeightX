// SPDX-License-Identifier: GPL-3.0-or-later
//! L1 Architecture Understanding (AutoArch Engine) — Rust port of `weightx.l1_arch.autoarch`.
//!
//! Shape-first tensor role inference (`q`, `k`, `v`, `o`, `qkv`, `gate`, `up`,
//! `down`, `norm_gain`, `norm_bias`, `embedding`, `lm_head`) with structural
//! stack detection and optional likelihood/spectral calibration disambiguation.

use std::collections::HashMap;
use std::path::Path;

use regex::Regex;
use serde_json::{json, Value};

use crate::error::Result;
use crate::linalg;
use crate::tensor::{self, Tensor};
use crate::workspace::WeightWorkspace;

/// (role, name hints) table — mirrors Python `NAME_HINTS`.
pub const NAME_HINTS: &[(&str, &[&str])] = &[
    ("q", &["q_proj", ".wq", "query", ".q.", "q_lin", "attention.wq", "attn.q"]),
    ("k", &["k_proj", ".wk", ".key", ".k.", "k_lin", "attention.wk", "attn.k"]),
    ("v", &["v_proj", ".wv", "value", ".v.", "v_lin", "attention.wv", "attn.v"]),
    ("o", &["o_proj", ".wo", "out_proj", "attention.wo", "attn.proj", "c_proj", "out_lin", "output.dense", "attn.o"]),
    ("qkv", &["qkv", "c_attn", "query_key_value", "in_proj_weight", "in_proj"]),
    ("gate", &["gate_proj", "w1", "gate", "fc_gate", "mlp.gate"]),
    ("up", &["up_proj", "w3", "w_up", "wi_1", "fc1", "c_fc", "intermediate.dense", "mlp.up", "mlp.fc1"]),
    ("down", &["down_proj", "w2", "w_down", "wo.", "fc2", "mlp.c_proj", "output.dense", "mlp.down", "mlp.fc2"]),
    ("norm", &["norm", "ln", "layernorm", "layer_norm", "rmsnorm"]),
    ("embed", &["embed", "wte", "tok_embeddings", "word_embeddings", "shared"]),
    ("pos", &["pos", "wpe", "position"]),
    ("lm_head", &["lm_head", "output", "head", "unembed", "logits"]),
];

pub fn has_hint(name: &str, role: &str) -> bool {
    let low = name.to_lowercase();
    NAME_HINTS
        .iter()
        .find(|(r, _)| *r == role)
        .map(|(_, hints)| hints.iter().any(|h| low.contains(h)))
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
pub struct TensorRoleInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub numel: usize,
    pub layer_idx: Option<usize>,
    pub role: String,
}

impl TensorRoleInfo {
    pub fn to_value(&self) -> Value {
        json!({
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "numel": self.numel,
            "layer_idx": self.layer_idx.map(|v| v as i64).unwrap_or(-1),
            "role": self.role,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CalibrationResolution {
    pub chosen_heads: usize,
    pub chosen_kv_heads: usize,
    pub chosen_head_dim: usize,
    pub chosen_activation: String,
    pub chosen_rope_base: f64,
    pub best_nll: f64,
    pub evaluated_configs: usize,
    pub identifiable: bool,
    pub margin_nll: f64,
    pub candidate_nlls: Vec<(String, f64)>,
}

impl CalibrationResolution {
    pub fn to_value(&self) -> Value {
        let r5 = |v: f64| (v * 1e5).round() / 1e5;
        let cands: serde_json::Map<String, Value> = self
            .candidate_nlls
            .iter()
            .map(|(k, v)| (k.clone(), json!(r5(*v))))
            .collect();
        json!({
            "chosen_heads": self.chosen_heads,
            "chosen_kv_heads": self.chosen_kv_heads,
            "chosen_head_dim": self.chosen_head_dim,
            "chosen_activation": self.chosen_activation,
            "chosen_rope_base": self.chosen_rope_base,
            "best_nll": r5(self.best_nll),
            "evaluated_configs": self.evaluated_configs,
            "identifiable": self.identifiable,
            "margin_nll": r5(self.margin_nll),
            "candidate_nlls": Value::Object(cands),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ArchitectureReport {
    pub family: String,
    pub confidence: f64,
    pub total_params: usize,
    pub d_model: Option<usize>,
    pub n_layers: usize,
    pub vocab_size: Option<usize>,
    pub mlp_type: Option<String>,
    pub mlp_hidden_dim: Option<usize>,
    pub stack_prefix: Option<String>,
    pub torch_orientation: bool,
    pub tied_embeddings: bool,
    pub norm_type: String,
    pub heads_candidates: Vec<(usize, usize, usize)>, // (heads, kv, head_dim)
    pub block_roles: HashMap<String, String>,
    pub global_roles: HashMap<String, String>,
    pub tensors: Vec<TensorRoleInfo>,
    pub evidence: Vec<String>,
    pub ambiguities: Vec<String>,
    pub calibration_resolution: Option<CalibrationResolution>,
    pub backend_engine: String,
}

impl ArchitectureReport {
    pub fn to_value(&self, include_tensor_table: bool) -> Value {
        let heads: Vec<Value> = self
            .heads_candidates
            .iter()
            .take(6)
            .map(|(h, kv, hd)| json!({"heads": h, "kv": kv, "head_dim": hd}))
            .collect();
        let mut out = json!({
            "family": self.family,
            "confidence": (self.confidence * 1e3).round() / 1e3,
            "total_params": self.total_params,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "mlp_type": self.mlp_type,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "stack_prefix": self.stack_prefix,
            "torch_orientation": self.torch_orientation,
            "tied_embeddings": self.tied_embeddings,
            "norm_type": self.norm_type,
            "heads_candidates": heads,
            "block_roles": self.block_roles,
            "global_roles": self.global_roles,
            "evidence": self.evidence,
            "ambiguities": self.ambiguities,
            "backend_engine": self.backend_engine,
        });
        if let Some(c) = &self.calibration_resolution {
            out["calibration_resolution"] = c.to_value();
        }
        if include_tensor_table {
            out["tensors"] = Value::Array(self.tensors.iter().map(|t| t.to_value()).collect());
        }
        out
    }

    pub fn to_toon(&self, include_tensor_table: bool) -> String {
        crate::toon::dumps_toon(&self.to_value(include_tensor_table))
    }
}

fn extract_layer_idx(name: &str, stack_prefix: Option<&str>) -> Option<usize> {
    if let Some(prefix) = stack_prefix {
        let with_dot = format!("{prefix}.");
        if let Some(rest) = name.strip_prefix(&with_dot) {
            let first = rest.split('.').next().unwrap_or("");
            if let Ok(v) = first.parse::<usize>() {
                return Some(v);
            }
        }
    }
    let re = Regex::new(r"(?:^|\.)(\d+)\.").unwrap();
    re.captures(name)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<usize>().ok())
}

/// Structural fallback stack detector (port of `_fallback_inspect`).
fn fallback_inspect(
    shapes: &[(String, Vec<usize>, String)],
) -> (Option<String>, usize, Option<usize>, Option<usize>) {
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (name, _, _) in shapes {
        let parts: Vec<&str> = name.split('.').collect();
        for (i, p) in parts.iter().enumerate() {
            if let Ok(v) = p.parse::<usize>() {
                groups.entry(parts[..i].join(".")).or_default().push(v);
                break;
            }
        }
    }
    let mut stack_prefix = None;
    let mut n_layers = 0usize;
    if let Some((prefix, idxs)) = groups.iter().max_by_key(|(_, v)| v.len()) {
        let mut uniq = idxs.clone();
        uniq.sort_unstable();
        uniq.dedup();
        stack_prefix = Some(prefix.clone());
        n_layers = uniq.len();
    }
    let mut dim_counts: HashMap<usize, usize> = HashMap::new();
    for (_, shp, _) in shapes {
        for &d in shp {
            *dim_counts.entry(d).or_insert(0) += 1;
        }
    }
    let d_model = dim_counts
        .iter()
        .max_by_key(|(_, c)| **c)
        .map(|(d, _)| *d);
    let vocab_size = shapes
        .iter()
        .filter(|(_, s, _)| s.len() == 2)
        .map(|(_, s, _)| *s.iter().max().unwrap_or(&0))
        .max();
    (stack_prefix, n_layers, d_model, vocab_size)
}

/// Inspect a checkpoint given `(name, shape, dtype)` metadata.
pub fn inspect_architecture(
    shapes: &[(String, Vec<usize>, String)],
    config: Option<&Value>,
) -> ArchitectureReport {
    let total_params: usize = shapes.iter().map(|(_, s, _)| s.iter().product::<usize>()).sum();
    let (stack_prefix, n_layers, d_model, vocab_size) = fallback_inspect(shapes);

    let family = if n_layers > 0 { "decoder_transformer" } else { "mlp" }.to_string();
    let confidence = if n_layers > 0 { 0.75 } else { 0.6 };
    let mut block_roles: HashMap<String, String> = HashMap::new();
    let mut global_roles: HashMap<String, String> = HashMap::new();
    let mut heads_candidates = Vec::new();
    if let Some(d) = d_model {
        for hd in [64usize, 128, 32, 16, 8] {
            if d % hd == 0 {
                heads_candidates.push((d / hd, d / hd, hd));
            }
        }
    }
    let evidence = vec![format!(
        "WeightX structural fallback: {n_layers} layers, d_model={}",
        d_model.map(|v| v.to_string()).unwrap_or_else(|| "None".into())
    )];
    let mut ambiguities: Vec<String> = Vec::new();
    if heads_candidates.len() > 1 {
        ambiguities.push("heads/head_dim ambiguous: multiple divisor candidates".into());
    }

    // Refine roles for unclassified suffixes inside the stack / globals.
    for (name, _, _) in shapes {
        let l_idx = extract_layer_idx(name, stack_prefix.as_deref());
        if let (Some(_), Some(prefix)) = (l_idx, stack_prefix.as_deref()) {
            let prefix_with_idx = format!("{prefix}.{}.", l_idx.unwrap());
            let suffix = name.strip_prefix(&prefix_with_idx).unwrap_or(name);
            if !block_roles.contains_key(suffix) {
                for hint_role in ["q", "k", "v", "o", "qkv", "gate", "up", "down", "norm"] {
                    if has_hint(suffix, hint_role) {
                        let role = if hint_role == "norm" {
                            if suffix.ends_with("bias") { "norm_bias" } else { "norm_gain" }
                        } else {
                            hint_role
                        };
                        block_roles.insert(suffix.to_string(), role.to_string());
                        break;
                    }
                }
            }
        } else if !global_roles.contains_key(name) {
            if has_hint(name, "embed") {
                global_roles.insert(name.clone(), "embedding".into());
            } else if has_hint(name, "lm_head") {
                global_roles.insert(name.clone(), "lm_head".into());
            } else if has_hint(name, "pos") {
                global_roles.insert(name.clone(), "pos_embedding".into());
            } else if has_hint(name, "norm") {
                let role = if name.ends_with("bias") { "final_norm_bias" } else { "final_norm_gain" };
                global_roles.insert(name.clone(), role.into());
            }
        }
    }

    let mlp_type = if block_roles.values().any(|r| r == "gate") { "gated" } else { "plain" };
    let tied_embeddings = !global_roles.values().any(|r| r == "lm_head");

    let mut tensor_infos = Vec::new();
    for (name, shp, dtype) in shapes {
        let l_idx = extract_layer_idx(name, stack_prefix.as_deref());
        let mut role = "other".to_string();
        if let (Some(_), Some(prefix)) = (l_idx, stack_prefix.as_deref()) {
            let prefix_with_idx = format!("{prefix}.{}.", l_idx.unwrap());
            let suffix = name.strip_prefix(&prefix_with_idx).unwrap_or(name);
            role = block_roles.get(suffix).cloned().unwrap_or_else(|| "block_param".into());
        } else if let Some(r) = global_roles.get(name) {
            role = r.clone();
        }
        tensor_infos.push(TensorRoleInfo {
            name: name.clone(),
            shape: shp.clone(),
            dtype: dtype.clone(),
            numel: shp.iter().product(),
            layer_idx: l_idx,
            role,
        });
    }

    // Config overrides (when a config.json is available).
    let mut d_model = d_model;
    let mut vocab_size = vocab_size;
    let mut n_layers = n_layers;
    if let Some(cfg) = config {
        let g = |k: &str| cfg.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        if let Some(v) = g("hidden_size").or_else(|| g("n_embd")) { d_model = Some(v); }
        if let Some(v) = g("vocab_size") { vocab_size = Some(v); }
        if let Some(v) = g("num_hidden_layers").or_else(|| g("n_layer")) { n_layers = v; }
        if let (Some(h), Some(hd)) = (g("num_attention_heads").or_else(|| g("n_head")), g("head_dim")) {
            heads_candidates.insert(0, (h, g("num_key_value_heads").unwrap_or(h), hd));
        }
    }

    // Norm family & MLP hidden dim from tensor names/shapes.
    let norm_type = if shapes.iter().any(|(n, _, _)| {
        n.contains("ln_1") || n.contains("ln_2") || n.contains("ln_f") || n.contains("layer_norm")
    }) {
        "layernorm"
    } else {
        "rmsnorm"
    };
    let mlp_hidden_dim = shapes.iter().find_map(|(n, s, _)| {
        if s.len() != 2 {
            return None;
        }
        if n.ends_with("gate_proj.weight") || n.ends_with("up_proj.weight") {
            Some(s[0])
        } else if n.ends_with("c_fc.weight") {
            Some(s[1]) // GPT-2 Conv1D stores [in, out]
        } else {
            None
        }
    });

    ArchitectureReport {
        family,
        confidence,
        total_params,
        d_model,
        n_layers,
        vocab_size,
        mlp_type: Some(mlp_type.into()),
        mlp_hidden_dim,
        stack_prefix,
        torch_orientation: true,
        tied_embeddings,
        norm_type: norm_type.into(),
        heads_candidates,
        block_roles,
        global_roles,
        tensors: tensor_infos,
        evidence,
        ambiguities,
        calibration_resolution: None,
        backend_engine: "weightx-native".into(),
    }
}

/// Inspect a `.safetensors` checkpoint without materializing tensor data.
pub fn inspect_safetensors(path: &Path) -> Result<ArchitectureReport> {
    let meta = tensor::safetensors_metadata(path)?;
    Ok(inspect_architecture(&meta, None))
}

/// Inspect the tensors currently held by a workspace.
pub fn inspect_workspace(ws: &WeightWorkspace) -> ArchitectureReport {
    let shapes: Vec<(String, Vec<usize>, String)> = ws
        .iter()
        .map(|(n, t)| (n.clone(), t.shape().to_vec(), "float32".to_string()))
        .collect();
    inspect_architecture(&shapes, None)
}

/// Resolve head-count ambiguity via Q/K per-head spectral block energy
/// (weights-only port of the calibration probe; `base_nll` may come from a
/// real forward pass when a runtime is available).
pub fn resolve_ambiguities_with_calibration(
    ws: &WeightWorkspace,
    heads_candidates: &[(usize, usize, usize)],
    d_model: usize,
    base_nll: f64,
    config: Option<&Value>,
) -> CalibrationResolution {
    let mut cands: Vec<(usize, usize, usize)> = heads_candidates.to_vec();
    if cands.is_empty() {
        for hd in [64usize, 32, 16, 8] {
            if d_model % hd == 0 {
                cands.push((d_model / hd, d_model / hd, hd));
            }
        }
    }
    if cands.is_empty() {
        cands.push((1, 1, d_model));
    }

    // First 2D query-role weight.
    let q_weight: Option<Tensor> = ws
        .iter()
        .find(|(n, t)| t.ndim() == 2 && has_hint(n, "q"))
        .map(|(_, t)| t.clone());

    let cfg_heads = config
        .and_then(|c| c.get("num_attention_heads").or_else(|| c.get("n_head")))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let mut candidate_nlls: Vec<(String, f64)> = Vec::new();
    for &(h, kv, hd) in &cands {
        let label = format!("heads={h},kv={kv},hd={hd}");
        let penalty = match &q_weight {
            Some(q) if q.shape()[0] == h * hd => {
                let m = tensor::as_2d(q);
                let cols = m.ncols();
                let norms: Vec<f64> = (0..h)
                    .map(|i| {
                        let mut s = 0.0f64;
                        for r in (i * hd)..((i + 1) * hd) {
                            for j in 0..cols {
                                let v = m[[r, j]] as f64;
                                s += v * v;
                            }
                        }
                        s.sqrt()
                    })
                    .collect();
                let mean = linalg::mean(&norms) + 1e-8;
                let sd = linalg::std_pop(&norms);
                let cv = sd / mean;
                match cfg_heads {
                    Some(ch) if ch == h => 0.0,
                    _ => 0.05 + 0.1 * cv,
                }
            }
            _ => 0.25,
        };
        candidate_nlls.push((label, base_nll + penalty));
    }

    let mut sorted = candidate_nlls.clone();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let (best_label, best_val) = sorted[0].clone();
    let second_val = sorted.get(1).map(|x| x.1).unwrap_or(best_val + 1.0);
    let margin = second_val - best_val;
    let best_idx = candidate_nlls.iter().position(|(l, _)| *l == best_label).unwrap_or(0);
    let (ch, ckv, chd) = cands[best_idx];

    let activation = config
        .and_then(|c| c.get("hidden_act"))
        .and_then(|v| v.as_str())
        .unwrap_or("silu")
        .to_string();
    let rope_base = config
        .and_then(|c| c.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0);

    CalibrationResolution {
        chosen_heads: ch,
        chosen_kv_heads: ckv,
        chosen_head_dim: chd,
        chosen_activation: activation,
        chosen_rope_base: rope_base,
        best_nll: best_val,
        evaluated_configs: candidate_nlls.len(),
        identifiable: margin >= 0.01 || cands.len() == 1,
        margin_nll: margin,
        candidate_nlls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_hints_and_stack_detection() {
        let shapes = vec![
            ("model.embed_tokens.weight".to_string(), vec![128, 64], "float32".into()),
            ("model.layers.0.self_attn.q_proj.weight".to_string(), vec![64, 64], "float32".into()),
            ("model.layers.0.mlp.down_proj.weight".to_string(), vec![64, 172], "float32".into()),
            ("model.layers.1.self_attn.q_proj.weight".to_string(), vec![64, 64], "float32".into()),
            ("lm_head.weight".to_string(), vec![128, 64], "float32".into()),
        ];
        let rep = inspect_architecture(&shapes, None);
        assert_eq!(rep.n_layers, 2);
        assert_eq!(rep.d_model, Some(64));
        assert_eq!(rep.stack_prefix.as_deref(), Some("model.layers"));
        assert_eq!(rep.block_roles.get("self_attn.q_proj.weight").unwrap(), "q");
        assert_eq!(rep.block_roles.get("mlp.down_proj.weight").unwrap(), "down");
        assert_eq!(rep.global_roles.get("lm_head.weight").unwrap(), "lm_head");
        assert!(!rep.tied_embeddings);
    }
}