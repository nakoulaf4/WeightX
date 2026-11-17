// SPDX-License-Identifier: GPL-3.0-or-later
//! Model runtime: architecture trait, config, and shared neural math.
//! Replaces the PyTorch `nn.Module` dependency of the Python harness.

pub mod llama;
pub mod gpt2;
pub mod minillama;

use std::collections::HashMap;

use ndarray::{Array2, Array3};
use serde_json::Value;

use crate::error::Result;
use crate::tensor::Tensor;
use crate::workspace::WeightWorkspace;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub layer_norm_eps: f64,
    pub rope_theta: f64,
    pub use_rope: bool,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: "llama".into(),
            vocab_size: 128,
            hidden_size: 64,
            intermediate_size: 172,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 16,
            rms_norm_eps: 1e-5,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            use_rope: true,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
        }
    }
}

impl ModelConfig {
    pub fn from_hf_json(v: &Value) -> Self {
        let g = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let gf = |k: &str| v.get(k).and_then(|x| x.as_f64());
        let model_type = v
            .get("model_type")
            .and_then(|x| x.as_str())
            .unwrap_or("llama")
            .to_string();
        let hidden = g("hidden_size").or_else(|| g("n_embd")).unwrap_or(64);
        let heads = g("num_attention_heads").or_else(|| g("n_head")).unwrap_or(4);
        let kv = g("num_key_value_heads").unwrap_or(heads);
        let head_dim = g("head_dim").unwrap_or(hidden / heads.max(1));
        let layers = g("num_hidden_layers").or_else(|| g("n_layer")).unwrap_or(4);
        let inter = g("intermediate_size")
            .or_else(|| g("n_inner"))
            .unwrap_or(hidden * 4);
        let vocab = g("vocab_size").unwrap_or(128);
        let is_gpt2 = model_type == "gpt2";
        Self {
            model_type: model_type.clone(),
            vocab_size: vocab,
            hidden_size: hidden,
            intermediate_size: inter,
            num_hidden_layers: layers,
            num_attention_heads: heads,
            num_key_value_heads: kv,
            head_dim,
            rms_norm_eps: gf("rms_norm_eps").unwrap_or(1e-5),
            layer_norm_eps: gf("layer_norm_eps").or_else(|| gf("layer_norm_epsilon")).unwrap_or(1e-5),
            rope_theta: gf("rope_theta").unwrap_or(10000.0),
            use_rope: !is_gpt2,
            hidden_act: v
                .get("hidden_act")
                .or_else(|| v.get("activation_function"))
                .and_then(|x| x.as_str())
                .unwrap_or(if is_gpt2 { "gelu_new" } else { "silu" })
                .to_string(),
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(is_gpt2),
        }
    }

    pub fn to_value(&self) -> Value {
        serde_json::json!({
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "hidden_act": self.hidden_act,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ForwardOut {
    /// `[batch, seq, vocab]`
    pub logits: Array3<f32>,
    /// Per-layer residual outputs (after each block), `[batch, seq, hidden]`.
    pub hidden_states: Vec<Array3<f32>>,
}

/// Architecture description + inference over workspace-owned weights.
pub trait ModelArch: Send + Sync {
    fn config(&self) -> &ModelConfig;
    fn forward(&self, ws: &WeightWorkspace, ids: &Array2<i64>, want_hidden: bool) -> Result<ForwardOut>;
    /// All parameter tensor names known to this architecture.
    fn weight_names(&self) -> Vec<String>;
    /// MLP output projection tensor names in depth order.
    fn mlp_out_names(&self) -> Vec<String>;
    /// Normalized Logit Lens: per-layer logits of the last token `[batch, vocab]`.
    fn lens_logits(&self, ws: &WeightWorkspace, hidden: &[Array3<f32>]) -> Result<Vec<Array2<f32>>>;
    /// Gradients of next-token cross-entropy loss w.r.t. every weight (empirical Fisher).
    fn weight_grads(&self, ws: &WeightWorkspace, ids: &Array2<i64>) -> Result<HashMap<String, Tensor>>;
    /// Keys (MLP-out inputs) and loss gradients at MLP outputs, concatenated over sequences.
    /// Keys/grads rows exclude the last position (mirrors `inp[0][0, :-1]`).
    fn mlp_keys_grads(
        &self,
        ws: &WeightWorkspace,
        seqs: &[Vec<i64>],
        need_grads: bool,
    ) -> Result<(HashMap<String, Array2<f64>>, HashMap<String, Array2<f64>>)>;
    fn supports_backward(&self) -> bool {
        true
    }
    /// Numeric backend identifier for UI/status reporting ("rust" or "torch").
    fn backend_name(&self) -> &'static str {
        "rust"
    }
    /// Optional full delegation of EAMIT to an external numeric backend
    /// (hybrid mode: PyTorch/numpy sidecar). `None` = use the native Rust path.
    fn eamit_override(
        &self,
        _ws: &mut WeightWorkspace,
        _target_tensors: &[String],
        _facts: &[crate::eamit::BatchFactSpec],
        _lambda_cov: f64,
        _apply_fisher_guard: bool,
        _align_cfg: &crate::eamit::EamitAlignConfig,
    ) -> Option<Result<crate::eamit::EamitResult>> {
        None
    }
}

// ---------------------------------------------------------------------------
// Shared neural math (f32)
// ---------------------------------------------------------------------------

pub fn softmax_rows(x: &Array2<f32>) -> Array2<f32> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    out
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

pub fn gelu_new(x: f32) -> f32 {
    // GPT-2 "gelu_new": 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

pub fn gelu_new_grad(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    let t = inner.tanh();
    let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
}

/// RMSNorm over the last axis: `x * rsqrt(mean(x^2) + eps) * weight`.
pub fn rmsnorm(x: &Array2<f32>, weight: &[f32], eps: f32) -> Array2<f32> {
    let d = x.ncols();
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (j, v) in row.iter_mut().enumerate() {
            *v = *v * inv * weight[j];
        }
    }
    out
}

/// LayerNorm over the last axis with weight & bias (GPT-2).
pub fn layernorm(x: &Array2<f32>, weight: &[f32], bias: &[f32], eps: f32) -> Array2<f32> {
    let d = x.ncols();
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (j, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv * weight[j] + bias[j];
        }
    }
    out
}

/// Cross-entropy per row: logits `[N, V]`, targets `[N]`. Returns per-row loss.
pub fn cross_entropy_rows(logits: &Array2<f32>, targets: &[i64]) -> Vec<f32> {
    let probs = softmax_rows(logits);
    targets
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let p = probs[[i, (t.max(0) as usize).min(probs.ncols() - 1)]];
            -(p.max(1e-12)).ln()
        })
        .collect()
}

/// dL/dlogits for cross-entropy (softmax - onehot), divided by N when `mean`.
pub fn cross_entropy_grad(logits: &Array2<f32>, targets: &[i64], mean: bool) -> Array2<f32> {
    let mut g = softmax_rows(logits);
    let n = targets.len().max(1) as f32;
    for (i, &t) in targets.iter().enumerate() {
        let j = (t.max(0) as usize).min(g.ncols() - 1);
        g[[i, j]] -= 1.0;
    }
    if mean {
        g.mapv_inplace(|v| v / n);
    }
    g
}

/// Linear layer `y = x @ W^T` for W stored `[out, in]`.
pub fn linear(x: &Array2<f32>, w: &Array2<f32>) -> Array2<f32> {
    x.dot(&w.t())
}

pub fn get_weight<'a>(ws: &'a WeightWorkspace, name: &str) -> Result<&'a Tensor> {
    ws.get_readonly(name)
}

pub fn weight_2d(ws: &WeightWorkspace, name: &str) -> Result<Array2<f32>> {
    let t = ws.get_readonly(name)?;
    let shp = t.shape();
    Ok(t
        .clone()
        .into_shape_with_order((shp[0], shp[1]))
        .unwrap()
        .to_owned())
}

pub fn weight_1d(ws: &WeightWorkspace, name: &str) -> Result<Vec<f32>> {
    Ok(ws.get_readonly(name)?.iter().copied().collect())
}