// SPDX-License-Identifier: GPL-3.0-or-later
//! Llama-family runtime (RMSNorm + RoPE + GQA + SwiGLU), forward and backward.
//! Also runs the built-in MiniLlama demo (RoPE disabled, MHA).

use std::collections::HashMap;

use ndarray::{Array2, Array3, Axis};

use crate::error::{Result, WeightXError};
use crate::model::*;
use crate::tensor::Tensor;
use crate::workspace::WeightWorkspace;

pub struct LlamaModel {
    pub cfg: ModelConfig,
}

impl LlamaModel {
    pub fn new(cfg: ModelConfig) -> Self {
        Self { cfg }
    }

    fn layer_names(&self, i: usize) -> LayerNames {
        let p = format!("model.layers.{i}");
        LayerNames {
            ln1: format!("{p}.input_layernorm.weight"),
            q: format!("{p}.self_attn.q_proj.weight"),
            k: format!("{p}.self_attn.k_proj.weight"),
            v: format!("{p}.self_attn.v_proj.weight"),
            o: format!("{p}.self_attn.o_proj.weight"),
            ln2: format!("{p}.post_attention_layernorm.weight"),
            gate: format!("{p}.mlp.gate_proj.weight"),
            up: format!("{p}.mlp.up_proj.weight"),
            down: format!("{p}.mlp.down_proj.weight"),
        }
    }

    fn head_weight(&self, ws: &WeightWorkspace) -> Result<Array2<f32>> {
        if ws.contains("lm_head.weight") {
            weight_2d(ws, "lm_head.weight")
        } else {
            weight_2d(ws, "model.embed_tokens.weight")
        }
    }
}

struct LayerNames {
    ln1: String,
    q: String,
    k: String,
    v: String,
    o: String,
    ln2: String,
    gate: String,
    up: String,
    down: String,
}

/// Per-layer forward cache needed for backward.
struct LayerCache {
    x_in: Array2<f32>,
    n1: Array2<f32>,
    q_rot: Vec<Array2<f32>>,  // per query head [S, hd]
    k_rot: Vec<Array2<f32>>,  // per kv head [S, hd]
    v_heads: Vec<Array2<f32>>,
    probs: Vec<Array2<f32>>,  // per query head [S, S]
    attn_concat: Array2<f32>, // [S, nh*hd]
    x2: Array2<f32>,
    n2: Array2<f32>,
    g: Array2<f32>,
    u: Array2<f32>,
    m: Array2<f32>,
}

struct ForwardCache {
    ids: Vec<i64>,
    layers: Vec<LayerCache>,
    x_final: Array2<f32>,
    xn: Array2<f32>,
}

impl ModelArch for LlamaModel {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    fn forward(&self, ws: &WeightWorkspace, ids: &Array2<i64>, want_hidden: bool) -> Result<ForwardOut> {
        let (out, _cache) = self.forward_impl(ws, ids, want_hidden, false)?;
        Ok(out)
    }

    fn weight_names(&self) -> Vec<String> {
        let mut names = vec!["model.embed_tokens.weight".to_string()];
        for i in 0..self.cfg.num_hidden_layers {
            let ln = self.layer_names(i);
            names.extend([ln.ln1, ln.q, ln.k, ln.v, ln.o, ln.ln2, ln.gate, ln.up, ln.down]);
        }
        names.push("model.norm.weight".to_string());
        names.push("lm_head.weight".to_string());
        names
    }

    fn mlp_out_names(&self) -> Vec<String> {
        (0..self.cfg.num_hidden_layers)
            .map(|i| format!("model.layers.{i}.mlp.down_proj.weight"))
            .collect()
    }

    fn lens_logits(&self, ws: &WeightWorkspace, hidden: &[Array3<f32>]) -> Result<Vec<Array2<f32>>> {
        let norm_w = weight_1d(ws, "model.norm.weight")?;
        let head = self.head_weight(ws)?;
        let n = hidden.len();
        let mut out = Vec::new();
        for (l_idx, hs) in hidden.iter().enumerate() {
            let b = hs.shape()[0];
            let s = hs.shape()[1];
            let last = hs.slice(ndarray::s![.., s - 1, ..]).to_owned();
            let normed = if l_idx < n - 1 {
                rmsnorm(&last, &norm_w, self.cfg.rms_norm_eps as f32)
            } else {
                last
            };
            let logits = linear(&normed, &head);
            let _ = b;
            out.push(logits);
        }
        Ok(out)
    }

    fn weight_grads(&self, ws: &WeightWorkspace, ids: &Array2<i64>) -> Result<HashMap<String, Tensor>> {
        let mut grads: HashMap<String, Array2<f32>> = HashMap::new();
        let mut add = |name: &str, g: Array2<f32>, grads: &mut HashMap<String, Array2<f32>>| {
            grads
                .entry(name.to_string())
                .and_modify(|e| *e = &*e + &g)
                .or_insert(g);
        };
        let b = ids.shape()[0];
        let s = ids.shape()[1];
        if s < 2 {
            return Ok(HashMap::new());
        }
        let n_pos = (b * (s - 1)) as f32;
        for bi in 0..b {
            let seq: Vec<i64> = ids.row(bi).to_vec();
            let single = Array2::from_shape_vec((1, s), seq.clone()).unwrap();
            let (_out, cache) = self.forward_impl(ws, &single, false, true)?;
            let logits = self.head_logits(ws, &cache)?;
            let shift = logits.slice(ndarray::s![..s - 1, ..]).to_owned();
            let targets: Vec<i64> = seq[1..].to_vec();
            let mut dlogits = cross_entropy_grad(&shift, &targets, false);
            dlogits.mapv_inplace(|v| v / n_pos);
            let gmap = self.backward_impl(ws, &cache, &dlogits)?;
            for (name, g) in gmap {
                // Tied embeddings: head gradient accumulates into the embedding matrix.
                if name == "lm_head.weight" && !ws.contains("lm_head.weight") {
                    add("model.embed_tokens.weight", g, &mut grads);
                } else {
                    add(&name, g, &mut grads);
                }
            }
        }
        Ok(grads
            .into_iter()
            .filter(|(k, _)| ws.contains(k))
            .map(|(k, v)| {
                let shp = ws.shape(&k).unwrap();
                let t = crate::tensor::from_vec(&shp, v.iter().copied().collect());
                (k, t)
            })
            .collect())
    }

    fn mlp_keys_grads(
        &self,
        ws: &WeightWorkspace,
        seqs: &[Vec<i64>],
        need_grads: bool,
    ) -> Result<(HashMap<String, Array2<f64>>, HashMap<String, Array2<f64>>)> {
        let mut keys: HashMap<String, Vec<Array2<f64>>> = HashMap::new();
        let mut grads: HashMap<String, Vec<Array2<f64>>> = HashMap::new();
        for seq in seqs {
            if seq.len() < 2 {
                continue;
            }
            let ids = Array2::from_shape_vec((1, seq.len()), seq.clone()).unwrap();
            let (_out, cache) = self.forward_impl(ws, &ids, false, need_grads)?;
            for (i, lc) in cache.layers.iter().enumerate() {
                let name = format!("model.layers.{i}.mlp.down_proj.weight");
                let s = lc.m.nrows();
                let k = lc.m.slice(ndarray::s![..s - 1, ..]).mapv(|v| v as f64);
                keys.entry(name.clone()).or_default().push(k);
            }
            if need_grads {
                let logits = self.head_logits(ws, &cache)?;
                let s = seq.len();
                let shift = logits.slice(ndarray::s![..s - 1, ..]).to_owned();
                let targets: Vec<i64> = seq[1..].to_vec();
                let dlogits = cross_entropy_grad(&shift, &targets, false); // sum reduction
                // gradient at each layer's down_proj output = residual stream grad at layer output
                let stream_grads = self.backward_stream_only(ws, &cache, &dlogits)?;
                for (i, sg) in stream_grads.iter().enumerate() {
                    let name = format!("model.layers.{i}.mlp.down_proj.weight");
                    let s = sg.nrows();
                    let g = sg.slice(ndarray::s![..s - 1, ..]).mapv(|v| v as f64);
                    grads.entry(name).or_default().push(g);
                }
            }
        }
        let concat = |m: HashMap<String, Vec<Array2<f64>>>| -> HashMap<String, Array2<f64>> {
            m.into_iter()
                .map(|(k, parts)| {
                    let cols = parts.first().map(|p| p.ncols()).unwrap_or(0);
                    let rows: usize = parts.iter().map(|p| p.nrows()).sum();
                    let mut data = Vec::with_capacity(rows * cols);
                    for p in &parts {
                        data.extend(p.iter().copied());
                    }
                    (k, Array2::from_shape_vec((rows, cols), data).unwrap())
                })
                .collect()
        };
        Ok((concat(keys), concat(grads)))
    }
}

impl LlamaModel {
    fn head_logits(&self, ws: &WeightWorkspace, cache: &ForwardCache) -> Result<Array2<f32>> {
        let head = self.head_weight(ws)?;
        Ok(linear(&cache.xn, &head))
    }

    fn forward_impl(
        &self,
        ws: &WeightWorkspace,
        ids: &Array2<i64>,
        want_hidden: bool,
        _need_cache: bool,
    ) -> Result<(ForwardOut, ForwardCache)> {
        let cfg = &self.cfg;
        let b = ids.shape()[0];
        let s = ids.shape()[1];
        let d = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let group = nh / nkv.max(1);

        let embed = weight_2d(ws, "model.embed_tokens.weight")?;
        let norm_w = weight_1d(ws, "model.norm.weight")?;
        let head = self.head_weight(ws)?;

        let mut all_logits = Array3::<f32>::zeros((b, s, cfg.vocab_size));
        let mut all_hidden: Vec<Array3<f32>> = if want_hidden {
            (0..cfg.num_hidden_layers).map(|_| Array3::zeros((b, s, d))).collect()
        } else {
            Vec::new()
        };
        // cache for batch 0 only (backward operates per-sequence)
        let mut cache0: Option<ForwardCache> = None;

        for bi in 0..b {
            let seq: Vec<i64> = ids.row(bi).to_vec();
            let mut x = Array2::<f32>::zeros((s, d));
            for (p, &tok) in seq.iter().enumerate() {
                let row = embed.row((tok.max(0) as usize).min(embed.nrows() - 1));
                for j in 0..d {
                    x[[p, j]] = row[j];
                }
            }
            let mut layer_caches = Vec::new();
            for li in 0..cfg.num_hidden_layers {
                let ln = self.layer_names(li);
                let ln1_w = weight_1d(ws, &ln.ln1)?;
                let ln2_w = weight_1d(ws, &ln.ln2)?;
                let wq = weight_2d(ws, &ln.q)?;
                let wk = weight_2d(ws, &ln.k)?;
                let wv = weight_2d(ws, &ln.v)?;
                let wo = weight_2d(ws, &ln.o)?;
                let wg = weight_2d(ws, &ln.gate)?;
                let wu = weight_2d(ws, &ln.up)?;
                let wd = weight_2d(ws, &ln.down)?;

                let x_in = x.clone();
                let n1 = rmsnorm(&x, &ln1_w, cfg.rms_norm_eps as f32);
                let q_full = add_opt_bias(linear(&n1, &wq), ws, &ln.q.replace(".weight", ".bias"));
                let k_full = add_opt_bias(linear(&n1, &wk), ws, &ln.k.replace(".weight", ".bias"));
                let v_full = add_opt_bias(linear(&n1, &wv), ws, &ln.v.replace(".weight", ".bias"));

                let mut q_heads: Vec<Array2<f32>> = (0..nh)
                    .map(|h| q_full.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned())
                    .collect();
                let mut k_heads: Vec<Array2<f32>> = (0..nkv)
                    .map(|h| k_full.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned())
                    .collect();
                let v_heads: Vec<Array2<f32>> = (0..nkv)
                    .map(|h| v_full.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned())
                    .collect();

                if cfg.use_rope {
                    for h in 0..nh {
                        rope_apply(&mut q_heads[h], cfg.rope_theta);
                    }
                    for h in 0..nkv {
                        rope_apply(&mut k_heads[h], cfg.rope_theta);
                    }
                }

                let scale = 1.0 / (hd as f32).sqrt();
                let mut probs_heads = Vec::new();
                let mut attn_concat = Array2::<f32>::zeros((s, nh * hd));
                for h in 0..nh {
                    let kvh = h / group;
                    let scores = q_heads[h].dot(&k_heads[kvh].t()).mapv(|v| v * scale);
                    let probs = causal_softmax(&scores);
                    let ctx = probs.dot(&v_heads[kvh]);
                    for p in 0..s {
                        for j in 0..hd {
                            attn_concat[[p, h * hd + j]] = ctx[[p, j]];
                        }
                    }
                    probs_heads.push(probs);
                }
                let o = linear(&attn_concat, &wo);
                let x2 = &x + &o;
                let n2 = rmsnorm(&x2, &ln2_w, cfg.rms_norm_eps as f32);
                let g = linear(&n2, &wg);
                let u = linear(&n2, &wu);
                let m = &g.mapv(silu) * &u;
                let down_out = linear(&m, &wd);
                x = &x2 + &down_out;

                if want_hidden {
                    for p in 0..s {
                        for j in 0..d {
                            all_hidden[li][[bi, p, j]] = x[[p, j]];
                        }
                    }
                }
                layer_caches.push(LayerCache {
                    x_in,
                    n1,
                    q_rot: q_heads,
                    k_rot: k_heads,
                    v_heads,
                    probs: probs_heads,
                    attn_concat,
                    x2,
                    n2,
                    g,
                    u,
                    m,
                });
            }
            let xn = rmsnorm(&x, &norm_w, cfg.rms_norm_eps as f32);
            let logits = linear(&xn, &head);
            for p in 0..s {
                for j in 0..cfg.vocab_size {
                    all_logits[[bi, p, j]] = logits[[p, j]];
                }
            }
            if bi == 0 {
                cache0 = Some(ForwardCache {
                    ids: seq,
                    layers: layer_caches,
                    x_final: x,
                    xn,
                });
            }
        }

        Ok((
            ForwardOut { logits: all_logits, hidden_states: all_hidden },
            cache0.unwrap_or(ForwardCache {
                ids: vec![],
                layers: vec![],
                x_final: Array2::zeros((0, d)),
                xn: Array2::zeros((0, d)),
            }),
        ))
    }
}

// ---------------------------------------------------------------------------
// RoPE (HF rotate_half variant)
// ---------------------------------------------------------------------------

fn rope_freqs(seq_len: usize, hd: usize, theta: f64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let half = hd / 2;
    let mut cos = vec![vec![0.0f32; hd]; seq_len];
    let mut sin = vec![vec![0.0f32; hd]; seq_len];
    for p in 0..seq_len {
        for j in 0..half {
            let inv_freq = 1.0 / theta.powf((2 * j) as f64 / hd as f64);
            let ang = p as f64 * inv_freq;
            let c = ang.cos() as f32;
            let s = ang.sin() as f32;
            cos[p][j] = c;
            cos[p][j + half] = c;
            sin[p][j] = s;
            sin[p][j + half] = s;
        }
    }
    (cos, sin)
}

fn rope_apply(x: &mut Array2<f32>, theta: f64) {
    let s = x.nrows();
    let hd = x.ncols();
    let half = hd / 2;
    let (cos, sin) = rope_freqs(s, hd, theta);
    for p in 0..s {
        for j in 0..half {
            let x1 = x[[p, j]];
            let x2 = x[[p, j + half]];
            x[[p, j]] = x1 * cos[p][j] - x2 * sin[p][j];
            x[[p, j + half]] = x2 * cos[p][j + half] + x1 * sin[p][j + half];
        }
    }
}

fn rope_backward(g: &Array2<f32>, theta: f64) -> Array2<f32> {
    let s = g.nrows();
    let hd = g.ncols();
    let half = hd / 2;
    let (cos, sin) = rope_freqs(s, hd, theta);
    let mut out = Array2::<f32>::zeros((s, hd));
    for p in 0..s {
        for j in 0..half {
            let g1 = g[[p, j]];
            let g2 = g[[p, j + half]];
            // adjoint of rotate_half: R^T(g) = [g[half:], -g[:half]]
            out[[p, j]] = g1 * cos[p][j] + g2 * sin[p][j + half];
            out[[p, j + half]] = g2 * cos[p][j + half] - g1 * sin[p][j];
        }
    }
    out
}

pub(crate) fn causal_softmax(scores: &Array2<f32>) -> Array2<f32> {
    let s = scores.nrows();
    let mut out = scores.clone();
    for i in 0..s {
        let mut mx = f32::NEG_INFINITY;
        for j in 0..=i {
            mx = mx.max(out[[i, j]]);
        }
        let mut sum = 0.0f32;
        for j in 0..s {
            if j <= i {
                out[[i, j]] = (out[[i, j]] - mx).exp();
                sum += out[[i, j]];
            } else {
                out[[i, j]] = 0.0;
            }
        }
        for j in 0..=i {
            out[[i, j]] /= sum;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Backward helpers
// ---------------------------------------------------------------------------

/// Add an optional bias vector (Qwen2 q/k/v projections) when present.
fn add_opt_bias(mut a: Array2<f32>, ws: &WeightWorkspace, name: &str) -> Array2<f32> {
    if let Ok(b) = weight_1d(ws, name) {
        if b.len() == a.ncols() {
            for mut row in a.rows_mut() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v += b[j];
                }
            }
        }
    }
    a
}

pub(crate) fn rmsnorm_backward(
    dy: &Array2<f32>,
    x: &Array2<f32>,
    weight: &[f32],
    eps: f32,
) -> (Array2<f32>, Vec<f32>) {
    let (s, d) = (x.nrows(), x.ncols());
    let mut dx = Array2::<f32>::zeros((s, d));
    let mut dw = vec![0.0f32; d];
    for p in 0..s {
        let ms: f32 = (0..d).map(|j| x[[p, j]] * x[[p, j]]).sum::<f32>() / d as f32;
        let r = 1.0 / (ms + eps).sqrt();
        let dyw: Vec<f32> = (0..d).map(|j| dy[[p, j]] * weight[j]).collect();
        let dot: f32 = (0..d).map(|j| dyw[j] * x[[p, j]]).sum();
        for j in 0..d {
            dx[[p, j]] = r * dyw[j] - (r * r * r / d as f32) * x[[p, j]] * dot;
            dw[j] += dy[[p, j]] * x[[p, j]] * r;
        }
    }
    (dx, dw)
}

pub(crate) fn layernorm_backward(
    dy: &Array2<f32>,
    x: &Array2<f32>,
    weight: &[f32],
    eps: f32,
) -> (Array2<f32>, Vec<f32>, Vec<f32>) {
    let (s, d) = (x.nrows(), x.ncols());
    let mut dx = Array2::<f32>::zeros((s, d));
    let mut dw = vec![0.0f32; d];
    let mut db = vec![0.0f32; d];
    for p in 0..s {
        let mean: f32 = (0..d).map(|j| x[[p, j]]).sum::<f32>() / d as f32;
        let var: f32 = (0..d).map(|j| (x[[p, j]] - mean) * (x[[p, j]] - mean)).sum::<f32>() / d as f32;
        let r = 1.0 / (var + eps).sqrt();
        let dyw: Vec<f32> = (0..d).map(|j| dy[[p, j]] * weight[j]).collect();
        let m1: f32 = dyw.iter().sum::<f32>() / d as f32;
        let m2: f32 = (0..d).map(|j| dyw[j] * (x[[p, j]] - mean)).sum::<f32>() / d as f32;
        for j in 0..d {
            dx[[p, j]] = r * (dyw[j] - m1 - (x[[p, j]] - mean) * r * r * m2);
            dw[j] += dy[[p, j]] * (x[[p, j]] - mean) * r;
            db[j] += dy[[p, j]];
        }
    }
    (dx, dw, db)
}

/// Softmax backward per row: `ds = p * (gp - sum(gp*p))`.
pub(crate) fn softmax_backward(probs: &Array2<f32>, gprobs: &Array2<f32>) -> Array2<f32> {
    let mut out = probs.clone();
    for i in 0..probs.nrows() {
        let dot: f32 = probs.row(i).iter().zip(gprobs.row(i)).map(|(p, g)| p * g).sum();
        for j in 0..probs.ncols() {
            out[[i, j]] = probs[[i, j]] * (gprobs[[i, j]] - dot);
        }
    }
    out
}

impl LlamaModel {
    /// Full backward: gradients for every weight. `dlogits` is [S-1, V].
    fn backward_impl(
        &self,
        ws: &WeightWorkspace,
        cache: &ForwardCache,
        dlogits: &Array2<f32>,
    ) -> Result<HashMap<String, Array2<f32>>> {
        let cfg = &self.cfg;
        let s = cache.x_final.nrows();
        let d = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let group = nh / nkv.max(1);
        let mut grads: HashMap<String, Array2<f32>> = HashMap::new();
        let mut g1d: HashMap<String, Vec<f32>> = HashMap::new();

        let head = self.head_weight(ws)?;
        let norm_w = weight_1d(ws, "model.norm.weight")?;
        let embed = weight_2d(ws, "model.embed_tokens.weight")?;

        // head
        let mut gxn = dlogits.dot(&head); // [S-1, d]
        let g_head = dlogits.t().dot(&cache.xn.slice(ndarray::s![..s - 1, ..]));
        grads.insert("lm_head.weight".into(), g_head);

        // pad gradient to full sequence (last position has no loss)
        let mut gx_full = Array2::<f32>::zeros((s, d));
        gx_full
            .slice_mut(ndarray::s![..s - 1, ..])
            .assign(&gxn);
        gxn = gx_full;

        // final norm
        let (mut gx, g_norm) = rmsnorm_backward(&gxn, &cache.x_final, &norm_w, cfg.rms_norm_eps as f32);
        g1d.insert("model.norm.weight".into(), g_norm);

        for li in (0..cfg.num_hidden_layers).rev() {
            let lc = &cache.layers[li];
            let ln = self.layer_names(li);
            let ln1_w = weight_1d(ws, &ln.ln1)?;
            let ln2_w = weight_1d(ws, &ln.ln2)?;
            let wq = weight_2d(ws, &ln.q)?;
            let wk = weight_2d(ws, &ln.k)?;
            let wv = weight_2d(ws, &ln.v)?;
            let wo = weight_2d(ws, &ln.o)?;
            let wg = weight_2d(ws, &ln.gate)?;
            let wu = weight_2d(ws, &ln.up)?;
            let wd = weight_2d(ws, &ln.down)?;

            // residual split: x_out = x2 + down_out
            let gx2 = gx.clone();
            let gdown = gx.clone();

            // down proj
            let gm = gdown.dot(&wd);
            grads.insert(ln.down.clone(), gdown.t().dot(&lc.m));

            // SwiGLU
            let gg = &gm * &lc.u * &lc.g.mapv(silu_grad);
            let gu = &gm * &lc.g.mapv(silu);
            let mut gn2 = gg.dot(&wg) + gu.dot(&wu);
            grads.insert(ln.gate.clone(), gg.t().dot(&lc.n2));
            grads.insert(ln.up.clone(), gu.t().dot(&lc.n2));

            // post-attn norm
            let (gx2_from_norm, g_ln2) =
                rmsnorm_backward(&gn2, &lc.x2, &ln2_w, cfg.rms_norm_eps as f32);
            gn2 = gx2_from_norm;
            g1d.insert(ln.ln2.clone(), g_ln2);
            let gx2_total = &gx2 + &gn2;

            // o proj
            let gattn = gx2_total.dot(&wo);
            grads.insert(ln.o.clone(), gx2_total.t().dot(&lc.attn_concat));

            // attention backward
            let scale = 1.0 / (hd as f32).sqrt();
            let mut gq_full = Array2::<f32>::zeros((s, nh * hd));
            let mut gk_full = Array2::<f32>::zeros((s, nkv * hd));
            let mut gv_full = Array2::<f32>::zeros((s, nkv * hd));
            for h in 0..nh {
                let kvh = h / group;
                let gattn_h = gattn.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned();
                let probs = &lc.probs[h];
                let gprobs = gattn_h.dot(&lc.v_heads[kvh].t());
                let gv_h = probs.t().dot(&gattn_h);
                let gscores = softmax_backward(probs, &gprobs).mapv(|v| v * scale);
                let gq_h = gscores.dot(&lc.k_rot[kvh]);
                let gk_h = gscores.t().dot(&lc.q_rot[h]);
                // accumulate kv grads across grouped query heads
                for p in 0..s {
                    for j in 0..hd {
                        gv_full[[p, kvh * hd + j]] += gv_h[[p, j]];
                        gk_full[[p, kvh * hd + j]] += gk_h[[p, j]];
                    }
                }
                let gq_h = if cfg.use_rope { rope_backward(&gq_h, cfg.rope_theta) } else { gq_h };
                for p in 0..s {
                    for j in 0..hd {
                        gq_full[[p, h * hd + j]] = gq_h[[p, j]];
                    }
                }
            }
            let gk_full = if cfg.use_rope {
                let mut out = Array2::<f32>::zeros((s, nkv * hd));
                for h in 0..nkv {
                    let gk_h = gk_full.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned();
                    let gk_h = rope_backward(&gk_h, cfg.rope_theta);
                    for p in 0..s {
                        for j in 0..hd {
                            out[[p, h * hd + j]] = gk_h[[p, j]];
                        }
                    }
                }
                out
            } else {
                gk_full
            };

            let mut gn1 = gq_full.dot(&wq) + gk_full.dot(&wk) + gv_full.dot(&wv);
            grads.insert(ln.q.clone(), gq_full.t().dot(&lc.n1));
            grads.insert(ln.k.clone(), gk_full.t().dot(&lc.n1));
            grads.insert(ln.v.clone(), gv_full.t().dot(&lc.n1));
            // Qwen2-style attention biases (present only in some checkpoints).
            for (wname, g) in [(&ln.q, &gq_full), (&ln.k, &gk_full), (&ln.v, &gv_full)] {
                let bname = wname.replace(".weight", ".bias");
                if ws.contains(&bname) {
                    g1d.insert(bname, g.columns().into_iter().map(|c| c.sum()).collect());
                }
            }

            let (gx_from_norm, g_ln1) =
                rmsnorm_backward(&gn1, &lc.x_in, &ln1_w, cfg.rms_norm_eps as f32);
            gn1 = gx_from_norm;
            g1d.insert(ln.ln1.clone(), g_ln1);
            gx = &gx2_total + &gn1;
        }

        // embedding scatter-add
        let mut g_embed = Array2::<f32>::zeros(embed.raw_dim());
        for (p, &tok) in cache.ids.iter().enumerate() {
            let row = (tok.max(0) as usize).min(embed.nrows() - 1);
            for j in 0..d {
                g_embed[[row, j]] += gx[[p, j]];
            }
        }
        grads.insert("model.embed_tokens.weight".into(), g_embed);

        // convert 1d grads to 2d [1, d] for uniform storage; reshaped by caller
        for (name, g) in g1d {
            grads.insert(name, Array2::from_shape_vec((1, g.len()), g).unwrap());
        }
        Ok(grads)
    }

    /// Backward that only returns residual-stream gradients at each layer output
    /// (== gradients at down_proj outputs), used by domain EAMIT calibration.
    fn backward_stream_only(
        &self,
        ws: &WeightWorkspace,
        cache: &ForwardCache,
        dlogits: &Array2<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        let cfg = &self.cfg;
        let s = cache.x_final.nrows();
        let d = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let group = nh / nkv.max(1);

        let head = self.head_weight(ws)?;
        let norm_w = weight_1d(ws, "model.norm.weight")?;

        let gxn = dlogits.dot(&head);
        let mut gx = Array2::<f32>::zeros((s, d));
        gx.slice_mut(ndarray::s![..s - 1, ..]).assign(&gxn);
        let (mut gx, _) = rmsnorm_backward(&gx, &cache.x_final, &norm_w, cfg.rms_norm_eps as f32);

        let mut stream_grads = vec![Array2::<f32>::zeros((s, d)); cfg.num_hidden_layers];
        for li in (0..cfg.num_hidden_layers).rev() {
            stream_grads[li] = gx.clone();
            let lc = &cache.layers[li];
            let ln = self.layer_names(li);
            let ln1_w = weight_1d(ws, &ln.ln1)?;
            let ln2_w = weight_1d(ws, &ln.ln2)?;
            let wq = weight_2d(ws, &ln.q)?;
            let wk = weight_2d(ws, &ln.k)?;
            let wv = weight_2d(ws, &ln.v)?;
            let wo = weight_2d(ws, &ln.o)?;
            let wg = weight_2d(ws, &ln.gate)?;
            let wu = weight_2d(ws, &ln.up)?;
            let wd = weight_2d(ws, &ln.down)?;

            let gx2 = gx.clone();
            let gdown = gx.clone();
            let gm = gdown.dot(&wd);
            let gg = &gm * &lc.u * &lc.g.mapv(silu_grad);
            let gu = &gm * &lc.g.mapv(silu);
            let gn2 = gg.dot(&wg) + gu.dot(&wu);
            let (gx2n, _) = rmsnorm_backward(&gn2, &lc.x2, &ln2_w, cfg.rms_norm_eps as f32);
            let gx2_total = &gx2 + &gx2n;

            let gattn = gx2_total.dot(&wo);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut gq_full = Array2::<f32>::zeros((s, nh * hd));
            let mut gk_full = Array2::<f32>::zeros((s, nkv * hd));
            let mut gv_full = Array2::<f32>::zeros((s, nkv * hd));
            for h in 0..nh {
                let kvh = h / group;
                let gattn_h = gattn.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned();
                let probs = &lc.probs[h];
                let gprobs = gattn_h.dot(&lc.v_heads[kvh].t());
                let gv_h = probs.t().dot(&gattn_h);
                let gscores = softmax_backward(probs, &gprobs).mapv(|v| v * scale);
                let gq_h = gscores.dot(&lc.k_rot[kvh]);
                let gk_h = gscores.t().dot(&lc.q_rot[h]);
                for p in 0..s {
                    for j in 0..hd {
                        gv_full[[p, kvh * hd + j]] += gv_h[[p, j]];
                        gk_full[[p, kvh * hd + j]] += gk_h[[p, j]];
                    }
                }
                let gq_h = if cfg.use_rope { rope_backward(&gq_h, cfg.rope_theta) } else { gq_h };
                for p in 0..s {
                    for j in 0..hd {
                        gq_full[[p, h * hd + j]] = gq_h[[p, j]];
                    }
                }
            }
            let gk_full = if cfg.use_rope {
                let mut out = Array2::<f32>::zeros((s, nkv * hd));
                for h in 0..nkv {
                    let gk_h = gk_full.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned();
                    let gk_h = rope_backward(&gk_h, cfg.rope_theta);
                    for p in 0..s {
                        for j in 0..hd {
                            out[[p, h * hd + j]] = gk_h[[p, j]];
                        }
                    }
                }
                out
            } else {
                gk_full
            };
            let gn1 = gq_full.dot(&wq) + gk_full.dot(&wk) + gv_full.dot(&wv);
            let (gx_n, _) = rmsnorm_backward(&gn1, &lc.x_in, &ln1_w, cfg.rms_norm_eps as f32);
            gx = &gx2_total + &gx_n;
        }
        Ok(stream_grads)
    }
}

#[allow(dead_code)]
fn _axis_helper(a: &Array2<f32>) -> ndarray::Array1<f32> {
    a.sum_axis(Axis(0))
}

#[allow(dead_code)]
fn _err_helper() -> WeightXError {
    WeightXError::Value("unused".into())
}