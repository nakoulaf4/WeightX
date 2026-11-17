// SPDX-License-Identifier: GPL-3.0-or-later
//! GPT-2 runtime (LayerNorm + Conv1D + causal MHA + GELU), forward and backward.

use std::collections::HashMap;

use ndarray::{Array2, Array3};

use crate::error::Result;
use crate::model::llama::{causal_softmax, layernorm_backward, softmax_backward};
use crate::model::*;
use crate::tensor::Tensor;
use crate::workspace::WeightWorkspace;

pub struct Gpt2Model {
    pub cfg: ModelConfig,
}

impl Gpt2Model {
    pub fn new(cfg: ModelConfig) -> Self {
        Self { cfg }
    }

    fn block_prefix(i: usize) -> String {
        format!("transformer.h.{i}")
    }
}

struct Gpt2LayerCache {
    x_in: Array2<f32>,
    n1: Array2<f32>,
    q: Array2<f32>,
    k: Array2<f32>,
    v: Array2<f32>,
    probs: Vec<Array2<f32>>, // per head [S, S]
    attn_concat: Array2<f32>,
    x2: Array2<f32>,
    n2: Array2<f32>,
    fc: Array2<f32>,   // pre-gelu [S, 4d]
    act: Array2<f32>,  // post-gelu (MLP keys)
}

struct Gpt2Cache {
    ids: Vec<i64>,
    layers: Vec<Gpt2LayerCache>,
    x_final: Array2<f32>,
    xn: Array2<f32>,
}

impl ModelArch for Gpt2Model {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    fn forward(&self, ws: &WeightWorkspace, ids: &Array2<i64>, want_hidden: bool) -> Result<ForwardOut> {
        let (out, _) = self.forward_impl(ws, ids, want_hidden)?;
        Ok(out)
    }

    fn weight_names(&self) -> Vec<String> {
        let mut names = vec![
            "transformer.wte.weight".to_string(),
            "transformer.wpe.weight".to_string(),
        ];
        for i in 0..self.cfg.num_hidden_layers {
            let p = Self::block_prefix(i);
            for suffix in [
                "ln_1.weight", "ln_1.bias",
                "attn.c_attn.weight", "attn.c_attn.bias",
                "attn.c_proj.weight", "attn.c_proj.bias",
                "ln_2.weight", "ln_2.bias",
                "mlp.c_fc.weight", "mlp.c_fc.bias",
                "mlp.c_proj.weight", "mlp.c_proj.bias",
            ] {
                names.push(format!("{p}.{suffix}"));
            }
        }
        names.push("transformer.ln_f.weight".to_string());
        names.push("transformer.ln_f.bias".to_string());
        names
    }

    fn mlp_out_names(&self) -> Vec<String> {
        (0..self.cfg.num_hidden_layers)
            .map(|i| format!("{}.mlp.c_proj.weight", Self::block_prefix(i)))
            .collect()
    }

    fn lens_logits(&self, ws: &WeightWorkspace, hidden: &[Array3<f32>]) -> Result<Vec<Array2<f32>>> {
        let nw = weight_1d(ws, "transformer.ln_f.weight")?;
        let nb = weight_1d(ws, "transformer.ln_f.bias")?;
        let wte = weight_2d(ws, "transformer.wte.weight")?;
        let n = hidden.len();
        let mut out = Vec::new();
        for (l_idx, hs) in hidden.iter().enumerate() {
            let s = hs.shape()[1];
            let last = hs.slice(ndarray::s![.., s - 1, ..]).to_owned();
            let normed = if l_idx < n - 1 {
                layernorm(&last, &nw, &nb, self.cfg.layer_norm_eps as f32)
            } else {
                last
            };
            out.push(normed.dot(&wte.t()));
        }
        Ok(out)
    }

    fn weight_grads(&self, ws: &WeightWorkspace, ids: &Array2<i64>) -> Result<HashMap<String, Tensor>> {
        let mut grads: HashMap<String, Array2<f32>> = HashMap::new();
        let b = ids.shape()[0];
        let s = ids.shape()[1];
        if s < 2 {
            return Ok(HashMap::new());
        }
        let n_pos = (b * (s - 1)) as f32;
        for bi in 0..b {
            let seq: Vec<i64> = ids.row(bi).to_vec();
            let single = Array2::from_shape_vec((1, s), seq.clone()).unwrap();
            let (_out, cache) = self.forward_impl(ws, &single, false)?;
            let wte = weight_2d(ws, "transformer.wte.weight")?;
            let logits = cache.xn.dot(&wte.t());
            let shift = logits.slice(ndarray::s![..s - 1, ..]).to_owned();
            let targets: Vec<i64> = seq[1..].to_vec();
            let mut dlogits = cross_entropy_grad(&shift, &targets, false);
            dlogits.mapv_inplace(|v| v / n_pos);
            let gmap = self.backward_impl(ws, &cache, &dlogits, true)?;
            for (name, g) in gmap {
                grads
                    .entry(name)
                    .and_modify(|e| *e = &*e + &g)
                    .or_insert(g);
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
            let (_out, cache) = self.forward_impl(ws, &ids, false)?;
            for (i, lc) in cache.layers.iter().enumerate() {
                let name = format!("{}.mlp.c_proj.weight", Self::block_prefix(i));
                let s = lc.act.nrows();
                keys.entry(name).or_default().push(lc.act.slice(ndarray::s![..s - 1, ..]).mapv(|v| v as f64));
            }
            if need_grads {
                let wte = weight_2d(ws, "transformer.wte.weight")?;
                let logits = cache.xn.dot(&wte.t());
                let s = seq.len();
                let shift = logits.slice(ndarray::s![..s - 1, ..]).to_owned();
                let targets: Vec<i64> = seq[1..].to_vec();
                let dlogits = cross_entropy_grad(&shift, &targets, false);
                let gmap = self.backward_impl(ws, &cache, &dlogits, false)?;
                for (i, _) in cache.layers.iter().enumerate() {
                    let name = format!("{}.mlp.c_proj.weight", Self::block_prefix(i));
                    let gname = format!("__stream__.{i}");
                    if let Some(g) = gmap.get(&gname) {
                        let s = g.nrows();
                        grads.entry(name).or_default().push(g.slice(ndarray::s![..s - 1, ..]).mapv(|v| v as f64));
                    }
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

impl Gpt2Model {
    fn forward_impl(
        &self,
        ws: &WeightWorkspace,
        ids: &Array2<i64>,
        want_hidden: bool,
    ) -> Result<(ForwardOut, Gpt2Cache)> {
        let cfg = &self.cfg;
        let b = ids.shape()[0];
        let s = ids.shape()[1];
        let d = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let hd = d / nh;

        let wte = weight_2d(ws, "transformer.wte.weight")?;
        let wpe = weight_2d(ws, "transformer.wpe.weight")?;
        let ln_f_w = weight_1d(ws, "transformer.ln_f.weight")?;
        let ln_f_b = weight_1d(ws, "transformer.ln_f.bias")?;

        let mut all_logits = Array3::<f32>::zeros((b, s, cfg.vocab_size));
        let mut all_hidden: Vec<Array3<f32>> = if want_hidden {
            (0..cfg.num_hidden_layers).map(|_| Array3::zeros((b, s, d))).collect()
        } else {
            Vec::new()
        };
        let mut cache0: Option<Gpt2Cache> = None;

        for bi in 0..b {
            let seq: Vec<i64> = ids.row(bi).to_vec();
            let mut x = Array2::<f32>::zeros((s, d));
            for p in 0..s {
                let tok = (seq[p].max(0) as usize).min(wte.nrows() - 1);
                let pos = p.min(wpe.nrows() - 1);
                for j in 0..d {
                    x[[p, j]] = wte[[tok, j]] + wpe[[pos, j]];
                }
            }
            let mut layer_caches = Vec::new();
            for li in 0..cfg.num_hidden_layers {
                let p = Self::block_prefix(li);
                let ln1_w = weight_1d(ws, &format!("{p}.ln_1.weight"))?;
                let ln1_b = weight_1d(ws, &format!("{p}.ln_1.bias"))?;
                let cattn_w = weight_2d(ws, &format!("{p}.attn.c_attn.weight"))?;
                let cattn_b = weight_1d(ws, &format!("{p}.attn.c_attn.bias"))?;
                let cproj_w = weight_2d(ws, &format!("{p}.attn.c_proj.weight"))?;
                let cproj_b = weight_1d(ws, &format!("{p}.attn.c_proj.bias"))?;
                let ln2_w = weight_1d(ws, &format!("{p}.ln_2.weight"))?;
                let ln2_b = weight_1d(ws, &format!("{p}.ln_2.bias"))?;
                let cfc_w = weight_2d(ws, &format!("{p}.mlp.c_fc.weight"))?;
                let cfc_b = weight_1d(ws, &format!("{p}.mlp.c_fc.bias"))?;
                let mlp_proj_w = weight_2d(ws, &format!("{p}.mlp.c_proj.weight"))?;
                let mlp_proj_b = weight_1d(ws, &format!("{p}.mlp.c_proj.bias"))?;

                let x_in = x.clone();
                let n1 = layernorm(&x, &ln1_w, &ln1_b, cfg.layer_norm_eps as f32);
                let qkv = conv1d(&n1, &cattn_w, &cattn_b);
                let q = qkv.slice(ndarray::s![.., ..d]).to_owned();
                let k = qkv.slice(ndarray::s![.., d..2 * d]).to_owned();
                let v = qkv.slice(ndarray::s![.., 2 * d..]).to_owned();

                let scale = 1.0 / (hd as f32).sqrt();
                let mut probs_heads = Vec::new();
                let mut attn_concat = Array2::<f32>::zeros((s, d));
                for h in 0..nh {
                    let qh = q.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned();
                    let kh = k.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned();
                    let vh = v.slice(ndarray::s![.., h * hd..(h + 1) * hd]).to_owned();
                    let scores = qh.dot(&kh.t()).mapv(|v| v * scale);
                    let probs = causal_softmax(&scores);
                    let ctx = probs.dot(&vh);
                    for pp in 0..s {
                        for j in 0..hd {
                            attn_concat[[pp, h * hd + j]] = ctx[[pp, j]];
                        }
                    }
                    probs_heads.push(probs);
                }
                let attn_out = conv1d(&attn_concat, &cproj_w, &cproj_b);
                let x2 = &x + &attn_out;
                let n2 = layernorm(&x2, &ln2_w, &ln2_b, cfg.layer_norm_eps as f32);
                let fc = conv1d(&n2, &cfc_w, &cfc_b);
                let act = fc.mapv(gelu_new);
                let mlp_out = conv1d(&act, &mlp_proj_w, &mlp_proj_b);
                x = &x2 + &mlp_out;

                if want_hidden {
                    for pp in 0..s {
                        for j in 0..d {
                            all_hidden[li][[bi, pp, j]] = x[[pp, j]];
                        }
                    }
                }
                layer_caches.push(Gpt2LayerCache {
                    x_in, n1, q, k, v, probs: probs_heads, attn_concat, x2, n2, fc, act,
                });
            }
            let xn = layernorm(&x, &ln_f_w, &ln_f_b, cfg.layer_norm_eps as f32);
            let logits = xn.dot(&wte.t());
            for pp in 0..s {
                for j in 0..cfg.vocab_size {
                    all_logits[[bi, pp, j]] = logits[[pp, j]];
                }
            }
            if bi == 0 {
                cache0 = Some(Gpt2Cache { ids: seq, layers: layer_caches, x_final: x, xn });
            }
        }
        Ok((
            ForwardOut { logits: all_logits, hidden_states: all_hidden },
            cache0.unwrap_or(Gpt2Cache {
                ids: vec![],
                layers: vec![],
                x_final: Array2::zeros((0, d)),
                xn: Array2::zeros((0, d)),
            }),
        ))
    }

    /// Full backward. When `full` is false, only `__stream__.{i}` gradients are returned.
    fn backward_impl(
        &self,
        ws: &WeightWorkspace,
        cache: &Gpt2Cache,
        dlogits: &Array2<f32>,
        full: bool,
    ) -> Result<HashMap<String, Array2<f32>>> {
        let cfg = &self.cfg;
        let s = cache.x_final.nrows();
        let d = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let hd = d / nh;
        let mut grads: HashMap<String, Array2<f32>> = HashMap::new();
        let mut g1d: HashMap<String, Vec<f32>> = HashMap::new();

        let wte = weight_2d(ws, "transformer.wte.weight")?;
        let ln_f_w = weight_1d(ws, "transformer.ln_f.weight")?;
        let ln_f_b = weight_1d(ws, "transformer.ln_f.bias")?;

        let gxn = dlogits.dot(&wte);
        let mut gx = Array2::<f32>::zeros((s, d));
        gx.slice_mut(ndarray::s![..s - 1, ..]).assign(&gxn);
        if full {
            let g_wte_head = dlogits.t().dot(&cache.xn.slice(ndarray::s![..s - 1, ..]));
            grads.insert("transformer.wte.weight".into(), g_wte_head);
        }
        let (mut gx, g_lnf_w, g_lnf_b) =
            layernorm_backward(&gx, &cache.x_final, &ln_f_w, cfg.layer_norm_eps as f32);
        if full {
            g1d.insert("transformer.ln_f.weight".into(), g_lnf_w);
            g1d.insert("transformer.ln_f.bias".into(), g_lnf_b);
        }

        for li in (0..cfg.num_hidden_layers).rev() {
            if !full {
                grads.insert(format!("__stream__.{li}"), gx.clone());
            }
            let lc = &cache.layers[li];
            let p = Self::block_prefix(li);
            let ln1_w = weight_1d(ws, &format!("{p}.ln_1.weight"))?;
            let ln1_b = weight_1d(ws, &format!("{p}.ln_1.bias"))?;
            let cattn_w = weight_2d(ws, &format!("{p}.attn.c_attn.weight"))?;
            let cproj_w = weight_2d(ws, &format!("{p}.attn.c_proj.weight"))?;
            let ln2_w = weight_1d(ws, &format!("{p}.ln_2.weight"))?;
            let ln2_b = weight_1d(ws, &format!("{p}.ln_2.bias"))?;
            let cfc_w = weight_2d(ws, &format!("{p}.mlp.c_fc.weight"))?;
            let mlp_proj_w = weight_2d(ws, &format!("{p}.mlp.c_proj.weight"))?;

            let gx2 = gx.clone();
            let gmlp_out = gx.clone();

            // MLP backward (Conv1D)
            let gact = gmlp_out.dot(&mlp_proj_w.t());
            let gfc = &gact * &lc.fc.mapv(gelu_new_grad);
            let gn2 = gfc.dot(&cfc_w.t());
            if full {
                // Conv1D weights are stored [in, out]: dW = x^T @ gy.
                grads.insert(format!("{p}.mlp.c_proj.weight"), lc.act.t().dot(&gmlp_out));
                g1d.insert(format!("{p}.mlp.c_proj.bias"), col_sum(&gmlp_out));
                grads.insert(format!("{p}.mlp.c_fc.weight"), lc.n2.t().dot(&gfc));
                g1d.insert(format!("{p}.mlp.c_fc.bias"), col_sum(&gfc));
            }
            let (gx2n, g_ln2_w, g_ln2_b) =
                layernorm_backward(&gn2, &lc.x2, &ln2_w, cfg.layer_norm_eps as f32);
            if full {
                g1d.insert(format!("{p}.ln_2.weight"), g_ln2_w);
                g1d.insert(format!("{p}.ln_2.bias"), g_ln2_b);
            }
            let gx2_total = &gx2 + &gx2n;

            // attention output proj: y = attn_concat @ W + b, dL/dy = gx2_total.
            // `gattn` below is dL/d(attn_concat) = gx2_total @ W^T (feeds the heads).
            let gattn = gx2_total.dot(&cproj_w.t());
            if full {
                grads.insert(format!("{p}.attn.c_proj.weight"), lc.attn_concat.t().dot(&gx2_total));
                g1d.insert(format!("{p}.attn.c_proj.bias"), col_sum(&gx2_total));
            }

            let scale = 1.0 / (hd as f32).sqrt();
            let mut gq = Array2::<f32>::zeros((s, d));
            let mut gk = Array2::<f32>::zeros((s, d));
            let mut gv = Array2::<f32>::zeros((s, d));
            for h in 0..nh {
                let sl = ndarray::s![.., h * hd..(h + 1) * hd];
                let gattn_h = gattn.slice(sl).to_owned();
                let qh = lc.q.slice(sl).to_owned();
                let kh = lc.k.slice(sl).to_owned();
                let vh = lc.v.slice(sl).to_owned();
                let probs = &lc.probs[h];
                let gprobs = gattn_h.dot(&vh.t());
                let gv_h = probs.t().dot(&gattn_h);
                let gscores = softmax_backward(probs, &gprobs).mapv(|v| v * scale);
                let gq_h = gscores.dot(&kh);
                let gk_h = gscores.t().dot(&qh);
                for pp in 0..s {
                    for j in 0..hd {
                        gq[[pp, h * hd + j]] = gq_h[[pp, j]];
                        gk[[pp, h * hd + j]] = gk_h[[pp, j]];
                        gv[[pp, h * hd + j]] = gv_h[[pp, j]];
                    }
                }
            }
            let gqkv = ndarray::concatenate(ndarray::Axis(1), &[gq.view(), gk.view(), gv.view()]).unwrap();
            let gn1 = gqkv.dot(&cattn_w.t());
            if full {
                grads.insert(format!("{p}.attn.c_attn.weight"), lc.n1.t().dot(&gqkv));
                g1d.insert(format!("{p}.attn.c_attn.bias"), col_sum(&gqkv));
            }
            let (gx_n, g_ln1_w, g_ln1_b) =
                layernorm_backward(&gn1, &lc.x_in, &ln1_w, cfg.layer_norm_eps as f32);
            if full {
                g1d.insert(format!("{p}.ln_1.weight"), g_ln1_w);
                g1d.insert(format!("{p}.ln_1.bias"), g_ln1_b);
            }
            gx = &gx2_total + &gx_n;
        }

        if full {
            // embedding + position gradients
            let wpe = weight_2d(ws, "transformer.wpe.weight")?;
            let mut g_wte = grads.remove("transformer.wte.weight").unwrap_or_else(|| Array2::zeros(wte.raw_dim()));
            let mut g_wpe = Array2::<f32>::zeros(wpe.raw_dim());
            for (pp, &tok) in cache.ids.iter().enumerate() {
                let row = (tok.max(0) as usize).min(wte.nrows() - 1);
                let pos = pp.min(wpe.nrows() - 1);
                for j in 0..d {
                    g_wte[[row, j]] += gx[[pp, j]];
                    g_wpe[[pos, j]] += gx[[pp, j]];
                }
            }
            grads.insert("transformer.wte.weight".into(), g_wte);
            grads.insert("transformer.wpe.weight".into(), g_wpe);
            for (name, g) in g1d {
                grads.insert(name, Array2::from_shape_vec((1, g.len()), g).unwrap());
            }
        }
        Ok(grads)
    }
}

fn conv1d(x: &Array2<f32>, w: &Array2<f32>, b: &[f32]) -> Array2<f32> {
    let mut y = x.dot(w);
    for mut row in y.rows_mut() {
        for (j, v) in row.iter_mut().enumerate() {
            *v += b[j];
        }
    }
    y
}

fn col_sum(x: &Array2<f32>) -> Vec<f32> {
    x.sum_axis(ndarray::Axis(0)).to_vec()
}