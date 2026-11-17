// SPDX-License-Identifier: GPL-3.0-or-later
//! Built-in MiniLlama demo stand: deterministic corrupted Llama-family model + splits.
//! Port of `examples/mvp_end_to_end_demo.py` model construction.

use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal, Uniform};

use crate::error::Result;
use crate::model::llama::LlamaModel;
use crate::model::{ModelArch, ModelConfig};
use crate::tensor::{self, Tensor};
use crate::vespasian::EvaluationSplit;
use crate::workspace::WeightWorkspace;

pub fn minillama_config() -> ModelConfig {
    ModelConfig {
        model_type: "minillama".into(),
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
        use_rope: false,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
    }
}

pub struct DemoBundle {
    pub model: LlamaModel,
    pub tensors: Vec<(String, Tensor)>,
    pub practice: EvaluationSplit,
    pub held_out: EvaluationSplit,
    pub calib_ids: Array2<i64>,
}

fn init_linear(rng: &mut ChaCha8Rng, out: usize, inn: usize) -> Tensor {
    let bound = 1.0 / (inn as f64).sqrt();
    let dist = Uniform::new(-bound as f32, bound as f32);
    let data: Vec<f32> = (0..out * inn).map(|_| dist.sample(rng)).collect();
    tensor::from_vec(&[out, inn], data)
}

fn init_embed(rng: &mut ChaCha8Rng, vocab: usize, d: usize) -> Tensor {
    let data: Vec<f32> = (0..vocab * d).map(|_| StandardNormal.sample(rng)).collect();
    tensor::from_vec(&[vocab, d], data)
}

/// Build the MiniLlama weight set (clean) with a deterministic seed.
pub fn build_minillama_tensors(cfg: &ModelConfig, seed: u64) -> Vec<(String, Tensor)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let d = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let mut t: Vec<(String, Tensor)> = Vec::new();
    t.push(("model.embed_tokens.weight".into(), init_embed(&mut rng, cfg.vocab_size, d)));
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        t.push((format!("{p}.input_layernorm.weight"), tensor::ones(&[d])));
        t.push((format!("{p}.self_attn.q_proj.weight"), init_linear(&mut rng, d, d)));
        t.push((format!("{p}.self_attn.k_proj.weight"), init_linear(&mut rng, d, d)));
        t.push((format!("{p}.self_attn.v_proj.weight"), init_linear(&mut rng, d, d)));
        t.push((format!("{p}.self_attn.o_proj.weight"), init_linear(&mut rng, d, d)));
        t.push((format!("{p}.post_attention_layernorm.weight"), tensor::ones(&[d])));
        t.push((format!("{p}.mlp.gate_proj.weight"), init_linear(&mut rng, inter, d)));
        t.push((format!("{p}.mlp.up_proj.weight"), init_linear(&mut rng, inter, d)));
        t.push((format!("{p}.mlp.down_proj.weight"), init_linear(&mut rng, d, inter)));
    }
    t.push(("model.norm.weight".into(), tensor::ones(&[d])));
    t.push(("lm_head.weight".into(), init_linear(&mut rng, cfg.vocab_size, d)));
    t
}

/// Create the corrupted demo model + practice/held-out splits (seed 2026 default).
pub fn build_corrupted_model_and_splits(seed: u64) -> Result<DemoBundle> {
    let cfg = minillama_config();
    let mut tensors = build_minillama_tensors(&cfg, seed);
    let model = LlamaModel::new(cfg.clone());

    // Deterministic sequences: 32 x 10, tokens in [4, vocab-4)
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));
    let dist = Uniform::new(4i64, (cfg.vocab_size - 4) as i64);
    let mut all_inputs = Array2::<i64>::zeros((32, 10));
    for v in all_inputs.iter_mut() {
        *v = dist.sample(&mut rng);
    }

    // Clean targets from the uncorrupted model
    let clean_ws = WeightWorkspace::from_tensors(tensors.clone(), None);
    let clean_out = model.forward(&clean_ws, &all_inputs, false)?;
    let s = all_inputs.shape()[1];
    let mut clean_targets = Vec::new();
    for b in 0..32 {
        let last = clean_out.logits.slice(ndarray::s![b, s - 1, ..]);
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (j, &v) in last.iter().enumerate() {
            if v > bv {
                bv = v;
                best = j;
            }
        }
        clean_targets.push(best as i64);
    }

    // Inject severe outlier fault in Layer 2 MLP down_proj rows 5, 11, 19
    for (name, t) in tensors.iter_mut() {
        if name == "model.layers.2.mlp.down_proj.weight" {
            let mut m2d = tensor::as_2d(t);
            for (row, factor) in [(5usize, 16.0f32), (11, 15.0), (19, 14.0)] {
                for j in 0..m2d.ncols() {
                    m2d[[row, j]] *= factor;
                }
            }
            *t = m2d.into_dyn().into_shape_with_order(t.shape()).unwrap().to_owned();
        }
    }

    let practice = EvaluationSplit::targets(
        "practice",
        all_inputs.slice(ndarray::s![..16, ..]).to_owned(),
        clean_targets[..16].to_vec(),
        Some(all_inputs.slice(ndarray::s![..8, ..]).to_owned()),
    );
    let held_out = EvaluationSplit::targets(
        "held_out_hidden",
        all_inputs.slice(ndarray::s![16.., ..]).to_owned(),
        clean_targets[16..].to_vec(),
        Some(all_inputs.slice(ndarray::s![16..24, ..]).to_owned()),
    );
    let calib_ids = all_inputs.slice(ndarray::s![..6, ..]).to_owned();

    Ok(DemoBundle { model, tensors, practice, held_out, calib_ids })
}

/// Clean (uncorrupted) MiniLlama for catalog `corrupt_demo=False`.
pub fn build_clean_model(seed: u64) -> (LlamaModel, Vec<(String, Tensor)>) {
    let cfg = minillama_config();
    let tensors = build_minillama_tensors(&cfg, seed);
    (LlamaModel::new(cfg), tensors)
}