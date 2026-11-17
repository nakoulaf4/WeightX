// SPDX-License-Identifier: GPL-3.0-or-later
//! Regression tests for audit fixes: GPT-2 grad orientation, Conv1D domain
//! calibration, string-typed plan params, empirical Fisher, Qwen2 biases, dtypes.

use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use weightx_core::domain::{self, DomainCalibOptions};
use weightx_core::fisher;
use weightx_core::model::gpt2::Gpt2Model;
use weightx_core::model::{minillama, ModelArch, ModelConfig};
use weightx_core::{eamit, plan, tensor};
use weightx_core::workspace::WeightWorkspace;

fn rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(123)
}

fn rand_tensor(r: &mut ChaCha8Rng, shape: &[usize], scale: f32) -> tensor::Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|_| -> f32 { StandardNormal.sample(r) })
        .map(|v| v * scale)
        .collect();
    tensor::from_vec(shape, data)
}

fn tiny_gpt2() -> (Gpt2Model, Vec<(String, tensor::Tensor)>) {
    let cfg = ModelConfig {
        model_type: "gpt2".into(),
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        use_rope: false,
        hidden_act: "gelu_new".into(),
        tie_word_embeddings: true,
        ..Default::default()
    };
    let mut r = rng();
    let mut t: Vec<(String, tensor::Tensor)> = Vec::new();
    fn add(t: &mut Vec<(String, tensor::Tensor)>, r: &mut ChaCha8Rng, name: &str, shape: &[usize], scale: f32) {
        t.push((name.to_string(), rand_tensor(r, shape, scale)));
    }
    fn add_ones(t: &mut Vec<(String, tensor::Tensor)>, name: &str, n: usize) {
        t.push((name.to_string(), tensor::from_vec(&[n], vec![1.0; n])));
    }
    fn add_zeros(t: &mut Vec<(String, tensor::Tensor)>, name: &str, n: usize) {
        t.push((name.to_string(), tensor::from_vec(&[n], vec![0.0; n])));
    }
    add(&mut t, &mut r, "transformer.wte.weight", &[16, 8], 0.1);
    add(&mut t, &mut r, "transformer.wpe.weight", &[16, 8], 0.1);
    let p = "transformer.h.0";
    add_ones(&mut t, &format!("{p}.ln_1.weight"), 8);
    add_zeros(&mut t, &format!("{p}.ln_1.bias"), 8);
    add(&mut t, &mut r, &format!("{p}.attn.c_attn.weight"), &[8, 24], 0.1);
    add_zeros(&mut t, &format!("{p}.attn.c_attn.bias"), 24);
    add(&mut t, &mut r, &format!("{p}.attn.c_proj.weight"), &[8, 8], 0.1);
    add_zeros(&mut t, &format!("{p}.attn.c_proj.bias"), 8);
    add_ones(&mut t, &format!("{p}.ln_2.weight"), 8);
    add_zeros(&mut t, &format!("{p}.ln_2.bias"), 8);
    add(&mut t, &mut r, &format!("{p}.mlp.c_fc.weight"), &[8, 32], 0.1);
    add_zeros(&mut t, &format!("{p}.mlp.c_fc.bias"), 32);
    add(&mut t, &mut r, &format!("{p}.mlp.c_proj.weight"), &[32, 8], 0.1);
    add_zeros(&mut t, &format!("{p}.mlp.c_proj.bias"), 8);
    add_ones(&mut t, "transformer.ln_f.weight", 8);
    add_zeros(&mut t, "transformer.ln_f.bias", 8);
    (Gpt2Model::new(cfg), t)
}

fn mean_ce_loss(model: &Gpt2Model, ws: &WeightWorkspace, ids: &Array2<i64>) -> f64 {
    let out = model.forward(ws, ids, false).unwrap();
    let s = out.logits.shape()[1];
    let shift = out.logits.slice(ndarray::s![0, ..s - 1, ..]).to_owned();
    let targets: Vec<i64> = ids.row(0).to_vec()[1..].to_vec();
    let losses = weightx_core::model::cross_entropy_rows(&shift, &targets);
    losses.iter().map(|v| *v as f64).sum::<f64>() / losses.len() as f64
}

#[test]
fn gpt2_conv1d_grads_match_finite_difference() {
    let (model, tensors) = tiny_gpt2();
    let ws = WeightWorkspace::from_tensors(tensors, None);
    let ids = Array2::from_shape_vec((1, 5), vec![1i64, 5, 3, 9, 2]).unwrap();
    let grads = model.weight_grads(&ws, &ids).unwrap();

    let eps = 1e-2f32;
    for name in [
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_proj.weight",
    ] {
        let g = grads.get(name).unwrap_or_else(|| panic!("missing grad {name}"));
        let w = ws.get_readonly(name).unwrap().clone();
        assert_eq!(g.shape(), w.shape(), "grad shape mismatch for {name}");
        // Check a few elements against central differences.
        for idx in [0usize, w.len() / 2, w.len() - 1] {
            let mut all: Vec<(String, tensor::Tensor)> =
                ws.iter().map(|(n, t)| (n.clone(), t.clone())).collect();
            for (n, t) in all.iter_mut() {
                if n == name {
                    t.iter_mut().nth(idx).map(|v| *v += eps);
                }
            }
            let ws_p = WeightWorkspace::from_tensors(all.clone(), None);
            let lp = mean_ce_loss(&model, &ws_p, &ids);
            for (n, t) in all.iter_mut() {
                if n == name {
                    t.iter_mut().nth(idx).map(|v| *v -= 2.0 * eps);
                }
            }
            let ws_m = WeightWorkspace::from_tensors(all, None);
            let lm = mean_ce_loss(&model, &ws_m, &ids);
            let fd = (lp - lm) / (2.0 * eps as f64);
            let analytical = g.iter().nth(idx).copied().unwrap() as f64;
            assert!(
                (analytical - fd).abs() < 5e-3 * (1.0 + fd.abs()),
                "{name}[{idx}]: analytical={analytical:.5} fd={fd:.5}"
            );
        }
    }
}

#[test]
fn gpt2_domain_calibration_finds_directions() {
    let (model, tensors) = tiny_gpt2();
    let mut ws = WeightWorkspace::from_tensors(tensors, None);
    let prac: Vec<Vec<i64>> = vec![vec![1, 5, 3, 9, 2], vec![4, 6, 2, 8, 1], vec![3, 3, 7, 5, 9]];
    let ctrl: Vec<Vec<i64>> = vec![vec![2, 4, 6, 8, 10], vec![5, 7, 9, 11, 13]];
    let opts = DomainCalibOptions { iterations: 1, apply_fisher_guard: false, ..Default::default() };
    let res = domain::eamit_domain_calibration(&mut ws, &model, &prac, &ctrl, &opts).unwrap();
    assert!(!res.target_tensors.is_empty());
    // Before the Conv1D fix, directions were empty and no step was recorded.
    assert!(!res.steps.is_empty(), "Conv1D domain calibration produced no steps");
    assert!(
        res.steps[0].alignment_kl_after <= res.steps[0].alignment_kl_before + 1e-9,
        "EAMIT domain calibration did not preserve/improve alignment KL"
    );
}

#[test]
fn plan_accepts_string_typed_numbers() {
    let w = tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut ws = WeightWorkspace::from_tensors(vec![("w".into(), w)], None);
    let yaml = "transforms:\n  - assert: { exists: { of: 'w' } }\n  - scale_: { target: 'w', by: \"2.0\" }\n";
    let rep = plan::execute_yaml_plan(&mut ws, yaml, false, true, false);
    assert!(rep.success, "{:?}", rep.error);
    let after = ws.get_readonly("w").unwrap();
    assert!((after[[0, 0]] - 2.0).abs() < 1e-6, "by: \"2.0\" was ignored");
    assert!((after[[1, 1]] - 8.0).abs() < 1e-6);
}

#[test]
fn plan_eamit_empty_targets_falls_back_to_target() {
    let w = tensor::from_vec(&[4, 4], vec![0.1; 16]);
    let mut ws = WeightWorkspace::from_tensors(vec![("w".into(), w)], None);
    let yaml = json!({
        "transforms": [
            {"assert": {"exists": {"of": "w"}}},
            {"eamit_batch_": {
                "targets": [],
                "target": "w",
                "facts": [{"id": "f0", "key": [1.0, 0.0, 0.0, 0.0], "value": [0.5, 0.0, 0.0, 0.0]}]
            }}
        ]
    });
    let rep = plan::execute_yaml_plan_value(&mut ws, &yaml, false, true, false);
    assert!(rep.success, "{:?}", rep.error);
    assert!(ws.is_modified("w"));
}

#[test]
fn eamit_embedding_alignment_reduces_kl_and_top_m_mse() {
    let mut r = rng();
    let w0 = rand_tensor(&mut r, &[16, 32], 0.05);
    let w1 = rand_tensor(&mut r, &[16, 32], 0.05);
    let mut ws = WeightWorkspace::from_tensors(
        vec![
            ("l0.weight".into(), w0),
            ("l1.weight".into(), w1),
        ],
        None,
    );
    // 18 clustered facts (3 clusters of 6 facts) where keys share high similarity
    let mut centers = Vec::new();
    for _ in 0..3 {
        let c: Vec<f64> = (0..32).map(|_| StandardNormal.sample(&mut r)).collect();
        centers.push(c);
    }
    let mut facts = Vec::new();
    for i in 0..18 {
        let c = &centers[i / 6];
        let mut key: Vec<f64> = c
            .iter()
            .map(|&v| v + 0.15 * <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut r))
            .collect();
        let kn = key.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut key {
            *x /= kn;
        }
        let val: Vec<f64> = (0..16)
            .map(|_| 0.25 * <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut r))
            .collect();
        facts.push(eamit::BatchFactSpec::new(format!("cf_{i}"), key, val));
    }
    let targets = vec!["l0.weight".to_string(), "l1.weight".to_string()];
    let res = eamit::apply_eamit_batch(&mut ws, &targets, &facts, 1e-2, false).unwrap();
    assert!(res.post_alignment_kl < res.pre_alignment_kl * 0.6, "pre={} post={}", res.pre_alignment_kl, res.post_alignment_kl);
    assert!(res.post_alignment_top_m_mse < res.pre_alignment_top_m_mse, "pre={} post={}", res.pre_alignment_top_m_mse, res.post_alignment_top_m_mse);
    assert!(
        res.post_edit_mse < res.pre_edit_mse * 0.45,
        "clustered random-target MSE reduction should stay >55%: pre={} post={}",
        res.pre_edit_mse,
        res.post_edit_mse
    );
}

#[test]
fn json_f64_vec_flattens_nested() {
    let v = json!([[1.0, 2.0], [3.0]]);
    let flat = eamit::json_f64_vec(Some(&v)).unwrap();
    assert_eq!(flat, vec![1.0, 2.0, 3.0]);
}

#[test]
fn empirical_fisher_from_calibration() {
    let bundle = minillama::build_corrupted_model_and_splits(2026).unwrap();
    let ws = WeightWorkspace::from_tensors(bundle.tensors, None);
    let fmap = fisher::compute_empirical_fisher(&ws, &bundle.model, &[bundle.calib_ids]).unwrap();
    assert!(!fmap.is_empty());
    assert!(fmap.contains_key("model.layers.2.mlp.down_proj.weight"));
    for t in fmap.values() {
        assert!(t.iter().all(|v| v.is_finite() && *v >= 0.0));
    }
}

#[test]
fn qwen2_attention_bias_changes_forward() {
    let bundle = minillama::build_corrupted_model_and_splits(7).unwrap();
    let ids = bundle.calib_ids.slice(ndarray::s![..1, ..]).to_owned();
    let ws0 = WeightWorkspace::from_tensors(bundle.tensors.clone(), None);
    let out0 = bundle.model.forward(&ws0, &ids, false).unwrap();
    let mut with_bias = bundle.tensors.clone();
    let d = bundle.model.config().hidden_size;
    with_bias.push((
        "model.layers.0.self_attn.q_proj.bias".into(),
        tensor::from_vec(&[d], vec![0.5; d]),
    ));
    let ws1 = WeightWorkspace::from_tensors(with_bias, None);
    let out1 = bundle.model.forward(&ws1, &ids, false).unwrap();
    let diff: f32 = out0.logits.iter().zip(out1.logits.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-3, "q_proj.bias was ignored by the Llama runtime");
}

#[test]
fn dtype_tracking_from_safetensors() {
    // Write a float16 safetensors file directly.
    let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| half_bits(*v).to_le_bytes())
        .collect();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2, 2], &data).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("f16.safetensors");
    let serialized = safetensors::tensor::serialize([("w", view)], &None).unwrap();
    std::fs::write(&path, serialized).unwrap();

    let (tensors, dtypes) = tensor::load_safetensors_with_dtypes(&path).unwrap();
    assert_eq!(dtypes.get("w").map(|s| s.as_str()), Some("float16"));
    let ws = WeightWorkspace::from_tensors_with_dtypes(tensors, dtypes, None);
    assert_eq!(ws.dtype("w").unwrap(), "float16");
    assert!((ws.get_readonly("w").unwrap()[[0, 0]] - 1.0).abs() < 1e-3);
}

fn half_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let frac = ((bits >> 13) & 0x3ff) as u16;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | frac
}

#[test]
fn knowledge_probes_cover_three_domains() {
    let probes = weightx_core::corpora::DEFAULT_KNOWLEDGE_PROBES;
    assert_eq!(probes.len(), 12);
    for domain in ["factual", "syntactic", "reasoning"] {
        assert_eq!(probes.iter().filter(|(d, _, _)| *d == domain).count(), 4);
    }
}
