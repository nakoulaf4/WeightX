// SPDX-License-Identifier: GPL-3.0-or-later
//! wx-bench — micro-benchmark harness for the Rust port (prints one JSON line per run).
//! Usage: wx-bench <command> [args...]   (see main() for commands)

use std::time::Instant;

use ndarray::Array2;
use serde_json::{json, Value};
use weightx_core::catalog::ModelCatalog;
use weightx_core::model::ModelArch;
use weightx_core::workspace::WeightWorkspace;
use weightx_core::{eamit, l2, plan, sandbox, tensor, toon};

fn lcg_ids(vocab: usize, seq: usize) -> Array2<i64> {
    let mut state: u64 = 42;
    let mut data = Vec::with_capacity(seq);
    for _ in 0..seq {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push(((state >> 33) % vocab as u64) as i64);
    }
    Array2::from_shape_vec((1, seq), data).unwrap()
}

fn load(model_id: &str) -> (WeightWorkspace, Option<std::sync::Arc<dyn ModelArch>>, f64) {
    let t0 = Instant::now();
    let mut cat = ModelCatalog::new();
    let loaded = cat.load_model(model_id).expect("load model");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut ws = WeightWorkspace::from_tensors(loaded.tensors, None);
    ws.arch = loaded.arch.clone();
    (ws, loaded.arch, ms)
}

fn stats(times: &[f64]) -> Value {
    let mut s = times.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    json!({"mean_ms": mean, "min_ms": s[0], "max_ms": s[s.len() - 1], "iters": s.len()})
}

fn arg(args: &[String], i: usize) -> String {
    args.get(i).cloned().unwrap_or_default()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = arg(&args, 1);
    let out: Value = match cmd.as_str() {
        "forward" => {
            let (ws, arch, load_ms) = load(&arg(&args, 2));
            let seq: usize = arg(&args, 3).parse().unwrap_or(64);
            let iters: usize = arg(&args, 4).parse().unwrap_or(5);
            let arch = arch.expect("arch");
            let vocab = arch.config().vocab_size;
            let ids = lcg_ids(vocab, seq);
            let out0 = arch.forward(&ws, &ids, false).expect("forward");
            let checksum: f64 = out0.logits.iter().map(|&v| v as f64).sum();
            let last = out0.logits.shape()[1] - 1;
            let first5: Vec<f32> = (0..5).map(|i| out0.logits[[0, last, i]]).collect();
            let mut times = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                let o = arch.forward(&ws, &ids, false).expect("forward");
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(&o);
            }
            json!({"cmd": "forward", "model": arg(&args, 2), "seq": seq, "load_ms": load_ms,
                   "logits_sum": checksum, "logits_first5": first5, "timing": stats(&times)})
        }
        "grads" => {
            let (ws, arch, load_ms) = load(&arg(&args, 2));
            let seq: usize = arg(&args, 3).parse().unwrap_or(64);
            let iters: usize = arg(&args, 4).parse().unwrap_or(3);
            let arch = arch.expect("arch");
            let ids = lcg_ids(arch.config().vocab_size, seq);
            let g0 = arch.weight_grads(&ws, &ids).expect("grads");
            let sq: f64 = g0.values().map(|t| t.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sum();
            let mut times = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                let g = arch.weight_grads(&ws, &ids).expect("grads");
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(&g);
            }
            json!({"cmd": "grads", "model": arg(&args, 2), "seq": seq, "load_ms": load_ms,
                   "grad_sq_sum": sq, "n_tensors": g0.len(), "timing": stats(&times)})
        }
        "eamit" => {
            let (mut ws, _arch, load_ms) = load(&arg(&args, 2));
            let facts_v: Value = serde_json::from_str(&std::fs::read_to_string(arg(&args, 3)).unwrap()).unwrap();
            let facts = eamit::facts_from_json(facts_v.as_array().unwrap()).unwrap();
            let targets: Vec<String> = arg(&args, 4).split(',').map(String::from).collect();
            let iters: usize = arg(&args, 5).parse().unwrap_or(5);
            ws.checkpoint("base");
            let mut times = Vec::new();
            let mut pre = 0.0;
            let mut post = 0.0;
            let mut pre_kl = 0.0;
            let mut post_kl = 0.0;
            for _ in 0..iters {
                ws.restore("base").unwrap();
                let t = Instant::now();
                let r = eamit::apply_eamit_batch(&mut ws, &targets, &facts, 1e-2, true).expect("eamit");
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                pre = r.pre_edit_mse;
                post = r.post_edit_mse;
                pre_kl = r.pre_alignment_kl;
                post_kl = r.post_alignment_kl;
            }
            json!({"cmd": "eamit", "model": arg(&args, 2), "facts": facts.len(), "layers": targets.len(),
                   "load_ms": load_ms, "pre_mse": pre, "post_mse": post,
                   "pre_alignment_kl": pre_kl, "post_alignment_kl": post_kl, "timing": stats(&times)})
        }
        "toon" => {
            let payload: Value = serde_json::from_str(&std::fs::read_to_string(arg(&args, 2)).unwrap()).unwrap();
            let iters: usize = arg(&args, 3).parse().unwrap_or(20);
            let mut enc = Vec::new();
            let mut dec = Vec::new();
            let mut text = String::new();
            for _ in 0..iters {
                let t = Instant::now();
                text = toon::dumps_toon(&payload);
                enc.push(t.elapsed().as_secs_f64() * 1000.0);
                let t = Instant::now();
                let v = toon::loads_toon(&text);
                dec.push(t.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(&v);
            }
            json!({"cmd": "toon", "chars": text.len(), "encode": stats(&enc), "decode": stats(&dec)})
        }
        "toon-dump" => {
            let payload: Value = serde_json::from_str(&std::fs::read_to_string(arg(&args, 2)).unwrap()).unwrap();
            let text = toon::dumps_toon(&payload);
            std::fs::write(arg(&args, 3), &text).unwrap();
            json!({"cmd": "toon-dump", "chars": text.len()})
        }
        "toon-load" => {
            let text = std::fs::read_to_string(arg(&args, 2)).unwrap();
            let v = toon::loads_toon(&text);
            json!({"cmd": "toon-load", "json": v})
        }
        "plan" => {
            let (mut ws, _arch, load_ms) = load(&arg(&args, 2));
            let iters: usize = arg(&args, 3).parse().unwrap_or(10);
            let yaml = r#"
transforms:
  - assert:
      exists: { of: "model.layers.0.self_attn.q_proj.weight" }
  - scale_:
      target: "model.layers.0.self_attn.q_proj.weight"
      by: 1.001
  - scale_:
      target: "model.layers.1.self_attn.q_proj.weight"
      by: 0.999
  - scale_:
      target: "model.layers.2.self_attn.q_proj.weight"
      by: 1.001
  - assert:
      finite: { of: "model.layers.0.self_attn.q_proj.weight" }
"#;
            ws.checkpoint("base");
            let mut times = Vec::new();
            let mut ok = true;
            for _ in 0..iters {
                ws.restore("base").unwrap();
                let t = Instant::now();
                let r = plan::execute_yaml_plan(&mut ws, yaml, false, true, false);
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                ok = r.success;
            }
            json!({"cmd": "plan", "model": arg(&args, 2), "load_ms": load_ms, "success": ok, "timing": stats(&times)})
        }
        "wsio" => {
            let path = arg(&args, 2);
            let iters: usize = arg(&args, 3).parse().unwrap_or(3);
            let tmp = std::env::temp_dir().join("wx_bench_export.safetensors");
            let mut loads = Vec::new();
            let mut exports = Vec::new();
            let mut params = 0usize;
            for _ in 0..iters {
                let t = Instant::now();
                let (ts, _ds) = tensor::load_safetensors_with_dtypes(std::path::Path::new(&path)).unwrap();
                loads.push(t.elapsed().as_secs_f64() * 1000.0);
                let ws = WeightWorkspace::from_tensors(ts, None);
                params = ws.total_params();
                let t = Instant::now();
                ws.export_safetensors(&tmp, false).unwrap();
                exports.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            json!({"cmd": "wsio", "params": params, "load": stats(&loads), "export": stats(&exports)})
        }
        "describe" => {
            let (ws, _arch, load_ms) = load(&arg(&args, 2));
            let name = arg(&args, 3);
            let iters: usize = arg(&args, 4).parse().unwrap_or(10);
            let mut times = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                let v = ws.describe_tensor(&name, 8).unwrap();
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(&v);
            }
            json!({"cmd": "describe", "model": arg(&args, 2), "load_ms": load_ms, "timing": stats(&times)})
        }
        "l2" => {
            let (ws, arch, load_ms) = load(&arg(&args, 2));
            let seq: usize = arg(&args, 3).parse().unwrap_or(32);
            let arch = arch.expect("arch");
            let ids = lcg_ids(arch.config().vocab_size, seq);
            let t = Instant::now();
            let rep = l2::profile_model_layers(&ws, Some(arch.as_ref()), Some(&ids), "bench", 2.5);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            json!({"cmd": "l2", "model": arg(&args, 2), "seq": seq, "load_ms": load_ms,
                   "layers": rep.num_layers, "elapsed_ms": ms})
        }
        "sandbox" => {
            let (mut ws, _arch, load_ms) = load(&arg(&args, 2));
            let name = arg(&args, 3);
            let iters: usize = arg(&args, 4).parse().unwrap_or(3);
            let code = format!("W['{name}'] = W['{name}'] * 1.0001\nprint('ok')");
            let sb = sandbox::IsolatedSandbox::default();
            let mut times = Vec::new();
            let mut ok = true;
            for _ in 0..iters {
                let t = Instant::now();
                let r = sb.run(&mut ws, &code, Some(&name));
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                ok = r.success;
            }
            json!({"cmd": "sandbox", "model": arg(&args, 2), "load_ms": load_ms, "success": ok, "timing": stats(&times)})
        }
        _ => {
            eprintln!("commands: forward|grads|eamit|toon|toon-dump|toon-load|plan|wsio|describe|l2|sandbox");
            std::process::exit(2);
        }
    };
    println!("{}", serde_json::to_string(&out).unwrap());
}