// SPDX-License-Identifier: GPL-3.0-or-later
//! Built-in MiniLlama end-to-end MVP demo (port of `examples/mvp_end_to_end_demo.py`).

use std::sync::Arc;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::{json, Value};

use crate::error::Result;
use crate::harness::{HarnessOptions, WeightXHarness};
use crate::eamit::{self, BatchFactSpec};
use crate::model::{minillama, ModelArch};
use crate::plan;
use crate::provider::AutonomousWSPProvider;
use crate::toon;

/// Execute the complete WeightX MVP demonstration and print a structured report.
pub fn run_mvp_demo() -> Result<Value> {
    println!("{}", "=".repeat(78));
    println!("WeightX (GPLv3) — Weight-Space Programming Harness End-to-End MVP Demo");
    println!("{}", "=".repeat(78));

    let bundle = minillama::build_corrupted_model_and_splits(2026)?;
    let cfg = bundle.model.config().clone();

    let opts = HarnessOptions {
        practice_split: Some(bundle.practice),
        held_out_split: Some(bundle.held_out),
        calibration_input_ids: Some(bundle.calib_ids),
        fisher_damping_alpha: 0.15,
        fisher_freeze_top_fraction: 0.05,
        improvement_threshold: 0.01,
        max_kl_drift: 2.0,
        ..Default::default()
    };
    let mut harness = WeightXHarness::new(
        bundle.tensors,
        Some(Arc::new(bundle.model)),
        Some(Box::new(AutonomousWSPProvider::new(false))),
        opts,
    );

    println!("\n[1] Initial Corrupted Held-Out Score: {:.4}", harness.vespasian.baseline_score);
    let run_res = harness.run_editing_cycle(
        "Localize and repair outlier neuron corruption in the causal LM using a declarative BrainSurgery-WX YAML plan.",
        6,
    )?;

    println!(
        "[2] Completed 5-Stage WSP Loop in {} turns ({} tool calls):",
        run_res.turns_executed, run_res.tool_calls_executed
    );
    println!("    • Baseline Held-Out Score : {:.4}", run_res.baseline_held_out_score);
    println!(
        "    • Post-Repair Best Score  : {:.4} (Delta: {:+.4})",
        run_res.final_best_held_out_score, run_res.net_improvement
    );
    println!(
        "    • Changed Parameters      : {} / {} ({:.4}%)",
        run_res.diff_summary.changed_params,
        run_res.diff_summary.total_params,
        run_res.diff_summary.changed_fraction() * 100.0
    );
    for v in &run_res.vespasian_history {
        println!("    • Vespasian Verdict #{}   : {}", v.iteration, v.verdict_reason);
    }

    // Demonstrate automatic rollback on a destructive YAML edit.
    println!("\n[3] Demonstrating Vespasian Automatic Rollback on a Destructive Edit:");
    let bad_plan = r#"
transforms:
  - assert:
      exists: { of: "model.layers.0.self_attn.q_proj.weight" }
  - scale_:
      target: "model.layers.0.self_attn.q_proj.weight"
      by: -50.0
  - assert:
      finite: { of: "model.layers.0.self_attn.q_proj.weight" }
"#;
    plan::execute_yaml_plan(&mut harness.workspace, bad_plan, false, true, false);
    let rollback_verdict = harness.vespasian.verify_and_decide(
        &mut harness.workspace,
        harness.arch.as_deref(),
        "Intentional destructive edit to verify automatic rollback",
    );
    println!("    • Candidate Score         : {:.4}", rollback_verdict.candidate_score);
    println!("    • Rollback Triggered      : {}", rollback_verdict.rolled_back);
    println!("    • Restored Best Score     : {:.4}", rollback_verdict.best_score_after);
    println!("    • Verdict                 : {}", rollback_verdict.verdict_reason);

    // EAMIT simultaneous batch associative editing (50 facts across layers 1, 2, 3).
    println!("\n[4] Demonstrating EAMIT Batch Editing (50 Facts Across 3 MLP Layers):");
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let d_in = cfg.intermediate_size;
    let d_out = cfg.hidden_size;
    let mut facts = Vec::new();
    for i in 0..50 {
        let mut key: Vec<f64> = (0..d_in).map(|_| StandardNormal.sample(&mut rng)).collect();
        let norm = key.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in &mut key {
            *v /= norm;
        }
        let val: Vec<f64> = (0..d_out)
            .map(|_| -> f64 { StandardNormal.sample(&mut rng) })
            .map(|v| v * 0.25)
            .collect();
        facts.push(BatchFactSpec::new(format!("fact_{i}"), key, val));
    }
    let eamit_targets: Vec<String> = [1, 2, 3]
        .iter()
        .map(|l| format!("model.layers.{l}.mlp.down_proj.weight"))
        .collect();
    let eamit_res = eamit::apply_eamit_batch(
        &mut harness.workspace,
        &eamit_targets,
        &facts,
        1e-2,
        true,
    )?;
    println!(
        "    • Batch Size              : {} facts across {} layers",
        eamit_res.num_facts,
        eamit_res.target_tensors.len()
    );
    println!("    • Pre-Edit Memory MSE     : {:.6}", eamit_res.pre_edit_mse);
    println!(
        "    • Post-Edit Memory MSE    : {:.6} ({:.1}% error reduction)",
        eamit_res.post_edit_mse,
        eamit_res.relative_error_reduction * 100.0
    );
    println!(
        "    • Alignment KL / Top-M    : {:.4} -> {:.4} / {:.4} -> {:.4}",
        eamit_res.pre_alignment_kl,
        eamit_res.post_alignment_kl,
        eamit_res.pre_alignment_top_m_mse,
        eamit_res.post_alignment_top_m_mse
    );

    let _ = harness.workspace.restore("vespasian_best");

    // TOON vs JSON token savings on the L3 tensor table.
    let tensor_table = json!({"tensors": harness.workspace.list_tensors_table(None, None)});
    let savings = toon::compare_toon_vs_json(&tensor_table);
    println!("\n[5] TOON Serialization Token Efficiency:");
    println!(
        "    • JSON: {} chars (~{} tokens) -> TOON: {} chars (~{} tokens) [{:.1}% token reduction]",
        savings.json_chars,
        savings.json_est_tokens,
        savings.toon_chars,
        savings.toon_est_tokens,
        savings.token_reduction_ratio * 100.0
    );
    println!("{}", "=".repeat(78));

    Ok(json!({
        "run_result": run_res.to_value(),
        "rollback_verdict": rollback_verdict.to_value(),
        "eamit_result": eamit_res.to_value(),
        "toon_savings": savings.to_value(),
    }))
}