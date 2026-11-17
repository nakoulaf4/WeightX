// SPDX-License-Identifier: GPL-3.0-or-later
//! End-to-end smoke tests: demo cycle, all 8 WSP skills, improver pipeline.

use weightx_core::improver;
use weightx_core::skills::{parse_skill_command, resolve_skill, WSP_SKILLS};
use weightx_core::studio::StudioSession;

fn quiet_session() -> StudioSession {
    let mut s = StudioSession::new();
    s.set_log(std::sync::Arc::new(|_msg: &str| {}));
    s.config.history.clear();
    s
}

#[test]
fn skill_resolution() {
    assert_eq!(resolve_skill("/skill coding-boost").unwrap().skill_id, "coding-boost");
    assert_eq!(resolve_skill("Улучшить в кодинге").unwrap().skill_id, "coding-boost");
    assert_eq!(resolve_skill("3").unwrap().skill_id, "outlier-repair");
    assert_eq!(resolve_skill("/memory").unwrap().skill_id, "memory-slim");
    let (sk, rest, explicit) = parse_skill_command("/skill coding-boost layer=2");
    assert!(explicit);
    assert_eq!(sk.unwrap().skill_id, "coding-boost");
    assert_eq!(rest, "layer=2");
    assert_eq!(WSP_SKILLS.len(), 8);
}

#[test]
fn all_skills_run_on_minillama() {
    let mut s = quiet_session();
    s.load_model("minillama-demo").unwrap();
    for sk in WSP_SKILLS {
        let res = s.run_skill(sk.skill_id, "").unwrap_or_else(|e| panic!("skill {} failed: {e}", sk.skill_id));
        assert!(!res.steps.is_empty(), "skill {} produced no steps", sk.skill_id);
        assert!(!res.to_markdown().is_empty());
        // The rollback stress-test must prove bit-exact restore.
        if sk.skill_id == "vespasian-guard" {
            assert!(res.accepted, "vespasian-guard stress test failed: {}", res.summary);
        }
    }
}

#[test]
fn studio_actions_and_status() {
    let mut s = quiet_session();
    s.load_model("minillama-demo").unwrap();
    let out = s.run_action("recon_profile", "");
    assert!(out.contains("L1 AutoArch"), "{out}");
    let out = s.run_action("knowledge_map", "");
    assert!(!out.is_empty());
    let out = s.run_action("chat", "Diagnose and repair the target model.");
    assert!(out.contains("baseline_held_out_score"), "{out}");
    let status = s.status();
    assert!(status.contains("Модель: minillama-demo"), "{status}");
    let tmp = tempfile::tempdir().unwrap();
    let ckpt = tmp.path().join("out.safetensors");
    let msg = s.export_checkpoint(ckpt.to_str().unwrap()).unwrap();
    assert!(ckpt.exists(), "{msg}");
}

#[test]
fn improve_small_model_minillama() {
    let tmp = tempfile::tempdir().unwrap();
    let rep = improver::improve_small_model("minillama-demo", true, Some(tmp.path()), 2026).unwrap();
    assert!(rep.total_params > 0);
    assert_eq!(rep.stages.len(), 2);
    assert!(rep.saved_checkpoint_path.is_some());
    assert!(rep.final_metrics.perplexity.is_finite());
}

#[test]
fn mvp_demo_runs() {
    let rep = weightx_core::demo::run_mvp_demo().unwrap();
    assert!(rep["run_result"]["turns_executed"].as_u64().unwrap() >= 4);
    assert!(rep["rollback_verdict"]["rolled_back"].as_bool().unwrap());
    assert!(rep["eamit_result"]["num_facts"].as_u64().unwrap() == 50);
    assert!(rep["eamit_result"]["post_alignment_kl"].as_f64().unwrap() <= rep["eamit_result"]["pre_alignment_kl"].as_f64().unwrap());
}