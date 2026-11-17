# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Suites 3, 4, and 5: Outlier Repair + Fisher Guard, SVD/TIES Compression Frontier,
L2 Knowledge Map (Gini KCI), and Vespasian Bit-Exact Rollback Proof.
"""

from __future__ import annotations

import gc
import hashlib
import time
from typing import Any

import numpy as np
import torch

from benchmarks.eval_suites import (
    build_arc_items,
    eval_lambada,
    eval_mc_acc,
    eval_wikitext2_ppl,
)
from benchmarks.prepare_datasets import load_cached
from weightx.catalog import ModelCatalog
from weightx.declarative.plan_engine import execute_yaml_plan
from weightx.execution.vespasian import EvaluationSplit, VespasianEngine
from weightx.l1_arch.autoarch import inspect_architecture
from weightx.l2_profile.knowledge_map import run_knowledge_map
from weightx.l2_profile.unilayer import profile_model_layers
from weightx.l3_weights.domain_eamit import find_mlp_output_modules, select_edit_layers, token_nll
from weightx.l3_weights.workspace import WeightWorkspace
from weightx.safety.iwck_fisher import FisherGuard


def _sha256_tensor(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _eval_general(model: Any, tok: Any, wt_rows: list[dict], lb_rows: list[dict], arc_items: list[Any]) -> dict[str, Any]:
    wt = eval_wikitext2_ppl(model, tok, wt_rows, max_tokens=65536)
    lb = eval_lambada(model, tok, lb_rows, max_items=400, batch_size=16)
    arc = eval_mc_acc(model, tok, arc_items, batch_size=16)
    return {"wikitext2": wt, "lambada": lb, "arc_easy": arc}


def run_model_repair_slim_kmap(model_id: str, *, device: str = "cuda") -> dict[str, Any]:
    """Run Suites 3, 4, and 5 cleanly on a single real pretrained model."""
    wt_rows = load_cached("wikitext2_test")
    lb_rows = load_cached("lambada_test")
    arc_items = build_arc_items(load_cached("arc_easy_test"), max_items=400)

    cat = ModelCatalog()
    t_load = time.perf_counter()
    model, tok, entry = cat.load_model_and_tokenizer(model_id, device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    load_s = round(time.perf_counter() - t_load, 2)
    total_params = sum(p.numel() for p in model.parameters())

    # Build deterministic calibration / Vespasian held-out splits from non-overlapping text
    calib_texts = [
        "Artificial intelligence systems encode knowledge across attention and multilayer perceptron weights.",
        "Closed form linear algebra enables predictable weight surgery without full gradient descent.",
        "The capital of France is Paris and the chemical symbol for water is H2O.",
        "Python functions process structured data using deterministic control flow.",
    ]
    calib_seqs = [
        torch.tensor(tok.encode(t, add_special_tokens=False)[:24], dtype=torch.long)
        for t in calib_texts
    ]
    min_c = min(len(s) for s in calib_seqs)
    calib_ids = torch.stack([s[:min_c] for s in calib_seqs]).to(device)

    held_wt_texts = [r["text"].strip() for r in wt_rows if len(r.get("text", "").strip()) > 120][:24]
    held_seqs = [torch.tensor(tok.encode(t, add_special_tokens=False)[:64], dtype=torch.long) for t in held_wt_texts]

    def _vesp_eval(m: Any) -> tuple[float, list[float]]:
        _, per = token_nll(m, held_seqs)
        items = [float(np.exp(-x)) for x in per]
        return float(np.mean(items)), items

    ref_ids = calib_ids

    # ------------------------------------------------------------------
    # Suite 5A: L1 + L2 Profiling & Knowledge Map (Gini KCI)
    # ------------------------------------------------------------------
    print(f"  [repair:{model_id}] L1+L2+KnowledgeMap...", flush=True)
    t_l2 = time.perf_counter()
    arch = inspect_architecture(model, calibration_input_ids=calib_ids)
    prof = profile_model_layers(model, calibration_input_ids=calib_ids, model_name=model_id, outlier_row_threshold=2.5)
    kmap = run_knowledge_map(model, tokenizer=tok)
    l2_time_s = round(time.perf_counter() - t_l2, 2)

    mean_eff_rank = float(np.mean([m["effective_rank"] for m in prof.per_layer_metrics])) if prof.per_layer_metrics else 0.0
    mean_stable_rank = float(np.mean([m["stable_rank"] for m in prof.per_layer_metrics])) if prof.per_layer_metrics else 0.0
    max_kurtosis = float(max((m["weight_kurtosis"] for m in prof.per_layer_metrics), default=0.0))

    suite5_kmap = {
        "family": arch.family,
        "num_layers": arch.n_layers,
        "d_model": arch.d_model,
        "n_heads": kmap.n_heads,
        "kci_overall": round(kmap.kci_overall, 4),
        "kci_per_domain": {k: round(v, 4) for k, v in kmap.kci_per_domain.items()},
        "concentration_regime": kmap.concentration_regime,
        "avg_crystallization_layer": round(kmap.avg_crystallization_layer, 2),
        "crystallization_fraction": round(kmap.crystallization_fraction, 3),
        "dominant_layers_by_domain": kmap.dominant_layers_by_domain,
        "bottleneck_layers": prof.bottleneck_layers[:5],
        "redundant_layers": prof.redundant_layers[:5],
        "natural_outlier_tensors": len(prof.outlier_neurons),
        "max_natural_row_ratio": round(max((o.max_ratio_to_median for o in prof.outlier_neurons), default=1.0), 3),
        "mean_effective_rank": round(mean_eff_rank, 2),
        "mean_stable_rank": round(mean_stable_rank, 2),
        "max_weight_kurtosis": round(max_kurtosis, 2),
        "profile_time_s": l2_time_s,
    }

    # Clean baseline general metrics (shared across Suites 3, 4, 5)
    t_base = time.perf_counter()
    clean_metrics = _eval_general(model, tok, wt_rows, lb_rows, arc_items)
    eval_s = round(time.perf_counter() - t_base, 2)

    # ------------------------------------------------------------------
    # Suite 5B: Vespasian Guard Safety Stress-Test (Bit-Exact Rollback)
    # ------------------------------------------------------------------
    ws_guard = WeightWorkspace(model)
    vesp_guard = VespasianEngine(
        ws_guard,
        held_out_split=EvaluationSplit(name="held_wt", eval_fn=_vesp_eval, reference_input_ids=ref_ids),
        improvement_threshold=1e-4,
        max_kl_drift=0.05,
        max_changed_fraction=0.10,
    )
    mlp_mods = find_mlp_output_modules(model)
    guard_target = f"{mlp_mods[len(mlp_mods)//2][0]}.weight"
    sha_before = _sha256_tensor(ws_guard.get_readonly(guard_target))
    arr_before = ws_guard.get_readonly(guard_target).copy()

    # Attack 1: Destructive scaling (-50x)
    sh = list(ws_guard.shape(guard_target))
    bad_scale_yaml = (
        "transforms:\n"
        f"  - assert: {{ shape: {{ of: '{guard_target}', is: {sh} }} }}\n"
        f"  - scale_: {{ target: '{guard_target}', by: -50.0 }}\n"
        f"  - assert: {{ finite: {{ of: '{guard_target}' }} }}\n"
    )
    execute_yaml_plan(ws_guard, bad_scale_yaml, apply_fisher_guard=False)
    v_scale = vesp_guard.verify_and_decide(hypothesis="Adversarial scale -50.0 stress test")
    arr_after_scale = ws_guard.get_readonly(guard_target)
    max_diff_scale = float(np.max(np.abs(arr_before - arr_after_scale)))
    sha_after_scale = _sha256_tensor(arr_after_scale)

    # Attack 2: NaN injection (must be blocked atomically by BrainSurgery-WX `assert finite`)
    nan_arr = ws_guard.get_readonly(guard_target).copy()
    nan_arr[0, 0] = np.nan
    ws_guard.set_tensor(guard_target, nan_arr, apply_fisher_guard=False)
    v_nan = vesp_guard.verify_and_decide(hypothesis="Adversarial NaN injection test")
    sha_after_nan = _sha256_tensor(ws_guard.get_readonly(guard_target))

    # Attack 3: Unbounded 100% weight overwrite (exceeds max_changed_fraction or KL drift)
    noise_arr = ws_guard.get_readonly(guard_target).copy() + np.random.default_rng(1).normal(0, 0.25, size=sh).astype(np.float32)
    ws_guard.set_tensor(guard_target, noise_arr, apply_fisher_guard=False)
    v_noise = vesp_guard.verify_and_decide(hypothesis="Adversarial KL/noise drift test")
    sha_after_noise = _sha256_tensor(ws_guard.get_readonly(guard_target))

    suite5_vespasian = {
        "target_tensor": guard_target,
        "sha256_before": sha_before[:16],
        "scale_attack_rolled_back": bool(v_scale.rolled_back),
        "scale_attack_max_abs_diff": max_diff_scale,
        "scale_attack_sha256_match": sha_before == sha_after_scale,
        "nan_attack_rolled_back": bool(v_nan.rolled_back),
        "nan_attack_nonfinite_detected": int(v_nan.nonfinite_params),
        "nan_attack_sha256_match": sha_before == sha_after_nan,
        "noise_attack_rolled_back": bool(v_noise.rolled_back),
        "noise_attack_kl_drift": round(float(v_noise.kl_divergence_ref), 4),
        "noise_attack_sha256_match": sha_before == sha_after_noise,
    }
    # ------------------------------------------------------------------
    # Suite 3: Outlier Neuron Fault Injection & Surgical Repair (+ Fisher Ablation)
    # ------------------------------------------------------------------
    print(f"  [repair:{model_id}] Suite 3 outlier fault injection & Fisher repair...", flush=True)
    del ws_guard
    gc.collect()

    fault_layer_mod = select_edit_layers(model, n_layers=3)[1]
    fault_tensor = f"{fault_layer_mod}.weight"
    param_map = dict(model.named_parameters())
    orig_w_cpu = param_map[fault_tensor].detach().clone().cpu()
    orig_w = orig_w_cpu.float().numpy()
    sh_f = list(orig_w.shape)

    # Inject realistic severe outlier row corruption on 3 rows (16x, 14x, 12x)
    rng = np.random.default_rng(2026)
    corrupt_rows = sorted(rng.choice(sh_f[0], size=min(3, sh_f[0]), replace=False).tolist())
    corrupt_w = orig_w.copy()
    factors = [16.0, 14.0, 12.0]
    for r_idx, fac in zip(corrupt_rows, factors):
        corrupt_w[r_idx, :] *= fac
    with torch.no_grad():
        param_map[fault_tensor].copy_(torch.from_numpy(corrupt_w).to(device=param_map[fault_tensor].device, dtype=param_map[fault_tensor].dtype))

    corrupted_metrics = _eval_general(model, tok, wt_rows, lb_rows, arc_items)
    prof_corrupt = profile_model_layers(model, calibration_input_ids=calib_ids, model_name=model_id, outlier_row_threshold=2.5)
    detected_tensors = [o.tensor_name for o in prof_corrupt.outlier_neurons]
    detected_rows_for_target: list[int] = []
    for o in prof_corrupt.outlier_neurons:
        if o.tensor_name == fault_tensor:
            detected_rows_for_target.extend(o.row_indices)
    loc_hit_tensor = fault_tensor in detected_tensors
    loc_row_recall = len(set(corrupt_rows) & set(detected_rows_for_target)) / max(len(corrupt_rows), 1)

    repair_yaml = (
        "transforms:\n"
        "  - assert:\n"
        "      all:\n"
        f"        - exists: {{ of: '{fault_tensor}' }}\n"
        f"        - shape: {{ of: '{fault_tensor}', is: {sh_f} }}\n"
        f"        - finite: {{ of: '{fault_tensor}' }}\n"
        "  - rescale_outlier_rows_:\n"
        f"      target: '{fault_tensor}'\n"
        "      min_ratio: 2.5\n"
        "  - assert:\n"
        f"      finite: {{ of: '{fault_tensor}' }}\n"
    )

    # 3A: Repair WITHOUT FisherGuard
    ws_nofg = WeightWorkspace(model, fisher_guard=None)
    t_r0 = time.perf_counter()
    execute_yaml_plan(ws_nofg, repair_yaml, apply_fisher_guard=False)
    repaired_nofg_metrics = _eval_general(model, tok, wt_rows, lb_rows, arc_items)
    t_nofg = round(time.perf_counter() - t_r0, 3)
    ws_nofg.restore("baseline")
    del ws_nofg
    gc.collect()

    # 3B: Repair WITH IWCKEdit FisherGuard + Vespasian verification
    fguard = FisherGuard.from_model(
        model,
        reference_batches=[calib_ids],
        damping_alpha=0.25,
        freeze_top_fraction=0.05,
        outlier_bypass_ratio=2.5,
        enabled=True,
    )
    ws_fg = WeightWorkspace(model, fisher_guard=fguard)
    vesp_fg = VespasianEngine(
        ws_fg,
        held_out_split=EvaluationSplit(name="held_wt", eval_fn=_vesp_eval, reference_input_ids=ref_ids),
        improvement_threshold=1e-4,
        max_kl_drift=2.0,
        max_changed_fraction=0.25,
    )
    t_r1 = time.perf_counter()
    plan_res = execute_yaml_plan(ws_fg, repair_yaml, apply_fisher_guard=True)
    v_rep = vesp_fg.verify_and_decide(hypothesis=f"Outlier repair on {fault_tensor}")
    repaired_fg_metrics = _eval_general(model, tok, wt_rows, lb_rows, arc_items)
    t_fg = round(time.perf_counter() - t_r1, 3)
    diff_fg = ws_fg.diff()

    # Restore exact clean baseline weights before Suite 4!
    with torch.no_grad():
        param_map[fault_tensor].copy_(orig_w_cpu.to(device=param_map[fault_tensor].device, dtype=param_map[fault_tensor].dtype))
    del ws_fg, vesp_fg, fguard
    gc.collect()

    def _recovery_pct(clean_v: float, corrupt_v: float, rep_v: float, lower_better: bool = True) -> float:
        denom = (corrupt_v - clean_v) if lower_better else (clean_v - corrupt_v)
        if abs(denom) < 1e-9:
            return 100.0
        num = (corrupt_v - rep_v) if lower_better else (rep_v - corrupt_v)
        return round(float(num / denom * 100.0), 2)

    suite3_outlier = {
        "injected_tensor": fault_tensor,
        "injected_rows": corrupt_rows,
        "scale_factors": factors,
        "localization_found_tensor": loc_hit_tensor,
        "localization_row_recall": round(loc_row_recall, 3),
        "clean": clean_metrics,
        "corrupted": corrupted_metrics,
        "repaired_with_fisher": repaired_fg_metrics,
        "repaired_no_fisher": repaired_nofg_metrics,
        "vespasian_accepted": bool(v_rep.accepted),
        "changed_params": int(diff_fg.changed_params),
        "changed_pct": round(diff_fg.changed_fraction * 100.0, 5),
        "wt2_ppl_recovery_pct_fisher": _recovery_pct(
            clean_metrics["wikitext2"]["ppl"], corrupted_metrics["wikitext2"]["ppl"], repaired_fg_metrics["wikitext2"]["ppl"], True
        ),
        "wt2_ppl_recovery_pct_no_fisher": _recovery_pct(
            clean_metrics["wikitext2"]["ppl"], corrupted_metrics["wikitext2"]["ppl"], repaired_nofg_metrics["wikitext2"]["ppl"], True
        ),
        "lambada_acc_recovery_pct_fisher": _recovery_pct(
            clean_metrics["lambada"]["acc"], corrupted_metrics["lambada"]["acc"], repaired_fg_metrics["lambada"]["acc"], False
        ),
        "repair_time_s": t_fg,
        "plan_mutations": int(plan_res.mutations_executed),
    }

    # ------------------------------------------------------------------
    # Suite 4: Spectral SVD Low-Rank Compression + TIES Trimming Frontier
    # ------------------------------------------------------------------
    print(f"  [repair:{model_id}] Suite 4 SVD/TIES compression sweep...", flush=True)
    target_mods = select_edit_layers(model, n_layers=4)
    target_tensors = [f"{m}.weight" for m in target_mods]
    ws_slim = WeightWorkspace(model)
    ws_slim.checkpoint("clean_slim")

    sweep_results: list[dict[str, Any]] = []
    for ratio in (0.70, 0.60, 0.50, 0.40):
        ws_slim.restore("clean_slim")
        orig_elems = 0
        fact_elems = 0
        yaml_steps = ["transforms:"]
        for tn in target_tensors:
            m_dim, n_dim = ws_slim.shape(tn)
            min_dim = min(m_dim, n_dim)
            r_keep = max(8, int(round(min_dim * ratio)))
            orig_elems += m_dim * n_dim
            fact_elems += r_keep * (m_dim + n_dim)
            yaml_steps.append(f"  - assert: {{ shape: {{ of: '{tn}', is: [{m_dim}, {n_dim}] }} }}")
            yaml_steps.append(f"  - svd_low_rank_: {{ target: '{tn}', rank: {r_keep} }}")
            yaml_steps.append(f"  - ties_trim_: {{ target: '{tn}', keep_top_fraction: 0.85 }}")
            yaml_steps.append(f"  - assert: {{ finite: {{ of: '{tn}' }} }}")
        yaml_slim = "\n".join(yaml_steps)

        t_s0 = time.perf_counter()
        execute_yaml_plan(ws_slim, yaml_slim, apply_fisher_guard=False)
        slim_metrics = _eval_general(model, tok, wt_rows, lb_rows, arc_items)
        t_slim = round(time.perf_counter() - t_s0, 2)

        target_param_saving_pct = (1.0 - fact_elems / max(orig_elems, 1)) * 100.0
        total_model_saving_pct = (orig_elems - fact_elems) / max(total_params, 1) * 100.0
        wt2_delta_pct = (slim_metrics["wikitext2"]["ppl"] - clean_metrics["wikitext2"]["ppl"]) / clean_metrics["wikitext2"]["ppl"] * 100.0
        lb_delta = slim_metrics["lambada"]["acc"] - clean_metrics["lambada"]["acc"]
        arc_delta = slim_metrics["arc_easy"]["acc_norm"] - clean_metrics["arc_easy"]["acc_norm"]

        sweep_results.append({
            "rank_ratio": ratio,
            "target_tensors": target_tensors,
            "target_factorized_param_saving_pct": round(target_param_saving_pct, 2),
            "total_model_param_saving_pct": round(total_model_saving_pct, 2),
            "metrics": slim_metrics,
            "wikitext2_ppl_delta_pct": round(wt2_delta_pct, 2),
            "lambada_acc_delta": round(lb_delta, 4),
            "arc_easy_acc_norm_delta": round(arc_delta, 4),
            "elapsed_s": t_slim,
        })

    ws_slim.restore("clean_slim")
    suite4_slim = {
        "clean_baseline": clean_metrics,
        "sweeps": sweep_results,
    }

    return {
        "model_id": model_id,
        "hf_repo_id": entry.hf_repo_id,
        "total_params": total_params,
        "load_time_s": load_s,
        "baseline_eval_time_s": eval_s,
        "suite5_knowledge_map": suite5_kmap,
        "suite5_vespasian_guard": suite5_vespasian,
        "suite3_outlier_repair": suite3_outlier,
        "suite4_memory_slim": suite4_slim,
    }
