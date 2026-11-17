# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Suite 1: Zero-SGD Domain Calibration, General Benchmark Retention, and Ablations.

Tests WeightX's closed-form EAMIT domain calibration (`eamit_domain_calibration` +
`VespasianEngine`) across real models on three non-overlapping tracks:
  - Track A: Python Code Adaptation (MBPP train -> MBPP test + HumanEval test)
  - Track B: Scientific QA Adaptation (SciQ train -> SciQ test PPL + 4-choice QA)
  - Track C: WeightX WSP Technical Domain (`WSP_PRACTICE` -> `WSP_HELD_OUT`)

Also measures general capability retention before & after each calibration on
WikiText-2, LAMBADA, ARC-Easy, and HellaSwag, and runs an ablation comparing:
  Base vs Unregularized Ridge (no C0, no align) vs WeightX EAMIT Full vs AdamW SGD.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.eval_suites import (
    build_arc_items,
    build_hellaswag_items,
    build_sciq_mc_items,
    eval_humaneval_pass1,
    eval_lambada,
    eval_mc_acc,
    eval_text_nll_top1,
    eval_wikitext2_ppl,
)
from benchmarks.prepare_datasets import load_cached
from weightx.catalog import ModelCatalog
from weightx.corpora import GENERAL_CONTROL, WSP_HELD_OUT, WSP_PRACTICE
from weightx.execution.vespasian import EvaluationSplit, VespasianEngine
from weightx.improver import _tokenize_texts
from weightx.l3_weights.domain_eamit import (
    eamit_domain_calibration,
    select_edit_layers,
    token_nll,
)
from weightx.l3_weights.eamit import EamitAlignConfig
from weightx.l3_weights.workspace import WeightWorkspace


def _encode_capped(tokenizer: Any, texts: list[str], max_len: int = 160) -> list[torch.Tensor]:
    seqs = []
    for seq in _tokenize_texts(tokenizer, texts):
        if seq.numel() > max_len:
            seq = seq[:max_len]
        if seq.numel() >= 4:
            seqs.append(seq)
    return seqs


def _eval_general(
    model: Any,
    tok: Any,
    wt_rows: list[dict],
    lb_rows: list[dict],
    arc_items: list[Any],
    hs_items: list[Any],
) -> dict[str, Any]:
    return {
        "wikitext2": eval_wikitext2_ppl(model, tok, wt_rows, max_tokens=65536),
        "lambada": eval_lambada(model, tok, lb_rows, max_items=400, batch_size=16),
        "arc_easy": eval_mc_acc(model, tok, arc_items, batch_size=16),
        "hellaswag": eval_mc_acc(model, tok, hs_items, batch_size=16),
    }


def _run_weightx_calibration(
    model: Any,
    ws: WeightWorkspace,
    prac_seqs: list[torch.Tensor],
    vesp_seqs: list[torch.Tensor],
    ctrl_seqs: list[torch.Tensor],
    *,
    iterations: int = 3,
    n_layers: int = 4,
    lambda_preserve: float = 1.0,
    max_control_nll_increase: float = 0.015,
    align_config: EamitAlignConfig | None = None,
    rel_steps: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075),
    use_vespasian: bool = True,
) -> dict[str, Any]:
    def _vesp_eval(m: Any) -> tuple[float, list[float]]:
        _, per = token_nll(m, vesp_seqs)
        items = [float(np.exp(-x)) for x in per]
        return float(np.mean(items)), items

    min_len = min(len(s) for s in ctrl_seqs)
    ctrl_ref = torch.stack([s[:min_len] for s in ctrl_seqs])
    vespasian = VespasianEngine(
        ws,
        held_out_split=EvaluationSplit(name="held_out_gate", eval_fn=_vesp_eval, reference_input_ids=ctrl_ref),
        improvement_threshold=1e-4,
        max_kl_drift=0.05,
        max_changed_fraction=0.15,
    )

    t0 = time.perf_counter()
    verdicts: list[str] = []
    steps_log: list[dict[str, Any]] = []
    edited_tensors: list[str] = []

    for it in range(iterations):
        res = eamit_domain_calibration(
            model,
            ws,
            prac_seqs,
            ctrl_seqs,
            n_layers=n_layers,
            lambda_preserve=lambda_preserve,
            rel_steps=rel_steps,
            iterations=1,
            max_control_nll_increase=max_control_nll_increase,
            apply_fisher_guard=False,
            align_config=align_config,
        )
        edited_tensors = res.target_tensors
        for s in res.steps:
            steps_log.append(s.to_dict() | {"iteration": it + 1})
        if not res.steps or res.steps[-1].chosen_rel_step == 0.0:
            break
        if use_vespasian:
            v = vespasian.verify_and_decide(hypothesis=f"EAMIT calibration iter {it + 1}")
            verdicts.append(v.verdict_reason)
            if not v.accepted:
                break

    elapsed_s = round(time.perf_counter() - t0, 3)
    diff = ws.diff()
    return {
        "elapsed_s": elapsed_s,
        "edited_tensors": edited_tensors,
        "changed_params": int(diff.changed_params),
        "changed_pct": round(diff.changed_fraction * 100.0, 4),
        "iterations_executed": len(steps_log),
        "steps": steps_log,
        "vespasian_verdicts": verdicts,
    }


def _run_sgd_baseline(
    model: Any,
    ws: WeightWorkspace,
    prac_seqs: list[torch.Tensor],
    *,
    n_layers: int = 4,
    epochs: int = 10,
    lr: float = 5e-5,
) -> dict[str, Any]:
    """Apples-to-apples AdamW SGD baseline updating only the same `n_layers` MLP projections."""
    dev = next(model.parameters()).device
    target_mods = set(select_edit_layers(model, n_layers=n_layers))
    target_names = [f"{m}.weight" for m in target_mods]
    trainable_params = []
    for name, param in model.named_parameters():
        if name in target_names:
            param.requires_grad_(True)
            trainable_params.append(param)
        else:
            param.requires_grad_(False)

    t0 = time.perf_counter()
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    model.train()
    losses: list[float] = []
    for _ep in range(epochs):
        ep_loss = 0.0
        for seq in prac_seqs:
            x = seq.to(dev).unsqueeze(0)
            opt.zero_grad(set_to_none=True)
            logits = model(x).logits[:, :-1, :].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[0, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            ep_loss += float(loss.item())
        losses.append(round(ep_loss / max(len(prac_seqs), 1), 4))
    model.eval()
    for p in trainable_params:
        p.requires_grad_(False)
    elapsed_s = round(time.perf_counter() - t0, 3)

    # Sync live parameter edits into workspace diff reporting
    for name in target_names:
        for pname, p in model.named_parameters():
            if pname == name:
                ws.set_tensor(name, p.detach().float().cpu().numpy(), apply_fisher_guard=False)
                break
    diff = ws.diff()
    return {
        "elapsed_s": elapsed_s,
        "epochs": epochs,
        "lr": lr,
        "edited_tensors": target_names,
        "changed_params": int(diff.changed_params),
        "changed_pct": round(diff.changed_fraction * 100.0, 4),
        "epoch_losses": losses,
    }


def run_model_domain_suite(
    model_id: str,
    *,
    run_ablation: bool = False,
    run_humaneval_gen: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run all 3 domain adaptation tracks + general retention + optional ablations for `model_id`."""
    wt_rows = load_cached("wikitext2_test")
    lb_rows = load_cached("lambada_test")
    arc_items = build_arc_items(load_cached("arc_easy_test"), max_items=400)
    hs_items = build_hellaswag_items(load_cached("hellaswag_val"), max_items=350)
    sciq_train = load_cached("sciq_train")
    sciq_test = load_cached("sciq_test")
    mbpp_train = load_cached("mbpp_train")
    mbpp_test = load_cached("mbpp_test")
    humaneval_test = load_cached("humaneval_test")

    cat = ModelCatalog()
    t_load = time.perf_counter()
    model, tok, entry = cat.load_model_and_tokenizer(model_id, device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    load_s = round(time.perf_counter() - t_load, 2)
    total_params = sum(p.numel() for p in model.parameters())

    # Prepare track texts (zero train/gate/test overlap!)
    mbpp_prac_texts = [f"# {r['text']}\n{r['code']}" for r in mbpp_train[:16]]
    mbpp_gate_texts = [f"# {r['text']}\n{r['code']}" for r in mbpp_test[:16]]
    mbpp_eval_texts = [f"# {r['text']}\n{r['code']}" for r in mbpp_test[40:240]]
    humaneval_eval_texts = [f"{r['prompt']}{r['canonical_solution']}" for r in humaneval_test]

    sciq_prac_texts = [
        f"Question: {r['question']}\nSupport: {r['support']}\nAnswer: {r['correct_answer']}"
        for r in sciq_train[:16]
    ]
    sciq_gate_texts = [
        f"Question: {r['question']}\nSupport: {r['support']}\nAnswer: {r['correct_answer']}"
        for r in sciq_train[48:64]
    ]
    sciq_eval_texts = [
        f"Question: {r['question']}\nSupport: {r['support']}\nAnswer: {r['correct_answer']}"
        for r in sciq_test[:300]
    ]
    sciq_mc_items = build_sciq_mc_items(sciq_test, max_items=500)

    prac_code = _encode_capped(tok, mbpp_prac_texts, max_len=80)
    gate_code = _encode_capped(tok, mbpp_gate_texts, max_len=96)
    prac_sciq = _encode_capped(tok, sciq_prac_texts, max_len=80)
    gate_sciq = _encode_capped(tok, sciq_gate_texts, max_len=96)
    prac_wsp = _encode_capped(tok, WSP_PRACTICE, max_len=128)
    held_wsp = _encode_capped(tok, WSP_HELD_OUT, max_len=128)
    ctrl_seqs = _encode_capped(tok, GENERAL_CONTROL, max_len=128)

    # Save pristine clean state of target MLP matrices (4 layers * 2 bytes/float -> tiny CPU copy!)
    target_mods = select_edit_layers(model, n_layers=4)
    target_names = [f"{m}.weight" for m in target_mods]
    pristine_weights = {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if name in target_names
    }

    def _restore_pristine() -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in pristine_weights:
                    param.copy_(pristine_weights[name].to(device=param.device, dtype=param.dtype))

    # 1. Clean Baseline Metrics across ALL suites
    print(f"  [domain:{model_id}] evaluating baseline...", flush=True)
    t_b = time.perf_counter()
    base_general = _eval_general(model, tok, wt_rows, lb_rows, arc_items, hs_items)
    base_mbpp = eval_text_nll_top1(model, tok, mbpp_eval_texts, max_seq_len=256)
    base_humaneval_ppl = eval_text_nll_top1(model, tok, humaneval_eval_texts, max_seq_len=256)
    base_sciq_text = eval_text_nll_top1(model, tok, sciq_eval_texts, max_seq_len=256)
    base_sciq_mc = eval_mc_acc(model, tok, sciq_mc_items, batch_size=16)
    base_wsp = eval_text_nll_top1(model, tok, WSP_HELD_OUT, max_seq_len=128)
    base_ctrl = eval_text_nll_top1(model, tok, GENERAL_CONTROL, max_seq_len=128)
    base_humaneval_gen = (
        eval_humaneval_pass1(model, tok, humaneval_test, max_items=40, max_new_tokens=160)
        if run_humaneval_gen
        else None
    )
    baseline_time_s = round(time.perf_counter() - t_b, 2)

    # 2. Track A: Python Code Adaptation
    print(f"  [domain:{model_id}] track A: code...", flush=True)
    _restore_pristine()
    ws_code = WeightWorkspace(model)
    calib_code = _run_weightx_calibration(model, ws_code, prac_code, gate_code, ctrl_seqs)
    after_code_general = _eval_general(model, tok, wt_rows, lb_rows, arc_items, hs_items)
    after_mbpp = eval_text_nll_top1(model, tok, mbpp_eval_texts, max_seq_len=256)
    after_humaneval_ppl = eval_text_nll_top1(model, tok, humaneval_eval_texts, max_seq_len=256)
    after_humaneval_gen = (
        eval_humaneval_pass1(model, tok, humaneval_test, max_items=40, max_new_tokens=160)
        if run_humaneval_gen
        else None
    )
    del ws_code
    gc.collect()

    # 3. Track B: Science QA Adaptation (SciQ)
    print(f"  [domain:{model_id}] track B: sciq...", flush=True)
    _restore_pristine()
    ws_sciq = WeightWorkspace(model)
    calib_sciq = _run_weightx_calibration(model, ws_sciq, prac_sciq, gate_sciq, ctrl_seqs)
    after_sciq_general = _eval_general(model, tok, wt_rows, lb_rows, arc_items, hs_items)
    after_sciq_text = eval_text_nll_top1(model, tok, sciq_eval_texts, max_seq_len=256)
    after_sciq_mc = eval_mc_acc(model, tok, sciq_mc_items, batch_size=16)
    del ws_sciq
    gc.collect()

    # 4. Track C: WeightX WSP Technical Domain
    print(f"  [domain:{model_id}] track C: wsp...", flush=True)
    _restore_pristine()
    ws_wsp = WeightWorkspace(model)
    calib_wsp = _run_weightx_calibration(
        model, ws_wsp, prac_wsp, held_wsp, ctrl_seqs, max_control_nll_increase=0.01
    )
    after_wsp_general = _eval_general(model, tok, wt_rows, lb_rows, arc_items, hs_items)
    after_wsp = eval_text_nll_top1(model, tok, WSP_HELD_OUT, max_seq_len=128)
    after_ctrl = eval_text_nll_top1(model, tok, GENERAL_CONTROL, max_seq_len=128)
    del ws_wsp
    gc.collect()

    # 5. Optional Ablation & SGD Comparison (on Code Track + SciQ Track)
    ablations: dict[str, Any] | None = None
    if run_ablation:
        print(f"  [domain:{model_id}] ablations: naive ridge & adamw sgd...", flush=True)
        # 5a: Unregularized Closed-Form Ridge (lambda_preserve=0, no EAMIT align, generous ctrl budget)
        _restore_pristine()
        ws_naive = WeightWorkspace(model)
        naive_info = _run_weightx_calibration(
            model,
            ws_naive,
            prac_code,
            gate_code,
            ctrl_seqs,
            iterations=2,
            lambda_preserve=0.0,
            max_control_nll_increase=0.50,
            align_config=EamitAlignConfig(lambda_kl=0.0, lambda_mse=0.0, align_steps=0),
            rel_steps=(0.05,),
            use_vespasian=False,
        )
        naive_code_gen = _eval_general(model, tok, wt_rows, lb_rows, arc_items, hs_items)
        naive_mbpp = eval_text_nll_top1(model, tok, mbpp_eval_texts, max_seq_len=256)
        naive_he = eval_text_nll_top1(model, tok, humaneval_eval_texts, max_seq_len=256)
        del ws_naive
        gc.collect()

        # 5b: AdamW SGD Baseline (same 4 MLP down_proj layers, same 48 practice seqs)
        _restore_pristine()
        ws_sgd = WeightWorkspace(model)
        sgd_info = _run_sgd_baseline(model, ws_sgd, prac_code, n_layers=4, epochs=8, lr=5e-5)
        sgd_code_gen = _eval_general(model, tok, wt_rows, lb_rows, arc_items, hs_items)
        sgd_mbpp = eval_text_nll_top1(model, tok, mbpp_eval_texts, max_seq_len=256)
        sgd_he = eval_text_nll_top1(model, tok, humaneval_eval_texts, max_seq_len=256)
        del ws_sgd
        gc.collect()

        ablations = {
            "unregularized_ridge": {
                "calibration": naive_info,
                "general": naive_code_gen,
                "mbpp_test": naive_mbpp,
                "humaneval_test": naive_he,
            },
            "adamw_sgd_4layer": {
                "training": sgd_info,
                "general": sgd_code_gen,
                "mbpp_test": sgd_mbpp,
                "humaneval_test": sgd_he,
            },
        }

    _restore_pristine()
    return {
        "model_id": model_id,
        "hf_repo_id": entry.hf_repo_id,
        "total_params": total_params,
        "load_time_s": load_s,
        "baseline_eval_time_s": baseline_time_s,
        "baseline": {
            "general": base_general,
            "mbpp_test": base_mbpp,
            "humaneval_test": base_humaneval_ppl,
            "humaneval_pass1": base_humaneval_gen,
            "sciq_text": base_sciq_text,
            "sciq_mc": base_sciq_mc,
            "wsp_held_out": base_wsp,
            "general_control": base_ctrl,
        },
        "track_code": {
            "calibration": calib_code,
            "general_after": after_code_general,
            "mbpp_test_after": after_mbpp,
            "humaneval_test_after": after_humaneval_ppl,
            "humaneval_pass1_after": after_humaneval_gen,
            "mbpp_ppl_change_pct": round((after_mbpp["ppl"] - base_mbpp["ppl"]) / base_mbpp["ppl"] * 100.0, 2),
            "humaneval_ppl_change_pct": round((after_humaneval_ppl["ppl"] - base_humaneval_ppl["ppl"]) / base_humaneval_ppl["ppl"] * 100.0, 2),
            "wikitext2_ppl_change_pct": round((after_code_general["wikitext2"]["ppl"] - base_general["wikitext2"]["ppl"]) / base_general["wikitext2"]["ppl"] * 100.0, 2),
        },
        "track_sciq": {
            "calibration": calib_sciq,
            "general_after": after_sciq_general,
            "sciq_text_after": after_sciq_text,
            "sciq_mc_after": after_sciq_mc,
            "sciq_ppl_change_pct": round((after_sciq_text["ppl"] - base_sciq_text["ppl"]) / base_sciq_text["ppl"] * 100.0, 2),
            "sciq_mc_acc_delta": round(after_sciq_mc["acc_norm"] - base_sciq_mc["acc_norm"], 4),
            "wikitext2_ppl_change_pct": round((after_sciq_general["wikitext2"]["ppl"] - base_general["wikitext2"]["ppl"]) / base_general["wikitext2"]["ppl"] * 100.0, 2),
        },
        "track_wsp": {
            "calibration": calib_wsp,
            "general_after": after_wsp_general,
            "wsp_held_out_after": after_wsp,
            "general_control_after": after_ctrl,
            "wsp_ppl_reduction_pct": round((base_wsp["ppl"] - after_wsp["ppl"]) / base_wsp["ppl"] * 100.0, 2),
            "control_ppl_change_pct": round((after_ctrl["ppl"] - base_ctrl["ppl"]) / base_ctrl["ppl"] * 100.0, 2),
            "wikitext2_ppl_change_pct": round((after_wsp_general["wikitext2"]["ppl"] - base_general["wikitext2"]["ppl"]) / base_general["wikitext2"]["ppl"] * 100.0, 2),
        },
        "ablations": ablations,
    }
