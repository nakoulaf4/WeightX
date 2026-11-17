# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Suite 2: Batch Factual Knowledge Editing (`EAMIT` on CounterFact & Scaling).

Evaluates WeightX's `apply_eamit_batch` (Embedding Alignment Model Editing in
Transformers, arXiv:2505.11876) on real CounterFact factual triples across
batch sizes N = 10, 50, 100, plus an ablation comparing full EAMIT alignment
(KL + top-M MSE) vs unaligned closed-form ridge (align_steps=0).
"""

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.eval_suites import (
    eval_counterfact_facts,
    eval_lambada,
    eval_wikitext2_ppl,
)
from benchmarks.prepare_datasets import load_cached
from weightx.catalog import ModelCatalog
from weightx.l3_weights.domain_eamit import select_edit_layers
from weightx.l3_weights.eamit import BatchFactSpec, EamitAlignConfig, apply_eamit_batch
from weightx.l3_weights.workspace import WeightWorkspace
from weightx.safety.iwck_fisher import FisherGuard


def _prepare_counterfact_pool(
    model: Any,
    tok: Any,
    rows: list[dict[str, Any]],
    *,
    max_items: int = 260,
) -> list[dict[str, Any]]:
    """Select CounterFact items where the base model currently prefers `target_true` over `target_new`."""
    dev = next(model.parameters()).device
    selected: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for r in rows:
            prompt = str(r["prompt"])
            subj = str(r["subject"]).strip()
            t_true = str(r["target_true"])
            t_new = str(r["target_false"])
            if not subj or subj not in prompt:
                continue
            true_ids = tok.encode(t_true, add_special_tokens=False)
            new_ids = tok.encode(t_new, add_special_tokens=False)
            if not true_ids or not new_ids or true_ids[0] == new_ids[0]:
                continue
            ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
            if tok.bos_token_id is not None:
                bos = torch.tensor([[tok.bos_token_id]], dtype=torch.long, device=dev)
                ids = torch.cat([bos, ids], dim=1)
            logits = model(ids).logits[0, -1].float()
            if float(logits[true_ids[0]].item()) > float(logits[new_ids[0]].item()):
                selected.append(
                    {
                        "prompt": prompt,
                        "subject": subj,
                        "target_true": t_true,
                        "target_new": t_new,
                        "relation_id": r.get("relation_id", ""),
                    }
                )
            if len(selected) >= max_items:
                break
    return selected


def _compute_counterfact_kv_specs(
    model: Any,
    tok: Any,
    items: list[dict[str, Any]],
    target_tensors: list[str],
    *,
    opt_steps: int = 18,
    opt_lr: float = 0.45,
) -> list[BatchFactSpec]:
    """Compute real MLP input keys `k_l` at subject-end and optimal activation-space delta `v*`."""
    dev = next(model.parameters()).device
    modules = dict(model.named_modules())
    mod_names = [t[: -len(".weight")] for t in target_tensors]
    anchor_mod = mod_names[-1]
    d_out = modules[anchor_mod].weight.shape[0]
    w_list = [modules[mn].weight.detach().float().cpu().numpy().astype(np.float64) for mn in mod_names]

    specs: list[BatchFactSpec] = []
    for idx, item in enumerate(items):
        prompt = item["prompt"]
        subj = item["subject"]
        t_new_id = tok.encode(item["target_new"], add_special_tokens=False)[0]
        t_true_id = tok.encode(item["target_true"], add_special_tokens=False)[0]

        sub_pos = prompt.find(subj)
        prefix_str = prompt[: sub_pos + len(subj)]
        full_ids = tok.encode(prompt, add_special_tokens=False)
        pref_ids = tok.encode(prefix_str, add_special_tokens=False)
        offset = 1 if tok.bos_token_id is not None else 0
        if tok.bos_token_id is not None:
            full_ids = [tok.bos_token_id] + full_ids
        subj_tok_idx = min(max(0, offset + len(pref_ids) - 1), len(full_ids) - 1)
        x = torch.tensor([full_ids], dtype=torch.long, device=dev)

        # 1. Capture base probabilities and per-layer MLP down_proj input keys at `subj_tok_idx`
        layer_keys: dict[str, np.ndarray] = {}
        hooks = []
        for mn, tn in zip(mod_names, target_tensors):
            def _mk_cap(tname: str):
                def _hk(_m: Any, inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
                    layer_keys[tname] = inp[0][0, subj_tok_idx].detach().float().cpu().numpy().astype(np.float64)
                return _hk
            hooks.append(modules[mn].register_forward_hook(_mk_cap(tn)))

        with torch.no_grad():
            base_logits = model(x).logits.float()
            base_probs_prefix = F.softmax(base_logits[:, : max(1, subj_tok_idx + 1), :], dim=-1)
        for h in hooks:
            h.remove()

        # 2. Optimize activation residual `delta` at `anchor_mod` output at `subj_tok_idx`
        delta = torch.zeros(d_out, device=dev, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=opt_lr)

        def _inject_hook(_m: Any, _inp: Any, out: torch.Tensor) -> torch.Tensor:
            patched = out.clone()
            patched[0, subj_tok_idx, :] = patched[0, subj_tok_idx, :] + delta
            return patched

        h_inj = modules[anchor_mod].register_forward_hook(_inject_hook)
        try:
            with torch.enable_grad():
                for _step in range(opt_steps):
                    opt.zero_grad(set_to_none=True)
                    logits = model(x).logits.float()
                    last_lp = F.log_softmax(logits[0, -1, :], dim=-1)
                    margin = last_lp[t_new_id] - last_lp[t_true_id]
                    loss_tgt = -last_lp[t_new_id] + 0.15 * F.relu(-margin + 2.5)
                    # KL preservation at prefix tokens
                    cand_lp = F.log_softmax(logits[:, : max(1, subj_tok_idx + 1), :], dim=-1)
                    kl = F.kl_div(cand_lp, base_probs_prefix, reduction="batchmean")
                    loss = loss_tgt + 0.2 * kl + 0.02 * delta.pow(2).mean()
                    loss.backward()
                    opt.step()
                    if _step >= 4 and float(margin.item()) > 3.0:
                        break
        finally:
            h_inj.remove()

        k_anchor = layer_keys[target_tensors[-1]]
        delta_np = delta.detach().cpu().numpy().astype(np.float64)
        init_out = sum(w @ k_anchor for w in w_list)
        v_target = init_out + delta_np

        specs.append(
            BatchFactSpec(
                fact_id=f"cf_{idx}",
                key_vector=k_anchor,
                target_value_vector=v_target,
                layer_inputs=layer_keys,
            )
        )
        if (idx + 1) % 25 == 0 or (idx + 1) == len(items):
            print(f"    kv-spec {idx + 1}/{len(items)}", flush=True)

    return specs


def _paraphrase_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefixes = [
        "According to updated records, ",
        "In fact, ",
        "Note that ",
        "Verified information states that ",
    ]
    out: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        pref = prefixes[i % len(prefixes)]
        p = it["prompt"]
        out.append({**it, "prompt": pref + p[0].lower() + p[1:] if p else pref + p})
    return out


def run_model_eamit_facts_suite(model_id: str, *, device: str = "cuda") -> dict[str, Any]:
    """Run CounterFact batch editing & associative memory scaling on `model_id`."""
    cf_rows = load_cached("counterfact_train")
    wt_rows = load_cached("wikitext2_test")
    lb_rows = load_cached("lambada_test")

    cat = ModelCatalog()
    t_load = time.perf_counter()
    model, tok, entry = cat.load_model_and_tokenizer(model_id, device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    load_s = round(time.perf_counter() - t_load, 2)

    target_mods = select_edit_layers(model, n_layers=4, lo=0.55, hi=0.78)
    target_tensors = [f"{m}.weight" for m in target_mods]
    pristine = {
        n: p.detach().clone().cpu()
        for n, p in model.named_parameters()
        if n in target_tensors
    }

    def _restore_pristine() -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in pristine:
                    p.copy_(pristine[n].to(device=p.device, dtype=p.dtype))

    # Build known-fact CounterFact pool
    pool = _prepare_counterfact_pool(model, tok, cf_rows, max_items=220)
    edit_limit = min(100, max(10, len(pool) // 2))
    edit_pool = pool[:edit_limit]
    neigh_pool = pool[edit_limit:]
    para_pool = _paraphrase_items(edit_pool)

    # Pre-compute (K, V*) specs once on pristine model
    print(f"  [facts:{model_id}] extracting CounterFact K/V* specs (pool={len(edit_pool)})...", flush=True)
    t_spec = time.perf_counter()
    all_specs = _compute_counterfact_kv_specs(model, tok, edit_pool, target_tensors)
    spec_time_s = round(time.perf_counter() - t_spec, 2)

    # Calibration Fisher guard from neighborhood prompts
    calib_rows = [
        torch.tensor(tok.encode(it["prompt"], add_special_tokens=False)[:6], dtype=torch.long)
        for it in neigh_pool
        if len(tok.encode(it["prompt"], add_special_tokens=False)) >= 6
    ][:8]
    calib_ids = torch.stack(calib_rows).to(device)
    fguard = FisherGuard.from_model(
        model,
        reference_batches=[calib_ids],
        damping_alpha=0.10,
        freeze_top_fraction=0.02,
        enabled=True,
    )

    base_wt = eval_wikitext2_ppl(model, tok, wt_rows, max_tokens=65536)
    base_lb = eval_lambada(model, tok, lb_rows, max_items=400, batch_size=16)
    base_neigh = eval_counterfact_facts(model, tok, neigh_pool)

    batch_runs: list[dict[str, Any]] = []
    for n_facts in (10, 50, 100):
        if n_facts > len(all_specs):
            continue
        print(f"  [facts:{model_id}] CounterFact batch N={n_facts}...", flush=True)
        _restore_pristine()
        ws = WeightWorkspace(model, fisher_guard=fguard)
        pre_edit = eval_counterfact_facts(model, tok, edit_pool[:n_facts])
        t0 = time.perf_counter()
        res = apply_eamit_batch(
            ws,
            target_tensors,
            all_specs[:n_facts],
            lambda_cov=0.2,
            apply_fisher_guard=True,
            align_config=EamitAlignConfig(lambda_kl=0.05, lambda_mse=0.25, top_m=8, align_steps=20),
        )
        edit_s = round(time.perf_counter() - t0, 3)

        post_edit = eval_counterfact_facts(model, tok, edit_pool[:n_facts])
        post_para = eval_counterfact_facts(model, tok, para_pool[:n_facts])
        post_neigh = eval_counterfact_facts(model, tok, neigh_pool)
        # Neighborhood retention: % where P(true) still > P(new) (1 - efficacy_score on neighborhood)
        neigh_specificity = round(1.0 - post_neigh["efficacy_score"], 4)

        retention = None
        if n_facts == 50:
            retention = {
                "wikitext2": eval_wikitext2_ppl(model, tok, wt_rows, max_tokens=65536),
                "lambada": eval_lambada(model, tok, lb_rows, max_items=400, batch_size=16),
            }

        batch_runs.append({
            "num_facts": n_facts,
            "edit_time_s": edit_s,
            "pre_edit_efficacy": pre_edit["efficacy_score"],
            "post_edit": post_edit,
            "paraphrase": post_para,
            "neighborhood_specificity": neigh_specificity,
            "eamit_telemetry": res.to_dict(),
            "retention_eval": retention,
        })
        del ws
        gc.collect()

    # Ablation at N=50: No Embedding Alignment (align_steps=0) vs Full EAMIT
    ablation_no_align = None
    if len(all_specs) >= 50:
        _restore_pristine()
        ws_noa = WeightWorkspace(model, fisher_guard=fguard)
        t0 = time.perf_counter()
        res_noa = apply_eamit_batch(
            ws_noa,
            target_tensors,
            all_specs[:50],
            lambda_cov=0.2,
            apply_fisher_guard=True,
            align_config=EamitAlignConfig(lambda_kl=0.0, lambda_mse=0.0, align_steps=0),
        )
        t_noa = round(time.perf_counter() - t0, 3)
        ablation_no_align = {
            "edit_time_s": t_noa,
            "post_edit": eval_counterfact_facts(model, tok, edit_pool[:50]),
            "paraphrase": eval_counterfact_facts(model, tok, para_pool[:50]),
            "neighborhood_specificity": round(1.0 - eval_counterfact_facts(model, tok, neigh_pool)["efficacy_score"], 4),
            "eamit_telemetry": res_noa.to_dict(),
            "wikitext2": eval_wikitext2_ppl(model, tok, wt_rows, max_tokens=65536),
            "lambada": eval_lambada(model, tok, lb_rows, max_items=400, batch_size=16),
        }
        del ws_noa
        gc.collect()

    # Part 2: Associative Key-Value Memory Scaling (N=10, 50, 100, 250, 500)
    _restore_pristine()
    ws_scale = WeightWorkspace(model, fisher_guard=fguard)
    ws_scale.checkpoint("clean")
    d_out_s, d_in_s = ws_scale.shape(target_tensors[0])
    rng = np.random.default_rng(7)
    scaling_rows: list[dict[str, Any]] = []
    for n_assoc in (10, 50, 100, 250, 500):
        ws_scale.restore("clean")
        keys = rng.standard_normal((n_assoc, d_in_s))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8
        vals = rng.standard_normal((n_assoc, d_out_s)) * 0.05
        facts = [BatchFactSpec(f"s_{i}", keys[i], vals[i]) for i in range(n_assoc)]
        t0 = time.perf_counter()
        r_s = apply_eamit_batch(ws_scale, target_tensors, facts, lambda_cov=1e-2, apply_fisher_guard=True)
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        scaling_rows.append({
            "num_facts": n_assoc,
            "elapsed_ms": ms,
            "pre_edit_mse": round(r_s.pre_edit_mse, 6),
            "post_edit_mse": round(r_s.post_edit_mse, 6),
            "error_reduction_pct": round(r_s.relative_error_reduction * 100.0, 2),
            "pre_alignment_kl": round(r_s.pre_alignment_kl, 5),
            "post_alignment_kl": round(r_s.post_alignment_kl, 5),
            "pre_top_m_mse": round(r_s.pre_alignment_top_m_mse, 5),
            "post_top_m_mse": round(r_s.post_alignment_top_m_mse, 5),
        })

    _restore_pristine()
    return {
        "model_id": model_id,
        "hf_repo_id": entry.hf_repo_id,
        "target_tensors": target_tensors,
        "load_time_s": load_s,
        "spec_extraction_time_s": spec_time_s,
        "pool_size_known_facts": len(pool),
        "baseline": {
            "wikitext2": base_wt,
            "lambada": base_lb,
            "neighborhood_base_specificity": round(1.0 - base_neigh["efficacy_score"], 4),
        },
        "counterfact_batches": batch_runs,
        "ablation_no_alignment_n50": ablation_no_align,
        "associative_scaling": scaling_rows,
    }
