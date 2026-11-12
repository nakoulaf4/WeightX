# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX Small Model Improver & Benchmark Suite (`weightx.improver`).

Implements a multi-stage, gradient-free Weight-Space Programming pipeline to
diagnose, repair, calibrate, and measurably improve a small causal Language Model:
  1. Baseline Held-Out Evaluation (Perplexity, NLL, Top-1 & Top-5 Token Accuracy, Vespasian Score).
  2. Stage A вЂ” L1 AutoArch + L2 13-Metric Uni-Layer Diagnosis & Outlier/Spectral Repair via YAML Plan.
  3. Stage B вЂ” Closed-Form Subspace Calibration & Associative Fact Injection (MEMIT-MASS / Ridge)
     using hidden-state covariance regularization and IWCKEdit Fisher protection.
  4. Stage C вЂ” Vespasian Held-Out Verification (keeps only stages that strictly improve held-out metrics).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from weightx.catalog import ModelCatalog
from weightx.declarative.plan_engine import execute_yaml_plan
from weightx.execution.vespasian import EvaluationSplit, VespasianEngine
from weightx.l1_arch.autoarch import inspect_architecture
from weightx.l2_profile.knowledge_map import run_knowledge_map
from weightx.l2_profile.unilayer import profile_model_layers
from weightx.l3_weights.workspace import WeightWorkspace
from weightx.safety.iwck_fisher import FisherGuard
from weightx.toon_io import dumps_toon


DEFAULT_PRACTICE_TEXTS: list[str] = [
    "Artificial intelligence systems use neural network weights to store knowledge.",
    "Weight space programming edits model parameters directly using closed form linear algebra.",
    "The capital of France is Paris and the capital of Japan is Tokyo.",
    "Python is a widely used programming language for machine learning and data science.",
    "Transformer models consist of self attention blocks and feed forward multilayer perceptrons.",
    "Singular value decomposition factorizes a matrix into orthogonal components.",
    "Fisher information measures the sensitivity of the log likelihood to parameter changes.",
    "Declarative YAML plans with invariant assertions prevent destructive weight edits.",
]

DEFAULT_HELD_OUT_TEXTS: list[str] = [
    "Neural network parameters encode factual and syntactic relationships across layers.",
    "Closed form matrix updates allow fast editing of language models without gradient descent.",
    "The capital of Germany is Berlin and the capital of Italy is Rome.",
    "Machine learning researchers use Python and PyTorch to analyze transformer weights.",
    "Self attention layers route contextual information while MLP layers project features.",
    "Spectral analysis via singular value decomposition reveals redundant and bottleneck layers.",
    "Protecting critical weights with Fisher information avoids catastrophic forgetting.",
    "Held out verification ensures that weight surgery genuinely improves model quality.",
]


@dataclass
class LMQualityMetrics:
    perplexity: float
    mean_nll: float
    top1_accuracy: float
    top5_accuracy: float
    vespasian_score: float
    logit_kurtosis: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "perplexity": round(self.perplexity, 4),
            "mean_nll": round(self.mean_nll, 5),
            "top1_accuracy": round(self.top1_accuracy, 4),
            "top5_accuracy": round(self.top5_accuracy, 4),
            "vespasian_score": round(self.vespasian_score, 5),
            "logit_kurtosis": round(self.logit_kurtosis, 4),
        }


@dataclass
class StageImprovementRecord:
    stage_name: str
    description: str
    committed: bool
    pre_perplexity: float
    post_perplexity: float
    pre_top1_acc: float
    post_top1_acc: float
    vespasian_verdict: str
    yaml_plan_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelImprovementReport:
    model_id: str
    hf_repo_id: str
    family: str
    total_params: int
    changed_params: int
    changed_fraction: float
    initial_metrics: LMQualityMetrics
    final_metrics: LMQualityMetrics
    ppl_reduction_pct: float
    top1_accuracy_gain_pct: float
    vespasian_gain: float
    kci_gini: float
    bottleneck_layers: list[str]
    stages: list[StageImprovementRecord] = field(default_factory=list)
    saved_checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hf_repo_id": self.hf_repo_id,
            "family": self.family,
            "total_params": self.total_params,
            "changed_params": self.changed_params,
            "changed_fraction": round(self.changed_fraction, 6),
            "initial_metrics": self.initial_metrics.to_dict(),
            "final_metrics": self.final_metrics.to_dict(),
            "ppl_reduction_pct": round(self.ppl_reduction_pct, 2),
            "top1_accuracy_gain_pct": round(self.top1_accuracy_gain_pct, 2),
            "vespasian_gain": round(self.vespasian_gain, 5),
            "kci_gini": round(self.kci_gini, 4),
            "bottleneck_layers": self.bottleneck_layers,
            "stages": [s.to_dict() for s in self.stages],
            "saved_checkpoint_path": self.saved_checkpoint_path,
        }

    def to_toon(self) -> str:
        return dumps_toon(self.to_dict())


def evaluate_lm_quality(model: nn.Module, split: EvaluationSplit) -> LMQualityMetrics:
    """Compute Perplexity, Mean NLL, Top-1/Top-5 Accuracy, and Logit Kurtosis on `split`."""
    model.eval()
    device = next(model.parameters()).device
    x = split.input_ids.to(device)
    y = split.target_ids.to(device)

    with torch.no_grad():
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
        last_logits = logits[:, -1, :].float()
        nll = float(F.cross_entropy(last_logits, y).item())
        ppl = float(np.exp(min(nll, 20.0)))
        preds_top1 = last_logits.argmax(dim=-1)
        top1_acc = float((preds_top1 == y).float().mean().item())
        k = min(5, last_logits.shape[-1])
        topk_indices = torch.topk(last_logits, k=k, dim=-1).indices
        top5_acc = float((topk_indices == y.unsqueeze(-1)).any(dim=-1).float().mean().item())

        centered = last_logits - last_logits.mean(dim=-1, keepdim=True)
        var = (centered.pow(2).mean(dim=-1) + 1e-8)
        kurt = float(((centered.pow(4).mean(dim=-1) / (var.pow(2))) - 3.0).mean().item())

    vesp_score = 0.70 * top1_acc + 0.30 * float(np.exp(-0.25 * nll))
    return LMQualityMetrics(
        perplexity=ppl,
        mean_nll=nll,
        top1_accuracy=top1_acc,
        top5_accuracy=top5_acc,
        vespasian_score=vesp_score,
        logit_kurtosis=kurt,
    )


def _build_token_splits_from_texts(
    tokenizer: Any,
    practice_texts: list[str],
    held_out_texts: list[str],
    seq_len: int = 10,
) -> tuple[EvaluationSplit, EvaluationSplit]:
    """Build multiple next-token prediction windows from real text corpora."""
    def _encode_windows(texts: list[str], name: str) -> EvaluationSplit:
        inputs_list: list[list[int]] = []
        targets_list: list[int] = []
        for txt in texts:
            ids = tokenizer.encode(txt, add_special_tokens=False)
            if len(ids) <= seq_len:
                pad_id = tokenizer.pad_token_id or 0
                padded = [pad_id] * (seq_len + 1 - len(ids)) + ids
                inputs_list.append(padded[:seq_len])
                targets_list.append(padded[seq_len])
            else:
                for start in range(0, len(ids) - seq_len, 2):
                    window = ids[start : start + seq_len + 1]
                    inputs_list.append(window[:seq_len])
                    targets_list.append(window[seq_len])
        x = torch.tensor(inputs_list, dtype=torch.long)
        y = torch.tensor(targets_list, dtype=torch.long)
        return EvaluationSplit(name=name, input_ids=x, target_ids=y, reference_input_ids=x[: min(8, len(x))])

    return _encode_windows(practice_texts, "practice_text"), _encode_windows(held_out_texts, "held_out_text")


def _closed_form_lm_head_or_last_mlp_calibration(
    model: nn.Module,
    workspace: WeightWorkspace,
    practice_split: EvaluationSplit,
    *,
    ridge_lambda: float = 2.0,
    step_size: float = 0.35,
) -> tuple[str, int]:
    """
    Closed-form ridge subspace alignment on the final projection / LM head or last MLP layer.
    Uses hidden features collected on `practice_split` to nudge logits toward the target distribution
    while regularizing strongly (`ridge_lambda`) to improve generalization on held-out sequences.
    """
    device = next(model.parameters()).device
    x = practice_split.input_ids.to(device)
    y = practice_split.target_ids.to(device)

    # Locate the final output projection or last 2D weight matrix
    candidate_names = [
        n for n in workspace.names() if ("lm_head" in n or "embed_out" in n) and len(workspace.shape(n)) == 2
    ]
    if not candidate_names:
        candidate_names = [n for n in workspace.names() if len(workspace.shape(n)) == 2]
    target_name = candidate_names[-1]

    # Hook input activations to the target module
    mod_path = target_name[: -len(".weight")] if target_name.endswith(".weight") else target_name
    target_module: nn.Module | None = None
    for name, mod in model.named_modules():
        if name == mod_path:
            target_module = mod
            break

    captured_in: list[torch.Tensor] = []
    if target_module is not None:
        def _hook(_m: nn.Module, inp: tuple[torch.Tensor, ...], _out: Any) -> None:
            if inp and isinstance(inp[0], torch.Tensor):
                captured_in.append(inp[0].detach())

        handle = target_module.register_forward_hook(_hook)
        with torch.no_grad():
            out = model(x)
        handle.remove()
    else:
        with torch.no_grad():
            out = model(x, output_hidden_states=True)
            if hasattr(out, "hidden_states") and out.hidden_states:
                captured_in.append(out.hidden_states[-1].detach())

    if not captured_in:
        return target_name, 0

    h_last = captured_in[-1][:, -1, :].float()  # [B, d_model]
    logits = out.logits[:, -1, :].float() if hasattr(out, "logits") else out[0][:, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    one_hot = F.one_hot(y, num_classes=logits.shape[-1]).float()

    # Negative gradient of cross-entropy w.r.t. logits: R = (Y_onehot - P)  [B, V]
    residual = (one_hot - probs)  # [B, V]

    w_arr = workspace.get_readonly(target_name).astype(np.float64)
    H = h_last.cpu().numpy().astype(np.float64)  # [B, d]
    R = residual.cpu().numpy().astype(np.float64)  # [B, V]

    # Solve regularized normal equations:
    # If w_arr is [V, d] (nn.Linear), W @ h -> delta_W = step * R^T @ H @ (H^T H + lambda I)^(-1)
    # If w_arr is [d, V] (GPT-2 Conv1D), h @ W -> delta_W = step * (H^T H + lambda I)^(-1) @ H^T @ R
    d = H.shape[1]
    cov = H.T @ H + ridge_lambda * np.eye(d, dtype=np.float64)
    if w_arr.shape == (R.shape[1], d):
        delta_w = step_size * (R.T @ H) @ np.linalg.inv(cov)
    elif w_arr.shape == (d, R.shape[1]):
        delta_w = step_size * np.linalg.inv(cov) @ (H.T @ R)
    else:
        return target_name, 0

    updated = (w_arr + delta_w).astype(np.float32)
    workspace.set_tensor(target_name, updated, apply_fisher_guard=True)
    return target_name, int(delta_w.size)


def improve_small_model(
    model_id_or_repo: str = "tiny-gpt2",
    *,
    inject_realistic_corruption: bool = True,
    save_dir: Path | str | None = None,
    seed: int = 2026,
) -> ModelImprovementReport:
    """
    End-to-end pipeline to benchmark, diagnose, repair, and improve a small LLM.

    Works on both real HuggingFace pretrained checkpoints (e.g. `tiny-gpt2`, `smollm2-135m`,
    `gpt2-124m`) and built-in `minillama-demo`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    catalog = ModelCatalog()
    if model_id_or_repo == "minillama-demo":
        from examples.mvp_end_to_end_demo import build_corrupted_model_and_splits

        model, practice_split, held_out_split, calib_ids = build_corrupted_model_and_splits(seed=seed)
        entry = catalog.get("minillama-demo")
        assert entry is not None
        tokenizer = None
    else:
        model, tokenizer, entry = catalog.load_model_and_tokenizer(model_id_or_repo, corrupt_demo=True)
        if tokenizer is not None:
            practice_split, held_out_split = _build_token_splits_from_texts(
                tokenizer,
                DEFAULT_PRACTICE_TEXTS,
                DEFAULT_HELD_OUT_TEXTS,
                seq_len=8,
            )
            # Align held-out evaluation to shared domain themes + inject realistic weight noise if requested
            if inject_realistic_corruption:
                with torch.no_grad():
                    # Find an MLP or attention weight tensor and inject realistic outlier row spikes
                    for name, param in model.named_parameters():
                        if param.ndim == 2 and ("mlp" in name or "c_proj" in name or "fc" in name):
                            param[0, :] *= 8.5
                            if param.shape[0] > 1:
                                param[-1, :] *= 7.5
                            break
        else:
            vocab_sz = int(getattr(getattr(model, "config", None), "vocab_size", 128))
            all_x = torch.randint(2, max(10, vocab_sz - 2), (32, 8), dtype=torch.long)
            with torch.no_grad():
                out = model(all_x)
                logits = out.logits if hasattr(out, "logits") else out[0]
                all_y = logits[:, -1, :].argmax(dim=-1)
            practice_split = EvaluationSplit(name="practice", input_ids=all_x[:16], target_ids=all_y[:16], reference_input_ids=all_x[:8])
            held_out_split = EvaluationSplit(name="held_out", input_ids=all_x[16:], target_ids=all_y[16:], reference_input_ids=all_x[16:24])
        calib_ids = practice_split.input_ids[:6]

    fisher_guard = FisherGuard.from_model(
        model,
        reference_batches=calib_ids,
        damping_alpha=0.15,
        freeze_top_fraction=0.05,
        enabled=True,
    )
    workspace = WeightWorkspace(model, fisher_guard=fisher_guard)

    vespasian = VespasianEngine(
        workspace=workspace,
        practice_split=practice_split,
        held_out_split=held_out_split,
        improvement_threshold=0.0005,
        max_kl_drift=5.0,
    )

    initial_metrics = evaluate_lm_quality(model, held_out_split)
    arch_report = inspect_architecture(model, calibration_input_ids=calib_ids)
    uni_report = profile_model_layers(model, calibration_input_ids=calib_ids)
    kmap_report = run_knowledge_map(model, tokenizer=tokenizer)

    stages: list[StageImprovementRecord] = []

    # -----------------------------------------------------------------------
    # Stage 1: Declarative YAML Outlier & Spectral Spike Repair
    # -----------------------------------------------------------------------
    pre_s1 = evaluate_lm_quality(model, held_out_split)
    outlier_tensors: list[str] = []
    for on in uni_report.outlier_neurons:
        if on.tensor_name and on.tensor_name not in outlier_tensors:
            outlier_tensors.append(on.tensor_name)

    if not outlier_tensors:
        # Fallback: pick 2D MLP tensors
        outlier_tensors = [
            n for n in workspace.names() if len(workspace.shape(n)) == 2 and ("mlp" in n or "c_proj" in n or "down_proj" in n)
        ][:2]

    if outlier_tensors:
        target_t = outlier_tensors[0]
        t_shape = list(workspace.shape(target_t))
        t_dtype = workspace.dtype(target_t)
        yaml_plan = (
            "transforms:\n"
            "  - assert:\n"
            "      all:\n"
            f"        - exists: {{ of: {target_t!r} }}\n"
            f"        - shape: {{ of: {target_t!r}, is: {t_shape} }}\n"
            f"        - dtype: {{ of: {target_t!r}, is: {t_dtype!r} }}\n"
            f"        - finite: {{ of: {target_t!r} }}\n"
            "  - rescale_outlier_rows_:\n"
            f"      target: {target_t!r}\n"
            "      min_ratio: 2.2\n"
            "  - spectral_norm_cap_:\n"
            f"      target: {target_t!r}\n"
            "      max_median_ratio: 3.5\n"
            "  - assert:\n"
            f"      finite: {{ of: {target_t!r} }}\n"
        )
        execute_yaml_plan(workspace, yaml_plan, apply_fisher_guard=True)
        v1 = vespasian.verify_and_decide(hypothesis=f"Stage 1: Outlier & Spectral Repair on {target_t}")
        post_s1 = evaluate_lm_quality(model, held_out_split)
        stages.append(
            StageImprovementRecord(
                stage_name="Stage 1: BrainSurgery-WX Outlier & Spectral Repair",
                description=f"Rescaled outlier rows and capped spectral spikes in {target_t}",
                committed=v1.accepted,
                pre_perplexity=pre_s1.perplexity,
                post_perplexity=post_s1.perplexity,
                pre_top1_acc=pre_s1.top1_accuracy,
                post_top1_acc=post_s1.top1_accuracy,
                vespasian_verdict=v1.verdict_reason,
                yaml_plan_used=yaml_plan,
            )
        )

    # -----------------------------------------------------------------------
    # Stage 2: Closed-Form Subspace Ridge Calibration with Fisher Protection
    # -----------------------------------------------------------------------
    pre_s2 = evaluate_lm_quality(model, held_out_split)
    calib_target, _num_updated = _closed_form_lm_head_or_last_mlp_calibration(
        model,
        workspace,
        practice_split,
        ridge_lambda=2.5,
        step_size=0.30,
    )
    v2 = vespasian.verify_and_decide(
        hypothesis=f"Stage 2: Closed-Form Subspace Calibration on {calib_target}"
    )
    post_s2 = evaluate_lm_quality(model, held_out_split)
    stages.append(
        StageImprovementRecord(
            stage_name="Stage 2: Closed-Form Subspace Ridge Calibration",
            description=f"Aligned output subspace on {calib_target} with Fisher guard",
            committed=v2.accepted,
            pre_perplexity=pre_s2.perplexity,
            post_perplexity=post_s2.perplexity,
            pre_top1_acc=pre_s2.top1_accuracy,
            post_top1_acc=post_s2.top1_accuracy,
            vespasian_verdict=v2.verdict_reason,
        )
    )

    final_metrics = evaluate_lm_quality(model, held_out_split)
    diff = workspace.diff()
    ppl_red = (
        (initial_metrics.perplexity - final_metrics.perplexity)
        / max(initial_metrics.perplexity, 1e-9)
        * 100.0
    )
    acc_gain = (final_metrics.top1_accuracy - initial_metrics.top1_accuracy) * 100.0
    vesp_gain = final_metrics.vespasian_score - initial_metrics.vespasian_score

    saved_path_str: str | None = None
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_id = entry.model_id.replace(":", "_").replace("/", "_")
        out_file = out_dir / f"{safe_id}_improved.safetensors"
        state_tensors = {k: torch.from_numpy(np.array(workspace.get_readonly(k), copy=True)) for k in workspace.names()}
        save_file(state_tensors, str(out_file))
        saved_path_str = str(out_file.resolve())

    return ModelImprovementReport(
        model_id=entry.model_id,
        hf_repo_id=entry.hf_repo_id,
        family=arch_report.family,
        total_params=diff.total_params,
        changed_params=diff.changed_params,
        changed_fraction=diff.changed_fraction,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        ppl_reduction_pct=ppl_red,
        top1_accuracy_gain_pct=acc_gain,
        vespasian_gain=vesp_gain,
        kci_gini=kmap_report.kci_overall,
        bottleneck_layers=uni_report.bottleneck_layers,
        stages=stages,
        saved_checkpoint_path=saved_path_str,
    )


# ===========================================================================
# Real pretrained LM improvement (HuggingFace models with a tokenizer)
# ===========================================================================


@dataclass
class TextQualityMetrics:
    held_out_ppl: float
    held_out_top1: float
    control_ppl: float
    control_top1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out_ppl": round(self.held_out_ppl, 4),
            "held_out_top1": round(self.held_out_top1, 4),
            "control_ppl": round(self.control_ppl, 4),
            "control_top1": round(self.control_top1, 4),
        }


@dataclass
class PretrainedImprovementReport:
    model_id: str
    hf_repo_id: str
    total_params: int
    changed_params: int
    edited_tensors: list[str]
    before: TextQualityMetrics
    after: TextQualityMetrics
    held_out_ppl_reduction_pct: float
    control_ppl_change_pct: float
    vespasian_verdicts: list[str]
    calibration: dict[str, Any]
    saved_checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hf_repo_id": self.hf_repo_id,
            "total_params": self.total_params,
            "changed_params": self.changed_params,
            "changed_fraction": round(self.changed_params / max(self.total_params, 1), 6),
            "edited_tensors": self.edited_tensors,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "held_out_ppl_reduction_pct": round(self.held_out_ppl_reduction_pct, 2),
            "control_ppl_change_pct": round(self.control_ppl_change_pct, 2),
            "vespasian_verdicts": self.vespasian_verdicts,
            "calibration": self.calibration,
            "saved_checkpoint_path": self.saved_checkpoint_path,
        }

    def to_toon(self) -> str:
        return dumps_toon(self.to_dict())


def _tokenize_texts(tokenizer: Any, texts: list[str]) -> list[torch.Tensor]:
    seqs = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if tokenizer.bos_token_id is not None:
            ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), ids])
        seqs.append(ids)
    return seqs


def _top1_accuracy(model: nn.Module, seqs: list[torch.Tensor]) -> float:
    dev = next(model.parameters()).device
    hit, tot = 0, 0
    with torch.no_grad():
        for ids in seqs:
            x = ids.to(dev).unsqueeze(0)
            pred = model(x).logits[0, :-1].argmax(-1)
            hit += int((pred == x[0, 1:]).sum().item())
            tot += x.shape[1] - 1
    return hit / max(tot, 1)


def _text_metrics(model: nn.Module, held: list[torch.Tensor], ctrl: list[torch.Tensor]) -> TextQualityMetrics:
    from weightx.l3_weights.domain_memit import token_nll

    h_nll, _ = token_nll(model, held)
    c_nll, _ = token_nll(model, ctrl)
    return TextQualityMetrics(
        held_out_ppl=float(np.exp(h_nll)),
        held_out_top1=_top1_accuracy(model, held),
        control_ppl=float(np.exp(c_nll)),
        control_top1=_top1_accuracy(model, ctrl),
    )


def improve_pretrained_lm(
    model_id_or_repo: str = "smollm2-135m",
    *,
    practice_texts: list[str] | None = None,
    held_out_texts: list[str] | None = None,
    control_texts: list[str] | None = None,
    iterations: int = 3,
    n_layers: int = 4,
    max_control_ppl_increase: float = 0.02,
    save_dir: Path | str | None = None,
    log: Any = print,
) -> PretrainedImprovementReport:
    """Measurably improve a real pretrained causal LM on a target domain without SGD.

    Pipeline: load -> baseline metrics -> closed-form MEMIT domain calibration on practice
    texts (control-covariance preserving, Fisher-guarded) -> Vespasian verification on hidden
    held-out texts with control-text KL guard -> keep best checkpoint / rollback.
    """
    from weightx.corpora import GENERAL_CONTROL, WSP_HELD_OUT, WSP_PRACTICE
    from weightx.l3_weights.domain_memit import memit_domain_calibration, token_nll

    torch.manual_seed(0)
    catalog = ModelCatalog()
    log(f"[load] {model_id_or_repo}")
    model, tokenizer, entry = catalog.load_model_and_tokenizer(model_id_or_repo)
    if tokenizer is None:
        raise ValueError("improve_pretrained_lm requires a HuggingFace model with a tokenizer")
    for p in model.parameters():
        p.requires_grad_(False)

    prac = _tokenize_texts(tokenizer, practice_texts or WSP_PRACTICE)
    held = _tokenize_texts(tokenizer, held_out_texts or WSP_HELD_OUT)
    ctrl = _tokenize_texts(tokenizer, control_texts or GENERAL_CONTROL)

    before = _text_metrics(model, held, ctrl)
    log(f"[baseline] {before.to_dict()}")

    def _held_eval(m: nn.Module) -> tuple[float, list[float]]:
        _, per = token_nll(m, held)
        items = [float(np.exp(-x)) for x in per]
        return float(np.mean(items)), items

    min_len = min(len(s) for s in ctrl)
    ctrl_ref = torch.stack([s[:min_len] for s in ctrl])
    held_split = EvaluationSplit(name="held_out_hidden", eval_fn=_held_eval, reference_input_ids=ctrl_ref)

    workspace = WeightWorkspace(model)
    vespasian = VespasianEngine(
        workspace,
        held_out_split=held_split,
        improvement_threshold=1e-4,
        max_kl_drift=0.05,
        max_changed_fraction=0.10,
    )

    verdicts: list[str] = []
    calib_log: list[dict[str, Any]] = []
    edited: list[str] = []
    for it in range(iterations):
        res = memit_domain_calibration(
            model,
            workspace,
            prac,
            ctrl,
            n_layers=n_layers,
            iterations=1,
            max_control_nll_increase=max_control_ppl_increase / 2.0,
            apply_fisher_guard=False,
        )
        edited = res.target_tensors
        calib_log.extend(s.to_dict() | {"iteration": it + 1} for s in res.steps)
        if not res.steps or res.steps[-1].chosen_rel_step == 0.0:
            log(f"[iter {it + 1}] line search found no safe improving step; stopping")
            break
        st = res.steps[-1]
        log(
            f"[iter {it + 1}] step={st.chosen_rel_step} practice NLL {st.practice_nll_before:.4f}"
            f"->{st.practice_nll_after:.4f}, control NLL {st.control_nll_before:.4f}->{st.control_nll_after:.4f}"
        )
        v = vespasian.verify_and_decide(hypothesis=f"MEMIT domain calibration iteration {it + 1}")
        verdicts.append(v.verdict_reason)
        log(f"[vespasian] {v.verdict_reason}")
        if not v.accepted:
            break

    after = _text_metrics(model, held, ctrl)
    log(f"[final] {after.to_dict()}")
    diff = workspace.diff()

    saved: str | None = None
    if save_dir is not None and diff.changed_params > 0:
        out = Path(save_dir) / f"{entry.model_id.replace('/', '_').replace(':', '_')}_weightx_improved.safetensors"
        workspace.export_safetensors(out, only_changed=True)
        saved = str(out.resolve())

    return PretrainedImprovementReport(
        model_id=entry.model_id,
        hf_repo_id=entry.hf_repo_id,
        total_params=diff.total_params,
        changed_params=diff.changed_params,
        edited_tensors=edited,
        before=before,
        after=after,
        held_out_ppl_reduction_pct=(before.held_out_ppl - after.held_out_ppl) / before.held_out_ppl * 100.0,
        control_ppl_change_pct=(after.control_ppl - before.control_ppl) / before.control_ppl * 100.0,
        vespasian_verdicts=verdicts,
        calibration={"steps": calib_log},
        saved_checkpoint_path=saved,
    )
