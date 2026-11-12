# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
Closed-Form MEMIT-Style Domain Calibration for Real Pretrained Causal LMs.

Extends WeightX MEMIT-MASS from synthetic key/value facts to *real text*:

1. Keys `K_l` are collected at the input of selected MLP output projections
   (`mlp.down_proj` for Llama/Qwen/SmolLM, `mlp.c_proj` for GPT-2, `dense_4h_to_h`, `fc2`).
2. Residual targets are obtained from the activation-space gradient of the
   next-token loss on the *visible* practice texts: `R_l = -dL/dO_l`
   (no gradient is ever taken w.r.t. weights; no SGD training loop).
3. A covariance-preserving closed-form update is solved per layer:

       dW_l = R_l^T K_l (K_l^T K_l + lambda * N * C0_l + eps*I)^-1

   where `C0_l` is the key covariance on *control* (general-domain) texts,
   which pushes the update orthogonal to the subspace used by general knowledge.
4. The update magnitude is chosen by a line search on the practice split under
   a control-perplexity budget. The caller then verifies on hidden held-out
   texts through the Vespasian engine and rolls back if it does not help.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from weightx.l3_weights.workspace import WeightWorkspace

_MLP_OUT_SUFFIXES = ("mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h", "mlp.fc2", "feed_forward.w2")


@dataclass
class DomainCalibrationStep:
    iteration: int
    layers: list[str]
    chosen_rel_step: float
    practice_nll_before: float
    practice_nll_after: float
    control_nll_before: float
    control_nll_after: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "layers": self.layers,
            "chosen_rel_step": self.chosen_rel_step,
            "practice_nll_before": round(self.practice_nll_before, 5),
            "practice_nll_after": round(self.practice_nll_after, 5),
            "control_nll_before": round(self.control_nll_before, 5),
            "control_nll_after": round(self.control_nll_after, 5),
        }


@dataclass
class DomainCalibrationResult:
    target_tensors: list[str]
    steps: list[DomainCalibrationStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"target_tensors": self.target_tensors, "steps": [s.to_dict() for s in self.steps]}


def find_mlp_output_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return `(module_name, module)` pairs for every MLP output projection, in depth order."""
    found: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if any(name.endswith(sfx) for sfx in _MLP_OUT_SUFFIXES) and hasattr(mod, "weight"):
            found.append((name, mod))
    return found


def select_edit_layers(model: nn.Module, n_layers: int = 4, lo: float = 0.45, hi: float = 0.9) -> list[str]:
    """Pick `n_layers` MLP output projections spread over the upper-middle depth band."""
    mods = find_mlp_output_modules(model)
    if not mods:
        return []
    total = len(mods)
    start = int(lo * (total - 1))
    end = max(start, int(hi * (total - 1)))
    idxs = np.unique(np.linspace(start, end, num=min(n_layers, end - start + 1)).round().astype(int))
    return [mods[i][0] for i in idxs]


def _is_conv1d(mod: nn.Module) -> bool:
    return type(mod).__name__ == "Conv1D"


def _input_embedding_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    for _n, m in model.named_modules():
        if isinstance(m, nn.Embedding):
            return m
    raise RuntimeError("Could not locate input embedding module")


def token_nll(model: nn.Module, token_seqs: list[torch.Tensor]) -> tuple[float, list[float]]:
    """Mean per-token NLL over all next-token positions, plus per-sequence NLLs."""
    model.eval()
    dev = next(model.parameters()).device
    total, count = 0.0, 0
    per_seq: list[float] = []
    with torch.no_grad():
        for ids in token_seqs:
            x = ids.to(dev).unsqueeze(0)
            logits = model(x).logits[0, :-1].float()
            loss = F.cross_entropy(logits, x[0, 1:], reduction="sum").item()
            n = x.shape[1] - 1
            total += loss
            count += n
            per_seq.append(loss / max(n, 1))
    return total / max(count, 1), per_seq


def _collect_keys_and_grads(
    model: nn.Module,
    module_names: list[str],
    token_seqs: list[torch.Tensor],
    *,
    need_grads: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    modules = dict(model.named_modules())
    dev = next(model.parameters()).device
    keys: dict[str, list[np.ndarray]] = {n: [] for n in module_names}
    grads: dict[str, list[np.ndarray]] = {n: [] for n in module_names}
    outs: dict[str, torch.Tensor] = {}
    handles = []

    def _mk(name: str):
        def _hook(_m: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            keys[name].append(inp[0][0, :-1].detach().float().cpu().numpy().astype(np.float64))
            if need_grads and out.requires_grad:
                out.retain_grad()
                outs[name] = out
        return _hook

    for n in module_names:
        handles.append(modules[n].register_forward_hook(_mk(n)))

    emb_handle = None
    if need_grads:
        def _emb_hook(_m: nn.Module, _inp: Any, out: torch.Tensor) -> torch.Tensor:
            return out.detach().requires_grad_(True)
        emb_handle = _input_embedding_module(model).register_forward_hook(_emb_hook)

    try:
        for ids in token_seqs:
            x = ids.to(dev).unsqueeze(0)
            if need_grads:
                with torch.enable_grad():
                    outs.clear()
                    logits = model(x).logits[0, :-1].float()
                    loss = F.cross_entropy(logits, x[0, 1:], reduction="sum")
                    loss.backward()
                for n in module_names:
                    g = outs[n].grad
                    grads[n].append(g[0, :-1].detach().float().cpu().numpy().astype(np.float64))
            else:
                with torch.no_grad():
                    model(x)
    finally:
        for h in handles:
            h.remove()
        if emb_handle is not None:
            emb_handle.remove()

    k_cat = {n: np.concatenate(v, axis=0) for n, v in keys.items()}
    g_cat = {n: np.concatenate(v, axis=0) for n, v in grads.items()} if need_grads else {}
    return k_cat, g_cat


def memit_domain_calibration(
    model: nn.Module,
    workspace: WeightWorkspace,
    practice_seqs: list[torch.Tensor],
    control_seqs: list[torch.Tensor],
    *,
    layer_names: list[str] | None = None,
    n_layers: int = 4,
    lambda_preserve: float = 1.0,
    rel_steps: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075),
    max_control_nll_increase: float = 0.01,
    iterations: int = 1,
    apply_fisher_guard: bool = True,
) -> DomainCalibrationResult:
    """Run one or more closed-form domain calibration iterations (see module docstring)."""
    mod_names = layer_names or select_edit_layers(model, n_layers=n_layers)
    modules = dict(model.named_modules())
    tensor_names = [f"{n}.weight" for n in mod_names]
    result = DomainCalibrationResult(target_tensors=tensor_names)
    if not mod_names:
        return result

    ctrl_keys, _ = _collect_keys_and_grads(model, mod_names, control_seqs, need_grads=False)

    for it in range(iterations):
        prac_nll0, _ = token_nll(model, practice_seqs)
        ctrl_nll0, _ = token_nll(model, control_seqs)
        k_prac, g_prac = _collect_keys_and_grads(model, mod_names, practice_seqs, need_grads=True)

        directions: dict[str, np.ndarray] = {}
        base_weights: dict[str, np.ndarray] = {}
        for mn, tn in zip(mod_names, tensor_names):
            K = k_prac[mn]                      # [N, d_in]
            R = -g_prac[mn]                     # [N, d_out]
            Kc = ctrl_keys[mn]                  # [Nc, d_in]
            n_p = K.shape[0]
            d_in = K.shape[1]
            C0 = (Kc.T @ Kc) / max(Kc.shape[0], 1)
            A = K.T @ K + lambda_preserve * n_p * C0
            A += (1e-3 * np.trace(A) / d_in + 1e-8) * np.eye(d_in)
            D_lin = np.linalg.solve(A, (R.T @ K).T).T   # [d_out, d_in] == R^T K A^-1 (A symmetric)
            W0 = workspace.get_readonly(tn).astype(np.float64)
            D = D_lin.T if _is_conv1d(modules[mn]) else D_lin
            if D.shape != W0.shape:
                continue
            dn = float(np.linalg.norm(D))
            if dn < 1e-12:
                continue
            directions[tn] = D * (float(np.linalg.norm(W0)) / dn)
            base_weights[tn] = W0

        if not directions:
            break

        workspace.checkpoint("__domain_calib_pre__")
        best = (0.0, prac_nll0, ctrl_nll0)
        for step in rel_steps:
            for tn, D in directions.items():
                workspace.set_tensor(
                    tn, (base_weights[tn] + step * D).astype(np.float32), apply_fisher_guard=apply_fisher_guard
                )
            p_nll, _ = token_nll(model, practice_seqs)
            c_nll, _ = token_nll(model, control_seqs)
            if c_nll <= ctrl_nll0 * (1.0 + max_control_nll_increase) and p_nll < best[1]:
                best = (step, p_nll, c_nll)
            workspace.restore("__domain_calib_pre__")

        if best[0] > 0.0:
            for tn, D in directions.items():
                workspace.set_tensor(
                    tn, (base_weights[tn] + best[0] * D).astype(np.float32), apply_fisher_guard=apply_fisher_guard
                )
        result.steps.append(
            DomainCalibrationStep(
                iteration=it + 1,
                layers=tensor_names,
                chosen_rel_step=best[0],
                practice_nll_before=prac_nll0,
                practice_nll_after=best[1],
                control_nll_before=ctrl_nll0,
                control_nll_after=best[2],
            )
        )
        if best[0] == 0.0:
            break
    return result
