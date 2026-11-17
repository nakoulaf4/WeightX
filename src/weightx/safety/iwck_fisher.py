# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
IWCKEdit / FisherEdit Weight Importance Guard.

Identifies parameters critical to general model capabilities via the empirical
diagonal Fisher Information Matrix:
    F_{ij} = E_{(x,y) ~ D_ref} [ (d log p(y|x) / d W_{ij})^2 ]
and protects them during declarative, batch, or sandbox weight edits using:
  1. Soft Fisher-weighted damping:
       Delta W_safe = Delta W / (1 + alpha * F_norm)
  2. Hard Top-p critical weight freeze (with automatic exception for anomalous
     outlier neurons whose norm exceeds `outlier_bypass_sigma` x median).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FisherProtectionReport:
    """Telemetry from applying IWCKEdit Fisher protection to a proposed weight update."""

    tensor_name: str
    raw_l2_delta: float
    protected_l2_delta: float
    damping_ratio: float
    frozen_critical_elements: int
    outlier_bypass_elements: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "raw_l2_delta": round(self.raw_l2_delta, 6),
            "protected_l2_delta": round(self.protected_l2_delta, 6),
            "damping_ratio": round(self.damping_ratio, 4),
            "frozen_critical_elements": self.frozen_critical_elements,
            "outlier_bypass_elements": self.outlier_bypass_elements,
        }


def compute_empirical_fisher(
    model: nn.Module,
    reference_batches: list[torch.Tensor] | torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Compute diagonal empirical Fisher information for all parameters of `model`.

    If `reference_batches` is provided, accumulates squared gradients of next-token
    cross-entropy loss over the reference batches. Otherwise computes a fast
    spectral-magnitude Fisher proxy `F_{ij} = W_{ij}^2 * ||W_{i,:}||_2`.
    """
    model.eval()
    fisher: dict[str, np.ndarray] = {}

    if reference_batches is not None:
        if isinstance(reference_batches, torch.Tensor):
            batches = [reference_batches]
        else:
            batches = list(reference_batches)

        device = next(model.parameters()).device
        accum: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, dtype=torch.float32)
            for name, param in model.named_parameters()
        }
        n_batches = 0
        # Temporarily enable grads in case the caller froze model parameters for WeightWorkspace inference.
        orig_req_grad = [(p, p.requires_grad) for p in model.parameters()]
        for p, _ in orig_req_grad:
            p.requires_grad_(True)

        try:
            for batch in batches:
                ids = batch.to(device)
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    out = model(ids)
                    logits = (
                        out.logits
                        if hasattr(out, "logits")
                        else (out[0] if isinstance(out, tuple) else out)
                    )
                    if logits.ndim == 3 and logits.shape[1] > 1:
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = ids[:, 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                    else:
                        loss = logits.float().pow(2).mean()
                    loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        g = param.grad.detach().float()
                        accum[name].add_(g * g)
                n_batches += 1

            model.zero_grad(set_to_none=True)
        finally:
            for p, req in orig_req_grad:
                p.requires_grad_(req)

        denom = float(max(1, n_batches))
        for name, tensor_f in accum.items():
            fisher[name] = (tensor_f / denom).cpu().numpy()
        return fisher

    # Weights-only Fisher proxy
    for name, param in model.named_parameters():
        w = param.detach().float().cpu().numpy()
        fisher[name] = _proxy_fisher_for_array(w)
    return fisher


def _proxy_fisher_for_array(w: np.ndarray) -> np.ndarray:
    arr = np.asarray(w, dtype=np.float32)
    sq = arr * arr
    if arr.ndim >= 2:
        m2d = arr.reshape(arr.shape[0], -1)
        row_norms = np.linalg.norm(m2d, axis=1, keepdims=True)
        med = float(np.median(row_norms)) + 1e-8
        # Normal-range rows with moderate-high singular alignment get higher Fisher weight;
        # extreme anomalous outlier spikes (>3x median) get low Fisher protection so they can be repaired!
        ratio = row_norms / med
        row_factor = np.where(ratio > 3.0, 0.05, np.clip(ratio, 0.1, 2.0))
        scaled = sq.reshape(arr.shape[0], -1) * row_factor
        return scaled.reshape(arr.shape).astype(np.float32)
    return sq


class FisherGuard:
    """IWCKEdit Fisher-guided protection filter for weight updates."""

    def __init__(
        self,
        fisher_map: dict[str, np.ndarray] | None = None,
        *,
        damping_alpha: float = 0.25,
        freeze_top_fraction: float = 0.05,
        outlier_bypass_ratio: float = 2.5,
        enabled: bool = True,
    ):
        self.fisher_map: dict[str, np.ndarray] = fisher_map or {}
        self.damping_alpha = float(damping_alpha)
        self.freeze_top_fraction = float(np.clip(freeze_top_fraction, 0.0, 0.9))
        self.outlier_bypass_ratio = float(outlier_bypass_ratio)
        self.enabled = enabled
        self.history: list[FisherProtectionReport] = []

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        reference_batches: list[torch.Tensor] | torch.Tensor | None = None,
        *,
        damping_alpha: float = 0.25,
        freeze_top_fraction: float = 0.05,
        outlier_bypass_ratio: float = 2.5,
        enabled: bool = True,
    ) -> FisherGuard:
        fmap = compute_empirical_fisher(model, reference_batches=reference_batches)
        return cls(
            fmap,
            damping_alpha=damping_alpha,
            freeze_top_fraction=freeze_top_fraction,
            outlier_bypass_ratio=outlier_bypass_ratio,
            enabled=enabled,
        )

    def get_fisher(self, name: str, original_weight: np.ndarray) -> np.ndarray:
        if name not in self.fisher_map or self.fisher_map[name].shape != original_weight.shape:
            self.fisher_map[name] = _proxy_fisher_for_array(original_weight)
        return self.fisher_map[name]

    def protect_update(
        self,
        name: str,
        original_weight: np.ndarray,
        proposed_weight: np.ndarray,
        *,
        allow_outlier_repair: bool = True,
    ) -> tuple[np.ndarray, FisherProtectionReport]:
        """Filter `proposed_weight` against the Fisher importance map of `original_weight`."""
        orig = np.asarray(original_weight, dtype=np.float32)
        prop = np.asarray(proposed_weight, dtype=np.float32)
        delta = prop - orig
        raw_l2 = float(np.linalg.norm(delta.astype(np.float64)))

        if not self.enabled or raw_l2 <= 1e-12:
            rep = FisherProtectionReport(
                tensor_name=name,
                raw_l2_delta=raw_l2,
                protected_l2_delta=raw_l2,
                damping_ratio=1.0,
                frozen_critical_elements=0,
                outlier_bypass_elements=0,
            )
            return prop.astype(original_weight.dtype, copy=False), rep

        f_mat = self.get_fisher(name, orig)
        f_mean = float(np.mean(f_mat))
        if f_mean <= 1e-20:
            f_norm = np.zeros_like(f_mat)
        else:
            f_norm = np.clip(f_mat / f_mean, 0.0, 20.0)

        # Identify anomalous outlier rows/elements in original_weight so repairs aren't blocked
        outlier_mask = np.zeros(orig.shape, dtype=bool)
        if allow_outlier_repair:
            if orig.ndim >= 2:
                m2d = orig.reshape(orig.shape[0], -1)
                row_norms = np.linalg.norm(m2d.astype(np.float64), axis=1)
                med = float(np.median(row_norms))
                if med > 1e-12:
                    bad_rows = row_norms >= (self.outlier_bypass_ratio * med)
                    outlier_mask = np.broadcast_to(
                        bad_rows[:, None], m2d.shape
                    ).reshape(orig.shape).copy()
            # Also bypass non-finite or extreme element spikes
            elem_med = float(np.median(np.abs(orig))) + 1e-8
            outlier_mask |= (~np.isfinite(orig)) | (np.abs(orig) >= self.outlier_bypass_ratio * 3.0 * elem_med)

        # 1. Soft IWCKEdit damping: Delta W / (1 + alpha * F_norm)
        damping = 1.0 / (1.0 + self.damping_alpha * f_norm)
        damping = np.where(outlier_mask, 1.0, damping)

        # 2. Hard Top-p critical weight freeze (excluding outlier repair positions)
        frozen_count = 0
        if self.freeze_top_fraction > 0.0 and f_norm.size >= 10:
            thresh = float(np.quantile(f_norm, 1.0 - self.freeze_top_fraction))
            if thresh > 0.0:
                freeze_mask = (f_norm >= thresh) & (~outlier_mask) & (np.abs(delta) > 0)
                frozen_count = int(np.count_nonzero(freeze_mask))
                damping = np.where(freeze_mask, 0.0, damping)

        safe_delta = delta * damping
        safe_weight = orig + safe_delta
        prot_l2 = float(np.linalg.norm(safe_delta.astype(np.float64)))
        changed_outlier = int(np.count_nonzero(outlier_mask & (np.abs(delta) > 1e-12)))

        rep = FisherProtectionReport(
            tensor_name=name,
            raw_l2_delta=raw_l2,
            protected_l2_delta=prot_l2,
            damping_ratio=prot_l2 / max(raw_l2, 1e-12),
            frozen_critical_elements=frozen_count,
            outlier_bypass_elements=changed_outlier,
        )
        self.history.append(rep)
        return safe_weight.astype(original_weight.dtype, copy=False), rep
