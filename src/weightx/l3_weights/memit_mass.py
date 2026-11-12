# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
L3 Batch Associative Memory Editing (MEMIT-MASS) & Closed-Form Weight Solvers.

Implements gradient-free batch knowledge editing across multiple layers simultaneously:
  1. `apply_closed_form_ridge`: Minimum-norm regularized update on a single matrix W
     so that W_new @ X ~= Y:
         Delta W = (Y - W X) X^T (X X^T + lambda I + gamma diag(F_col))^{-1}
  2. `apply_memit_mass_batch`: Multi-layer distributed associative memory update
     (Meng et al. MEMIT + IWCKEdit Fisher regularization) capable of editing up to
     thousands of key-value facts in one simultaneous batch across target layers
     without the O(M^2) interference and catastrophic forgetting of sequential edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from weightx.l3_weights.workspace import WeightWorkspace
from weightx.toon_io import dumps_toon


@dataclass
class BatchFactSpec:
    """A single key-value associative memory edit specification for MEMIT-MASS."""

    fact_id: str
    key_vector: np.ndarray | list[float]
    target_value_vector: np.ndarray | list[float]
    layer_inputs: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class MemitMassResult:
    """Telemetry returned after executing a MEMIT-MASS multi-layer batch edit."""

    num_facts: int
    target_tensors: list[str]
    pre_edit_mse: float
    post_edit_mse: float
    relative_error_reduction: float
    per_tensor_l2_deltas: dict[str, float]
    fisher_protected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_facts": self.num_facts,
            "target_tensors": self.target_tensors,
            "pre_edit_mse": round(self.pre_edit_mse, 6),
            "post_edit_mse": round(self.post_edit_mse, 6),
            "relative_error_reduction_pct": round(self.relative_error_reduction * 100.0, 2),
            "per_tensor_l2_deltas": {k: round(v, 6) for k, v in self.per_tensor_l2_deltas.items()},
            "fisher_protected": self.fisher_protected,
        }

    def to_toon(self) -> str:
        return dumps_toon(self.to_dict())


def apply_closed_form_ridge(
    workspace: WeightWorkspace,
    tensor_name: str,
    keys_matrix: np.ndarray,
    values_matrix: np.ndarray,
    *,
    lambda_reg: float = 1e-2,
    apply_fisher_guard: bool = True,
) -> dict[str, Any]:
    """Solve for minimum-norm closed-form weight update `Delta W` so `(W + Delta W) @ K ~= V`.

    Args:
        workspace: Active `WeightWorkspace`.
        tensor_name: Target 2D weight tensor `(d_out, d_in)`.
        keys_matrix: Input key vectors shaped `(d_in, M)` or `(M, d_in)`.
        values_matrix: Desired output vectors shaped `(d_out, M)` or `(M, d_out)`.
        lambda_reg: Ridge regularization coefficient.
        apply_fisher_guard: Whether to pass the update through `FisherGuard`.
    """
    W = workspace.get_readonly(tensor_name).astype(np.float64)
    if W.ndim != 2:
        raise ValueError(f"closed_form_ridge requires a 2D tensor, got shape {W.shape}")
    d_out, d_in = W.shape

    K = np.asarray(keys_matrix, dtype=np.float64)
    V = np.asarray(values_matrix, dtype=np.float64)
    if K.ndim == 1:
        K = K[:, None]
    if V.ndim == 1:
        V = V[:, None]
    if K.shape[0] != d_in and K.shape[1] == d_in:
        K = K.T
    if V.shape[0] != d_out and V.shape[1] == d_out:
        V = V.T

    if K.shape[0] != d_in or V.shape[0] != d_out or K.shape[1] != V.shape[1]:
        raise ValueError(
            f"Dimension mismatch for {tensor_name} {W.shape}: K={K.shape}, V={V.shape}"
        )

    residual = V - (W @ K)
    pre_mse = float(np.mean(residual**2))

    # Incorporate column-level Fisher regularization into normal equations if FisherGuard is present
    fisher_diag = np.zeros(d_in, dtype=np.float64)
    if apply_fisher_guard and workspace.fisher_guard is not None:
        f_mat = workspace.fisher_guard.get_fisher(tensor_name, W.astype(np.float32))
        col_f = np.mean(f_mat.astype(np.float64), axis=0)
        if np.mean(col_f) > 1e-20:
            fisher_diag = 0.05 * lambda_reg * (col_f / np.mean(col_f))

    gram = (K @ K.T) + (lambda_reg * np.eye(d_in, dtype=np.float64)) + np.diag(fisher_diag)
    delta_W = (residual @ K.T) @ np.linalg.inv(gram)
    W_new = (W + delta_W).astype(np.float32)

    f_rep = workspace.set_tensor(
        tensor_name,
        W_new,
        apply_fisher_guard=apply_fisher_guard,
        allow_outlier_repair=False,
    )
    W_final = workspace.get_readonly(tensor_name).astype(np.float64)
    post_mse = float(np.mean((V - (W_final @ K)) ** 2))

    return {
        "tensor": tensor_name,
        "num_keys": int(K.shape[1]),
        "pre_mse": round(pre_mse, 6),
        "post_mse": round(post_mse, 6),
        "l2_delta": round(float(np.linalg.norm(W_final - W)), 6),
        "fisher_report": f_rep.to_dict() if f_rep is not None else None,
    }


def apply_memit_mass_batch(
    workspace: WeightWorkspace,
    target_tensors: list[str],
    facts: list[BatchFactSpec],
    *,
    lambda_cov: float = 1e-2,
    apply_fisher_guard: bool = True,
) -> MemitMassResult:
    """Execute MEMIT-MASS simultaneous batch editing across multiple layer matrices.

    Distributes the associative memory residual across `target_tensors` (`l_1, ..., l_L`)
    so that each layer absorbs `1 / (L - idx)` of the remaining residual for all `M`
    facts simultaneously:
        Delta W_l = R_l K_l^T (K_l K_l^T + lambda_cov * I + diag(F_l))^{-1}
    """
    if not target_tensors:
        raise ValueError("target_tensors must contain at least one tensor name")
    if not facts:
        raise ValueError("facts list must not be empty")

    num_layers = len(target_tensors)
    num_facts = len(facts)

    # Verify shapes of all target tensors
    shapes = [workspace.shape(t) for t in target_tensors]
    for t, shp in zip(target_tensors, shapes):
        if len(shp) != 2:
            raise ValueError(f"MEMIT-MASS target {t!r} must be 2D, got {shp}")

    d_out, d_in = shapes[-1]

    # Stack keys (d_in, M) and target values (d_out, M)
    K_base = np.column_stack(
        [np.asarray(f.key_vector, dtype=np.float64).reshape(d_in) for f in facts]
    )
    V_target = np.column_stack(
        [np.asarray(f.target_value_vector, dtype=np.float64).reshape(d_out) for f in facts]
    )

    # Compute initial cumulative output across the target layers
    def _cumulative_output() -> np.ndarray:
        total = np.zeros((d_out, num_facts), dtype=np.float64)
        for tname in target_tensors:
            W_l = workspace.get_readonly(tname).astype(np.float64)
            if W_l.shape == (d_out, d_in):
                total += W_l @ K_base
        return total

    init_out = _cumulative_output()
    remaining_residual = V_target - init_out
    pre_mse = float(np.mean(remaining_residual**2))

    per_tensor_deltas: dict[str, float] = {}

    for idx, tname in enumerate(target_tensors):
        W_l = workspace.get_readonly(tname).astype(np.float64)
        l_out, l_in = W_l.shape
        if l_out != d_out or l_in != d_in:
            continue

        # Per-layer key matrix K_l (allows layer-specific keys if provided in fact.layer_inputs)
        K_l = np.column_stack(
            [
                np.asarray(f.layer_inputs.get(tname, f.key_vector), dtype=np.float64).reshape(l_in)
                for f in facts
            ]
        )

        # Spread residual evenly across remaining target layers (MEMIT formula)
        layers_left = float(num_layers - idx)
        layer_target_residual = remaining_residual / layers_left

        # Fisher column regularizer
        fisher_diag = np.zeros(l_in, dtype=np.float64)
        if apply_fisher_guard and workspace.fisher_guard is not None:
            f_mat = workspace.fisher_guard.get_fisher(tname, W_l.astype(np.float32))
            col_f = np.mean(f_mat.astype(np.float64), axis=0)
            if np.mean(col_f) > 1e-20:
                fisher_diag = 0.05 * lambda_cov * (col_f / np.mean(col_f))

        gram = (K_l @ K_l.T) + (lambda_cov * np.eye(l_in, dtype=np.float64)) + np.diag(fisher_diag)
        delta_W_l = (layer_target_residual @ K_l.T) @ np.linalg.inv(gram)

        W_proposed = (W_l + delta_W_l).astype(np.float32)
        workspace.set_tensor(
            tname,
            W_proposed,
            apply_fisher_guard=apply_fisher_guard,
            allow_outlier_repair=False,
        )
        W_actual = workspace.get_readonly(tname).astype(np.float64)
        actual_delta = W_actual - W_l
        per_tensor_deltas[tname] = float(np.linalg.norm(actual_delta))

        # Update remaining residual after this layer's actual contribution
        remaining_residual = remaining_residual - (actual_delta @ K_l)

    final_out = _cumulative_output()
    post_mse = float(np.mean((V_target - final_out) ** 2))
    rel_red = (pre_mse - post_mse) / max(pre_mse, 1e-12)

    return MemitMassResult(
        num_facts=num_facts,
        target_tensors=list(target_tensors),
        pre_edit_mse=pre_mse,
        post_edit_mse=post_mse,
        relative_error_reduction=rel_red,
        per_tensor_l2_deltas=per_tensor_deltas,
        fisher_protected=bool(apply_fisher_guard and workspace.fisher_guard is not None),
    )
