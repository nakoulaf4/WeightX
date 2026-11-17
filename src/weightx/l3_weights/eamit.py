# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
L3 Batch Associative Memory Editing via EAMIT (EAMET — Embedding Alignment Model
Editing in Transformers, arXiv:2505.11876) & Closed-Form Weight Solvers.

Implements embedding-aligned closed-form batch knowledge editing across layers:
  1. `align_residuals_to_keys`: EAMET embedding alignment optimization aligning
     target residual embeddings `R` to key embeddings `K` via:
       - Key-key & residual-residual cosine similarity distributions `P_k`, `P_r`
       - Global KL alignment loss: `L_KL = KL(P_r || P_k)`
       - Fine-grained top-M similarity MSE loss: `L_MSE = (1/M) sum ||S_r - S_k||^2`
       - Scale-invariant residual fidelity preservation: `||R - R_raw||_F^2`
  2. `apply_closed_form_ridge`: Minimum-norm regularized update on a single matrix W
     so that W_new @ K ~= V:
         Delta W = (Y - W X) X^T (X X^T + lambda I + gamma diag(F_col))^{-1}
  3. `apply_eamit_batch`: Multi-layer EAMIT update combining embedding alignment
     optimization with distributed covariance-preserving closed-form updates and
     IWCKEdit Fisher regularization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from weightx.l3_weights.workspace import WeightWorkspace
from weightx.toon_io import dumps_toon


@dataclass(frozen=True)
class EamitAlignConfig:
    """Hyperparameters for EAMIT key-residual embedding alignment optimization."""

    lambda_kl: float = 0.05
    lambda_mse: float = 0.25
    top_m: int = 8
    tau: float = 0.1
    align_steps: int = 20
    align_lr: float = 0.03
    max_block_size: int = 512


@dataclass(frozen=True)
class EamitAlignStats:
    """Alignment telemetry before and after EAMIT embedding alignment."""

    pre_kl: float
    post_kl: float
    pre_top_m_mse: float
    post_top_m_mse: float
    steps_executed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_kl": round(self.pre_kl, 6),
            "post_kl": round(self.post_kl, 6),
            "pre_top_m_mse": round(self.pre_top_m_mse, 6),
            "post_top_m_mse": round(self.post_top_m_mse, 6),
            "steps_executed": self.steps_executed,
        }


@dataclass
class BatchFactSpec:
    """A single key-value associative memory edit specification for EAMIT."""

    fact_id: str
    key_vector: np.ndarray | list[float]
    target_value_vector: np.ndarray | list[float]
    layer_inputs: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EamitResult:
    """Telemetry returned after executing an EAMIT multi-layer batch edit."""

    num_facts: int
    target_tensors: list[str]
    pre_edit_mse: float
    post_edit_mse: float
    relative_error_reduction: float
    pre_alignment_kl: float
    post_alignment_kl: float
    pre_alignment_top_m_mse: float
    post_alignment_top_m_mse: float
    per_tensor_l2_deltas: dict[str, float]
    fisher_protected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_facts": self.num_facts,
            "target_tensors": self.target_tensors,
            "pre_edit_mse": round(self.pre_edit_mse, 6),
            "post_edit_mse": round(self.post_edit_mse, 6),
            "relative_error_reduction_pct": round(self.relative_error_reduction * 100.0, 2),
            "pre_alignment_kl": round(self.pre_alignment_kl, 6),
            "post_alignment_kl": round(self.post_alignment_kl, 6),
            "pre_alignment_top_m_mse": round(self.pre_alignment_top_m_mse, 6),
            "post_alignment_top_m_mse": round(self.post_alignment_top_m_mse, 6),
            "per_tensor_l2_deltas": {k: round(v, 6) for k, v in self.per_tensor_l2_deltas.items()},
            "fisher_protected": self.fisher_protected,
        }

    def to_toon(self) -> str:
        return dumps_toon(self.to_dict())


def _offdiag_softmax(sim: np.ndarray, tau: float) -> np.ndarray:
    """Row-wise softmax over off-diagonal similarity entries (diagonal forced to 0)."""
    m = sim.shape[0]
    scaled = sim / max(tau, 1e-6)
    np.fill_diagonal(scaled, -np.inf)
    mx = np.max(scaled, axis=1, keepdims=True)
    ex = np.exp(scaled - mx)
    np.fill_diagonal(ex, 0.0)
    denom = np.sum(ex, axis=1, keepdims=True)
    fallback = np.full((m, m), 1.0 / max(m - 1, 1), dtype=np.float64)
    np.fill_diagonal(fallback, 0.0)
    return np.where(denom > 0.0, ex / np.maximum(denom, 1e-30), fallback)


def _align_block(
    k_blk: np.ndarray,
    r_blk: np.ndarray,
    cfg: EamitAlignConfig,
) -> tuple[np.ndarray, EamitAlignStats]:
    """Align residual columns `r_blk (d_out, n)` to key columns `k_blk (d_in, n)`."""
    n = k_blk.shape[1]
    d_out = r_blk.shape[0]
    if n < 2 or d_out == 0:
        return r_blk.copy(), EamitAlignStats(0.0, 0.0, 0.0, 0.0, 0)

    k_norms = np.linalg.norm(k_blk, axis=0)
    r_norms = np.linalg.norm(r_blk, axis=0)
    active = np.where((k_norms > 1e-12) & (r_norms > 1e-12))[0]
    m = int(active.size)
    if m < 2:
        return r_blk.copy(), EamitAlignStats(0.0, 0.0, 0.0, 0.0, 0)

    k_act = k_blk[:, active]
    r_act = r_blk[:, active]
    k_hat = k_act / np.maximum(k_norms[active][None, :], 1e-12)
    s_k = np.clip(k_hat.T @ k_hat, -1.0, 1.0)
    p_k = _offdiag_softmax(s_k, cfg.tau)
    log_p_k = np.log(p_k + 1e-12)

    top_m = max(1, min(int(cfg.top_m), m - 1))
    top_mask = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        # Deterministic descending similarity, ascending index tie-break
        others = [j for j in range(m) if j != i]
        others.sort(key=lambda j: (-float(s_k[i, j]), j))
        for j in others[:top_m]:
            top_mask[i, j] = 1.0

    scales = np.maximum(r_norms[active] / np.sqrt(float(d_out)), 1e-12)
    r_tilde_raw = r_act / scales[None, :]
    r_tilde = r_tilde_raw.copy()
    tau = max(float(cfg.tau), 1e-6)
    eps_norm = 1e-12

    def _eval(rt: np.ndarray) -> tuple[float, np.ndarray, float, float]:
        r_sq = np.sum(rt * rt, axis=0) + eps_norm
        r_n = np.sqrt(r_sq)
        r_hat = rt / r_n[None, :]
        s_r = r_hat.T @ r_hat
        p_r = _offdiag_softmax(s_r, tau)

        # KL(P_r || P_k)
        log_ratio = np.where(p_r > 0.0, np.log(p_r + 1e-12) - log_p_k, 0.0)
        np.fill_diagonal(log_ratio, 0.0)
        kl_rows_raw = np.sum(p_r * log_ratio, axis=1)
        kl_mean = float(np.mean(np.maximum(kl_rows_raw, 0.0)))

        # Top-M MSE
        diff_top = (s_r - s_k) * top_mask
        top_m_mse = float(np.sum(diff_top * diff_top) / float(m * top_m))

        # Fidelity
        diff_fid = rt - r_tilde_raw
        loss_fid = float(np.sum(diff_fid * diff_fid) / float(m * d_out))
        total_loss = loss_fid + cfg.lambda_kl * kl_mean + cfg.lambda_mse * top_m_mse

        # Analytical gradient w.r.t. S_r (off-diagonal)
        g_kl = (cfg.lambda_kl / (float(m) * tau)) * p_r * (log_ratio - kl_rows_raw[:, None])
        g_mse = (2.0 * cfg.lambda_mse / float(m * top_m)) * diff_top
        g_mat = g_kl + g_mse
        np.fill_diagonal(g_mat, 0.0)
        g_sym = g_mat + g_mat.T

        u = r_hat @ g_sym
        dot_col = np.sum(r_hat * u, axis=0)
        grad_align = (u - r_hat * dot_col[None, :]) / r_n[None, :]
        grad_fid = (2.0 / float(m * d_out)) * diff_fid
        return total_loss, grad_fid + grad_align, kl_mean, top_m_mse

    loss0, grad, pre_kl, pre_mse = _eval(r_tilde)
    if cfg.align_steps <= 0 or (cfg.lambda_kl <= 0.0 and cfg.lambda_mse <= 0.0):
        return r_blk.copy(), EamitAlignStats(pre_kl, pre_kl, pre_mse, pre_mse, 0)

    best_loss = loss0
    best_rt = r_tilde.copy()
    best_kl = pre_kl
    best_mse = pre_mse

    m_adam = np.zeros_like(r_tilde)
    v_adam = np.zeros_like(r_tilde)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    steps_done = 0

    for step in range(1, int(cfg.align_steps) + 1):
        steps_done = step
        m_adam = beta1 * m_adam + (1.0 - beta1) * grad
        v_adam = beta2 * v_adam + (1.0 - beta2) * (grad * grad)
        m_hat = m_adam / (1.0 - beta1**step)
        v_hat = v_adam / (1.0 - beta2**step)
        r_tilde = r_tilde - cfg.align_lr * m_hat / (np.sqrt(v_hat) + adam_eps)

        loss_t, grad, kl_t, mse_t = _eval(r_tilde)
        if loss_t < best_loss:
            best_loss = loss_t
            best_rt = r_tilde.copy()
            best_kl = kl_t
            best_mse = mse_t

    r_out = r_blk.copy()
    r_out[:, active] = best_rt * scales[None, :]
    return r_out, EamitAlignStats(
        pre_kl=pre_kl,
        post_kl=best_kl,
        pre_top_m_mse=pre_mse,
        post_top_m_mse=best_mse,
        steps_executed=steps_done,
    )


def align_residuals_to_keys(
    keys_matrix: np.ndarray,
    residuals_matrix: np.ndarray,
    config: EamitAlignConfig | None = None,
) -> tuple[np.ndarray, EamitAlignStats]:
    """Run EAMIT key-residual embedding alignment optimization (`K: [d_in, N]`, `R: [d_out, N]`).

    Aligns pairwise cosine similarity structure of residual vectors `R` with key
    vectors `K` using KL divergence on off-diagonal similarity distributions plus
    top-M cosine-similarity MSE, while preserving residual fidelity.
    """
    cfg = config or EamitAlignConfig()
    k = np.asarray(keys_matrix, dtype=np.float64)
    r = np.asarray(residuals_matrix, dtype=np.float64)
    if k.ndim != 2 or r.ndim != 2 or k.shape[1] != r.shape[1]:
        raise ValueError(f"Expected 2D matrices with matching columns, got K={k.shape}, R={r.shape}")

    n = k.shape[1]
    if n < 2:
        return r.copy(), EamitAlignStats(0.0, 0.0, 0.0, 0.0, 0)

    bs = max(2, int(cfg.max_block_size))
    if n <= bs:
        return _align_block(k, r, cfg)

    out = r.copy()
    starts = list(range(0, n, bs))
    if len(starts) >= 2 and n - starts[-1] < 2:
        starts.pop()

    pre_kl_sum = 0.0
    post_kl_sum = 0.0
    pre_mse_sum = 0.0
    post_mse_sum = 0.0
    total_w = 0
    steps_max = 0

    for idx, s in enumerate(starts):
        e = n if idx == len(starts) - 1 else min(s + bs, n)
        blk_out, st = _align_block(k[:, s:e], r[:, s:e], cfg)
        out[:, s:e] = blk_out
        w = e - s
        pre_kl_sum += st.pre_kl * w
        post_kl_sum += st.post_kl * w
        pre_mse_sum += st.pre_top_m_mse * w
        post_mse_sum += st.post_top_m_mse * w
        total_w += w
        steps_max = max(steps_max, st.steps_executed)

    inv_w = 1.0 / max(total_w, 1)
    return out, EamitAlignStats(
        pre_kl=pre_kl_sum * inv_w,
        post_kl=post_kl_sum * inv_w,
        pre_top_m_mse=pre_mse_sum * inv_w,
        post_top_m_mse=post_mse_sum * inv_w,
        steps_executed=steps_max,
    )


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


def apply_eamit_batch(
    workspace: WeightWorkspace,
    target_tensors: list[str],
    facts: list[BatchFactSpec],
    *,
    lambda_cov: float = 1e-2,
    apply_fisher_guard: bool = True,
    align_config: EamitAlignConfig | None = None,
) -> EamitResult:
    """Execute EAMIT simultaneous batch editing across multiple layer matrices.

    Stages:
      1. Prepare stacked keys `K` and raw associative residuals `R_raw = V* - sum W_l K`.
      2. Align `R` to `K` via EAMIT embedding alignment optimization (KL + top-M MSE).
      3. Distribute the aligned residual across `target_tensors` (`l_1, ..., l_L`):
             Delta W_l = R_l K_l^T (K_l K_l^T + lambda_cov * I + diag(F_l))^{-1}
    """
    if not target_tensors:
        raise ValueError("target_tensors must contain at least one tensor name")
    if not facts:
        raise ValueError("facts list must not be empty")

    num_layers = len(target_tensors)
    num_facts = len(facts)

    shapes = [workspace.shape(t) for t in target_tensors]
    for t, shp in zip(target_tensors, shapes):
        if len(shp) != 2:
            raise ValueError(f"EAMIT target {t!r} must be 2D, got {shp}")

    d_out, d_in = shapes[-1]

    K_base = np.column_stack(
        [np.asarray(f.key_vector, dtype=np.float64).reshape(d_in) for f in facts]
    )
    V_target = np.column_stack(
        [np.asarray(f.target_value_vector, dtype=np.float64).reshape(d_out) for f in facts]
    )

    def _cumulative_output() -> np.ndarray:
        total = np.zeros((d_out, num_facts), dtype=np.float64)
        for tname in target_tensors:
            W_l = workspace.get_readonly(tname).astype(np.float64)
            if W_l.shape == (d_out, d_in):
                total += W_l @ K_base
        return total

    init_out = _cumulative_output()
    raw_residual = V_target - init_out
    pre_mse = float(np.mean(raw_residual**2))

    remaining_residual, align_stats = align_residuals_to_keys(
        K_base, raw_residual, align_config or EamitAlignConfig()
    )

    per_tensor_deltas: dict[str, float] = {}

    for idx, tname in enumerate(target_tensors):
        W_l = workspace.get_readonly(tname).astype(np.float64)
        l_out, l_in = W_l.shape
        if l_out != d_out or l_in != d_in:
            continue

        K_l = np.column_stack(
            [
                np.asarray(f.layer_inputs.get(tname, f.key_vector), dtype=np.float64).reshape(l_in)
                for f in facts
            ]
        )

        layers_left = float(num_layers - idx)
        layer_target_residual = remaining_residual / layers_left

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

        remaining_residual = remaining_residual - (actual_delta @ K_l)

    final_out = _cumulative_output()
    post_mse = float(np.mean((V_target - final_out) ** 2))
    rel_red = (pre_mse - post_mse) / max(pre_mse, 1e-12)

    return EamitResult(
        num_facts=num_facts,
        target_tensors=list(target_tensors),
        pre_edit_mse=pre_mse,
        post_edit_mse=post_mse,
        relative_error_reduction=rel_red,
        pre_alignment_kl=align_stats.pre_kl,
        post_alignment_kl=align_stats.post_kl,
        pre_alignment_top_m_mse=align_stats.pre_top_m_mse,
        post_alignment_top_m_mse=align_stats.post_top_m_mse,
        per_tensor_l2_deltas=per_tensor_deltas,
        fisher_protected=bool(apply_fisher_guard and workspace.fisher_guard is not None),
    )
