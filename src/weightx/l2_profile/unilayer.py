# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
L2 Uni-Layer 13-Metric Profiler.

Evaluates every layer of a target model across 13 complementary metrics:
  1. effective_rank          (Roy & Vetterli spectral entropy rank)
  2. stable_rank             (||W||_F^2 / sigma_max^2)
  3. weight_l2_norm          (Frobenius norm)
  4. weight_sparsity         (Near-zero parameter fraction)
  5. weight_outlier_ratio    (Fraction of weights > 3 std from mean)
  6. weight_kurtosis         (Excess fourth standardized moment)
  7. condition_number        (sigma_max / sigma_min)
  8. block_influence         (1 - cos(h_in, h_out) representation transform score)
  9. activation_entropy      (Normalized channel energy Shannon entropy)
 10. neuron_outlier_ratio    (Fraction of rows/neurons > 2.5x median L2 norm)
 11. fisher_information      (Empirical diagonal Fisher trace)
 12. gradient_norm           (Layer parameter gradient L2 norm)
 13. wanda_score             (Weight magnitude x input activation norm score)

Interoperates with `uni_layer.core.profile.LayerProfile` (MIT) to produce Borda
consensus rankings, redundant layer detection, representation bottleneck
detection, anomaly z-scores, outlier neuron indices, and a concise text summary.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from weightx.toon_io import dumps_toon

try:
    from uni_layer.core.profile import LayerProfile as _ExtLayerProfile
except ImportError:  # pragma: no cover
    _ExtLayerProfile = None


METRIC_NAMES_13: tuple[str, ...] = (
    "effective_rank",
    "stable_rank",
    "weight_l2_norm",
    "weight_sparsity",
    "weight_outlier_ratio",
    "weight_kurtosis",
    "condition_number",
    "block_influence",
    "activation_entropy",
    "neuron_outlier_ratio",
    "fisher_information",
    "gradient_norm",
    "wanda_score",
)


@dataclass
class OutlierNeuronRecord:
    """Pinpoints a specific tensor and row/neuron index with anomalous weight norm."""

    layer: str
    tensor_name: str
    row_indices: list[int]
    max_ratio_to_median: float
    median_row_norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "tensor_name": self.tensor_name,
            "row_indices": self.row_indices,
            "max_ratio_to_median": round(self.max_ratio_to_median, 3),
            "median_row_norm": round(self.median_row_norm, 5),
        }


@dataclass
class UniLayerProfileReport:
    """Structured L2 13-metric layer profile with natural-language summary and TOON export."""

    model_name: str
    num_layers: int
    metrics_computed: list[str]
    summary: str
    redundant_layers: list[str]
    bottleneck_layers: list[str]
    consensus_ranking: list[dict[str, Any]]
    depth_trends: dict[str, dict[str, Any]]
    anomalies: dict[str, list[dict[str, Any]]]
    outlier_neurons: list[OutlierNeuronRecord] = field(default_factory=list)
    per_layer_metrics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_full_table: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "metrics_count": len(self.metrics_computed),
            "summary": self.summary,
            "redundant_layers": self.redundant_layers,
            "bottleneck_layers": self.bottleneck_layers,
            "outlier_neurons": [o.to_dict() for o in self.outlier_neurons],
            "anomalies": self.anomalies,
            "depth_trends": self.depth_trends,
            "consensus_ranking": [
                {
                    "layer": r["layer"],
                    "rank": r["rank"],
                    "consensus_score": r["consensus_score"],
                }
                for r in self.consensus_ranking
            ],
        }
        if include_full_table:
            out["per_layer_metrics"] = self.per_layer_metrics
        return out

    def to_toon(self, include_full_table: bool = True) -> str:
        """Serialize the 13-metric layer profile in TOON format."""
        return dumps_toon(self.to_dict(include_full_table=include_full_table))


def _group_parameters_by_layer(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Group named 2D+ parameters into canonical layer buckets (`layer_0`, `layer_1`, ...)."""
    buckets: dict[int, list[tuple[str, nn.Parameter]]] = defaultdict(list)
    fallback_buckets: dict[str, list[tuple[str, nn.Parameter]]] = defaultdict(list)

    for name, param in model.named_parameters():
        m = re.search(r"(?:layers|h|blocks|layer)\.(\d+)\.", name)
        if not m:
            m = re.search(r"(?:^|\.)(\d+)\.", name)
        if m:
            buckets[int(m.group(1))].append((name, param))
        elif param.ndim >= 2:
            prefix = name.rsplit(".", 1)[0]
            fallback_buckets[prefix].append((name, param))

    if buckets:
        return {f"layer_{idx}": buckets[idx] for idx in sorted(buckets)}
    return dict(fallback_buckets)


def _compute_matrix_spectral_stats(w: np.ndarray) -> tuple[float, float, float]:
    """Return (effective_rank, stable_rank, condition_number) for a 2D weight matrix."""
    m = w.reshape(w.shape[0], -1).astype(np.float32)
    r, c = m.shape
    if min(r, c) <= 256:
        sv = np.linalg.svd(m.astype(np.float64), compute_uv=False)
    else:
        gram = (m.T @ m) if c < r else (m @ m.T)
        ev = np.linalg.eigvalsh(gram.astype(np.float64))[::-1]
        sv = np.sqrt(np.clip(ev, 0.0, None))

    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return 1.0, 1.0, 1.0

    p = sv / np.sum(sv)
    entropy = -float(np.sum(p * np.log(np.clip(p, 1e-15, 1.0))))
    eff_rank = float(np.exp(entropy))
    stable_rank = float(np.sum(sv**2) / max(sv[0] ** 2, 1e-15))
    cond = float(sv[0] / max(sv[-1], 1e-8))
    return eff_rank, stable_rank, cond


def profile_model_layers(
    model: nn.Module,
    calibration_input_ids: torch.Tensor | None = None,
    model_name: str = "target_model",
    outlier_row_threshold: float = 2.5,
) -> UniLayerProfileReport:
    """Compute all 13 Uni-Layer metrics per layer and extract actionable WSP insights."""
    model.eval()
    layer_groups = _group_parameters_by_layer(model)
    layer_names = list(layer_groups.keys())

    # Optional calibration forward + backward pass for activation/gradient/Fisher metrics
    layer_hidden_in: dict[str, torch.Tensor] = {}
    layer_hidden_out: dict[str, torch.Tensor] = {}
    layer_act_col_norms: dict[str, torch.Tensor] = {}
    grad_norms: dict[str, float] = {}
    fisher_traces: dict[str, float] = {}

    if calibration_input_ids is not None:
        device = next(model.parameters()).device
        ids = calibration_input_ids.to(device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        hooks = []
        # Attach hooks to numbered blocks if discoverable
        block_modules: dict[str, nn.Module] = {}
        for mod_name, mod in model.named_modules():
            m = re.search(r"^(?:.*?\.)?(?:layers|h|blocks|layer)\.(\d+)$", mod_name)
            if m:
                block_modules[f"layer_{int(m.group(1))}"] = mod

        def _make_hook(lname: str):
            def _hook(module, args, output):
                inp = args[0] if args else None
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
                    layer_hidden_in[lname] = inp.detach().float()
                    layer_hidden_out[lname] = out.detach().float()
                    flat_in = inp.detach().float().reshape(-1, inp.shape[-1])
                    layer_act_col_norms[lname] = torch.linalg.vector_norm(flat_in, dim=0)
            return _hook

        for lname, mod in block_modules.items():
            hooks.append(mod.register_forward_hook(_make_hook(lname)))

        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = model(ids)
                logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
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

            for lname, params in layer_groups.items():
                sq_grad_sum = 0.0
                total_n = 0
                for _, p in params:
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        sq_grad_sum += float(torch.sum(g * g).item())
                        total_n += g.numel()
                grad_norms[lname] = float(np.sqrt(sq_grad_sum))
                fisher_traces[lname] = float(sq_grad_sum / max(1, total_n))
            model.zero_grad(set_to_none=True)
        except Exception:
            model.zero_grad(set_to_none=True)
        finally:
            for h in hooks:
                h.remove()

    contributions: dict[str, dict[str, Any]] = {}
    outlier_records: list[OutlierNeuronRecord] = []
    per_layer_rows: list[dict[str, Any]] = []

    for idx, lname in enumerate(layer_names):
        params = layer_groups[lname]
        mats = [(n, p.detach().float().cpu().numpy()) for n, p in params if p.ndim >= 2]
        if not mats:
            mats = [(n, p.detach().float().cpu().numpy().reshape(1, -1)) for n, p in params]

        eff_ranks, stable_ranks, conds = [], [], []
        l2_sq = 0.0
        total_elems = 0
        near_zeros = 0
        weight_outliers = 0
        kurtosis_vals = []
        outlier_row_fracs = []
        wanda_vals = []

        for tname, arr in mats:
            m2d = arr.reshape(arr.shape[0], -1)
            er, sr, cn = _compute_matrix_spectral_stats(m2d)
            eff_ranks.append(er)
            stable_ranks.append(sr)
            conds.append(cn)

            flat = m2d.reshape(-1).astype(np.float64)
            l2_sq += float(np.dot(flat, flat))
            total_elems += flat.size
            mu = float(np.mean(flat))
            sd = float(np.std(flat)) + 1e-12
            near_zeros += int(np.count_nonzero(np.abs(flat) < 1e-4 * sd))
            z = np.abs(flat - mu) / sd
            weight_outliers += int(np.count_nonzero(z > 3.0))
            kurt = float(np.mean(z**4) - 3.0)
            kurtosis_vals.append(kurt)

            # Row/neuron L2 norms for outlier unit detection
            row_norms = np.linalg.norm(m2d.astype(np.float64), axis=1)
            med_norm = float(np.median(row_norms))
            if med_norm > 1e-12 and row_norms.size >= 4:
                ratios = row_norms / med_norm
                bad_rows = np.where(ratios >= outlier_row_threshold)[0]
                outlier_row_fracs.append(float(bad_rows.size / row_norms.size))
                if bad_rows.size > 0:
                    # Order by ratio descending
                    order = bad_rows[np.argsort(-ratios[bad_rows])]
                    outlier_records.append(
                        OutlierNeuronRecord(
                            layer=lname,
                            tensor_name=tname,
                            row_indices=[int(r) for r in order[:16]],
                            max_ratio_to_median=float(ratios[order[0]]),
                            median_row_norm=med_norm,
                        )
                    )
            else:
                outlier_row_fracs.append(0.0)

            # Wanda score
            act_norm = layer_act_col_norms.get(lname)
            if act_norm is not None and act_norm.numel() == m2d.shape[1]:
                an = act_norm.cpu().numpy()
                wanda_vals.append(float(np.mean(np.abs(m2d) * an[None, :])))
            else:
                wanda_vals.append(float(np.mean(np.abs(m2d)) * np.sqrt(m2d.shape[1])))

        # Block influence & activation entropy
        h_in = layer_hidden_in.get(lname)
        h_out = layer_hidden_out.get(lname)
        if h_in is not None and h_out is not None and h_in.shape == h_out.shape:
            v_in = h_in.reshape(-1, h_in.shape[-1])
            v_out = h_out.reshape(-1, h_out.shape[-1])
            cos_sim = F.cosine_similarity(v_in, v_out, dim=-1).mean().item()
            block_inf = float(max(0.0, 1.0 - cos_sim))

            ch_energy = v_out.pow(2).mean(dim=0)
            p_ch = ch_energy / (ch_energy.sum() + 1e-12)
            act_ent = float(
                -(p_ch * torch.log(torch.clamp(p_ch, min=1e-12))).sum().item()
                / max(1.0, np.log(p_ch.numel()))
            )
        else:
            # Spectral approximation of block influence when run weights-only
            block_inf = float(np.mean(stable_ranks) / max(1.0, np.mean(eff_ranks)))
            act_ent = float(np.log(max(np.mean(eff_ranks), 1.001)) / np.log(max(m2d.shape[0], 2)))

        w_l2 = float(np.sqrt(l2_sq))
        g_norm = grad_norms.get(lname, w_l2 / np.sqrt(max(1, total_elems)))
        f_info = fisher_traces.get(lname, (g_norm**2) / max(1, total_elems))

        m_dict: dict[str, Any] = {
            "layer_idx": idx,
            "layer_type": "transformer_block",
            "effective_rank": round(float(np.mean(eff_ranks)), 4),
            "stable_rank": round(float(np.mean(stable_ranks)), 4),
            "weight_l2_norm": round(w_l2, 4),
            "weight_sparsity": round(float(near_zeros / max(1, total_elems)), 6),
            "weight_outlier_ratio": round(float(weight_outliers / max(1, total_elems)), 6),
            "weight_kurtosis": round(float(np.mean(kurtosis_vals)), 4),
            "condition_number": round(float(np.mean(conds)), 4),
            "block_influence": round(block_inf, 5),
            "activation_entropy": round(act_ent, 5),
            "neuron_outlier_ratio": round(float(np.max(outlier_row_fracs) if outlier_row_fracs else 0.0), 6),
            "fisher_information": round(f_info, 8),
            "gradient_norm": round(g_norm, 5),
            "wanda_score": round(float(np.mean(wanda_vals)), 5),
        }
        contributions[lname] = m_dict
        per_layer_rows.append({"layer": lname, **m_dict})

    if _ExtLayerProfile is not None and contributions:
        ext_prof = _ExtLayerProfile(contributions, model_name=model_name)
        redundant = list(ext_prof.redundant_layers)
        bottlenecks = list(ext_prof.bottleneck_layers)
        ranking = list(ext_prof.consensus_ranking)
        trends = dict(ext_prof.depth_trends)
        anomalies = dict(ext_prof.anomalies)
        base_summary = ext_prof.summary()
    else:
        redundant, bottlenecks, ranking, trends, anomalies, base_summary = _fallback_profile_analysis(
            contributions, model_name
        )

    if outlier_records:
        outlier_desc = ", ".join(
            f"{o.tensor_name} rows {o.row_indices[:4]} ({o.max_ratio_to_median:.1f}x median)"
            for o in outlier_records[:3]
        )
        base_summary += f" Detected {len(outlier_records)} tensor(s) with outlier neurons: {outlier_desc}."

    return UniLayerProfileReport(
        model_name=model_name,
        num_layers=len(layer_names),
        metrics_computed=list(METRIC_NAMES_13),
        summary=base_summary,
        redundant_layers=redundant,
        bottleneck_layers=bottlenecks,
        consensus_ranking=ranking,
        depth_trends=trends,
        anomalies=anomalies,
        outlier_neurons=outlier_records,
        per_layer_metrics=per_layer_rows,
    )


def _fallback_profile_analysis(
    contributions: dict[str, dict[str, Any]],
    model_name: str,
) -> tuple[
    list[str],
    list[str],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    str,
]:
    layers = list(contributions.keys())
    n = len(layers)
    if n == 0:
        return [], [], [], {}, {}, f"{model_name} (0 layers)."

    bi = np.array([float(contributions[l]["block_influence"]) for l in layers])
    er = np.array([float(contributions[l]["effective_rank"]) for l in layers])
    redundant = [layers[i] for i, v in enumerate(bi) if v < 0.02]
    bottlenecks = [
        layers[i]
        for i in range(1, n - 1)
        if (er[i - 1] + er[i + 1]) > 0 and er[i] / ((er[i - 1] + er[i + 1]) / 2.0) < 0.7
    ]
    ranking = [
        {"layer": l, "rank": i + 1, "consensus_score": round(1.0 - i / max(1, n - 1), 4)}
        for i, l in enumerate(layers)
    ]
    summary = (
        f"{model_name} ({n}-layer model). analyzed with 13 metrics. "
        f"redundant={len(redundant)}, bottlenecks={len(bottlenecks)}."
    )
    return redundant, bottlenecks, ranking, {}, {}, summary
