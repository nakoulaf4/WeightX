# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
L2 Knowledge Localization: Logit Lens, Domain Attention Probing, and Knowledge Concentration Index (KCI).

Clean-room GPLv3 implementation designed for corporate and open-source compliance
(avoiding the AGPL-3.0 + commercial patent restrictions of `egora-diagnostics`).
Also improves technical accuracy over naive Logit Lens implementations by:
  1. Applying the model's final LayerNorm / RMSNorm before projecting intermediate
     hidden states through the unembedding matrix (`lm_head`), preventing pre-norm
     scale distortion in Pre-LN Transformers.
  2. Tracking both ground-truth target token rank/probability and final-layer
     prediction convergence across depth.
  3. Computing the Gini Knowledge Concentration Index (KCI) over per-head and
     per-layer sensitivities across factual, syntactic, and reasoning probes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from weightx.toon_io import dumps_toon


DEFAULT_KNOWLEDGE_PROBES: dict[str, list[tuple[str, str]]] = {
    "factual": [
        ("The capital of France is", "Paris"),
        ("The chemical symbol for water is", "H2O"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("Albert Einstein developed the theory of", "relativity"),
    ],
    "syntactic": [
        ("The cat sat on the", "mat"),
        ("She goes to school every", "day"),
        ("Neither the students nor the teacher", "was"),
        ("Despite the rain, they decided to", "go"),
    ],
    "reasoning": [
        ("If all cats are animals and all animals breathe, then all cats", "breathe"),
        ("2 + 2 = 4, so 4 - 2 =", "2"),
        ("The opposite of hot is", "cold"),
        ("If A is bigger than B and B is bigger than C, then A is bigger than", "C"),
    ],
}


@dataclass
class LogitLensLayerStat:
    """Per-layer Logit Lens metrics aggregated across probes."""

    layer_idx: int
    top1_convergence_rate: float
    target_prob_mean: float
    target_reciprocal_rank: float
    predictive_entropy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "top1_convergence_rate": round(self.top1_convergence_rate, 4),
            "target_prob_mean": round(self.target_prob_mean, 5),
            "target_reciprocal_rank": round(self.target_reciprocal_rank, 4),
            "predictive_entropy": round(self.predictive_entropy, 4),
        }


@dataclass
class KnowledgeMapReport:
    """Complete L2 Knowledge Map and Knowledge Concentration Index (KCI) report."""

    n_layers: int
    n_heads: int
    kci_overall: float
    kci_per_domain: dict[str, float]
    concentration_regime: str
    avg_crystallization_layer: float
    crystallization_fraction: float
    dominant_layers_by_domain: dict[str, list[int]]
    logit_lens_layers: list[LogitLensLayerStat] = field(default_factory=list)
    head_summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "kci_overall": round(self.kci_overall, 4),
            "kci_per_domain": {k: round(v, 4) for k, v in self.kci_per_domain.items()},
            "concentration_regime": self.concentration_regime,
            "avg_crystallization_layer": round(self.avg_crystallization_layer, 3),
            "crystallization_fraction": round(self.crystallization_fraction, 3),
            "dominant_layers_by_domain": self.dominant_layers_by_domain,
            "head_summary": self.head_summary,
            "logit_lens_layers": [s.to_dict() for s in self.logit_lens_layers],
        }

    def to_toon(self) -> str:
        """Render the Knowledge Map in compact TOON format."""
        return dumps_toon(self.to_dict())


def compute_gini_kci(contributions: np.ndarray | list[float]) -> float:
    """Compute the Gini coefficient over non-negative contribution scores in [0, 1].

    High KCI (-> 1.0) indicates knowledge is concentrated in a few fragile heads/layers;
    Low KCI (-> 0.0) indicates knowledge is evenly distributed across the network.
    """
    arr = np.asarray(contributions, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, 0.0, None)
    if arr.size == 0 or np.sum(arr) <= 1e-15:
        return 0.0
    sorted_vals = np.sort(arr)
    n = sorted_vals.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(idx * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def _find_final_norm_and_lm_head(
    model: nn.Module,
) -> tuple[nn.Module | None, nn.Module | Callable[[torch.Tensor], torch.Tensor] | None]:
    """Locate the final normalization layer and unembedding projection in a causal LM."""
    final_norm: nn.Module | None = None
    for attr_path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm", "norm", "ln_f"):
        cur: Any = model
        for part in attr_path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if isinstance(cur, nn.Module):
            final_norm = cur
            break

    lm_head: Any = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    if lm_head is None:
        # Check for tied input embeddings
        for emb_path in ("model.embed_tokens", "transformer.wte", "embed_tokens", "wte"):
            cur = model
            for part in emb_path.split("."):
                cur = getattr(cur, part, None)
                if cur is None:
                    break
            if isinstance(cur, nn.Embedding):
                weight = cur.weight
                lm_head = lambda x, w=weight: F.linear(x, w)
                break
    return final_norm, lm_head


def _collect_layer_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run a forward pass and collect per-layer residual states even for custom models."""
    with torch.no_grad():
        try:
            out = model(input_ids, output_hidden_states=True)
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                logits = out.logits if hasattr(out, "logits") else out[0]
                return logits, list(out.hidden_states[1:])
        except TypeError:
            pass

        # Fallback: attach forward hooks to numbered blocks
        collected: list[torch.Tensor] = []
        hooks = []
        for name, mod in model.named_modules():
            if any(
                name.endswith(f"{prefix}.{i}") or name == f"{prefix}.{i}"
                for prefix in ("layers", "h", "blocks", "layer")
                for i in range(256)
            ):
                def _hk(_m, _inp, output):
                    t = output[0] if isinstance(output, tuple) else output
                    if isinstance(t, torch.Tensor):
                        collected.append(t.detach())
                hooks.append(mod.register_forward_hook(_hk))

        try:
            out = model(input_ids)
            logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
        finally:
            for h in hooks:
                h.remove()
        return logits, collected


def run_logit_lens(
    model: nn.Module,
    probe_batches: list[tuple[torch.Tensor, int, str]],
) -> tuple[list[LogitLensLayerStat], float]:
    """Run normalized Logit Lens across `probe_batches` of `(input_ids, target_token_id, domain)`."""
    model.eval()
    final_norm, lm_head = _find_final_norm_and_lm_head(model)
    device = next(model.parameters()).device

    per_layer_conv: dict[int, list[float]] = {}
    per_layer_prob: dict[int, list[float]] = {}
    per_layer_rr: dict[int, list[float]] = {}
    per_layer_ent: dict[int, list[float]] = {}
    first_crystal_layers: list[int] = []

    for input_ids, target_id, _domain in probe_batches:
        ids = input_ids.to(device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        logits, hidden_states = _collect_layer_hidden_states(model, ids)
        if not hidden_states or lm_head is None:
            continue

        final_top1 = int(logits[0, -1].argmax(dim=-1).item())
        n_layers = len(hidden_states)
        found_first = None

        for l_idx, hs in enumerate(hidden_states):
            last_h = hs[0, -1:, :]
            # Apply final norm to intermediate hidden state except on the last layer if already normed
            if final_norm is not None and l_idx < n_layers - 1:
                try:
                    last_h = final_norm(last_h)
                except Exception:
                    pass
            layer_logits = lm_head(last_h).reshape(-1).float()
            probs = F.softmax(layer_logits, dim=-1)

            pred_id = int(probs.argmax(dim=-1).item())
            converged = 1.0 if (pred_id == target_id or pred_id == final_top1) else 0.0
            if converged > 0.5 and found_first is None:
                found_first = l_idx

            tgt_idx = min(max(0, int(target_id)), probs.numel() - 1)
            tgt_prob = float(probs[tgt_idx].item())

            # Rank of target token (1 = highest)
            higher_count = int((layer_logits > layer_logits[tgt_idx]).sum().item())
            rr = 1.0 / (higher_count + 1.0)

            ent = float(-(probs * torch.log(torch.clamp(probs, min=1e-12))).sum().item())

            per_layer_conv.setdefault(l_idx, []).append(converged)
            per_layer_prob.setdefault(l_idx, []).append(tgt_prob)
            per_layer_rr.setdefault(l_idx, []).append(rr)
            per_layer_ent.setdefault(l_idx, []).append(ent)

        if found_first is not None:
            first_crystal_layers.append(found_first)

    stats: list[LogitLensLayerStat] = []
    for l_idx in sorted(per_layer_conv):
        stats.append(
            LogitLensLayerStat(
                layer_idx=l_idx,
                top1_convergence_rate=float(np.mean(per_layer_conv[l_idx])),
                target_prob_mean=float(np.mean(per_layer_prob[l_idx])),
                target_reciprocal_rank=float(np.mean(per_layer_rr[l_idx])),
                predictive_entropy=float(np.mean(per_layer_ent[l_idx])),
            )
        )

    avg_crystal = float(np.mean(first_crystal_layers)) if first_crystal_layers else 0.0
    return stats, avg_crystal


def run_knowledge_map(
    model: nn.Module,
    tokenizer: Any | None = None,
    token_probes: list[tuple[torch.Tensor, int, str]] | None = None,
    n_heads_hint: int | None = None,
) -> KnowledgeMapReport:
    """Execute the full L2 Knowledge Map (Logit Lens + Per-Head Domain Probing + Gini KCI).

    Works with either a HuggingFace `tokenizer` + text prompts OR pre-tokenized
    `token_probes` (`[(input_ids, target_token_id, domain), ...]`) for tokenizer-free models.
    """
    model.eval()
    device = next(model.parameters()).device

    cfg = getattr(model, "config", None)
    n_heads = (
        n_heads_hint
        or getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "n_head", None)
        or 4
    )

    # Build probe batches
    probe_batches: list[tuple[torch.Tensor, int, str]] = []
    if token_probes is not None:
        probe_batches = list(token_probes)
    elif tokenizer is not None:
        for domain, pairs in DEFAULT_KNOWLEDGE_PROBES.items():
            for prompt_text, answer_text in pairs:
                enc_p = tokenizer(prompt_text, return_tensors="pt")
                enc_a = tokenizer(" " + answer_text, add_special_tokens=False)
                ans_ids = enc_a["input_ids"] if isinstance(enc_a, dict) else enc_a.input_ids
                first_ans = int(ans_ids[0][0]) if isinstance(ans_ids[0], (list, torch.Tensor)) else int(ans_ids[0])
                probe_batches.append((enc_p["input_ids"], first_ans, domain))
    else:
        # Synthetic deterministic token probes across 3 domains
        vocab_limit = getattr(cfg, "vocab_size", 64)
        rng = np.random.default_rng(42)
        for domain in ("factual", "syntactic", "reasoning"):
            for _ in range(4):
                seq = torch.tensor(rng.integers(1, max(4, vocab_limit), size=(1, 8)), dtype=torch.long)
                tgt = int(rng.integers(1, max(4, vocab_limit)))
                probe_batches.append((seq, tgt, domain))

    lens_stats, avg_crystal = run_logit_lens(model, probe_batches)
    n_layers = max(1, len(lens_stats))

    # Measure per-layer, per-head residual update norms across domains
    domain_names = sorted({d for _, _, d in probe_batches})
    domain_head_scores: dict[str, np.ndarray] = {
        d: np.zeros((n_layers, n_heads), dtype=np.float64) for d in domain_names
    }
    domain_counts: dict[str, int] = {d: 0 for d in domain_names}

    for input_ids, _tgt, domain in probe_batches:
        ids = input_ids.to(device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        _, hidden_states = _collect_layer_hidden_states(model, ids)
        if not hidden_states:
            continue

        prev_h = hidden_states[0][0, -1].float()
        for l_idx in range(min(n_layers, len(hidden_states))):
            cur_h = hidden_states[l_idx][0, -1].float()
            delta = cur_h if l_idx == 0 else (cur_h - prev_h)
            prev_h = cur_h
            head_dim = max(1, delta.numel() // n_heads)
            for h_i in range(n_heads):
                chunk = delta[h_i * head_dim : (h_i + 1) * head_dim]
                if chunk.numel() > 0:
                    domain_head_scores[domain][l_idx, h_i] += float(chunk.norm().item())
        domain_counts[domain] += 1

    kci_per_domain: dict[str, float] = {}
    dominant_layers: dict[str, list[int]] = {}
    head_summary: dict[str, int] = {f"n_{d}_heads": 0 for d in domain_names}

    for d in domain_names:
        if domain_counts[d] > 0:
            domain_head_scores[d] /= domain_counts[d]
        kci_per_domain[d] = compute_gini_kci(domain_head_scores[d].reshape(-1))
        layer_totals = domain_head_scores[d].sum(axis=1)
        top_l = np.argsort(-layer_totals)[: min(3, n_layers)]
        dominant_layers[d] = [int(x) for x in top_l]

    for l_idx in range(n_layers):
        for h_i in range(n_heads):
            best_d = max(domain_names, key=lambda d: domain_head_scores[d][l_idx, h_i])
            head_summary[f"n_{best_d}_heads"] = head_summary.get(f"n_{best_d}_heads", 0) + 1

    kci_overall = float(np.mean(list(kci_per_domain.values()))) if kci_per_domain else 0.0
    regime = (
        "concentrated_fragile"
        if kci_overall >= 0.45
        else ("moderate_localized" if kci_overall >= 0.20 else "distributed_robust")
    )

    return KnowledgeMapReport(
        n_layers=n_layers,
        n_heads=n_heads,
        kci_overall=kci_overall,
        kci_per_domain=kci_per_domain,
        concentration_regime=regime,
        avg_crystallization_layer=avg_crystal,
        crystallization_fraction=float(avg_crystal / max(1, n_layers)),
        dominant_layers_by_domain=dominant_layers,
        logit_lens_layers=lens_stats,
        head_summary=head_summary,
    )
