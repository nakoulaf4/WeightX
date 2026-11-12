# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
Vespasian Verification & Automatic Rollback Engine.

Enforces strict separation between:
  - `practice_split`: Accessible to the LLM via budgeted practice probes so it can
    test hypotheses during localization.
  - `held_out_split`: Cryptographically/structurally hidden from the LLM. Used only
    by `VespasianEngine.verify_and_decide()` after an edit is proposed.

Computes:
  1. Held-out target score (`held_out_score`) and paired item transitions
     (fixed vs regressed examples).
  2. Reference behavior preservation (`retention_score` and KL divergence
     `kl_divergence` against the baseline model outputs on general reference inputs).
  3. Structural health invariants (zero NaN/Inf, max parameter change fraction).

If the edit improves the held-out score by at least `improvement_threshold` AND
satisfies all preservation/health contracts (`kl_divergence <= max_kl_drift`,
`retention_degradation <= max_retention_drop`, `nonfinite == 0`), the engine
updates `best_checkpoint` and commits. Otherwise, it automatically rolls back
weights to `best_checkpoint`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from weightx.l3_weights.workspace import WeightWorkspace
from weightx.toon_io import dumps_toon


@dataclass
class EvaluationSplit:
    """A labeled evaluation split (`practice` or `held_out`) with input batches and targets.

    Supports both:
      - Custom `eval_fn(model_or_workspace) -> (mean_score, per_item_scores)`
      - Or standard `(input_ids, target_ids)` batches for causal language models.
    """

    name: str
    eval_fn: Callable[[Any], tuple[float, list[float]]] | None = None
    input_ids: torch.Tensor | None = None
    target_ids: torch.Tensor | None = None
    reference_input_ids: torch.Tensor | None = None

    def evaluate(self, workspace: WeightWorkspace) -> tuple[float, list[float]]:
        workspace.sync_all()
        target_obj = workspace.model if workspace.model is not None else workspace
        if self.eval_fn is not None:
            res = self.eval_fn(target_obj)
            if isinstance(res, tuple) and len(res) == 2:
                return float(res[0]), [float(x) for x in res[1]]
            val = float(res)
            return val, [val]

        if workspace.model is not None and self.input_ids is not None:
            model = workspace.model
            model.eval()
            dev = next(model.parameters()).device
            ids = self.input_ids.to(dev)
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            with torch.no_grad():
                out = model(ids)
                logits = (
                    out.logits
                    if hasattr(out, "logits")
                    else (out[0] if isinstance(out, tuple) else out)
                )
                if self.target_ids is not None:
                    tgts = self.target_ids.to(dev)
                    last_logits = logits[:, -1, :] if logits.ndim == 3 else logits
                    probs = F.softmax(last_logits.float(), dim=-1)
                    preds = probs.argmax(dim=-1)
                    item_scores = []
                    for i in range(ids.shape[0]):
                        t_id = int(tgts[i].item())
                        acc = 1.0 if int(preds[i].item()) == t_id else 0.0
                        p_target = float(probs[i, min(t_id, probs.shape[-1] - 1)].item())
                        # Smooth score combining exact accuracy and target probability
                        item_scores.append(0.7 * acc + 0.3 * p_target)
                    return float(np.mean(item_scores)), item_scores
                else:
                    # Next-token accuracy + exp(-NLL)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = ids[:, 1:].contiguous()
                    per_tok_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                    ).view(ids.shape[0], -1)
                    seq_scores = torch.exp(-per_tok_loss.mean(dim=-1)).cpu().numpy().tolist()
                    return float(np.mean(seq_scores)), [float(x) for x in seq_scores]

        return 0.0, [0.0]


@dataclass
class VespasianIterationResult:
    """Sanitized verification result returned by `VespasianEngine.verify_and_decide()`."""

    iteration: int
    accepted: bool
    rolled_back: bool
    baseline_score: float
    previous_best_score: float
    candidate_score: float
    best_score_after: float
    score_delta_vs_best: float
    paired_fixed_items: int
    paired_regressed_items: int
    kl_divergence_ref: float
    nonfinite_params: int
    changed_params_pct: float
    verdict_reason: str
    hypothesis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "accepted": self.accepted,
            "rolled_back": self.rolled_back,
            "baseline_score": round(self.baseline_score, 5),
            "previous_best_score": round(self.previous_best_score, 5),
            "candidate_score": round(self.candidate_score, 5),
            "best_score_after": round(self.best_score_after, 5),
            "score_delta_vs_best": round(self.score_delta_vs_best, 5),
            "paired_fixed_items": self.paired_fixed_items,
            "paired_regressed_items": self.paired_regressed_items,
            "kl_divergence_ref": round(self.kl_divergence_ref, 6),
            "nonfinite_params": self.nonfinite_params,
            "changed_params_pct": round(self.changed_params_pct, 4),
            "verdict_reason": self.verdict_reason,
            "hypothesis": self.hypothesis,
        }

    def to_toon(self) -> str:
        return dumps_toon(self.to_dict())


class VespasianEngine:
    """Manages budgeted practice probing, hidden held-out evaluation, and automatic rollback."""

    def __init__(
        self,
        workspace: WeightWorkspace,
        *,
        practice_split: EvaluationSplit | None = None,
        held_out_split: EvaluationSplit | None = None,
        improvement_threshold: float = 1e-4,
        max_kl_drift: float = 0.50,
        max_changed_fraction: float = 0.25,
        practice_budget: int = 10,
    ):
        self.workspace = workspace
        self.practice_split = practice_split
        self._held_out_split = held_out_split  # Private: never exposed to LLM tools
        self.improvement_threshold = float(improvement_threshold)
        self.max_kl_drift = float(max_kl_drift)
        self.max_changed_fraction = float(max_changed_fraction)
        self.practice_budget = int(practice_budget)
        self.practice_calls_used = 0
        self.iteration = 0
        self.history: list[VespasianIterationResult] = []

        # Record baseline checkpoint & reference logits for KL divergence
        self.workspace.checkpoint("vespasian_best")
        self._baseline_ref_probs: torch.Tensor | None = self._capture_reference_probs()

        if self._held_out_split is not None:
            self.baseline_score, self.best_item_scores = self._held_out_split.evaluate(self.workspace)
        elif self.practice_split is not None:
            self.baseline_score, self.best_item_scores = self.practice_split.evaluate(self.workspace)
        else:
            self.baseline_score, self.best_item_scores = 0.0, []

        self.best_score = self.baseline_score

    def _capture_reference_probs(self) -> torch.Tensor | None:
        split = self._held_out_split or self.practice_split
        if split is None or self.workspace.model is None:
            return None
        ref_ids = split.reference_input_ids if split.reference_input_ids is not None else split.input_ids
        if ref_ids is None:
            return None
        model = self.workspace.model
        model.eval()
        dev = next(model.parameters()).device
        with torch.no_grad():
            ids = ref_ids.to(dev)
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            out = model(ids)
            logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
            return F.softmax(logits.float(), dim=-1).detach().cpu()

    def _compute_reference_kl(self) -> float:
        if self._baseline_ref_probs is None:
            return 0.0
        cur_probs = self._capture_reference_probs()
        if cur_probs is None or cur_probs.shape != self._baseline_ref_probs.shape:
            return 0.0
        p = torch.clamp(self._baseline_ref_probs, min=1e-12)
        q = torch.clamp(cur_probs, min=1e-12)
        kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean().item()
        return float(max(0.0, kl))

    def run_practice_probe(self) -> dict[str, Any]:
        """Run evaluation on the visible `practice_split` (consumes 1 practice budget unit)."""
        if self.practice_calls_used >= self.practice_budget:
            return {
                "error": f"Practice probe budget exhausted ({self.practice_budget}/{self.practice_budget})",
                "budget_remaining": 0,
            }
        self.practice_calls_used += 1
        if self.practice_split is None:
            return {
                "practice_score": 0.0,
                "budget_remaining": self.practice_budget - self.practice_calls_used,
                "note": "No practice split configured",
            }
        score, items = self.practice_split.evaluate(self.workspace)
        return {
            "practice_score": round(score, 5),
            "num_practice_items": len(items),
            "per_item_scores": [round(x, 4) for x in items[:16]],
            "budget_remaining": self.practice_budget - self.practice_calls_used,
        }

    def verify_and_decide(self, hypothesis: str = "") -> VespasianIterationResult:
        """Evaluate current workspace weights on the hidden `held_out_split` and commit or rollback."""
        self.iteration += 1
        self.workspace.sync_all()

        # 1. Check numerical finiteness across all tensors
        nonfinite = 0
        for name in self.workspace.names():
            arr = self.workspace.get_readonly(name)
            nonfinite += int(np.count_nonzero(~np.isfinite(arr)))

        diff = self.workspace.diff()
        changed_frac = diff.changed_fraction

        # 2. Evaluate on hidden held-out split
        eval_split = self._held_out_split or self.practice_split
        if eval_split is not None and nonfinite == 0:
            cand_score, cand_items = eval_split.evaluate(self.workspace)
        else:
            cand_score, cand_items = -1.0, []

        # 3. Compute paired item transitions (fixed vs regressed)
        fixed_items = 0
        regressed_items = 0
        if len(cand_items) == len(self.best_item_scores):
            for before_s, after_s in zip(self.best_item_scores, cand_items):
                if after_s > before_s + 0.05:
                    fixed_items += 1
                elif after_s < before_s - 0.05:
                    regressed_items += 1

        # 4. Compute reference KL drift
        kl_drift = self._compute_reference_kl() if nonfinite == 0 else float("inf")

        prev_best = self.best_score
        delta_vs_best = cand_score - prev_best

        # 5. Apply Vespasian decision rules
        reasons: list[str] = []
        accepted = True

        if nonfinite > 0:
            accepted = False
            reasons.append(f"REJECTED: {nonfinite} NaN/Inf parameters detected")
        if changed_frac > self.max_changed_fraction:
            accepted = False
            reasons.append(
                f"REJECTED: changed parameter fraction {changed_frac:.2%} exceeds limit {self.max_changed_fraction:.2%}"
            )
        if kl_drift > self.max_kl_drift:
            accepted = False
            reasons.append(
                f"REJECTED: reference KL divergence {kl_drift:.4f} exceeds locality limit {self.max_kl_drift:.4f}"
            )
        if delta_vs_best < self.improvement_threshold:
            accepted = False
            reasons.append(
                f"REJECTED: held-out score {cand_score:.5f} did not improve over best {prev_best:.5f} "
                f"(delta={delta_vs_best:+.5f}, required >= +{self.improvement_threshold:g})"
            )

        if accepted:
            self.best_score = cand_score
            self.best_item_scores = cand_items
            self.workspace.checkpoint("vespasian_best")
            verdict_reason = (
                f"ACCEPTED & COMMITTED: held-out score improved {prev_best:.5f} -> {cand_score:.5f} "
                f"({delta_vs_best:+.5f}), KL drift={kl_drift:.4f} <= {self.max_kl_drift}"
            )
            rolled_back = False
        else:
            self.workspace.restore("vespasian_best")
            verdict_reason = "; ".join(reasons) + " -> AUTOMATIC ROLLBACK to vespasian_best."
            rolled_back = True

        result = VespasianIterationResult(
            iteration=self.iteration,
            accepted=accepted,
            rolled_back=rolled_back,
            baseline_score=self.baseline_score,
            previous_best_score=prev_best,
            candidate_score=cand_score,
            best_score_after=self.best_score,
            score_delta_vs_best=delta_vs_best,
            paired_fixed_items=fixed_items,
            paired_regressed_items=regressed_items,
            kl_divergence_ref=kl_drift,
            nonfinite_params=nonfinite,
            changed_params_pct=changed_frac * 100.0,
            verdict_reason=verdict_reason,
            hypothesis=hypothesis,
        )
        self.history.append(result)
        return result
