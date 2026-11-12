# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX Harness Orchestrator (`WeightXHarness`).

Coordinates the three isolated architectural layers:
  1. Provider Layer (`LLMProvider` — OpenAI/OpenRouter/vLLM/Ollama/LM Studio, Anthropic,
     CLI agents Claude Code/Codex/OpenCode/Antigravity/DSH, or custom callables)
  2. Tool Layer (`ToolRegistry` — L1 AutoArch, L2 Uni-Layer & Knowledge Map,
     L3 Workspace/Slices/BrainSurgery-WX YAML/MEMIT-MASS/Sandbox, Vespasian verify)
  3. Execution & Verification Layer (`WeightWorkspace`, `FisherGuard`, `IsolatedSandbox`,
     `VespasianEngine` with hidden held-out split and automatic rollback).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from weightx.execution.sandbox import IsolatedSandbox
from weightx.execution.vespasian import EvaluationSplit, VespasianEngine, VespasianIterationResult
from weightx.l3_weights.workspace import WeightWorkspace, WorkspaceDiffSummary
from weightx.providers.base import LLMMessage, LLMProvider
from weightx.providers.custom import AutonomousWSPProvider
from weightx.safety.iwck_fisher import FisherGuard
from weightx.toon_io import dumps_toon
from weightx.tools.registry import ToolRegistry, build_default_tool_registry


SYSTEM_PROMPT_WEIGHTX = """You are a Weight-Space Programming (WSP) agent operating inside the WeightX Harness.
You do not have direct access to the filesystem, raw model objects, or held-out evaluation tests.
Follow the 5-stage WeightX editing cycle strictly:
  1. RECON: Call `l1_autoarch_inspect`, `l2_unilayer_profile`, and `l2_knowledge_map` to obtain a compact TOON profile of the model architecture, 13 layer metrics, and Knowledge Concentration Index (KCI).
  2. LOCALIZATION: Form a precise hypothesis (e.g., anomalous outlier neurons or localized factual projections in specific layers) and inspect only the relevant tensors via `l3_describe_tensor` and `l3_read_tensor_slice`.
  3. PLANNING: Construct a declarative BrainSurgery-WX YAML plan (`transforms: [...]`) that includes `assert` steps (`exists`, `shape`, `dtype`, `finite`, `norm`) before and after mutations.
  4. EXECUTION: Apply the YAML plan via `l3_apply_yaml_plan` (or `l3_batch_edit_memit` for batch factual edits) and check the visible practice split via `l3_practice_probe`.
  5. VERIFICATION & ROLLBACK (Vespasian): Call `l3_verify_and_commit` with your hypothesis so the harness evaluates the edit on the hidden held-out test split and either commits the new best checkpoint or automatically rolls back regressions.
"""


@dataclass
class HarnessRunResult:
    """Complete audit report returned by `WeightXHarness.run_editing_cycle()`."""

    goal: str
    provider_name: str
    model_name: str
    turns_executed: int
    tool_calls_executed: int
    baseline_held_out_score: float
    final_best_held_out_score: float
    net_improvement: float
    final_assistant_summary: str
    diff_summary: WorkspaceDiffSummary
    vespasian_history: list[VespasianIterationResult] = field(default_factory=list)
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "provider": self.provider_name,
            "llm_model": self.model_name,
            "turns_executed": self.turns_executed,
            "tool_calls_executed": self.tool_calls_executed,
            "baseline_held_out_score": round(self.baseline_held_out_score, 5),
            "final_best_held_out_score": round(self.final_best_held_out_score, 5),
            "net_improvement": round(self.net_improvement, 5),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "final_assistant_summary": self.final_assistant_summary,
            "diff_summary": self.diff_summary.to_dict(),
            "vespasian_history": [v.to_dict() for v in self.vespasian_history],
            "tool_call_log": self.tool_call_log,
        }

    def to_toon(self) -> str:
        return dumps_toon(self.to_dict())


class WeightXHarness:
    """Main WeightX 3-layer harness for safe, verified LLM weight programming."""

    def __init__(
        self,
        target_model: nn.Module | Mapping[str, Any],
        *,
        provider: LLMProvider | None = None,
        practice_split: EvaluationSplit | None = None,
        held_out_split: EvaluationSplit | None = None,
        calibration_input_ids: torch.Tensor | None = None,
        reference_fisher_batches: list[torch.Tensor] | torch.Tensor | None = None,
        tokenizer: Any | None = None,
        fisher_damping_alpha: float = 0.25,
        fisher_freeze_top_fraction: float = 0.05,
        improvement_threshold: float = 1e-4,
        max_kl_drift: float = 0.50,
        max_changed_fraction: float = 0.25,
        use_toon: bool = True,
    ):
        self.provider: LLMProvider = provider or AutonomousWSPProvider()
        self.calibration_input_ids = calibration_input_ids

        # Initialize IWCKEdit Fisher Guard
        ref_batches = reference_fisher_batches if reference_fisher_batches is not None else calibration_input_ids
        if isinstance(target_model, nn.Module):
            self.fisher_guard = FisherGuard.from_model(
                target_model,
                reference_batches=ref_batches,
                damping_alpha=fisher_damping_alpha,
                freeze_top_fraction=fisher_freeze_top_fraction,
                enabled=True,
            )
        else:
            self.fisher_guard = FisherGuard(
                damping_alpha=fisher_damping_alpha,
                freeze_top_fraction=fisher_freeze_top_fraction,
                enabled=True,
            )

        # Layer 3: WeightWorkspace + IsolatedSandbox + VespasianEngine
        self.workspace = WeightWorkspace(target_model, fisher_guard=self.fisher_guard)
        self.sandbox = IsolatedSandbox(apply_fisher_guard=True)
        self.vespasian = VespasianEngine(
            self.workspace,
            practice_split=practice_split,
            held_out_split=held_out_split,
            improvement_threshold=improvement_threshold,
            max_kl_drift=max_kl_drift,
            max_changed_fraction=max_changed_fraction,
        )

        # Layer 2: Isolated Tool Registry
        self.tools: ToolRegistry = build_default_tool_registry(
            self.workspace,
            self.vespasian,
            sandbox=self.sandbox,
            calibration_input_ids=calibration_input_ids,
            tokenizer=tokenizer,
            use_toon=use_toon,
        )

    def register_custom_tool(self, tool_def: Any) -> None:
        """Register an additional custom `ToolDefinition` into the isolated Tool Layer."""
        self.tools.register(tool_def)

    def run_editing_cycle(
        self,
        goal: str = "Diagnose and repair weight anomalies in the target model without degrading reference quality.",
        *,
        max_turns: int = 8,
    ) -> HarnessRunResult:
        """Execute the multi-turn LLM-driven Recon -> Localize -> YAML Plan -> Execute -> Verify loop."""
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=SYSTEM_PROMPT_WEIGHTX),
            LLMMessage(role="user", content=f"Target WSP Goal: {goal}"),
        ]

        total_in_toks = 0
        total_out_toks = 0
        turns = 0
        tool_calls_count = 0
        final_summary = ""

        for _ in range(max_turns):
            turns += 1
            resp = self.provider.generate(messages, tools=self.tools)
            total_in_toks += resp.input_tokens
            total_out_toks += resp.output_tokens

            messages.append(
                LLMMessage(
                    role="assistant",
                    content=resp.content,
                    tool_calls=resp.tool_calls,
                )
            )

            if not resp.tool_calls:
                final_summary = resp.content
                break

            for tc in resp.tool_calls:
                tool_calls_count += 1
                tool_out = self.tools.invoke(tc.name, tc.arguments)
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=tool_out,
                        tool_call_id=tc.call_id,
                        name=tc.name,
                    )
                )
            final_summary = resp.content

        # Ensure final model state holds the Vespasian best verified checkpoint
        self.workspace.restore("vespasian_best")

        return HarnessRunResult(
            goal=goal,
            provider_name=self.provider.provider_name,
            model_name=self.provider.model_name,
            turns_executed=turns,
            tool_calls_executed=tool_calls_count,
            baseline_held_out_score=self.vespasian.baseline_score,
            final_best_held_out_score=self.vespasian.best_score,
            net_improvement=self.vespasian.best_score - self.vespasian.baseline_score,
            final_assistant_summary=final_summary,
            diff_summary=self.workspace.diff(),
            vespasian_history=list(self.vespasian.history),
            tool_call_log=list(self.tools.call_log),
            total_input_tokens=total_in_toks,
            total_output_tokens=total_out_toks,
        )
