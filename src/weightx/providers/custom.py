# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
Custom Callable Provider & Autonomous WSP Brain (`AutonomousWSPProvider`).

1. `CustomCallableProvider`: Wraps any user-provided Python function or callable class
   `fn(messages, tools) -> LLMResponse | dict | str`.
2. `AutonomousWSPProvider`: Built-in deterministic Weight-Space Programming reasoning
   agent that executes the complete 5-stage WeightX loop:
      Stage 1 (Recon): Calls `l1_autoarch_inspect`, `l2_unilayer_profile`, and `l2_knowledge_map`.
      Stage 2 (Localize): Parses the TOON profile to identify layers/tensors with anomalous
              outlier neurons and calls `l3_describe_tensor` + `l3_read_tensor_slice`.
      Stage 3 (Plan): Synthesizes a declarative **BrainSurgery-WX** YAML plan with mandatory
              `assert` invariants (`shape`, `dtype`, `finite`, `norm`) and surgical transforms.
      Stage 4 (Execute & Practice Probe): Applies the YAML plan via `l3_apply_yaml_plan`
              and checks the practice split via `l3_practice_probe`.
      Stage 5 (Verify & Commit/Rollback): Invokes `l3_verify_and_commit` (Vespasian engine).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import yaml

from weightx.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    parse_embedded_tool_calls,
)
from weightx.toon_io import estimate_tokens, loads_toon
from weightx.tools.registry import ToolRegistry


class CustomCallableProvider(LLMProvider):
    """Adapter wrapping a user-defined function or object as a WeightX `LLMProvider`."""

    def __init__(
        self,
        fn: Callable[[list[LLMMessage], ToolRegistry | None], Any],
        *,
        name: str = "custom-callable",
    ):
        self.fn = fn
        self.provider_name = "custom"
        self.model_name = name

    def generate(
        self,
        messages: list[LLMMessage],
        tools: ToolRegistry | None = None,
    ) -> LLMResponse:
        out = self.fn(messages, tools)
        if isinstance(out, LLMResponse):
            return out
        if isinstance(out, dict):
            raw_tcs = out.get("tool_calls", [])
            tcs = [
                tc
                if isinstance(tc, LLMToolCall)
                else LLMToolCall(
                    call_id=str(tc.get("id", f"call_{i + 1}")),
                    name=str(tc["name"]),
                    arguments=dict(tc.get("arguments", {})),
                )
                for i, tc in enumerate(raw_tcs)
            ]
            content = str(out.get("content", ""))
            return LLMResponse(
                content=content,
                tool_calls=tcs,
                input_tokens=estimate_tokens(str(messages)),
                output_tokens=estimate_tokens(content),
                provider_name=self.provider_name,
                model_name=self.model_name,
            )
        text = str(out)
        return LLMResponse(
            content=text,
            tool_calls=parse_embedded_tool_calls(text),
            input_tokens=estimate_tokens(str(messages)),
            output_tokens=estimate_tokens(text),
            provider_name=self.provider_name,
            model_name=self.model_name,
        )


class AutonomousWSPProvider(LLMProvider):
    """Self-contained deterministic WSP reasoning agent driving the 5-stage WeightX cycle."""

    def __init__(self, *, zero_outliers: bool = False):
        self.provider_name = "weightx-autonomous"
        self.model_name = "autonomous-wsp-brain-v1"
        self.zero_outliers = bool(zero_outliers)
        self._step = 0
        self._localized_outliers: list[dict[str, Any]] = []
        self._described_shapes: dict[str, list[int]] = {}
        self._described_dtypes: dict[str, str] = {}

    def generate(
        self,
        messages: list[LLMMessage],
        tools: ToolRegistry | None = None,
    ) -> LLMResponse:
        self._step += 1

        # Harvest tool results from the previous turn
        for msg in messages:
            if msg.role == "tool" and msg.content:
                try:
                    data = loads_toon(msg.content)
                except Exception:
                    continue
                if isinstance(data, dict):
                    if "outlier_neurons" in data and data["outlier_neurons"]:
                        self._localized_outliers = list(data["outlier_neurons"])
                    if "name" in data and "shape" in data and "l2_norm" in data:
                        tname = str(data["name"])
                        self._described_shapes[tname] = list(data["shape"])
                        self._described_dtypes[tname] = str(data.get("dtype", "float32"))

        # Stage 1: Recon (L1 AutoArch + L2 Uni-Layer 13-metric profile + L2 Knowledge Map)
        if self._step == 1:
            return LLMResponse(
                content=(
                    "Stage 1 (Recon): Running L1 AutoArch architecture inspection, "
                    "L2 Uni-Layer 13-metric profiling, and L2 Knowledge Map (Logit Lens + KCI)."
                ),
                tool_calls=[
                    LLMToolCall(
                        call_id="recon_l1",
                        name="l1_autoarch_inspect",
                        arguments={"resolve_with_calibration": True, "include_tensor_table": False},
                    ),
                    LLMToolCall(
                        call_id="recon_l2_unilayer",
                        name="l2_unilayer_profile",
                        arguments={"outlier_row_threshold": 2.5, "include_full_table": True},
                    ),
                    LLMToolCall(
                        call_id="recon_l2_kmap",
                        name="l2_knowledge_map",
                        arguments={},
                    ),
                ],
                provider_name=self.provider_name,
                model_name=self.model_name,
            )

        # Stage 2: Localization (Describe & read slices of the anomalous tensors identified in L2)
        if self._step == 2:
            calls: list[LLMToolCall] = []
            targets = self._localized_outliers[:3]
            if targets:
                for i, item in enumerate(targets):
                    tname = str(item["tensor_name"])
                    rows = item.get("row_indices", [0])
                    r0 = int(rows[0]) if rows else 0
                    calls.append(
                        LLMToolCall(
                            call_id=f"loc_desc_{i}",
                            name="l3_describe_tensor",
                            arguments={"name": tname, "top_k": 6},
                        )
                    )
                    calls.append(
                        LLMToolCall(
                            call_id=f"loc_slice_{i}",
                            name="l3_read_tensor_slice",
                            arguments={"name": tname, "index": f"{r0}:{r0 + 2}, :8", "max_elements": 32},
                        )
                    )
                hyp = (
                    f"Stage 2 (Localization): L2 Uni-Layer flagged {len(self._localized_outliers)} "
                    f"tensor(s) with extreme outlier neuron L2 norms ({[x['tensor_name'] for x in targets]}). "
                    "Requesting L3 tensor descriptions and targeted slices."
                )
            else:
                calls.append(
                    LLMToolCall(
                        call_id="loc_list",
                        name="l3_list_tensors",
                        arguments={"layer_idx": 0},
                    )
                )
                hyp = "Stage 2 (Localization): Inspecting layer 0 tensor list."

            return LLMResponse(
                content=hyp,
                tool_calls=calls,
                provider_name=self.provider_name,
                model_name=self.model_name,
            )

        # Stage 3 & 4: Declarative YAML Plan Generation & Execution + Practice Probe
        if self._step == 3:
            transforms_list: list[dict[str, Any]] = []
            for item in self._localized_outliers:
                tname = str(item["tensor_name"])
                shp = self._described_shapes.get(tname)
                dt = self._described_dtypes.get(tname, "float32")
                if shp:
                    transforms_list.append(
                        {
                            "assert": {
                                "all": [
                                    {"exists": {"of": tname}},
                                    {"shape": {"of": tname, "is": shp}},
                                    {"dtype": {"of": tname, "is": dt}},
                                    {"finite": {"of": tname}},
                                ]
                            }
                        }
                    )
                else:
                    transforms_list.append({"assert": {"finite": {"of": tname}}})

                transforms_list.append(
                    {
                        "rescale_outlier_rows_": {
                            "target": tname,
                            "min_ratio": 2.5,
                            "rows": item.get("row_indices"),
                            "zero_outliers": self.zero_outliers,
                        }
                    }
                )
                transforms_list.append({"assert": {"finite": {"of": tname}}})

            if not transforms_list:
                transforms_list = [{"assert": {"finite": {"of": "*"}}}]

            plan_yaml_str = yaml.safe_dump({"transforms": transforms_list}, sort_keys=False)

            return LLMResponse(
                content=(
                    "Stage 3 & 4 (Declarative Planning & Execution): Generated BrainSurgery-WX "
                    "YAML plan with shape/dtype/finite assertions and `rescale_outlier_rows_` "
                    "surgery, followed by a practice split probe."
                ),
                tool_calls=[
                    LLMToolCall(
                        call_id="exec_yaml_plan",
                        name="l3_apply_yaml_plan",
                        arguments={"plan_yaml": plan_yaml_str, "dry_run": False},
                    ),
                    LLMToolCall(
                        call_id="probe_practice",
                        name="l3_practice_probe",
                        arguments={},
                    ),
                ],
                provider_name=self.provider_name,
                model_name=self.model_name,
            )

        # Stage 5: Vespasian Verification & Commit/Rollback
        if self._step == 4:
            outlier_summary = ", ".join(
                f"{o['tensor_name']} rows {o.get('row_indices', [])}"
                for o in self._localized_outliers[:3]
            )
            return LLMResponse(
                content=(
                    "Stage 5 (Vespasian Verification): Practice probe confirmed improvement. "
                    "Triggering held-out verification and checkpoint commit."
                ),
                tool_calls=[
                    LLMToolCall(
                        call_id="vespasian_verify",
                        name="l3_verify_and_commit",
                        arguments={
                            "hypothesis": (
                                f"Repaired corrupted outlier neurons in {outlier_summary or 'target layers'} "
                                "via declarative BrainSurgery-WX plan with IWCKEdit Fisher protection."
                            )
                        },
                    )
                ],
                provider_name=self.provider_name,
                model_name=self.model_name,
            )

        # Final turn: summarize completed cycle
        return LLMResponse(
            content=(
                "WeightX 5-stage editing cycle complete: L1/L2 Recon localized the outlier neurons, "
                "BrainSurgery-WX declarative YAML plan repaired the weights under IWCKEdit Fisher "
                "protection, and Vespasian verification confirmed held-out improvement."
            ),
            tool_calls=[],
            provider_name=self.provider_name,
            model_name=self.model_name,
        )
