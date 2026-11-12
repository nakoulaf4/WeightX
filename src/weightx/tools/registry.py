# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX Isolated Tool Layer (`ToolRegistry` & `@weightx_tool`).

Provides the LLM with a strictly typed, JSON-Schema-documented set of tools across
L1 (AutoArch), L2 (Uni-Layer 13-metric profiler & Knowledge Map / KCI), L3 (Tensor
list/describe/slice, BrainSurgery-WX declarative YAML plans, MEMIT-MASS batch editing,
isolated sandbox), and Verification (practice probe & Vespasian held-out commit/rollback).

The LLM never receives direct access to the host filesystem, the `torch.nn.Module`
instance, or the hidden held-out test split. All tool responses are serialized in
compact TOON (Token-Oriented Object Notation) by default to save ~40% context tokens.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from weightx.declarative.plan_engine import execute_yaml_plan
from weightx.execution.sandbox import IsolatedSandbox
from weightx.execution.vespasian import VespasianEngine
from weightx.l1_arch.autoarch import inspect_architecture
from weightx.l2_profile.knowledge_map import run_knowledge_map
from weightx.l2_profile.unilayer import profile_model_layers
from weightx.l3_weights.memit_mass import BatchFactSpec, apply_memit_mass_batch
from weightx.l3_weights.workspace import WeightWorkspace
from weightx.toon_io import dumps_toon


@dataclass
class ToolDefinition:
    """Metadata and callable handler for a single isolated WeightX tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    layer: str = "L3"
    parallel_safe: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def weightx_tool(
    name: str,
    description: str,
    /,
    *,
    layer: str = "L3",
    parallel_safe: bool = False,
    **properties: dict[str, Any],
) -> Callable[[Callable[..., Any]], ToolDefinition]:
    """Decorator to register a function as a structured WeightX tool with JSON Schema."""
    props_clean: dict[str, Any] = {}
    required: list[str] = []
    for k, spec in properties.items():
        spec_copy = dict(spec)
        is_opt = bool(spec_copy.pop("optional", False))
        props_clean[k] = spec_copy
        if not is_opt:
            required.append(k)

    schema = {
        "type": "object",
        "properties": props_clean,
        "required": required,
        "additionalProperties": False,
    }

    def _decorator(fn: Callable[..., Any]) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description=description,
            parameters=schema,
            handler=fn,
            layer=layer,
            parallel_safe=parallel_safe,
        )

    return _decorator


class ToolRegistry:
    """Container of isolated tools exposed to the LLM brain."""

    def __init__(self, *, use_toon: bool = True):
        self._tools: dict[str, ToolDefinition] = {}
        self.use_toon = use_toon
        self.call_log: list[dict[str, Any]] = []

    def register(self, tool_def: ToolDefinition) -> None:
        self._tools[tool_def.name] = tool_def

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Unknown WeightX tool {name!r}. Available: {list(self._tools)}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def to_prompt_catalog(self) -> str:
        """Render a concise tool catalog in TOON format for CLI agents and text-mode LLMs."""
        catalog = [
            {
                "name": t.name,
                "layer": t.layer,
                "description": t.description,
                "required_args": ",".join(t.parameters.get("required", [])),
            }
            for t in self._tools.values()
        ]
        return dumps_toon({"available_tools": catalog})

    def invoke(self, name: str, arguments: dict[str, Any] | str | None = None) -> str:
        """Invoke tool `name` safely and serialize its return value in TOON (or JSON)."""
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return dumps_toon({"status": "error", "tool": name, "error": f"Invalid JSON args: {exc}"})
        else:
            args = dict(arguments or {})

        if name not in self._tools:
            return dumps_toon(
                {
                    "status": "error",
                    "tool": name,
                    "error": f"Unknown tool {name!r}. Available: {self.names()}",
                }
            )

        tool_def = self._tools[name]
        try:
            raw_result = tool_def.handler(**args)
            self.call_log.append({"tool": name, "args": args, "status": "ok"})
            if isinstance(raw_result, str):
                return raw_result
            if self.use_toon:
                return dumps_toon(raw_result)
            return json.dumps(raw_result, indent=2, ensure_ascii=False)
        except Exception as exc:
            tb_short = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.call_log.append({"tool": name, "args": args, "status": "error", "error": tb_short})
            return dumps_toon({"status": "error", "tool": name, "error": tb_short})


def build_default_tool_registry(
    workspace: WeightWorkspace,
    vespasian: VespasianEngine | None = None,
    *,
    sandbox: IsolatedSandbox | None = None,
    calibration_input_ids: torch.Tensor | None = None,
    tokenizer: Any | None = None,
    use_toon: bool = True,
) -> ToolRegistry:
    """Construct the full L1 + L2 + L3 + Verification tool suite bound to `workspace`."""
    reg = ToolRegistry(use_toon=use_toon)
    sbx = sandbox or IsolatedSandbox()

    # 1. L1 AutoArch
    @weightx_tool(
        "l1_autoarch_inspect",
        "Inspect the target model's architecture (family, d_model, layer count, tensor roles q/k/v/o/gate/up/down/norm/embed, and calibration NLL head disambiguation).",
        layer="L1",
        parallel_safe=True,
        resolve_with_calibration={
            "type": "boolean",
            "description": "Whether to resolve head/RoPE ambiguities using calibration data.",
            "optional": True,
        },
        include_tensor_table={
            "type": "boolean",
            "description": "Whether to include the full per-tensor role table.",
            "optional": True,
        },
    )
    def _l1_autoarch(
        resolve_with_calibration: bool = True,
        include_tensor_table: bool = False,
    ) -> dict[str, Any]:
        target = workspace.model if workspace.model is not None else {
            n: workspace.get_readonly(n) for n in workspace.names()
        }
        calib = calibration_input_ids if resolve_with_calibration else None
        rep = inspect_architecture(target, calibration_input_ids=calib)
        return rep.to_dict(include_tensor_table=include_tensor_table)

    # 2. L2 Uni-Layer 13-Metric Profile
    @weightx_tool(
        "l2_unilayer_profile",
        "Profile all model layers across 13 metrics (effective_rank, stable_rank, block_influence, fisher_information, neuron_outlier_ratio, etc.) and identify redundant, bottleneck, and anomalous outlier layers.",
        layer="L2",
        parallel_safe=True,
        outlier_row_threshold={
            "type": "number",
            "description": "Row L2 norm ratio to median above which a neuron is flagged as an outlier (default 2.5).",
            "optional": True,
        },
        include_full_table={
            "type": "boolean",
            "description": "Whether to include the full 13-metric table per layer.",
            "optional": True,
        },
    )
    def _l2_unilayer(
        outlier_row_threshold: float = 2.5,
        include_full_table: bool = True,
    ) -> dict[str, Any]:
        if workspace.model is None:
            return {"error": "l2_unilayer_profile requires a PyTorch nn.Module workspace"}
        workspace.sync_all()
        rep = profile_model_layers(
            workspace.model,
            calibration_input_ids=calibration_input_ids,
            outlier_row_threshold=float(outlier_row_threshold),
        )
        return rep.to_dict(include_full_table=include_full_table)

    # 3. L2 Knowledge Map (Logit Lens + KCI)
    @weightx_tool(
        "l2_knowledge_map",
        "Run Logit Lens and Domain Attention Probing to compute the Knowledge Map and Gini Knowledge Concentration Index (KCI) across factual, syntactic, and reasoning domains.",
        layer="L2",
        parallel_safe=True,
    )
    def _l2_kmap() -> dict[str, Any]:
        if workspace.model is None:
            return {"error": "l2_knowledge_map requires a PyTorch nn.Module workspace"}
        workspace.sync_all()
        rep = run_knowledge_map(workspace.model, tokenizer=tokenizer)
        return rep.to_dict()

    # 4. L3 List Tensors
    @weightx_tool(
        "l3_list_tensors",
        "List parameter tensors in the model matching an optional glob/regex pattern or layer index.",
        layer="L3",
        parallel_safe=True,
        pattern={
            "type": "string",
            "description": "Optional substring, glob, or regex pattern (e.g. '*layers.2*').",
            "optional": True,
        },
        layer_idx={
            "type": "integer",
            "description": "Optional integer layer index to filter by.",
            "optional": True,
        },
    )
    def _l3_list(pattern: str | None = None, layer_idx: int | None = None) -> dict[str, Any]:
        rows = workspace.list_tensors_table(pattern=pattern, layer_idx=layer_idx)
        return {"matched_count": len(rows), "tensors": rows}

    # 5. L3 Describe Tensor
    @weightx_tool(
        "l3_describe_tensor",
        "Compute statistical moments, quantiles, largest/smallest row & column L2 norms (outlier neurons), and leading SVD singular values for one tensor.",
        layer="L3",
        parallel_safe=True,
        name={"type": "string", "description": "Exact tensor name to describe."},
        top_k={
            "type": "integer",
            "description": "Number of largest/smallest row norms and singular values to report (default 8).",
            "optional": True,
        },
    )
    def _l3_describe(name: str, top_k: int = 8) -> dict[str, Any]:
        return workspace.describe_tensor(name, top_k=int(top_k))

    # 6. L3 Read Tensor Slice
    @weightx_tool(
        "l3_read_tensor_slice",
        "Read a multi-dimensional slice of a weight tensor (e.g. '0:8, :16' or '185:186, :') without overflowing context.",
        layer="L3",
        parallel_safe=True,
        name={"type": "string", "description": "Exact tensor name."},
        index={
            "type": "string",
            "description": "NumPy-style slice string such as '0:8, :16'.",
            "optional": True,
        },
        max_elements={
            "type": "integer",
            "description": "Maximum number of scalar elements to return (default 256).",
            "optional": True,
        },
    )
    def _l3_slice(
        name: str,
        index: str | None = None,
        max_elements: int = 256,
    ) -> dict[str, Any]:
        return workspace.read_slice(name, index, max_elements=int(max_elements))

    # 7. L3 Declarative BrainSurgery-WX YAML Plan
    @weightx_tool(
        "l3_apply_yaml_plan",
        "Validate and apply a declarative BrainSurgery-WX YAML plan. The plan MUST include `assert` steps (shape, dtype, norm, finite) and can use transforms like `rescale_outlier_rows_`, `scale_`, `fill_`, `clamp_`, `phlora_`, `closed_form_ridge_`, or `memit_mass_`.",
        layer="L3",
        parallel_safe=False,
        plan_yaml={
            "type": "string",
            "description": "YAML text containing `transforms: [...]` with assertions and weight edits.",
        },
        dry_run={
            "type": "boolean",
            "description": "If true, validates and simulates the plan then rolls back immediately.",
            "optional": True,
        },
    )
    def _l3_yaml_plan(plan_yaml: str, dry_run: bool = False) -> dict[str, Any]:
        rep = execute_yaml_plan(
            workspace,
            plan_yaml,
            dry_run=bool(dry_run),
            require_assertions=True,
            apply_fisher_guard=True,
        )
        return rep.to_dict()

    # 8. L3 MEMIT-MASS Batch Editing
    @weightx_tool(
        "l3_batch_edit_memit",
        "Apply MEMIT-MASS simultaneous batch associative memory editing across one or more 2D layer matrices with IWCKEdit Fisher regularization.",
        layer="L3",
        parallel_safe=False,
        target_tensors={
            "type": "array",
            "items": {"type": "string"},
            "description": "List of 2D weight tensor names to distribute the batch update across.",
        },
        facts={
            "type": "array",
            "items": {"type": "object"},
            "description": "List of `{id, key, value}` fact objects where `key` has length d_in and `value` has length d_out.",
        },
        lambda_cov={
            "type": "number",
            "description": "Covariance regularization strength (default 0.01).",
            "optional": True,
        },
    )
    def _l3_memit(
        target_tensors: list[str],
        facts: list[dict[str, Any]],
        lambda_cov: float = 1e-2,
    ) -> dict[str, Any]:
        specs = [
            BatchFactSpec(
                fact_id=str(f.get("id", idx)),
                key_vector=np.asarray(f["key"], dtype=np.float64),
                target_value_vector=np.asarray(f["value"], dtype=np.float64),
            )
            for idx, f in enumerate(facts)
        ]
        res = apply_memit_mass_batch(
            workspace,
            target_tensors,
            specs,
            lambda_cov=float(lambda_cov),
            apply_fisher_guard=True,
        )
        return res.to_dict()

    # 9. L3 Isolated Python Sandbox
    @weightx_tool(
        "l3_run_sandbox",
        "Run constrained NumPy/math analysis or surgical weight edits in an AST-guarded, process-isolated sandbox (`W` holds writable NumPy arrays matching `tensor_pattern`).",
        layer="L3",
        parallel_safe=False,
        code={
            "type": "string",
            "description": "Python code using `W` (dict of NumPy arrays), `np`, and `math`. No OS/network/file imports allowed.",
        },
        tensor_pattern={
            "type": "string",
            "description": "Glob or substring pattern restricting which tensors are mounted into `W`.",
            "optional": True,
        },
    )
    def _l3_sandbox(code: str, tensor_pattern: str | None = None) -> dict[str, Any]:
        res = sbx.run(workspace, code, tensor_pattern=tensor_pattern)
        return res.to_dict()

    # 10. Practice Probe & Vespasian Verification/Rollback
    @weightx_tool(
        "l3_practice_probe",
        "Run a budgeted evaluation on the visible practice split to check whether a localized edit improved behavior before committing.",
        layer="Execution",
        parallel_safe=False,
    )
    def _l3_practice() -> dict[str, Any]:
        return vespasian.run_practice_probe()

    @weightx_tool(
        "l3_verify_and_commit",
        "Run the Vespasian Verification & Rollback engine on the hidden held-out test suite. Automatically commits if held-out score improves and KL/health contracts pass, or automatically rolls back to the best checkpoint on regression.",
        layer="Verification",
        parallel_safe=False,
        hypothesis={
            "type": "string",
            "description": "Concise explanation of the localized root cause and the weight edit applied.",
            "optional": True,
        },
    )
    def _l3_verify(hypothesis: str = "") -> dict[str, Any]:
        res = vespasian.verify_and_decide(hypothesis=hypothesis)
        return res.to_dict()

    for t in (
        _l1_autoarch,
        _l2_unilayer,
        _l2_kmap,
        _l3_list,
        _l3_describe,
        _l3_slice,
        _l3_yaml_plan,
        _l3_memit,
        _l3_sandbox,
        _l3_practice,
        _l3_verify,
    ):
        reg.register(t)

    return reg
