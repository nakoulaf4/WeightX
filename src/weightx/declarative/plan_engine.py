# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
BrainSurgery-WX: Declarative YAML Plan Compiler, Assertion Validator, and Executor.

Supports the BrainSurgery YAML plan format (`transforms: [...]` with `alias::pattern::[slice]`
references and `assert` expressions `exists`, `count`, `shape`, `dimensions`, `dtype`,
`equal`, `iszero`, `all`, `any`, `not`) and extends it for Weight-Space Programming with:
  - Norm and numerical invariant assertions (`norm`, `finite`, `rel_delta`),
  - Mandatory invariant policy (`require_assertions=True` by default for LLM plans),
  - Zero-filesystem lockdown (blocks `load`, `save`, and external file paths in LLM plans),
  - First-class WSP & batch transforms (`rescale_outlier_rows_`, `phlora_`,
    `prune_magnitude_`, `rank_one_update_`, `closed_form_ridge_`, `memit_mass_`),
  - Automatic IWCKEdit Fisher protection and atomic rollback on any assertion failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from weightx.l3_weights.memit_mass import (
    BatchFactSpec,
    apply_closed_form_ridge,
    apply_memit_mass_batch,
)
from weightx.l3_weights.workspace import WeightWorkspace, parse_safe_slice
from weightx.toon_io import dumps_toon


class PlanValidationError(ValueError):
    """Raised when a declarative YAML surgery plan fails syntax, policy, or assertion checks."""


FORBIDDEN_FILESYSTEM_OPS = frozenset({"load", "save", "execute"})

MUTATION_OPS = frozenset(
    {
        "copy",
        "assign",
        "scale",
        "scale_",
        "fill",
        "fill_",
        "zeroes",
        "ones",
        "add",
        "add_",
        "subtract",
        "subtract_",
        "multiply",
        "multiply_",
        "clamp",
        "clamp_",
        "phlora_",
        "svd_low_rank_",
        "spectral_norm_cap_",
        "project_out_direction_",
        "abliterate_direction_",
        "lerp_",
        "slerp_",
        "ties_trim_",
        "prune_magnitude_",
        "rescale_outlier_rows_",
        "replace_nonfinite_",
        "rank_one_update_",
        "closed_form_ridge_",
        "memit_mass_",
    }
)

READONLY_OPS = frozenset({"assert", "dump", "diff"})

ALL_SUPPORTED_OPS = READONLY_OPS | MUTATION_OPS


@dataclass
class StepExecutionResult:
    """Execution telemetry for a single step in a YAML plan."""

    step_index: int
    op_name: str
    matched_tensors: int
    status: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_index,
            "op": self.op_name,
            "matched": self.matched_tensors,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class PlanExecutionReport:
    """Complete execution report for a BrainSurgery-WX YAML plan."""

    success: bool
    dry_run: bool
    assertions_verified: int
    mutations_executed: int
    steps: list[StepExecutionResult] = field(default_factory=list)
    diff_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "dry_run": self.dry_run,
            "assertions_verified": self.assertions_verified,
            "mutations_executed": self.mutations_executed,
            "error": self.error,
            "steps": [s.to_dict() for s in self.steps],
            "diff_summary": self.diff_summary,
        }

    def to_toon(self) -> str:
        return dumps_toon(self.to_dict())


def parse_tensor_ref(ref_str: str) -> tuple[str | None, str, tuple[Any, ...]]:
    """Parse a BrainSurgery tensor reference `'alias::pattern::[slice]'` or `'pattern::[slice]'`.

    Returns `(alias, pattern, slice_tuple)`.
    """
    if not isinstance(ref_str, str) or not ref_str.strip():
        raise PlanValidationError(f"Tensor reference must be a non-empty string, got {ref_str!r}")
    parts = ref_str.strip().split("::")
    alias: str | None = None
    slice_tup: tuple[Any, ...] = ()

    if len(parts) == 1:
        pattern = parts[0]
    elif len(parts) == 2:
        if parts[1].strip().startswith("["):
            pattern = parts[0]
            slice_tup = parse_safe_slice(parts[1].strip())
        else:
            alias = parts[0]
            pattern = parts[1]
    elif len(parts) == 3:
        alias = parts[0]
        pattern = parts[1]
        slice_tup = parse_safe_slice(parts[2].strip())
    else:
        raise PlanValidationError(f"Invalid tensor reference format: {ref_str!r}")

    # Also support inline bracket suffix like "layer.0.weight[0:4, :]"
    if not slice_tup and "[" in pattern and pattern.endswith("]"):
        b_idx = pattern.index("[")
        candidate_slice = pattern[b_idx:]
        try:
            slice_tup = parse_safe_slice(candidate_slice)
            pattern = pattern[:b_idx]
        except ValueError:
            pass

    return alias, pattern, slice_tup


def _resolve_targets(workspace: WeightWorkspace, ref_str: str) -> tuple[list[str], tuple[Any, ...]]:
    _alias, pattern, slice_tup = parse_tensor_ref(ref_str)
    matched = workspace.names(pattern=pattern)
    if not matched:
        raise PlanValidationError(f"Reference {ref_str!r} matched 0 tensors in workspace")
    return matched, slice_tup


def _compare_scalar(actual: float, spec: dict[str, Any], label: str) -> None:
    eps = float(spec.get("eps", 1e-5))
    if "is" in spec:
        expected = float(spec["is"])
        if abs(actual - expected) > eps:
            raise PlanValidationError(f"Assertion failed ({label}): {actual:.6g} != {expected:.6g} (eps={eps})")
    if "le" in spec and not (actual <= float(spec["le"]) + eps):
        raise PlanValidationError(f"Assertion failed ({label}): {actual:.6g} > le={spec['le']}")
    if "lt" in spec and not (actual < float(spec["lt"])):
        raise PlanValidationError(f"Assertion failed ({label}): {actual:.6g} >= lt={spec['lt']}")
    if "ge" in spec and not (actual + eps >= float(spec["ge"])):
        raise PlanValidationError(f"Assertion failed ({label}): {actual:.6g} < ge={spec['ge']}")
    if "gt" in spec and not (actual > float(spec["gt"])):
        raise PlanValidationError(f"Assertion failed ({label}): {actual:.6g} <= gt={spec['gt']}")


def evaluate_assertion(workspace: WeightWorkspace, expr: dict[str, Any]) -> tuple[int, str]:
    """Evaluate a BrainSurgery-WX assertion expression against `workspace`.

    Supported assertion keys:
      - `exists`: `str` or `{"of": str}`
      - `count`: `{"of": str, "is": int}`
      - `shape`: `{"of": str, "is": [int, ...]}`
      - `dimensions`: `{"of": str, "is"|"ge"|"le"|"gt"|"lt": int}`
      - `dtype`: `{"of": str, "is": str}`
      - `equal`: `{"left": str, "right": str, "eps": float}`
      - `iszero`: `{"of": str, "eps": float}`
      - `norm`: `{"of": str, "ord": "l2"|"fro"|"max"|"spectral", "is"|"le"|"ge"|"lt"|"gt": float}`
      - `finite`: `str` or `{"of": str}`
      - `rel_delta`: `{"of": str, "le": float}`
      - `all`: `[expr, ...]`
      - `any`: `[expr, ...]`
      - `not`: `expr`
    """
    if not isinstance(expr, dict) or len(expr) != 1:
        raise PlanValidationError(f"Each assert expression must be a single-key mapping, got {expr!r}")

    op, payload = next(iter(expr.items()))

    if op == "all":
        if not isinstance(payload, list) or not payload:
            raise PlanValidationError("assert.all requires a non-empty list of assertions")
        total = 0
        details = []
        for sub in payload:
            c, d = evaluate_assertion(workspace, sub)
            total += c
            details.append(d)
        return total, f"all({len(payload)} checks passed)"

    if op == "any":
        if not isinstance(payload, list) or not payload:
            raise PlanValidationError("assert.any requires a non-empty list of assertions")
        errors = []
        for sub in payload:
            try:
                c, d = evaluate_assertion(workspace, sub)
                return c, f"any({d})"
            except PlanValidationError as exc:
                errors.append(str(exc))
        raise PlanValidationError(f"assert.any failed all branches: {'; '.join(errors)}")

    if op == "not":
        if not isinstance(payload, dict):
            raise PlanValidationError("assert.not requires a nested assertion mapping")
        try:
            evaluate_assertion(workspace, payload)
        except PlanValidationError:
            return 1, "not(inner failed as expected)"
        raise PlanValidationError(f"assert.not failed: inner assertion {payload!r} succeeded")

    if op == "exists":
        ref = payload if isinstance(payload, str) else payload.get("of", "")
        matched, _ = _resolve_targets(workspace, ref)
        return len(matched), f"exists({ref} -> {len(matched)} tensors)"

    if op == "count":
        ref = payload["of"]
        _alias, pattern, _ = parse_tensor_ref(ref)
        matched = workspace.names(pattern=pattern)
        _compare_scalar(float(len(matched)), payload, f"count({ref})")
        return len(matched), f"count({ref}) == {len(matched)}"

    if op == "shape":
        ref = payload["of"]
        expected_shape = tuple(int(x) for x in payload["is"])
        matched, slice_tup = _resolve_targets(workspace, ref)
        for name in matched:
            arr = workspace.get_readonly(name)
            sub = arr[slice_tup] if slice_tup else arr
            if tuple(sub.shape) != expected_shape:
                raise PlanValidationError(
                    f"Shape assertion failed for {name!r}: expected {list(expected_shape)}, got {list(sub.shape)}"
                )
        return len(matched), f"shape({ref}) == {list(expected_shape)}"

    if op == "dimensions":
        ref = payload["of"]
        matched, slice_tup = _resolve_targets(workspace, ref)
        for name in matched:
            arr = workspace.get_readonly(name)
            sub = arr[slice_tup] if slice_tup else arr
            _compare_scalar(float(sub.ndim), payload, f"dimensions({name})")
        return len(matched), f"dimensions({ref}) verified"

    if op == "dtype":
        ref = payload["of"]
        expected = str(payload["is"]).replace("torch.", "").lower()
        matched, _ = _resolve_targets(workspace, ref)
        for name in matched:
            actual = workspace.dtype(name).replace("torch.", "").lower()
            if actual != expected and not (expected in ("float", "float32", "fp32") and actual in ("float", "float32")):
                raise PlanValidationError(
                    f"Dtype assertion failed for {name!r}: expected {expected!r}, got {actual!r}"
                )
        return len(matched), f"dtype({ref}) == {expected}"

    if op == "equal":
        left_names, left_slice = _resolve_targets(workspace, payload["left"])
        right_names, right_slice = _resolve_targets(workspace, payload["right"])
        eps = float(payload.get("eps", 1e-5))
        if len(left_names) != len(right_names):
            raise PlanValidationError(
                f"assert.equal matched {len(left_names)} left vs {len(right_names)} right tensors"
            )
        for ln, rn in zip(left_names, right_names):
            la = workspace.get_readonly(ln)
            ra = workspace.get_readonly(rn)
            lsub = la[left_slice] if left_slice else la
            rsub = ra[right_slice] if right_slice else ra
            if lsub.shape != rsub.shape:
                raise PlanValidationError(f"assert.equal shape mismatch: {ln} {lsub.shape} vs {rn} {rsub.shape}")
            max_diff = float(np.max(np.abs(lsub.astype(np.float64) - rsub.astype(np.float64))))
            if max_diff > eps:
                raise PlanValidationError(
                    f"assert.equal failed for {ln} vs {rn}: max_abs_diff={max_diff:.6g} > eps={eps}"
                )
        return len(left_names), f"equal({payload['left']}, {payload['right']})"

    if op == "iszero":
        ref = payload if isinstance(payload, str) else payload["of"]
        eps = float(payload.get("eps", 1e-6)) if isinstance(payload, dict) else 1e-6
        matched, slice_tup = _resolve_targets(workspace, ref)
        for name in matched:
            arr = workspace.get_readonly(name)
            sub = arr[slice_tup] if slice_tup else arr
            mx = float(np.max(np.abs(sub))) if sub.size else 0.0
            if mx > eps:
                raise PlanValidationError(f"assert.iszero failed for {name!r}: max_abs={mx:.6g} > {eps}")
        return len(matched), f"iszero({ref})"

    if op == "finite":
        ref = payload if isinstance(payload, str) else payload.get("of", "*")
        matched, slice_tup = _resolve_targets(workspace, ref)
        for name in matched:
            arr = workspace.get_readonly(name)
            sub = arr[slice_tup] if slice_tup else arr
            bad = int(np.count_nonzero(~np.isfinite(sub)))
            if bad > 0:
                raise PlanValidationError(f"assert.finite failed for {name!r}: {bad} NaN/Inf elements found")
        return len(matched), f"finite({ref})"

    if op == "norm":
        ref = payload["of"]
        ord_kind = str(payload.get("ord", "l2")).lower()
        matched, slice_tup = _resolve_targets(workspace, ref)
        last_val = 0.0
        for name in matched:
            arr = workspace.get_readonly(name)
            sub = (arr[slice_tup] if slice_tup else arr).astype(np.float64)
            if ord_kind in ("l2", "fro"):
                val = float(np.linalg.norm(sub.reshape(-1)))
            elif ord_kind in ("max", "inf"):
                val = float(np.max(np.abs(sub))) if sub.size else 0.0
            elif ord_kind == "l1":
                val = float(np.sum(np.abs(sub)))
            elif ord_kind == "spectral":
                m2d = sub.reshape(sub.shape[0], -1)
                val = float(np.linalg.svd(m2d, compute_uv=False)[0])
            else:
                raise PlanValidationError(f"Unsupported norm ord={ord_kind!r}")
            _compare_scalar(val, payload, f"norm({name}, ord={ord_kind})")
            last_val = val
        return len(matched), f"norm({ref}, {ord_kind}={last_val:.4g})"

    if op == "rel_delta":
        ref = payload["of"]
        matched, _ = _resolve_targets(workspace, ref)
        for name in matched:
            cur = workspace.get_readonly(name).astype(np.float64)
            orig = workspace.get_original(name).astype(np.float64)
            rel = float(np.linalg.norm(cur - orig) / max(np.linalg.norm(orig), 1e-12))
            _compare_scalar(rel, payload, f"rel_delta({name})")
        return len(matched), f"rel_delta({ref}) verified"

    raise PlanValidationError(f"Unknown assertion operator: {op!r}")


def compile_and_validate_plan(
    plan_yaml_or_dict: str | dict[str, Any] | list[dict[str, Any]],
    *,
    require_assertions: bool = True,
    allow_filesystem_ops: bool = False,
) -> list[tuple[str, Any]]:
    """Parse and validate a BrainSurgery-WX YAML plan into a list of `(op_name, payload)` steps."""
    if isinstance(plan_yaml_or_dict, str):
        try:
            raw = yaml.safe_load(plan_yaml_or_dict)
        except yaml.YAMLError as exc:
            raise PlanValidationError(f"Invalid YAML syntax in surgery plan: {exc}") from exc
    else:
        raw = plan_yaml_or_dict

    if isinstance(raw, list):
        raw_transforms = raw
    elif isinstance(raw, dict):
        if not allow_filesystem_ops and ("output" in raw and raw["output"] is not None):
            raise PlanValidationError(
                "Security Policy Violation: LLM surgery plans cannot specify filesystem 'output' paths"
            )
        raw_transforms = raw.get("transforms")
    else:
        raise PlanValidationError("YAML plan must be a mapping with 'transforms' or a list of transforms")

    if not isinstance(raw_transforms, list) or not raw_transforms:
        raise PlanValidationError("YAML plan must contain a non-empty 'transforms' list")

    steps: list[tuple[str, Any]] = []
    assertion_count = 0
    mutation_count = 0

    for idx, entry in enumerate(raw_transforms):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise PlanValidationError(
                f"Transform #{idx} must be a single-key mapping (e.g. `- assert: ...`), got {entry!r}"
            )
        op_name, payload = next(iter(entry.items()))
        if not allow_filesystem_ops and op_name in FORBIDDEN_FILESYSTEM_OPS:
            raise PlanValidationError(
                f"Security Policy Violation: operation {op_name!r} at step #{idx} is forbidden in isolated mode"
            )
        if op_name not in ALL_SUPPORTED_OPS:
            raise PlanValidationError(
                f"Unknown transform operation {op_name!r} at step #{idx}. "
                f"Supported: {sorted(ALL_SUPPORTED_OPS)}"
            )
        if op_name == "assert":
            assertion_count += 1
        elif op_name in MUTATION_OPS:
            mutation_count += 1
        steps.append((op_name, payload))

    if require_assertions and mutation_count > 0 and assertion_count == 0:
        raise PlanValidationError(
            "Invariant Policy Violation: every mutation plan must contain at least one `assert` step "
            "(validating shape, dtype, norm, or finite invariants)."
        )

    return steps


def _apply_mutation_step(
    workspace: WeightWorkspace,
    op_name: str,
    payload: dict[str, Any],
    *,
    apply_fisher_guard: bool = True,
) -> tuple[int, str]:
    """Execute a single validated mutation transform against `workspace`."""
    if not isinstance(payload, dict):
        raise PlanValidationError(f"Transform {op_name!r} payload must be a mapping, got {payload!r}")

    if op_name in ("scale", "scale_"):
        ref = payload.get("target") or payload.get("from") or payload.get("of")
        dest_ref = payload.get("to") if op_name == "scale" else ref
        factor = float(payload.get("by", payload.get("factor", 1.0)))
        src_names, src_slice = _resolve_targets(workspace, str(ref))
        dst_names, dst_slice = _resolve_targets(workspace, str(dest_ref or ref))
        for s_name, d_name in zip(src_names, dst_names):
            arr = workspace.get_readonly(s_name).copy()
            out = workspace.get_readonly(d_name).copy()
            s_sub = arr[src_slice] if src_slice else arr
            if dst_slice:
                out[dst_slice] = s_sub * factor
            else:
                out = s_sub * factor
            workspace.set_tensor(d_name, out, apply_fisher_guard=apply_fisher_guard)
        return len(dst_names), f"{op_name}(by={factor}) on {len(dst_names)} tensor(s)"

    if op_name in ("fill", "fill_", "zeroes", "ones"):
        ref = payload.get("target") or payload.get("to") or payload.get("of")
        matched, slice_tup = _resolve_targets(workspace, str(ref))
        if op_name == "zeroes":
            val = 0.0
        elif op_name == "ones":
            val = 1.0
        else:
            val = float(payload.get("value", 0.0))
        for name in matched:
            out = workspace.get_readonly(name).copy()
            if slice_tup:
                out[slice_tup] = val
            else:
                out[...] = val
            workspace.set_tensor(name, out, apply_fisher_guard=apply_fisher_guard)
        return len(matched), f"{op_name}(value={val}) on {len(matched)} tensor(s)"

    if op_name in ("clamp", "clamp_"):
        ref = payload.get("target") or payload.get("from") or payload.get("of")
        lo = payload.get("min")
        hi = payload.get("max")
        matched, slice_tup = _resolve_targets(workspace, str(ref))
        for name in matched:
            out = workspace.get_readonly(name).copy()
            sub = out[slice_tup] if slice_tup else out
            clamped = np.clip(
                sub,
                float(lo) if lo is not None else None,
                float(hi) if hi is not None else None,
            )
            if slice_tup:
                out[slice_tup] = clamped
            else:
                out = clamped
            workspace.set_tensor(name, out, apply_fisher_guard=apply_fisher_guard)
        return len(matched), f"{op_name}(min={lo}, max={hi}) on {len(matched)} tensor(s)"

    if op_name in ("add", "add_", "subtract", "subtract_", "multiply", "multiply_"):
        ref = payload.get("target") or payload.get("to")
        matched, slice_tup = _resolve_targets(workspace, str(ref))
        scalar_val = payload.get("value", payload.get("by"))
        from_ref = payload.get("from") or payload.get("from_b")
        for name in matched:
            out = workspace.get_readonly(name).copy()
            lhs = out[slice_tup] if slice_tup else out
            if scalar_val is not None and not isinstance(scalar_val, str):
                rhs = np.asarray(scalar_val, dtype=np.float32)
            elif from_ref is not None:
                f_names, f_slice = _resolve_targets(workspace, str(from_ref))
                f_arr = workspace.get_readonly(f_names[0])
                rhs = f_arr[f_slice] if f_slice else f_arr
            else:
                raise PlanValidationError(f"{op_name} requires 'value' or 'from'")

            if op_name.startswith("add"):
                res = lhs + rhs
            elif op_name.startswith("subtract"):
                res = lhs - rhs
            else:
                res = lhs * rhs

            if slice_tup:
                out[slice_tup] = res
            else:
                out = res
            workspace.set_tensor(name, out, apply_fisher_guard=apply_fisher_guard)
        return len(matched), f"{op_name} on {len(matched)} tensor(s)"

    if op_name in ("copy", "assign"):
        src_names, src_slice = _resolve_targets(workspace, str(payload["from"]))
        dst_names, dst_slice = _resolve_targets(workspace, str(payload["to"]))
        for s_name, d_name in zip(src_names, dst_names):
            s_arr = workspace.get_readonly(s_name)
            s_sub = s_arr[src_slice] if src_slice else s_arr
            out = workspace.get_readonly(d_name).copy()
            if dst_slice:
                out[dst_slice] = s_sub
            else:
                out[...] = s_sub
            workspace.set_tensor(d_name, out, apply_fisher_guard=apply_fisher_guard)
        return len(dst_names), f"{op_name}({payload['from']} -> {payload['to']})"

    if op_name in ("phlora_", "svd_low_rank_"):
        ref = payload.get("target") or payload.get("of")
        rank = payload.get("rank")
        energy_fraction = payload.get("energy_fraction")
        matched, _ = _resolve_targets(workspace, str(ref))
        for name in matched:
            arr = workspace.get_readonly(name).astype(np.float64)
            if arr.ndim != 2:
                raise PlanValidationError(f"{op_name} requires 2D tensor, got {name} with shape {arr.shape}")
            u, s, vh = np.linalg.svd(arr, full_matrices=False)
            if energy_fraction is not None:
                ef = float(energy_fraction)
                cum_e = np.cumsum(s**2) / max(float(np.sum(s**2)), 1e-12)
                eff_rank = int(np.searchsorted(cum_e, ef) + 1)
                eff_rank = max(1, min(eff_rank, len(s)))
            else:
                eff_rank = int(rank) if rank is not None else max(1, len(s) // 2)
            if eff_rank < min(arr.shape):
                approx = (u[:, :eff_rank] * s[:eff_rank]) @ vh[:eff_rank, :]
                workspace.set_tensor(
                    name,
                    approx.astype(np.float32),
                    apply_fisher_guard=apply_fisher_guard,
                    allow_outlier_repair=True,
                )
        return len(matched), f"{op_name}(rank={rank or energy_fraction}) on {len(matched)} tensor(s)"

    if op_name == "spectral_norm_cap_":
        ref = payload.get("target") or payload.get("of")
        max_sigma = payload.get("max_sigma")
        max_median_ratio = float(payload.get("max_median_ratio", 3.0))
        matched, _ = _resolve_targets(workspace, str(ref))
        capped_count = 0
        for name in matched:
            arr = workspace.get_readonly(name).astype(np.float64)
            if arr.ndim != 2:
                continue
            u, s, vh = np.linalg.svd(arr, full_matrices=False)
            med_s = float(np.median(s)) if s.size else 1.0
            cap = float(max_sigma) if max_sigma is not None else max(med_s * max_median_ratio, 1e-6)
            if float(s[0]) > cap:
                s_capped = np.minimum(s, cap)
                reconstructed = (u * s_capped) @ vh
                workspace.set_tensor(
                    name,
                    reconstructed.astype(np.float32),
                    apply_fisher_guard=apply_fisher_guard,
                    allow_outlier_repair=True,
                )
                capped_count += 1
        return len(matched), f"spectral_norm_cap_(capped {capped_count}/{len(matched)} tensor(s))"

    if op_name in ("project_out_direction_", "abliterate_direction_"):
        ref = payload.get("target") or payload.get("of")
        alpha = float(payload.get("alpha", 1.0))
        direction_raw = payload.get("direction")
        matched, _ = _resolve_targets(workspace, str(ref))
        for name in matched:
            arr = workspace.get_readonly(name).astype(np.float64)
            if arr.ndim != 2:
                raise PlanValidationError(f"{op_name} requires 2D tensor, got {name}")
            if direction_raw is not None:
                d = np.asarray(direction_raw, dtype=np.float64).reshape(-1)
            else:
                # Default: extract top principal output direction u[:, 0]
                u, _s, _vh = np.linalg.svd(arr, full_matrices=False)
                d = u[:, 0]
            norm_d = float(np.linalg.norm(d))
            if norm_d > 1e-12:
                d_hat = d / norm_d
                if d_hat.shape[0] == arr.shape[0]:
                    proj = np.outer(d_hat, d_hat @ arr)
                elif d_hat.shape[0] == arr.shape[1]:
                    proj = np.outer(arr @ d_hat, d_hat)
                else:
                    raise PlanValidationError(
                        f"{op_name} direction dim {d_hat.shape[0]} does not match {name} shape {arr.shape}"
                    )
                updated = (arr - alpha * proj).astype(np.float32)
                workspace.set_tensor(name, updated, apply_fisher_guard=apply_fisher_guard)
        return len(matched), f"{op_name}(alpha={alpha}) on {len(matched)} tensor(s)"

    if op_name in ("lerp_", "slerp_"):
        ref = payload.get("target") or payload.get("to")
        other_ref = payload.get("from")
        t_val = float(payload.get("t", 0.5))
        matched, _ = _resolve_targets(workspace, str(ref))
        other_names, _ = _resolve_targets(workspace, str(other_ref)) if other_ref else (matched, ())
        for name, o_name in zip(matched, other_names):
            a = workspace.get_readonly(name).astype(np.float64)
            b = (
                workspace.get_readonly(o_name).astype(np.float64)
                if other_ref
                else workspace.get_original(name).astype(np.float64)
            )
            if op_name == "lerp_":
                res = (1.0 - t_val) * a + t_val * b
            else:
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na < 1e-12 or nb < 1e-12:
                    res = (1.0 - t_val) * a + t_val * b
                else:
                    cos_omega = float(np.clip(np.sum((a / na) * (b / nb)), -1.0, 1.0))
                    omega = float(np.arccos(cos_omega))
                    if abs(omega) < 1e-6:
                        res = (1.0 - t_val) * a + t_val * b
                    else:
                        so = np.sin(omega)
                        res = (np.sin((1.0 - t_val) * omega) / so) * a + (np.sin(t_val * omega) / so) * b
            workspace.set_tensor(name, res.astype(np.float32), apply_fisher_guard=apply_fisher_guard)
        return len(matched), f"{op_name}(t={t_val}) on {len(matched)} tensor(s)"

    if op_name == "ties_trim_":
        ref = payload.get("target") or payload.get("of")
        keep_top_fraction = float(payload.get("keep_top_fraction", 0.2))
        matched, _ = _resolve_targets(workspace, str(ref))
        for name in matched:
            cur = workspace.get_readonly(name).astype(np.float64)
            orig = workspace.get_original(name).astype(np.float64)
            delta = cur - orig
            flat = delta.reshape(-1)
            k_keep = max(1, int(keep_top_fraction * flat.size))
            if k_keep < flat.size:
                thresh = float(np.partition(np.abs(flat), -k_keep)[-k_keep])
                mask = np.abs(delta) >= thresh
                trimmed = orig + delta * mask
                workspace.set_tensor(name, trimmed.astype(np.float32), apply_fisher_guard=apply_fisher_guard)
        return len(matched), f"ties_trim_(keep_top_fraction={keep_top_fraction}) on {len(matched)} tensor(s)"

    if op_name == "prune_magnitude_":
        ref = payload.get("target") or payload.get("of")
        fraction = float(payload["fraction"])
        matched, slice_tup = _resolve_targets(workspace, str(ref))
        for name in matched:
            out = workspace.get_readonly(name).copy()
            region = (out[slice_tup] if slice_tup else out).reshape(-1).copy()
            k = int(fraction * region.size)
            if k > 0:
                order = np.argsort(np.abs(region), kind="stable")[:k]
                region[order] = 0.0
                if slice_tup:
                    out[slice_tup] = region.reshape(out[slice_tup].shape)
                else:
                    out = region.reshape(out.shape)
                workspace.set_tensor(name, out, apply_fisher_guard=apply_fisher_guard)
        return len(matched), f"prune_magnitude_(fraction={fraction}) on {len(matched)} tensor(s)"

    if op_name == "rescale_outlier_rows_":
        ref = payload.get("target") or payload.get("of")
        min_ratio = float(payload.get("min_ratio", 2.5))
        explicit_rows = payload.get("rows")
        also_zero = bool(payload.get("zero_outliers", False))
        matched, _ = _resolve_targets(workspace, str(ref))
        rescaled_rows_total = 0
        for name in matched:
            arr = workspace.get_readonly(name).astype(np.float64).copy()
            if arr.ndim < 2:
                continue
            m2d = arr.reshape(arr.shape[0], -1)
            row_norms = np.linalg.norm(m2d, axis=1)
            med = float(np.median(row_norms))
            if med <= 1e-12:
                continue
            if explicit_rows is not None:
                rows_to_fix = [int(r) for r in explicit_rows if 0 <= int(r) < m2d.shape[0]]
            else:
                rows_to_fix = [int(r) for r in np.where(row_norms >= min_ratio * med)[0]]

            for r in rows_to_fix:
                if also_zero:
                    m2d[r, :] = 0.0
                elif row_norms[r] > 0:
                    m2d[r, :] *= med / row_norms[r]
                rescaled_rows_total += 1

            workspace.set_tensor(
                name,
                m2d.reshape(arr.shape).astype(np.float32),
                apply_fisher_guard=apply_fisher_guard,
                allow_outlier_repair=True,
            )
            # Also rescale corresponding 1D bias if present
            if name.endswith(".weight"):
                bias_name = name[: -len(".weight")] + ".bias"
                if bias_name in workspace.names():
                    b_arr = workspace.get_readonly(bias_name).astype(np.float64).copy()
                    if b_arr.ndim == 1 and b_arr.shape[0] == m2d.shape[0]:
                        for r in rows_to_fix:
                            if also_zero:
                                b_arr[r] = 0.0
                            elif row_norms[r] > 0:
                                b_arr[r] *= med / row_norms[r]
                        workspace.set_tensor(
                            bias_name,
                            b_arr.astype(np.float32),
                            apply_fisher_guard=apply_fisher_guard,
                            allow_outlier_repair=True,
                        )
        return len(matched), f"rescale_outlier_rows_(fixed {rescaled_rows_total} row(s))"

    if op_name == "replace_nonfinite_":
        ref = payload.get("target") or payload.get("of", "*")
        val = float(payload.get("value", 0.0))
        matched, _ = _resolve_targets(workspace, str(ref))
        fixed = 0
        for name in matched:
            arr = workspace.get_readonly(name).copy()
            mask = ~np.isfinite(arr)
            if np.any(mask):
                fixed += int(np.count_nonzero(mask))
                arr[mask] = val
                workspace.set_tensor(
                    name, arr, apply_fisher_guard=apply_fisher_guard, allow_outlier_repair=True
                )
        return len(matched), f"replace_nonfinite_(fixed {fixed} elements)"

    if op_name == "rank_one_update_":
        ref = payload.get("target") or payload.get("of")
        matched, _ = _resolve_targets(workspace, str(ref))
        u = np.asarray(payload["u"], dtype=np.float64).reshape(-1, 1)
        v = np.asarray(payload["v"], dtype=np.float64).reshape(1, -1)
        alpha = float(payload.get("alpha", 1.0))
        for name in matched:
            arr = workspace.get_readonly(name).astype(np.float64)
            if arr.shape != (u.shape[0], v.shape[1]):
                raise PlanValidationError(
                    f"rank_one_update_ shape mismatch for {name}: {arr.shape} vs ({u.shape[0]}, {v.shape[1]})"
                )
            updated = (arr + alpha * (u @ v)).astype(np.float32)
            workspace.set_tensor(name, updated, apply_fisher_guard=apply_fisher_guard)
        return len(matched), f"rank_one_update_(alpha={alpha})"

    if op_name == "closed_form_ridge_":
        target = str(payload["target"])
        keys = np.asarray(payload["keys"], dtype=np.float64)
        values = np.asarray(payload["values"], dtype=np.float64)
        lam = float(payload.get("lambda_reg", 1e-2))
        info = apply_closed_form_ridge(
            workspace,
            target,
            keys,
            values,
            lambda_reg=lam,
            apply_fisher_guard=apply_fisher_guard,
        )
        return 1, f"closed_form_ridge_({target}, post_mse={info['post_mse']})"

    if op_name == "memit_mass_":
        raw_targets = payload.get("targets") or payload.get("target")
        if isinstance(raw_targets, str):
            targets, _ = _resolve_targets(workspace, raw_targets)
        else:
            targets = list(raw_targets)
        raw_facts = payload["facts"]
        facts = [
            BatchFactSpec(
                fact_id=str(f.get("id", i)),
                key_vector=np.asarray(f["key"], dtype=np.float64),
                target_value_vector=np.asarray(f["value"], dtype=np.float64),
            )
            for i, f in enumerate(raw_facts)
        ]
        lam = float(payload.get("lambda_cov", 1e-2))
        res = apply_memit_mass_batch(
            workspace,
            targets,
            facts,
            lambda_cov=lam,
            apply_fisher_guard=apply_fisher_guard,
        )
        return len(targets), (
            f"memit_mass_({res.num_facts} facts across {len(targets)} layers, "
            f"mse {res.pre_edit_mse:.4g}->{res.post_edit_mse:.4g})"
        )

    raise PlanValidationError(f"Unsupported mutation op: {op_name!r}")


def execute_yaml_plan(
    workspace: WeightWorkspace,
    plan_yaml_or_dict: str | dict[str, Any] | list[dict[str, Any]],
    *,
    dry_run: bool = False,
    require_assertions: bool = True,
    apply_fisher_guard: bool = True,
) -> PlanExecutionReport:
    """Compile, validate, and atomically execute a BrainSurgery-WX YAML plan on `workspace`.

    If any assertion or transform fails during execution (or if `dry_run=True`),
    all tensor modifications made during the plan are automatically rolled back
    to the pre-plan checkpoint.
    """
    try:
        compiled_steps = compile_and_validate_plan(
            plan_yaml_or_dict,
            require_assertions=require_assertions,
            allow_filesystem_ops=False,
        )
    except PlanValidationError as exc:
        return PlanExecutionReport(
            success=False,
            dry_run=dry_run,
            assertions_verified=0,
            mutations_executed=0,
            error=str(exc),
        )

    ckpt_label = "__pre_yaml_plan__"
    workspace.checkpoint(ckpt_label)

    step_results: list[StepExecutionResult] = []
    assertions_verified = 0
    mutations_executed = 0

    try:
        for idx, (op_name, payload) in enumerate(compiled_steps):
            if op_name == "assert":
                matched_count, detail = evaluate_assertion(workspace, payload)
                assertions_verified += 1
                step_results.append(
                    StepExecutionResult(
                        step_index=idx,
                        op_name="assert",
                        matched_tensors=matched_count,
                        status="verified",
                        detail=detail,
                    )
                )
            elif op_name in ("dump", "diff"):
                step_results.append(
                    StepExecutionResult(
                        step_index=idx,
                        op_name=op_name,
                        matched_tensors=len(workspace.names()),
                        status="ok",
                        detail=f"{op_name} inspected",
                    )
                )
            else:
                matched_count, detail = _apply_mutation_step(
                    workspace,
                    op_name,
                    payload,
                    apply_fisher_guard=apply_fisher_guard,
                )
                mutations_executed += 1
                step_results.append(
                    StepExecutionResult(
                        step_index=idx,
                        op_name=op_name,
                        matched_tensors=matched_count,
                        status="applied",
                        detail=detail,
                    )
                )

        diff_dict = workspace.diff().to_dict()
        if dry_run:
            workspace.restore(ckpt_label)

        return PlanExecutionReport(
            success=True,
            dry_run=dry_run,
            assertions_verified=assertions_verified,
            mutations_executed=mutations_executed,
            steps=step_results,
            diff_summary=diff_dict,
        )
    except Exception as exc:
        workspace.restore(ckpt_label)
        return PlanExecutionReport(
            success=False,
            dry_run=dry_run,
            assertions_verified=assertions_verified,
            mutations_executed=mutations_executed,
            steps=step_results,
            diff_summary=workspace.diff().to_dict(),
            error=f"{type(exc).__name__}: {exc}",
        )
