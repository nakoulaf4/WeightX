# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
L3 WeightWorkspace: isolated tensor store with describe, safe slicing, sparse checkpoints,
cryptographic provenance (SHA-256), and optional IWCKEdit Fisher protection.

Integrates with `weightspace` (Apache-2.0) for fast parallel moments/quantiles/SVD
and extends it with PyTorch live synchronization, TOON tabular output, and FisherGuard.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from weightx.safety.iwck_fisher import FisherGuard, FisherProtectionReport
from weightx.toon_io import dumps_toon

try:
    from weightspace import render as _wsp_render
except ImportError:  # pragma: no cover
    _wsp_render = None


_SLICE_TOKEN_RE = re.compile(r"^\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*)?)?$")


def parse_safe_slice(index_expr: str | None) -> tuple[Any, ...]:
    """Parse a NumPy-style slice expression like `'0:8, :, 16:32'` without `eval`."""
    if index_expr is None or not str(index_expr).strip():
        return ()
    if _wsp_render is not None:
        return _wsp_render.parse_index(index_expr)

    text = str(index_expr).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    out: list[Any] = []
    for part in text.split(","):
        part = part.strip()
        if part in ("", ":"):
            out.append(slice(None))
            continue
        if part == "...":
            out.append(Ellipsis)
            continue
        m = _SLICE_TOKEN_RE.match(part)
        if not m:
            raise ValueError(f"Invalid slice component: {part!r}")
        if ":" not in part:
            out.append(int(m.group(1)))
        else:
            start, stop, step = (int(g) if g is not None else None for g in m.groups())
            out.append(slice(start, stop, step))
    return tuple(out)


@dataclass
class TensorDiffRecord:
    """Change telemetry for a single modified tensor."""

    name: str
    numel: int
    changed_count: int
    l2_delta: float
    max_abs_delta: float
    rel_delta: float
    sha256_after: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "numel": self.numel,
            "changed_count": self.changed_count,
            "l2_delta": round(self.l2_delta, 6),
            "max_abs_delta": round(self.max_abs_delta, 6),
            "rel_delta": round(self.rel_delta, 6),
            "sha256_after": self.sha256_after[:16],
        }


@dataclass
class WorkspaceDiffSummary:
    """Aggregate diff report across all touched tensors in a WeightWorkspace."""

    total_params: int
    changed_params: int
    tensors: list[TensorDiffRecord] = field(default_factory=list)

    @property
    def changed_fraction(self) -> float:
        return self.changed_params / max(1, self.total_params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_params": self.total_params,
            "changed_params": self.changed_params,
            "changed_pct": round(self.changed_fraction * 100.0, 5),
            "tensors_changed": len(self.tensors),
            "tensors": [t.to_dict() for t in self.tensors],
        }

    def to_toon(self) -> str:
        return dumps_toon(self.to_dict())


def _sha256_array(arr: np.ndarray) -> str:
    contig = np.ascontiguousarray(arr)
    return hashlib.sha256(contig.tobytes()).hexdigest()


class WeightWorkspace:
    """Isolated L3 workspace managing tensor reads, slices, edits, checkpoints, and rollback."""

    def __init__(
        self,
        model_or_tensors: nn.Module | Mapping[str, torch.Tensor | np.ndarray],
        *,
        fisher_guard: FisherGuard | None = None,
    ):
        self._model: nn.Module | None = model_or_tensors if isinstance(model_or_tensors, nn.Module) else None
        self._param_refs: dict[str, torch.Tensor] = {}
        self._live: dict[str, np.ndarray] = {}
        self._dtypes: dict[str, str] = {}
        self._original: dict[str, np.ndarray] = {}
        self._checkpoints: dict[str, dict[str, np.ndarray]] = {}
        self.fisher_guard: FisherGuard | None = fisher_guard

        if isinstance(model_or_tensors, nn.Module):
            for name, param in model_or_tensors.named_parameters():
                self._param_refs[name] = param
                self._dtypes[name] = str(param.dtype).replace("torch.", "")
                self._live[name] = param.detach().float().cpu().numpy().copy()
            for name, buf in model_or_tensors.named_buffers():
                if buf.is_floating_point():
                    self._param_refs[name] = buf
                    self._dtypes[name] = str(buf.dtype).replace("torch.", "")
                    self._live[name] = buf.detach().float().cpu().numpy().copy()
        elif isinstance(model_or_tensors, Mapping):
            for name, val in model_or_tensors.items():
                if isinstance(val, torch.Tensor):
                    self._param_refs[name] = val
                    self._dtypes[name] = str(val.dtype).replace("torch.", "")
                    self._live[name] = val.detach().float().cpu().numpy().copy()
                else:
                    arr = np.asarray(val)
                    self._dtypes[name] = str(arr.dtype)
                    self._live[name] = arr.astype(np.float32, copy=True)
        else:
            raise TypeError(f"Unsupported workspace target: {type(model_or_tensors).__name__}")

        self._names = list(self._live.keys())
        self.checkpoint("baseline")

    @property
    def model(self) -> nn.Module | None:
        return self._model

    def names(self, pattern: str | None = None, layer_idx: int | None = None) -> list[str]:
        """List parameter names matching a glob/substring/regex `pattern` and/or `layer_idx`."""
        result = self._names
        if layer_idx is not None:
            pat = re.compile(rf"(?:^|\.)({layer_idx})\.")
            result = [n for n in result if pat.search(n)]
        if pattern:
            matched = [
                n
                for n in result
                if fnmatch.fnmatch(n, pattern) or (pattern in n)
            ]
            if not matched:
                try:
                    rx = re.compile(pattern)
                    matched = [n for n in result if rx.search(n)]
                except re.error:
                    pass
            result = matched
        return list(result)

    def shape(self, name: str) -> tuple[int, ...]:
        if name not in self._live:
            raise KeyError(f"Unknown tensor {name!r}")
        return tuple(self._live[name].shape)

    def dtype(self, name: str) -> str:
        if name not in self._dtypes:
            raise KeyError(f"Unknown tensor {name!r}")
        return self._dtypes[name]

    def total_params(self) -> int:
        return sum(int(arr.size) for arr in self._live.values())

    def get_readonly(self, name: str) -> np.ndarray:
        if name not in self._live:
            raise KeyError(f"Unknown tensor {name!r}")
        view = self._live[name].view()
        view.flags.writeable = False
        return view

    def get_original(self, name: str) -> np.ndarray:
        if name in self._original:
            view = self._original[name].view()
            view.flags.writeable = False
            return view
        return self.get_readonly(name)

    def list_tensors_table(
        self,
        pattern: str | None = None,
        layer_idx: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a uniform list of tensor descriptors ideal for TOON tabular serialization."""
        rows: list[dict[str, Any]] = []
        for name in self.names(pattern=pattern, layer_idx=layer_idx):
            arr = self._live[name]
            rows.append(
                {
                    "name": name,
                    "shape": "x".join(str(d) for d in arr.shape),
                    "dtype": self._dtypes[name],
                    "numel": int(arr.size),
                    "l2_norm": round(float(np.linalg.norm(arr.reshape(-1).astype(np.float64))), 4),
                    "modified": name in self._original,
                }
            )
        return rows

    def describe_tensor(self, name: str, top_k: int = 8) -> dict[str, Any]:
        """Compute structured statistical, row/col L2 norm, and SVD spectral summary of `name`."""
        arr = self.get_readonly(name)
        flat = arr.reshape(-1).astype(np.float64)
        n = int(flat.size)
        if n == 0:
            return {"name": name, "shape": list(arr.shape), "numel": 0}

        mean = float(np.mean(flat))
        std = float(np.std(flat))
        lo = float(np.min(flat))
        hi = float(np.max(flat))
        abs_mean = float(np.mean(np.abs(flat)))
        l2_norm = float(np.linalg.norm(flat))
        frac_zero = float(np.count_nonzero(flat == 0.0) / n)
        qs = [round(float(x), 5) for x in np.quantile(flat, [0.01, 0.25, 0.5, 0.75, 0.99])]
        nonfinite = int(np.count_nonzero(~np.isfinite(flat)))

        info: dict[str, Any] = {
            "name": name,
            "shape": list(arr.shape),
            "dtype": self._dtypes[name],
            "numel": n,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(lo, 6),
            "max": round(hi, 6),
            "abs_mean": round(abs_mean, 6),
            "l2_norm": round(l2_norm, 5),
            "zeros_pct": round(frac_zero * 100.0, 3),
            "nonfinite_count": nonfinite,
            "quantiles_1_25_50_75_99": qs,
        }

        if arr.ndim >= 2:
            m2d = arr.reshape(arr.shape[0], -1).astype(np.float64)
            row_norms = np.linalg.norm(m2d, axis=1)
            col_norms = np.linalg.norm(m2d, axis=0)
            r_med = float(np.median(row_norms))
            c_med = float(np.median(col_norms))

            r_order = np.argsort(row_norms)[::-1]
            c_order = np.argsort(col_norms)[::-1]
            info["row_norm_median"] = round(r_med, 5)
            info["col_norm_median"] = round(c_med, 5)
            info["largest_rows"] = [
                {
                    "idx": int(i),
                    "l2": round(float(row_norms[i]), 5),
                    "ratio_to_median": round(float(row_norms[i] / max(r_med, 1e-12)), 3),
                }
                for i in r_order[:top_k]
            ]
            info["smallest_rows"] = [
                {"idx": int(i), "l2": round(float(row_norms[i]), 5)}
                for i in r_order[::-1][: min(4, top_k)]
            ]
            info["largest_cols"] = [
                {"idx": int(i), "l2": round(float(col_norms[i]), 5)}
                for i in c_order[: min(4, top_k)]
            ]

            if min(m2d.shape) <= 2048 and m2d.size <= 64_000_000:
                if _wsp_render is not None:
                    sv = _wsp_render.singular_values(m2d)
                else:
                    sv = np.linalg.svd(m2d, compute_uv=False)
                energy = np.cumsum(sv**2) / max(float(np.sum(sv**2)), 1e-30)
                rank_90 = int(np.searchsorted(energy, 0.90) + 1)
                info["top_singular_values"] = [round(float(v), 4) for v in sv[:top_k]]
                info["rank_90pct_energy"] = f"{rank_90}/{int(sv.size)}"

        return info

    def read_slice(
        self,
        name: str,
        index_expr: str | None = None,
        *,
        max_elements: int = 512,
        precision: int = 4,
    ) -> dict[str, Any]:
        """Read a bounded multi-dimensional slice of tensor `name` safely without `eval`."""
        arr = self.get_readonly(name)
        idx = parse_safe_slice(index_expr)
        sliced = arr[idx] if idx else arr
        truncated = False
        flat = sliced.reshape(-1)
        if flat.size > max_elements:
            truncated = True
            flat_sample = flat[:max_elements]
            text = " ".join(f"{float(v):.{precision}g}" for v in flat_sample)
        else:
            if _wsp_render is not None:
                text = _wsp_render.fmt_array(sliced, precision=precision)
            else:
                text = np.array2string(sliced, precision=precision, separator=" ")

        return {
            "name": name,
            "full_shape": list(arr.shape),
            "slice_expr": index_expr or ":",
            "slice_shape": list(sliced.shape),
            "slice_numel": int(sliced.size),
            "truncated": truncated,
            "values": text,
        }

    # ------------------------------------------------------------------
    # Mutation, Fisher Protection, Synchronization, and Checkpointing
    # ------------------------------------------------------------------

    def _ensure_snapshot(self, name: str) -> None:
        if name not in self._live:
            raise KeyError(f"Unknown tensor {name!r}")
        if name not in self._original:
            self._original[name] = self._live[name].copy()

    def set_tensor(
        self,
        name: str,
        new_value: np.ndarray | torch.Tensor,
        *,
        apply_fisher_guard: bool = True,
        allow_outlier_repair: bool = True,
    ) -> FisherProtectionReport | None:
        """Write `new_value` into tensor `name`, applying IWCKEdit Fisher protection if enabled."""
        self._ensure_snapshot(name)
        cur = self._live[name]
        arr = (
            new_value.detach().float().cpu().numpy()
            if isinstance(new_value, torch.Tensor)
            else np.asarray(new_value, dtype=np.float32)
        )
        if arr.shape != cur.shape:
            arr = np.broadcast_to(arr, cur.shape).copy()

        report: FisherProtectionReport | None = None
        if apply_fisher_guard and self.fisher_guard is not None and self.fisher_guard.enabled:
            arr, report = self.fisher_guard.protect_update(
                name,
                self._original[name],
                arr,
                allow_outlier_repair=allow_outlier_repair,
            )

        self._live[name][...] = arr
        self.sync_tensor(name)
        return report

    def sync_tensor(self, name: str) -> None:
        """Push in-memory NumPy array `name` back to the wrapped PyTorch parameter/buffer."""
        if name in self._param_refs:
            target_t = self._param_refs[name]
            with torch.no_grad():
                src = torch.from_numpy(self._live[name]).to(
                    device=target_t.device, dtype=target_t.dtype
                )
                target_t.copy_(src)

    def sync_all(self) -> None:
        """Synchronize all touched tensors to the live PyTorch model."""
        for name in self._live:
            self.sync_tensor(name)

    def diff(self) -> WorkspaceDiffSummary:
        """Compute exact parameter change counts and L2/max norms against baseline."""
        records: list[TensorDiffRecord] = []
        changed_total = 0
        for name, orig in self._original.items():
            cur = self._live[name]
            delta = cur.astype(np.float64) - orig.astype(np.float64)
            mask = np.abs(delta) > 1e-12
            n_changed = int(np.count_nonzero(mask))
            if n_changed == 0:
                continue
            l2_d = float(np.linalg.norm(delta.reshape(-1)))
            max_d = float(np.max(np.abs(delta)))
            base_l2 = float(np.linalg.norm(orig.reshape(-1).astype(np.float64)))
            rel_d = l2_d / max(base_l2, 1e-12)
            records.append(
                TensorDiffRecord(
                    name=name,
                    numel=int(cur.size),
                    changed_count=n_changed,
                    l2_delta=l2_d,
                    max_abs_delta=max_d,
                    rel_delta=rel_d,
                    sha256_after=_sha256_array(cur),
                )
            )
            changed_total += n_changed

        return WorkspaceDiffSummary(
            total_params=self.total_params(),
            changed_params=changed_total,
            tensors=records,
        )

    def checkpoint(self, label: str) -> None:
        """Save a named snapshot of all currently modified tensors."""
        snap: dict[str, np.ndarray] = {}
        for name in self._original:
            snap[name] = self._live[name].copy()
        self._checkpoints[label] = snap

    def restore(self, label: str) -> None:
        """Restore weights to checkpoint `label` and synchronize the live PyTorch model."""
        if label not in self._checkpoints:
            raise KeyError(f"Unknown checkpoint {label!r}; available={list(self._checkpoints)}")
        snap = self._checkpoints[label]
        for name, orig in list(self._original.items()):
            if name in snap:
                self._live[name][...] = snap[name]
            else:
                self._live[name][...] = orig
            self.sync_tensor(name)

    def revert_to_original(self, name: str | None = None) -> None:
        """Revert one or all touched tensors back to their initial unmodified values."""
        targets = [name] if name else list(self._original.keys())
        for n in targets:
            if n in self._original:
                self._live[n][...] = self._original[n]
                self.sync_tensor(n)
                del self._original[n]

    def export_safetensors(self, path: str | Path, only_changed: bool = False) -> Path:
        """Save workspace weights (or only modified tensors) to a `.safetensors` file."""
        from safetensors.numpy import save_file

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tensors_to_save = (
            {k: self._live[k] for k in self._original}
            if only_changed and self._original
            else dict(self._live)
        )
        save_file(tensors_to_save, str(out_path))
        return out_path
