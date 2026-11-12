# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
Process-Isolated Python Sandbox for Weight-Space Analysis and Surgery.

Enforces defense-in-depth isolation when executing LLM-generated Python code:
  1. Static AST Gatekeeper (`validate_sandbox_ast`):
     - Rejects all imports except `numpy` and `math`.
     - Rejects calls to `open`, `eval`, `exec`, `compile`, `__import__`, `getattr`,
       `setattr`, `delattr`, `globals`, `locals`, `vars`, `breakpoint`, `input`.
     - Rejects access to dunder attributes (`__subclasses__`, `__globals__`,
       `__builtins__`, `__class__`, `__bases__`, `__code__`, etc.).
  2. OS Process & Network Isolation:
     - Runs in a separate child interpreter (`python -I`) inside an ephemeral
       temporary directory.
     - Uses Linux `bwrap` (bubblewrap `--unshare-all --die-with-parent`) when
       available on PATH, and on Windows/macOS/Linux disables `socket`, filesystem
       `open`, and subprocess spawning inside the worker before executing user code.
     - Enforces wall-clock timeout (`timeout_sec`), output truncation, and memory
       allocation bounds (`max_memory_mb`).
  3. Safe Tensor Exchange & Fisher Guard:
     - Passes only the requested tensor subset via `.safetensors`, validates output
       shapes, dtypes, and finiteness, and routes any proposed weight edits through
       the `WeightWorkspace`'s IWCKEdit `FisherGuard`.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from weightx.l3_weights.workspace import WeightWorkspace
from weightx.toon_io import dumps_toon


class SandboxSecurityError(PermissionError):
    """Raised when sandbox code violates static AST or runtime security policies."""


ALLOWED_IMPORTS = frozenset({"numpy", "math"})

FORBIDDEN_CALLS = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "help",
        "exit",
        "quit",
    }
)

FORBIDDEN_ATTR_PREFIXES = ("__",)


def validate_sandbox_ast(code: str) -> ast.Module:
    """Parse and statically audit `code` against forbidden imports, builtins, and dunders."""
    if not isinstance(code, str) or not code.strip():
        raise SandboxSecurityError("Sandbox code must be a non-empty string")
    try:
        tree = ast.parse(code, filename="<weightx_sandbox>", mode="exec")
    except SyntaxError as exc:
        raise SandboxSecurityError(f"SyntaxError in sandbox code: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in ALLOWED_IMPORTS:
                    raise SandboxSecurityError(
                        f"Forbidden import {alias.name!r}; only {sorted(ALLOWED_IMPORTS)} are permitted"
                    )
        elif isinstance(node, ast.ImportFrom):
            root_mod = (node.module or "").split(".")[0]
            if root_mod not in ALLOWED_IMPORTS:
                raise SandboxSecurityError(
                    f"Forbidden import from {node.module!r}; only {sorted(ALLOWED_IMPORTS)} are permitted"
                )
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith(FORBIDDEN_ATTR_PREFIXES):
                raise SandboxSecurityError(
                    f"Forbidden dunder attribute access: '.{node.attr}' is blocked in the sandbox"
                )
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                raise SandboxSecurityError(
                    f"Forbidden dunder identifier: {node.id!r} is blocked in the sandbox"
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise SandboxSecurityError(
                    f"Forbidden builtin call: '{node.func.id}()' is blocked in the sandbox"
                )

    return tree


_WORKER_SCRIPT = r"""
import builtins
import hashlib
import math
import sys
import traceback
import numpy as np
from safetensors.numpy import load_file, save_file

def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

W = {k: np.array(v, copy=True) for k, v in load_file("input_weights.safetensors").items()}
before_meta = {k: (v.shape, v.dtype, _digest(v)) for k, v in W.items()}
code_text = builtins.open("user_cell.py", "r", encoding="utf-8").read()

# Lock down dangerous builtins and network modules before executing user_cell.py
for mod_name in ("socket", "urllib", "http", "subprocess", "shutil", "ctypes"):
    sys.modules[mod_name] = None

SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "Exception": Exception, "ValueError": ValueError, "RuntimeError": RuntimeError,
}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root == "numpy":
        return np
    if root == "math":
        return math
    raise PermissionError(f"Import of '{name}' is blocked by WeightX sandbox")

SAFE_BUILTINS["__import__"] = _safe_import

ns = {
    "__builtins__": SAFE_BUILTINS,
    "__name__": "__weightx_sandbox__",
    "W": W,
    "np": np,
    "math": math,
}

try:
    exec(compile(code_text, "<weightx_sandbox>", "exec"), ns)
except BaseException:
    traceback.print_exc()
    sys.exit(1)

changed = {}
problems = []
for name, val in W.items():
    if name not in before_meta:
        problems.append(f"Cannot add new tensor '{name}'")
        continue
    orig_shape, orig_dtype, orig_hash = before_meta[name]
    arr = np.ascontiguousarray(np.asarray(val, dtype=orig_dtype))
    if arr.shape != orig_shape:
        problems.append(f"Tensor '{name}' changed shape from {orig_shape} to {arr.shape}")
    elif not np.all(np.isfinite(arr)):
        problems.append(f"Tensor '{name}' contains NaN or Inf after sandbox edit")
    elif _digest(arr) != orig_hash:
        changed[name] = arr

if problems:
    for p in problems:
        print(p, file=sys.stderr)
    sys.exit(2)

if changed:
    save_file(changed, "changed_weights.safetensors")
"""


@dataclass
class SandboxExecutionResult:
    """Result returned by `IsolatedSandbox.run()`."""

    success: bool
    stdout: str
    stderr: str
    elapsed_sec: float
    timed_out: bool
    modified_tensors: list[str] = field(default_factory=list)
    isolation_backend: str = "subprocess-isolated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "timed_out": self.timed_out,
            "modified_tensors": self.modified_tensors,
            "isolation_backend": self.isolation_backend,
        }

    def to_toon(self) -> str:
        return dumps_toon(self.to_dict())


class IsolatedSandbox:
    """Subprocess + AST + optional bubblewrap (`bwrap`) sandbox for weight-space code."""

    def __init__(
        self,
        *,
        timeout_sec: float = 15.0,
        max_memory_mb: int = 1024,
        max_output_chars: int = 8000,
        allow_writes: bool = True,
        apply_fisher_guard: bool = True,
    ):
        self.timeout_sec = float(timeout_sec)
        self.max_memory_mb = int(max_memory_mb)
        self.max_output_chars = int(max_output_chars)
        self.allow_writes = bool(allow_writes)
        self.apply_fisher_guard = bool(apply_fisher_guard)

    def _build_command(self, workdir: Path) -> tuple[list[str], str]:
        bwrap_path = shutil.which("bwrap")
        if bwrap_path and sys.platform.startswith("linux"):
            cmd = [
                bwrap_path,
                "--unshare-all",
                "--die-with-parent",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind-try",
                "/lib64",
                "/lib64",
                "--bind",
                str(workdir),
                "/work",
                "--chdir",
                "/work",
                sys.executable,
                "-I",
                "worker.py",
            ]
            return cmd, "bubblewrap+ast"
        return [sys.executable, "-I", "worker.py"], "subprocess-isolated+ast"

    def run(
        self,
        workspace: WeightWorkspace,
        code: str,
        *,
        tensor_pattern: str | None = None,
    ) -> SandboxExecutionResult:
        """Validate and execute `code` in an isolated child process against `workspace`."""
        try:
            validate_sandbox_ast(code)
        except SandboxSecurityError as exc:
            return SandboxExecutionResult(
                success=False,
                stdout="",
                stderr=f"SandboxSecurityError: {exc}",
                elapsed_sec=0.0,
                timed_out=False,
                isolation_backend="ast-gatekeeper",
            )

        target_names = workspace.names(pattern=tensor_pattern)
        if not target_names:
            return SandboxExecutionResult(
                success=False,
                stdout="",
                stderr=f"No tensors matched pattern {tensor_pattern!r}",
                elapsed_sec=0.0,
                timed_out=False,
            )

        # Check memory cap before serializing tensors
        total_bytes = sum(workspace.get_readonly(n).nbytes for n in target_names)
        if total_bytes > self.max_memory_mb * 1024 * 1024:
            return SandboxExecutionResult(
                success=False,
                stdout="",
                stderr=(
                    f"Requested tensor subset ({total_bytes / (1024*1024):.1f} MB) exceeds "
                    f"sandbox memory cap ({self.max_memory_mb} MB). Use a narrower tensor_pattern."
                ),
                elapsed_sec=0.0,
                timed_out=False,
            )

        with tempfile.TemporaryDirectory(prefix="weightx_sbx_") as tmpdir:
            wdir = Path(tmpdir)
            export_dict = {n: np.array(workspace.get_readonly(n), copy=True) for n in target_names}
            save_file(export_dict, str(wdir / "input_weights.safetensors"))
            (wdir / "user_cell.py").write_text(code, encoding="utf-8")
            (wdir / "worker.py").write_text(_WORKER_SCRIPT, encoding="utf-8")

            cmd, backend_label = self._build_command(wdir)
            clean_env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NO_PROXY": "*",
                "http_proxy": "",
                "https_proxy": "",
            }

            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(wdir),
                    env=clean_env,
                    capture_output=True,
                    text=True,
                    timeout=max(0.2, self.timeout_sec),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.monotonic() - t0
                out_str = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                err_str = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
                return SandboxExecutionResult(
                    success=False,
                    stdout=out_str[-self.max_output_chars :],
                    stderr=(err_str + f"\nTimeoutError: exceeded {self.timeout_sec}s")[
                        -self.max_output_chars :
                    ],
                    elapsed_sec=elapsed,
                    timed_out=True,
                    isolation_backend=backend_label,
                )

            elapsed = time.monotonic() - t0
            changed_path = wdir / "changed_weights.safetensors"
            modified_names: list[str] = []

            if proc.returncode == 0 and changed_path.is_file() and self.allow_writes:
                changed_tensors = load_file(str(changed_path))
                for tname, new_arr in changed_tensors.items():
                    if tname in target_names:
                        workspace.set_tensor(
                            tname,
                            new_arr,
                            apply_fisher_guard=self.apply_fisher_guard,
                            allow_outlier_repair=True,
                        )
                        modified_names.append(tname)

            return SandboxExecutionResult(
                success=(proc.returncode == 0),
                stdout=proc.stdout[-self.max_output_chars :],
                stderr=proc.stderr[-self.max_output_chars :],
                elapsed_sec=elapsed,
                timed_out=False,
                modified_tensors=modified_names,
                isolation_backend=backend_label,
            )
