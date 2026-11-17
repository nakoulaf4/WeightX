// SPDX-License-Identifier: GPL-3.0-or-later
//! Process-isolated sandbox for LLM-generated weight-space code.
//! Rust port of `weightx.execution.sandbox`.
//!
//! Defense in depth:
//!   1. Static gatekeeper: only `numpy`/`math` imports, no dangerous builtins,
//!      no dunder attributes/identifiers (scanner approximation of the AST audit).
//!   2. Child interpreter (`python -I`) in an ephemeral directory with locked
//!      down builtins, blocked network modules, wall-clock timeout and output
//!      truncation.
//!   3. Tensor exchange via `.safetensors`; shape/finiteness validation; edits
//!      routed through the workspace Fisher guard.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use regex::Regex;
use serde_json::{json, Value};

use crate::error::Result;
use crate::tensor;
use crate::workspace::WeightWorkspace;

const ALLOWED_IMPORTS: &[&str] = &["numpy", "math"];

const FORBIDDEN_CALLS: &[&str] = &[
    "open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "breakpoint", "help", "exit", "quit",
];

/// Static security audit of sandbox code (port of `validate_sandbox_ast`).
pub fn validate_sandbox_code(code: &str) -> std::result::Result<(), String> {
    if code.trim().is_empty() {
        return Err("Sandbox code must be a non-empty string".into());
    }
    let import_re = Regex::new(r"(?m)(?:^|;)\s*import\s+([A-Za-z0-9_.,\s]+)").unwrap();
    for cap in import_re.captures_iter(code) {
        for alias in cap[1].split(',') {
            let root = alias
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("")
                .split('.')
                .next()
                .unwrap_or("");
            if !root.is_empty() && !ALLOWED_IMPORTS.contains(&root) {
                return Err(format!(
                    "Forbidden import '{root}'; only ['math', 'numpy'] are permitted"
                ));
            }
        }
    }
    let from_re = Regex::new(r"(?m)(?:^|;)\s*from\s+([A-Za-z0-9_.]+)\s+import").unwrap();
    for cap in from_re.captures_iter(code) {
        let root = cap[1].split('.').next().unwrap_or("");
        if !ALLOWED_IMPORTS.contains(&root) {
            return Err(format!(
                "Forbidden import from '{}'; only ['math', 'numpy'] are permitted",
                &cap[1]
            ));
        }
    }
    let dunder_attr = Regex::new(r"\.__[A-Za-z0-9_]*").unwrap();
    if let Some(m) = dunder_attr.find(code) {
        return Err(format!(
            "Forbidden dunder attribute access: '{}' is blocked in the sandbox",
            m.as_str()
        ));
    }
    let dunder_ident = Regex::new(r"\b__[A-Za-z0-9_]+__\b").unwrap();
    if let Some(m) = dunder_ident.find(code) {
        return Err(format!(
            "Forbidden dunder identifier: '{}' is blocked in the sandbox",
            m.as_str()
        ));
    }
    for call in FORBIDDEN_CALLS {
        let re = Regex::new(&format!(r"\b{}\s*\(", regex::escape(call))).unwrap();
        if re.is_match(code) {
            return Err(format!("Forbidden builtin call: '{call}()' is blocked in the sandbox"));
        }
    }
    Ok(())
}

const WORKER_SCRIPT: &str = r#"
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
"#;

#[derive(Debug, Clone)]
pub struct SandboxExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub elapsed_sec: f64,
    pub timed_out: bool,
    pub modified_tensors: Vec<String>,
    pub isolation_backend: String,
}

impl SandboxExecutionResult {
    pub fn to_value(&self) -> Value {
        json!({
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_sec": (self.elapsed_sec * 1e4).round() / 1e4,
            "timed_out": self.timed_out,
            "modified_tensors": self.modified_tensors,
            "isolation_backend": self.isolation_backend,
        })
    }
    pub fn to_toon(&self) -> String {
        crate::toon::dumps_toon(&self.to_value())
    }
}

pub struct IsolatedSandbox {
    pub timeout_sec: f64,
    pub max_memory_mb: usize,
    pub max_output_chars: usize,
    pub allow_writes: bool,
    pub apply_fisher_guard: bool,
    pub python_override: Option<PathBuf>,
}

impl Default for IsolatedSandbox {
    fn default() -> Self {
        Self {
            timeout_sec: 15.0,
            max_memory_mb: 1024,
            max_output_chars: 8000,
            allow_writes: true,
            apply_fisher_guard: true,
            python_override: None,
        }
    }
}

/// Locate a Python interpreter. Single canonical order shared with the torch
/// sidecar: explicit override, env `WEIGHTX_PYTHON`, the repo `.venv`, then PATH.
/// The `.venv` step matters because `numpy`/`safetensors` are usually installed
/// only there; without it the sandbox would pick a bare PATH interpreter and the
/// worker would die on `import safetensors`.
pub fn find_python(override_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = override_path {
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    if let Ok(p) = std::env::var("WEIGHTX_PYTHON") {
        let pb = PathBuf::from(&p);
        if pb.exists() {
            return Some(pb);
        }
    }
    // Repo `.venv` (same fallback the torch sidecar uses), before bare PATH.
    if let Some(root) = crate::torch_backend::repo_root() {
        let venv = if cfg!(windows) {
            root.join(".venv").join("Scripts").join("python.exe")
        } else {
            root.join(".venv").join("bin").join("python")
        };
        if venv.exists() {
            return Some(venv);
        }
    }
    let path_var = std::env::var("PATH").unwrap_or_default();
    let sep = if cfg!(windows) { ';' } else { ':' };
    let candidates: &[&str] = if cfg!(windows) {
        &["python.exe", "python3.exe"]
    } else {
        &["python3", "python"]
    };
    for dir in path_var.split(sep) {
        for cand in candidates {
            let p = Path::new(dir).join(cand);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

impl IsolatedSandbox {
    pub fn run(
        &self,
        ws: &mut WeightWorkspace,
        code: &str,
        tensor_pattern: Option<&str>,
    ) -> SandboxExecutionResult {
        let fail = |stderr: String, backend: &str| SandboxExecutionResult {
            success: false,
            stdout: String::new(),
            stderr,
            elapsed_sec: 0.0,
            timed_out: false,
            modified_tensors: vec![],
            isolation_backend: backend.into(),
        };

        if let Err(e) = validate_sandbox_code(code) {
            return fail(format!("SandboxSecurityError: {e}"), "static-gatekeeper");
        }

        let target_names = ws.names(tensor_pattern, None);
        if target_names.is_empty() {
            return fail(
                format!("No tensors matched pattern {tensor_pattern:?}"),
                "subprocess-isolated+scanner",
            );
        }

        let total_bytes: usize = target_names
            .iter()
            .filter_map(|n| ws.get_readonly(n).ok())
            .map(|t| t.len() * 4)
            .sum();
        if total_bytes > self.max_memory_mb * 1024 * 1024 {
            return fail(
                format!(
                    "Requested tensor subset ({:.1} MB) exceeds sandbox memory cap ({} MB). Use a narrower tensor_pattern.",
                    total_bytes as f64 / (1024.0 * 1024.0),
                    self.max_memory_mb
                ),
                "subprocess-isolated+scanner",
            );
        }

        let Some(python) = find_python(self.python_override.as_deref()) else {
            return fail(
                "No Python interpreter found for the sandbox. Install Python with numpy+safetensors or set WEIGHTX_PYTHON.".into(),
                "subprocess-isolated+scanner",
            );
        };

        let tmp = match tempfile::tempdir() {
            Ok(t) => t,
            Err(e) => return fail(format!("Cannot create sandbox temp dir: {e}"), "subprocess-isolated+scanner"),
        };
        let wdir = tmp.path();

        let export: Vec<(String, tensor::Tensor)> = target_names
            .iter()
            .filter_map(|n| ws.get_readonly(n).ok().map(|t| (n.clone(), t.clone())))
            .collect();
        if let Err(e) = tensor::save_safetensors(&wdir.join("input_weights.safetensors"), &export) {
            return fail(format!("Failed to export tensors: {e}"), "subprocess-isolated+scanner");
        }
        if std::fs::write(wdir.join("user_cell.py"), code).is_err() {
            return fail("Failed to write user cell".into(), "subprocess-isolated+scanner");
        }
        if std::fs::write(wdir.join("worker.py"), WORKER_SCRIPT).is_err() {
            return fail("Failed to write worker".into(), "subprocess-isolated+scanner");
        }

        let t0 = Instant::now();
        let (mut child, backend) = spawn_worker(&python, wdir);
        let mut child = match child {
            Ok(c) => c,
            Err(e) => return fail(format!("Failed to spawn sandbox worker: {e}"), &backend),
        };

        let timeout = Duration::from_secs_f64(self.timeout_sec.max(0.2));
        let mut timed_out = false;
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if t0.elapsed() >= timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        timed_out = true;
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => break,
            }
        }
        let mut stdout = String::new();
        let mut stderr = String::new();
        if let Some(mut out) = child.stdout.take() {
            let _ = out.read_to_string(&mut stdout);
        }
        if let Some(mut err) = child.stderr.take() {
            let _ = err.read_to_string(&mut stderr);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let status_ok = !timed_out && child.wait().map(|s| s.success()).unwrap_or(false);

        let tail = |s: String| {
            if s.len() > self.max_output_chars {
                s[s.len() - self.max_output_chars..].to_string()
            } else {
                s
            }
        };

        if timed_out {
            return SandboxExecutionResult {
                success: false,
                stdout: tail(stdout),
                stderr: tail(format!("{stderr}\nTimeoutError: exceeded {}s", self.timeout_sec)),
                elapsed_sec: elapsed,
                timed_out: true,
                modified_tensors: vec![],
                isolation_backend: backend.clone(),
            };
        }

        let mut modified_names = Vec::new();
        let changed_path = wdir.join("changed_weights.safetensors");
        if status_ok && changed_path.is_file() && self.allow_writes {
            if let Ok(changed) = tensor::load_safetensors(&changed_path) {
                for (tname, new_arr) in changed {
                    if target_names.contains(&tname) {
                        if ws
                            .set_tensor(&tname, &new_arr, self.apply_fisher_guard, true)
                            .is_ok()
                        {
                            modified_names.push(tname);
                        }
                    }
                }
            }
        }

        SandboxExecutionResult {
            success: status_ok,
            stdout: tail(stdout),
            stderr: tail(stderr),
            elapsed_sec: elapsed,
            timed_out: false,
            modified_tensors: modified_names,
            isolation_backend: backend,
        }
    }
}

/// Spawn the isolated worker. On Linux with bubblewrap available, run it inside
/// `bwrap --unshare-all` with read-only system mounts (port of `_build_command`).
fn spawn_worker(python: &Path, wdir: &Path) -> (std::io::Result<std::process::Child>, String) {
    let base_env = |cmd: &mut Command| {
        cmd.env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("SYSTEMROOT", std::env::var("SYSTEMROOT").unwrap_or_default())
            .env("PYTHONHASHSEED", "0")
            .env("OMP_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("NO_PROXY", "*")
            .env("http_proxy", "")
            .env("https_proxy", "")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
    };
    if cfg!(target_os = "linux") {
        if let Some(bwrap) = find_on_path("bwrap") {
            let mut cmd = Command::new(bwrap);
            cmd.args([
                "--unshare-all",
                "--die-with-parent",
                "--ro-bind", "/usr", "/usr",
                "--ro-bind", "/lib", "/lib",
                "--ro-bind-try", "/lib64", "/lib64",
                "--bind",
            ]);
            cmd.arg(wdir).arg("/work").arg("--chdir").arg("/work");
            cmd.arg(python).arg("-I").arg("worker.py");
            base_env(&mut cmd);
            return (cmd.spawn(), "bubblewrap+scanner".to_string());
        }
    }
    let mut cmd = Command::new(python);
    cmd.arg("-I").arg("worker.py").current_dir(wdir);
    base_env(&mut cmd);
    (cmd.spawn(), "subprocess-isolated+scanner".to_string())
}

fn find_on_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var("PATH").ok()?;
    for dir in path_var.split(':') {
        let p = Path::new(dir).join(binary);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[allow(dead_code)]
fn _result_helper(r: &Result<()>) -> bool {
    r.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gatekeeper() {
        assert!(validate_sandbox_code("import os").is_err());
        assert!(validate_sandbox_code("from subprocess import run").is_err());
        assert!(validate_sandbox_code("x = open('f')").is_err());
        assert!(validate_sandbox_code("y = W.__class__").is_err());
        assert!(validate_sandbox_code("import numpy as np\nW['a'] = np.zeros_like(W['a'])").is_ok());
        assert!(validate_sandbox_code("import math\nprint(math.sqrt(4))").is_ok());
        assert!(validate_sandbox_code("").is_err());
    }
}