// SPDX-License-Identifier: GPL-3.0-or-later
//! Hybrid backend: heavy numerics (forward/backward, real-size EAMIT) delegated
//! to a persistent PyTorch sidecar, while Rust keeps CLI, orchestration, TOON,
//! plans and workspace I/O (see docs/benchmark_rust_vs_python.md, option A).
//!
//! Selection (`WEIGHTX_BACKEND` or `wx --backend`):
//!   - `rust`  — pure Rust core (previous behavior);
//!   - `torch` — force the sidecar (falls back to Rust if unavailable);
//!   - `auto`  — sidecar for models >= 5M params, Rust for small/demo models.
//!
//! The sidecar is a JSONL child process (`torch_worker.py`) that reuses the
//! reference Python `weightx` package. Weights are synced incrementally via
//! safetensors; large outputs (logits/grads/keys) come back as files.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use ndarray::{Array2, Array3};
use serde_json::{json, Value};

use crate::error::{Result, WeightXError};
use crate::eamit::{BatchFactSpec, EamitAlignConfig, EamitResult};
use crate::model::{ForwardOut, ModelArch, ModelConfig};
use crate::tensor::{self, Tensor};
use crate::workspace::WeightWorkspace;

const WORKER_SRC: &str = include_str!("torch_worker.py");
const AUTO_THRESHOLD_PARAMS: usize = 5_000_000;

// ---------------------------------------------------------------------------
// Backend selection & environment discovery
// ---------------------------------------------------------------------------

pub fn backend_pref() -> String {
    std::env::var("WEIGHTX_BACKEND")
        .unwrap_or_else(|_| "auto".into())
        .to_lowercase()
}

/// Locate the repository root (contains `src/weightx`).
pub fn repo_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("WEIGHTX_HOME") {
        let pb = PathBuf::from(p);
        if pb.join("src").join("weightx").is_dir() {
            return Some(pb);
        }
    }
    let mut starts: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(p) = exe.parent() {
            starts.push(p.to_path_buf());
        }
    }
    for start in starts {
        let mut cur: Option<&Path> = Some(start.as_path());
        while let Some(dir) = cur {
            if dir.join("src").join("weightx").join("__init__.py").is_file() {
                return Some(dir.to_path_buf());
            }
            cur = dir.parent();
        }
    }
    None
}

/// Python for the sidecar. Delegates to the single canonical discovery order in
/// `sandbox::find_python` (WEIGHTX_PYTHON, repo .venv, PATH) so the sidecar and
/// the L3 sandbox always resolve the same interpreter.
pub fn sidecar_python() -> Option<PathBuf> {
    crate::sandbox::find_python(None)
}

fn probe_cache() -> &'static Mutex<HashMap<PathBuf, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Check that the interpreter can import torch + safetensors + weightx.
pub fn torch_available() -> bool {
    let Some(py) = sidecar_python() else { return false };
    if let Some(&v) = probe_cache().lock().unwrap().get(&py) {
        return v;
    }
    let src = repo_root()
        .map(|r| r.join("src").to_string_lossy().to_string())
        .unwrap_or_default();
    let code = format!(
        "import sys; sys.path.insert(0, r'{}'); import torch, safetensors, numpy, weightx",
        src.replace('\\', "\\\\")
    );
    let ok = Command::new(&py)
        .arg("-I")
        .arg("-c")
        .arg(&code)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    probe_cache().lock().unwrap().insert(py, ok);
    ok
}

/// Wrap a native architecture into the hybrid torch backend when appropriate.
pub fn maybe_wrap_torch(
    native: Arc<dyn ModelArch>,
    total_params: usize,
    hf_config: Option<Value>,
) -> Arc<dyn ModelArch> {
    let pref = backend_pref();
    if pref == "rust" {
        return native;
    }
    let forced = pref == "torch";
    if !forced && total_params < AUTO_THRESHOLD_PARAMS {
        return native;
    }
    if !torch_available() {
        if forced {
            eprintln!(
                "[weightx] WARNING: WEIGHTX_BACKEND=torch, but Python+torch sidecar is \
                 unavailable; falling back to the Rust core."
            );
        }
        return native;
    }
    Arc::new(TorchArch::new(native, hf_config))
}

// ---------------------------------------------------------------------------
// Sidecar process state
// ---------------------------------------------------------------------------

struct SidecarState {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    reader: Option<Arc<Mutex<BufReader<ChildStdout>>>>,
    dir: Option<tempfile::TempDir>,
    initialized: bool,
    synced_rev: u64,
    synced_modified: Vec<String>,
}

impl SidecarState {
    fn new() -> Self {
        Self {
            child: None,
            stdin: None,
            reader: None,
            dir: None,
            initialized: false,
            synced_rev: 0,
            synced_modified: Vec::new(),
        }
    }

    fn kill(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        self.stdin = None;
        self.reader = None;
        self.dir = None;
        self.initialized = false;
        self.synced_modified.clear();
    }

    fn exchange_dir(&self) -> Result<PathBuf> {
        self.dir
            .as_ref()
            .map(|d| d.path().to_path_buf())
            .ok_or_else(|| WeightXError::Value("torch sidecar is not running".into()))
    }
}

impl Drop for SidecarState {
    fn drop(&mut self) {
        self.kill();
    }
}

fn sidecar_timeout() -> Duration {
    let secs = std::env::var("WEIGHTX_TORCH_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(900.0);
    Duration::from_secs_f64(secs.max(1.0))
}

impl SidecarState {
    fn spawn(&mut self) -> Result<()> {
        let py = sidecar_python().ok_or_else(|| {
            WeightXError::Value("No Python interpreter for the torch sidecar (set WEIGHTX_PYTHON)".into())
        })?;
        let root = repo_root().ok_or_else(|| {
            WeightXError::Value("Cannot locate WeightX repo root for the torch sidecar (set WEIGHTX_HOME)".into())
        })?;
        let tmp = tempfile::tempdir()
            .map_err(|e| WeightXError::Value(format!("sidecar temp dir: {e}")))?;
        let wdir = tmp.path().to_path_buf();
        std::fs::write(wdir.join("torch_worker.py"), WORKER_SRC)?;
        let device = std::env::var("WEIGHTX_TORCH_DEVICE").unwrap_or_else(|_| "auto".into());
        let mut child = Command::new(&py)
            .arg("-I")
            .arg(wdir.join("torch_worker.py"))
            .arg(root.join("src"))
            .arg(&wdir)
            .arg(&device)
            .current_dir(&wdir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| WeightXError::Value(format!("spawn torch sidecar: {e}")))?;
        self.stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("piped stdout");
        self.reader = Some(Arc::new(Mutex::new(BufReader::new(stdout))));
        self.child = Some(child);
        self.dir = Some(tmp);
        self.initialized = false;
        Ok(())
    }

    /// One JSONL request/response round-trip with a wall-clock timeout.
    fn call(&mut self, req: &Value) -> Result<Value> {
        if self.child.is_none() {
            self.spawn()?;
        }
        let line = serde_json::to_string(req)? + "\n";
        {
            let stdin = self
                .stdin
                .as_mut()
                .ok_or_else(|| WeightXError::Value("sidecar stdin closed".into()))?;
            stdin
                .write_all(line.as_bytes())
                .and_then(|_| stdin.flush())
                .map_err(|e| WeightXError::Value(format!("sidecar write: {e}")))?;
        }
        let reader = self.reader.clone().expect("reader");
        let (tx, rx) = std::sync::mpsc::channel::<std::io::Result<String>>();
        std::thread::spawn(move || {
            let mut buf = String::new();
            let res = reader.lock().unwrap().read_line(&mut buf).map(|_| buf);
            let _ = tx.send(res);
        });
        let resp_line = match rx.recv_timeout(sidecar_timeout()) {
            Ok(Ok(l)) if !l.trim().is_empty() => l,
            Ok(Ok(_)) => {
                self.kill();
                return Err(WeightXError::Value("torch sidecar closed stdout (crashed?)".into()));
            }
            Ok(Err(e)) => {
                self.kill();
                return Err(WeightXError::Value(format!("torch sidecar read: {e}")));
            }
            Err(_) => {
                self.kill();
                return Err(WeightXError::Value(format!(
                    "torch sidecar timed out after {}s (WEIGHTX_TORCH_TIMEOUT)",
                    sidecar_timeout().as_secs()
                )));
            }
        };
        let resp: Value = serde_json::from_str(resp_line.trim())
            .map_err(|e| WeightXError::Value(format!("sidecar bad response: {e}")))?;
        if resp.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            let err = resp.get("error").and_then(|v| v.as_str()).unwrap_or("unknown");
            return Err(WeightXError::Value(format!("torch sidecar error: {err}")));
        }
        Ok(resp)
    }
}

// ---------------------------------------------------------------------------
// TorchArch: ModelArch implemented over the sidecar
// ---------------------------------------------------------------------------

pub struct TorchArch {
    native: Arc<dyn ModelArch>,
    hf_config: Option<Value>,
    inner: Mutex<SidecarState>,
}

impl TorchArch {
    pub fn new(native: Arc<dyn ModelArch>, hf_config: Option<Value>) -> Self {
        Self {
            native,
            hf_config,
            inner: Mutex::new(SidecarState::new()),
        }
    }

    /// Bring the sidecar's weight copy up to date with the workspace.
    fn ensure_sync(&self, st: &mut SidecarState, ws: &WeightWorkspace) -> Result<()> {
        if !st.initialized {
            if st.child.is_none() {
                st.spawn()?;
            }
            let dir = st.exchange_dir()?;
            let wpath = dir.join("weights.safetensors");
            ws.export_safetensors(&wpath, false)?;
            let cfg = self.native.config();
            let mut cfgv = cfg.to_value();
            cfgv["tie_word_embeddings"] = json!(cfg.tie_word_embeddings);
            cfgv["layer_norm_eps"] = json!(cfg.layer_norm_eps);
            st.call(&json!({
                "op": "init",
                "weights": wpath.to_string_lossy(),
                "config": cfgv,
                "hf_config": self.hf_config,
            }))?;
            st.initialized = true;
            st.synced_rev = ws.revision();
            st.synced_modified = ws.all_names().into_iter().filter(|n| ws.is_modified(n)).collect();
            return Ok(());
        }
        if ws.revision() == st.synced_rev {
            return Ok(());
        }
        let current: Vec<String> = ws.all_names().into_iter().filter(|n| ws.is_modified(n)).collect();
        let mut to_export: Vec<String> = current.clone();
        for n in &st.synced_modified {
            if !current.contains(n) {
                to_export.push(n.clone()); // reverted since last sync
            }
        }
        let dir = st.exchange_dir()?;
        let wpath = dir.join("sync.safetensors");
        if to_export.is_empty() {
            // Revision changed but no tensor differs from original: reset all
            // previously-synced modifications by exporting the full state.
            ws.export_safetensors(&wpath, false)?;
        } else {
            let pairs: Vec<(String, Tensor)> = to_export
                .iter()
                .filter_map(|n| ws.get_readonly(n).ok().map(|t| (n.clone(), t.clone())))
                .collect();
            tensor::save_safetensors(&wpath, &pairs)?;
        }
        st.call(&json!({"op": "sync", "weights": wpath.to_string_lossy()}))?;
        st.synced_rev = ws.revision();
        st.synced_modified = current;
        Ok(())
    }

    fn locked_forward(
        &self,
        ws: &WeightWorkspace,
        ids: &Array2<i64>,
        want_hidden: bool,
    ) -> Result<ForwardOut> {
        let mut st = self.inner.lock().unwrap();
        self.ensure_sync(&mut st, ws)?;
        let dir = st.exchange_dir()?;
        let ids_v: Vec<Vec<i64>> = ids.rows().into_iter().map(|r| r.to_vec()).collect();
        let resp = st.call(&json!({"op": "forward", "ids": ids_v, "want_hidden": want_hidden}))?;
        let shape: Vec<usize> = resp["logits_shape"]
            .as_array()
            .ok_or_else(|| WeightXError::Value("sidecar: no logits_shape".into()))?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let logits_t = tensor::load_safetensors(&dir.join("logits.safetensors"))?
            .into_iter()
            .next()
            .map(|(_, t)| t)
            .ok_or_else(|| WeightXError::Value("sidecar: empty logits file".into()))?;
        let logits = logits_t
            .into_shape_with_order((shape[0], shape[1], shape[2]))
            .map_err(|e| WeightXError::Value(format!("logits shape: {e}")))?
            .to_owned();
        let mut hidden_states = Vec::new();
        if want_hidden {
            let n = resp["n_hidden"].as_u64().unwrap_or(0) as usize;
            let hs = tensor::load_safetensors(&dir.join("hidden.safetensors"))?;
            let map: HashMap<String, Tensor> = hs.into_iter().collect();
            for i in 0..n {
                let t = map
                    .get(&format!("h{i}"))
                    .ok_or_else(|| WeightXError::Value(format!("sidecar: missing h{i}")))?;
                let shp = t.shape();
                hidden_states.push(
                    t.clone()
                        .into_shape_with_order((shp[0], shp[1], shp[2]))
                        .unwrap()
                        .to_owned(),
                );
            }
        }
        Ok(ForwardOut { logits, hidden_states })
    }
}

impl ModelArch for TorchArch {
    fn config(&self) -> &ModelConfig {
        self.native.config()
    }

    fn backend_name(&self) -> &'static str {
        "torch"
    }

    fn forward(&self, ws: &WeightWorkspace, ids: &Array2<i64>, want_hidden: bool) -> Result<ForwardOut> {
        self.locked_forward(ws, ids, want_hidden)
    }

    fn weight_names(&self) -> Vec<String> {
        self.native.weight_names()
    }

    fn mlp_out_names(&self) -> Vec<String> {
        self.native.mlp_out_names()
    }

    fn lens_logits(&self, ws: &WeightWorkspace, hidden: &[Array3<f32>]) -> Result<Vec<Array2<f32>>> {
        // Cheap head projection stays in Rust, applied to sidecar hidden states.
        self.native.lens_logits(ws, hidden)
    }

    fn weight_grads(&self, ws: &WeightWorkspace, ids: &Array2<i64>) -> Result<HashMap<String, Tensor>> {
        let mut st = self.inner.lock().unwrap();
        self.ensure_sync(&mut st, ws)?;
        let dir = st.exchange_dir()?;
        let ids_v: Vec<Vec<i64>> = ids.rows().into_iter().map(|r| r.to_vec()).collect();
        st.call(&json!({"op": "grads", "ids": ids_v}))?;
        Ok(tensor::load_safetensors(&dir.join("grads.safetensors"))?
            .into_iter()
            .collect())
    }

    fn mlp_keys_grads(
        &self,
        ws: &WeightWorkspace,
        seqs: &[Vec<i64>],
        need_grads: bool,
    ) -> Result<(HashMap<String, Array2<f64>>, HashMap<String, Array2<f64>>)> {
        let mut st = self.inner.lock().unwrap();
        self.ensure_sync(&mut st, ws)?;
        let dir = st.exchange_dir()?;
        st.call(&json!({"op": "keys_grads", "seqs": seqs, "need_grads": need_grads}))?;
        let keys = load_f64_2d(&dir.join("keys.safetensors"))?;
        let grads = if need_grads {
            load_f64_2d(&dir.join("kgrads.safetensors"))?
        } else {
            HashMap::new()
        };
        Ok((keys, grads))
    }

    fn eamit_override(
        &self,
        ws: &mut WeightWorkspace,
        target_tensors: &[String],
        facts: &[BatchFactSpec],
        lambda_cov: f64,
        apply_fisher_guard: bool,
        align_cfg: &EamitAlignConfig,
    ) -> Option<Result<EamitResult>> {
        Some(self.run_eamit(ws, target_tensors, facts, lambda_cov, apply_fisher_guard, align_cfg))
    }
}

impl TorchArch {
    /// Full EAMIT delegation: LAPACK-grade solve + embedding alignment in the Python sidecar,
    /// Fisher-guarded application and metrics on the Rust side.
    fn run_eamit(
        &self,
        ws: &mut WeightWorkspace,
        target_tensors: &[String],
        facts: &[BatchFactSpec],
        lambda_cov: f64,
        apply_fisher_guard: bool,
        align_cfg: &EamitAlignConfig,
    ) -> Result<EamitResult> {
        use crate::linalg::{self, matmul};

        let num_facts = facts.len();
        let mut shapes = Vec::new();
        for t in target_tensors {
            let shp = ws.shape(t)?;
            if shp.len() != 2 {
                return Err(WeightXError::Value(format!(
                    "EAMIT target {t:?} must be 2D, got {shp:?}"
                )));
            }
            shapes.push(shp);
        }
        let (d_out, d_in) = (shapes[shapes.len() - 1][0], shapes[shapes.len() - 1][1]);

        let mut k_base = Array2::<f64>::zeros((d_in, num_facts));
        let mut v_target = Array2::<f64>::zeros((d_out, num_facts));
        for (m, f) in facts.iter().enumerate() {
            if f.key_vector.len() != d_in || f.target_value_vector.len() != d_out {
                return Err(WeightXError::Value(format!(
                    "Fact {} dims mismatch: key len {} (need {d_in}), value len {} (need {d_out})",
                    f.fact_id, f.key_vector.len(), f.target_value_vector.len()
                )));
            }
            for i in 0..d_in {
                k_base[[i, m]] = f.key_vector[i];
            }
            for i in 0..d_out {
                v_target[[i, m]] = f.target_value_vector[i];
            }
        }
        let cumulative_output = |ws: &WeightWorkspace| -> Result<Array2<f64>> {
            let mut total = Array2::<f64>::zeros((d_out, num_facts));
            for tname in target_tensors {
                let w_l = linalg::to_f64_2d(ws.get_readonly(tname)?);
                if w_l.nrows() == d_out && w_l.ncols() == d_in {
                    total = total + matmul(&w_l, &k_base);
                }
            }
            Ok(total)
        };
        let init_out = cumulative_output(ws)?;
        let pre_res = &v_target - &init_out;
        let pre_mse = pre_res.iter().map(|x| x * x).sum::<f64>() / pre_res.len() as f64;

        // Fisher column means (same formula as the native Rust EAMIT path).
        let mut fisher_diag: HashMap<String, Vec<f64>> = HashMap::new();
        if apply_fisher_guard {
            let weights: Vec<(String, Tensor)> = target_tensors
                .iter()
                .filter_map(|t| ws.get_readonly(t).ok().map(|w| (t.clone(), w.clone())))
                .collect();
            if let Some(guard) = ws.fisher_guard.as_mut() {
                for (tname, w_t) in &weights {
                    let f_mat = guard.get_fisher(tname, w_t);
                    let f2d = linalg::to_f64_2d(&f_mat);
                    let l_out = f2d.nrows().max(1);
                    let col_f: Vec<f64> = (0..f2d.ncols())
                        .map(|j| f2d.column(j).iter().sum::<f64>() / l_out as f64)
                        .collect();
                    if linalg::mean(&col_f) > 1e-20 {
                        fisher_diag.insert(tname.clone(), col_f);
                    }
                }
            }
        }

        let facts_v: Vec<Value> = facts
            .iter()
            .map(|f| {
                json!({
                    "id": f.fact_id,
                    "key": f.key_vector,
                    "value": f.target_value_vector,
                    "layer_inputs": f.layer_inputs,
                })
            })
            .collect();

        let (edited, align_resp): (Vec<(String, Tensor)>, Value) = {
            let mut st = self.inner.lock().unwrap();
            self.ensure_sync(&mut st, ws)?;
            let dir = st.exchange_dir()?;
            let resp = st.call(&json!({
                "op": "eamit",
                "targets": target_tensors,
                "facts": facts_v,
                "lambda_cov": lambda_cov,
                "fisher_diag": fisher_diag,
                "align_config": {
                    "lambda_kl": align_cfg.lambda_kl,
                    "lambda_mse": align_cfg.lambda_mse,
                    "top_m": align_cfg.top_m,
                    "tau": align_cfg.tau,
                    "align_steps": align_cfg.align_steps,
                    "align_lr": align_cfg.align_lr,
                    "max_block_size": align_cfg.max_block_size,
                },
            }))?;
            (tensor::load_safetensors(&dir.join("edited.safetensors"))?, resp)
        };

        let mut per_tensor_deltas = HashMap::new();
        for (tname, new_t) in &edited {
            let w_before = linalg::to_f64_2d(ws.get_readonly(tname)?);
            ws.set_tensor(tname, new_t, apply_fisher_guard, false)?;
            let w_after = linalg::to_f64_2d(ws.get_readonly(tname)?);
            per_tensor_deltas.insert(tname.clone(), linalg::l2_norm_flat(&(&w_after - &w_before)));
        }

        let final_out = cumulative_output(ws)?;
        let post_res = &v_target - &final_out;
        let post_mse = post_res.iter().map(|x| x * x).sum::<f64>() / post_res.len() as f64;
        let rel_red = (pre_mse - post_mse) / pre_mse.max(1e-12);

        Ok(EamitResult {
            num_facts,
            target_tensors: target_tensors.to_vec(),
            pre_edit_mse: pre_mse,
            post_edit_mse: post_mse,
            relative_error_reduction: rel_red,
            pre_alignment_kl: align_resp["pre_alignment_kl"].as_f64().unwrap_or(0.0),
            post_alignment_kl: align_resp["post_alignment_kl"].as_f64().unwrap_or(0.0),
            pre_alignment_top_m_mse: align_resp["pre_alignment_top_m_mse"].as_f64().unwrap_or(0.0),
            post_alignment_top_m_mse: align_resp["post_alignment_top_m_mse"].as_f64().unwrap_or(0.0),
            per_tensor_l2_deltas: per_tensor_deltas,
            fisher_protected: apply_fisher_guard && ws.fisher_guard.is_some(),
        })
    }
}

/// Load a safetensors file of F64 matrices (keys/grads exchange).
fn load_f64_2d(path: &Path) -> Result<HashMap<String, Array2<f64>>> {
    let data = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&data)?;
    let mut out = HashMap::new();
    for (name, view) in st.tensors() {
        if view.dtype() != safetensors::Dtype::F64 {
            return Err(WeightXError::Value(format!(
                "sidecar exchange tensor {name}: expected F64, got {:?}",
                view.dtype()
            )));
        }
        let shp = view.shape();
        if shp.len() != 2 {
            return Err(WeightXError::Value(format!(
                "sidecar exchange tensor {name}: expected 2D, got {shp:?}"
            )));
        }
        let vals: Vec<f64> = view
            .data()
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        out.insert(
            name.to_string(),
            Array2::from_shape_vec((shp[0], shp[1]), vals)
                .map_err(|e| WeightXError::Value(e.to_string()))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::llama::LlamaModel;

    fn test_env_ready() -> bool {
        // Tests run from the repo; make sure discovery works even if cwd differs.
        if std::env::var("WEIGHTX_HOME").is_err() {
            if let Some(root) = repo_root() {
                std::env::set_var("WEIGHTX_HOME", root);
            }
        }
        torch_available()
    }

    fn tiny_ws() -> (WeightWorkspace, Arc<dyn ModelArch>) {
        let cfg = ModelConfig {
            model_type: "llama".into(),
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            ..Default::default()
        };
        let arch = LlamaModel::new(cfg.clone());
        let mut tensors: Vec<(String, Tensor)> = Vec::new();
        let mut seed: u64 = 12345;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        for name in arch.weight_names() {
            let shape: Vec<usize> = if name.contains("layernorm") || name == "model.norm.weight" {
                vec![cfg.hidden_size]
            } else if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj") || name.contains("o_proj") {
                vec![cfg.hidden_size, cfg.hidden_size]
            } else if name.contains("gate_proj") || name.contains("up_proj") {
                vec![cfg.intermediate_size, cfg.hidden_size]
            } else if name.contains("down_proj") {
                vec![cfg.hidden_size, cfg.intermediate_size]
            } else {
                vec![cfg.vocab_size, cfg.hidden_size]
            };
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| rnd() * 0.1).collect();
            tensors.push((name, tensor::from_vec(&shape, data)));
        }
        let ws = WeightWorkspace::from_tensors(tensors, None);
        (ws, Arc::new(arch))
    }

    fn ids(seq: usize) -> Array2<i64> {
        Array2::from_shape_vec((1, seq), (0..seq).map(|i| (i * 7 % 32) as i64).collect()).unwrap()
    }

    #[test]
    fn hybrid_forward_parity() {
        if !test_env_ready() {
            eprintln!("skip: python+torch unavailable");
            return;
        }
        let (ws, native) = tiny_ws();
        let torch = TorchArch::new(native.clone(), None);
        let x = ids(8);
        let a = native.forward(&ws, &x, true).unwrap();
        let b = torch.forward(&ws, &x, true).unwrap();
        assert_eq!(a.logits.shape(), b.logits.shape());
        let sa: f64 = a.logits.iter().map(|&v| v as f64).sum();
        let sb: f64 = b.logits.iter().map(|&v| v as f64).sum();
        let rel = (sa - sb).abs() / sa.abs().max(1e-9);
        assert!(rel < 1e-4, "forward parity rel={rel}");
        assert_eq!(a.hidden_states.len(), b.hidden_states.len());
    }

    #[test]
    fn hybrid_grads_parity() {
        if !test_env_ready() {
            eprintln!("skip: python+torch unavailable");
            return;
        }
        let (ws, native) = tiny_ws();
        let torch = TorchArch::new(native.clone(), None);
        let x = ids(8);
        let ga = native.weight_grads(&ws, &x).unwrap();
        let gb = torch.weight_grads(&ws, &x).unwrap();
        let sq = |m: &HashMap<String, Tensor>| -> f64 {
            m.values().map(|t| t.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sum()
        };
        let (a, b) = (sq(&ga), sq(&gb));
        let rel = (a - b).abs() / a.abs().max(1e-9);
        assert!(rel < 1e-3, "grads parity rel={rel} (a={a}, b={b})");
    }

    #[test]
    fn hybrid_eamit_parity() {
        if !test_env_ready() {
            eprintln!("skip: python+torch unavailable");
            return;
        }
        let (mut ws_a, native) = tiny_ws();
        let mut ws_b = WeightWorkspace::from_tensors(
            ws_a.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            None,
        );
        let torch = TorchArch::new(native.clone(), None);
        ws_b.arch = Some(Arc::new(torch));
        let targets = vec![
            "model.layers.0.mlp.down_proj.weight".to_string(),
            "model.layers.1.mlp.down_proj.weight".to_string(),
        ];
        let facts: Vec<BatchFactSpec> = (0..5)
            .map(|i| {
                let key: Vec<f64> = (0..32).map(|j| ((i * 31 + j * 17) % 13) as f64 / 13.0 - 0.5).collect();
                let val: Vec<f64> = (0..16).map(|j| ((i * 7 + j * 5) % 11) as f64 / 11.0 - 0.5).collect();
                BatchFactSpec::new(format!("f{i}"), key, val)
            })
            .collect();
        let ra = crate::eamit::apply_eamit_batch(&mut ws_a, &targets, &facts, 1e-2, false).unwrap();
        let rb = crate::eamit::apply_eamit_batch(&mut ws_b, &targets, &facts, 1e-2, false).unwrap();
        let rel = (ra.post_edit_mse - rb.post_edit_mse).abs() / ra.post_edit_mse.abs().max(1e-12);
        assert!(rel < 1e-4, "eamit parity rel={rel} (a={}, b={})", ra.post_edit_mse, rb.post_edit_mse);
        let rel_kl = (ra.post_alignment_kl - rb.post_alignment_kl).abs() / ra.post_alignment_kl.abs().max(1e-9);
        assert!(rel_kl < 1e-4, "eamit KL parity rel={rel_kl}");
        assert!(rb.post_edit_mse < rb.pre_edit_mse * 0.5);
    }
}