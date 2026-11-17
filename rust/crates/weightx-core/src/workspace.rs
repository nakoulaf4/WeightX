// SPDX-License-Identifier: GPL-3.0-or-later
//! L3 WeightWorkspace: isolated tensor store with describe, safe slicing, checkpoints,
//! SHA-256 provenance, diff telemetry, and IWCKEdit Fisher protection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use ndarray::IxDyn;
use regex::Regex;
use serde_json::{json, Value};

use crate::error::{Result, WeightXError};
use crate::fisher::{FisherGuard, FisherProtectionReport};
use crate::linalg::{self, median, quantile};
use crate::tensor::{self, Tensor};

// ---------------------------------------------------------------------------
// Safe slice parsing (no eval)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum SliceItem {
    All,
    Ellipsis,
    Index(i64),
    Range { start: Option<i64>, stop: Option<i64>, step: Option<i64> },
}

pub fn parse_safe_slice(index_expr: Option<&str>) -> Result<Vec<SliceItem>> {
    let text = match index_expr {
        None => return Ok(vec![]),
        Some(t) => t.trim(),
    };
    if text.is_empty() {
        return Ok(vec![]);
    }
    let text = text.strip_prefix('[').and_then(|t| t.strip_suffix(']')).unwrap_or(text);
    let mut out = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() || part == ":" {
            out.push(SliceItem::All);
            continue;
        }
        if part == "..." {
            out.push(SliceItem::Ellipsis);
            continue;
        }
        let re = Regex::new(r"^\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*)?)?$").unwrap();
        let m = re
            .captures(part)
            .ok_or_else(|| WeightXError::Value(format!("Invalid slice component: {part:?}")))?;
        if !part.contains(':') {
            let v: i64 = m.get(1).unwrap().as_str().parse().unwrap();
            out.push(SliceItem::Index(v));
        } else {
            let g = |i: usize| m.get(i).map(|x| x.as_str().parse::<i64>().unwrap());
            out.push(SliceItem::Range { start: g(1), stop: g(2), step: g(3) });
        }
    }
    Ok(out)
}

fn dim_indices(item: &SliceItem, dim: usize) -> Result<(Vec<usize>, bool)> {
    let d = dim as i64;
    let norm = |i: i64| -> Result<usize> {
        let idx = if i < 0 { d + i } else { i };
        if idx < 0 || idx >= d {
            return Err(WeightXError::Value(format!("Index {i} out of bounds for dim {dim}")));
        }
        Ok(idx as usize)
    };
    match item {
        SliceItem::All => Ok(((0..dim).collect(), false)),
        SliceItem::Index(i) => Ok((vec![norm(*i)?], true)),
        SliceItem::Range { start, stop, step } => {
            let step = step.unwrap_or(1);
            if step == 0 {
                return Err(WeightXError::Value("Slice step cannot be zero".into()));
            }
            let (s, e) = if step > 0 {
                (start.unwrap_or(0), stop.unwrap_or(d))
            } else {
                (start.unwrap_or(d - 1), stop.unwrap_or(-d - 1))
            };
            let s = if s < 0 { (d + s).max(0) } else { s.min(d) };
            let e = if e < 0 { (d + e).max(-1) } else { e.min(d) };
            let mut idxs = Vec::new();
            if step > 0 {
                let mut i = s;
                while i < e {
                    idxs.push(i as usize);
                    i += step;
                }
            } else {
                let mut i = s;
                while i > e {
                    if i >= 0 && i < d {
                        idxs.push(i as usize);
                    }
                    i += step;
                }
            }
            Ok((idxs, false))
        }
        SliceItem::Ellipsis => Ok(((0..dim).collect(), false)),
    }
}

/// Apply parsed slice items to a tensor (numpy semantics), returning an owned tensor.
pub fn apply_slice(t: &Tensor, items: &[SliceItem]) -> Result<Tensor> {
    if items.is_empty() {
        return Ok(t.clone());
    }
    let ndim = t.ndim();
    let n_ellipsis = items.iter().filter(|i| **i == SliceItem::Ellipsis).count();
    if n_ellipsis > 1 {
        return Err(WeightXError::Value("Only one ellipsis allowed".into()));
    }
    let mut expanded: Vec<SliceItem> = Vec::new();
    let concrete = items.iter().filter(|i| **i != SliceItem::Ellipsis).count();
    for it in items {
        if *it == SliceItem::Ellipsis {
            for _ in 0..(ndim.saturating_sub(concrete)) {
                expanded.push(SliceItem::All);
            }
        } else {
            expanded.push(it.clone());
        }
    }
    while expanded.len() < ndim {
        expanded.push(SliceItem::All);
    }
    if expanded.len() > ndim {
        return Err(WeightXError::Value("Too many slice indices".into()));
    }
    let mut per_dim: Vec<Vec<usize>> = Vec::new();
    let mut out_shape: Vec<usize> = Vec::new();
    for (d, it) in expanded.iter().enumerate() {
        let (idxs, removes) = dim_indices(it, t.shape()[d])?;
        if !removes {
            out_shape.push(idxs.len());
        }
        per_dim.push(idxs);
    }
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let total: usize = per_dim.iter().map(|v| v.len()).product();
    let mut data = Vec::with_capacity(total);
    let mut counters = vec![0usize; ndim];
    for _ in 0..total {
        let idx: Vec<usize> = counters
            .iter()
            .enumerate()
            .map(|(d, &c)| per_dim[d][c])
            .collect();
        data.push(t[IxDyn(&idx)]);
        // increment mixed-radix counter
        for d in (0..ndim).rev() {
            counters[d] += 1;
            if counters[d] < per_dim[d].len() {
                break;
            }
            counters[d] = 0;
        }
    }
    // If all dims removed (scalar), numpy returns 0-d; keep shape [] semantics via [1]
    let shape = if expanded.iter().all(|i| matches!(i, SliceItem::Index(_))) {
        vec![]
    } else {
        out_shape
    };
    if shape.is_empty() {
        Ok(Tensor::from_shape_vec(IxDyn(&[]), data).unwrap())
    } else {
        Ok(tensor::from_vec(&shape, data))
    }
}

// ---------------------------------------------------------------------------
// Diff telemetry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TensorDiffRecord {
    pub name: String,
    pub numel: usize,
    pub changed_count: usize,
    pub l2_delta: f64,
    pub max_abs_delta: f64,
    pub rel_delta: f64,
    pub sha256_after: String,
}

impl TensorDiffRecord {
    pub fn to_value(&self) -> Value {
        json!({
            "name": self.name,
            "numel": self.numel,
            "changed_count": self.changed_count,
            "l2_delta": (self.l2_delta * 1e6).round() / 1e6,
            "max_abs_delta": (self.max_abs_delta * 1e6).round() / 1e6,
            "rel_delta": (self.rel_delta * 1e6).round() / 1e6,
            "sha256_after": &self.sha256_after[..16],
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct WorkspaceDiffSummary {
    pub total_params: usize,
    pub changed_params: usize,
    pub tensors: Vec<TensorDiffRecord>,
}

impl WorkspaceDiffSummary {
    pub fn changed_fraction(&self) -> f64 {
        self.changed_params as f64 / self.total_params.max(1) as f64
    }
    pub fn to_value(&self) -> Value {
        json!({
            "total_params": self.total_params,
            "changed_params": self.changed_params,
            "changed_pct": (self.changed_fraction() * 100.0 * 1e5).round() / 1e5,
            "tensors_changed": self.tensors.len(),
            "tensors": self.tensors.iter().map(|t| t.to_value()).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

pub struct WeightWorkspace {
    live: IndexMap<String, Tensor>,
    dtypes: HashMap<String, String>,
    original: HashMap<String, Tensor>,
    checkpoints: HashMap<String, HashMap<String, Tensor>>,
    pub fisher_guard: Option<FisherGuard>,
    /// Optional runtime architecture (set by the harness). Used by the hybrid
    /// torch backend to delegate heavy numerics (EAMIT on real-size models).
    pub arch: Option<std::sync::Arc<dyn crate::model::ModelArch>>,
    revision: u64,
}

impl WeightWorkspace {
    pub fn from_tensors(tensors: Vec<(String, Tensor)>, fisher_guard: Option<FisherGuard>) -> Self {
        Self::from_tensors_with_dtypes(tensors, HashMap::new(), fisher_guard)
    }

    /// Build a workspace preserving the original on-disk dtype names (values are f32).
    pub fn from_tensors_with_dtypes(
        tensors: Vec<(String, Tensor)>,
        source_dtypes: HashMap<String, String>,
        fisher_guard: Option<FisherGuard>,
    ) -> Self {
        let mut live = IndexMap::new();
        let mut dtypes = HashMap::new();
        for (name, t) in tensors {
            let dt = source_dtypes.get(&name).cloned().unwrap_or_else(|| "float32".to_string());
            dtypes.insert(name.clone(), dt);
            live.insert(name, t);
        }
        let mut ws = Self {
            live,
            dtypes,
            original: HashMap::new(),
            checkpoints: HashMap::new(),
            fisher_guard,
            arch: None,
            revision: 0,
        };
        ws.checkpoint("baseline");
        ws
    }

    pub fn from_map(map: HashMap<String, Tensor>, fisher_guard: Option<FisherGuard>) -> Self {
        let mut pairs: Vec<_> = map.into_iter().collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        Self::from_tensors(pairs, fisher_guard)
    }

    pub fn names(&self, pattern: Option<&str>, layer_idx: Option<i64>) -> Vec<String> {
        let mut result: Vec<String> = self.live.keys().cloned().collect();
        if let Some(idx) = layer_idx {
            let re = Regex::new(&format!(r"(?:^|\.)({})\.", idx)).unwrap();
            result.retain(|n| re.is_match(n));
        }
        if let Some(pat) = pattern {
            if !pat.is_empty() {
                let glob = glob::Pattern::new(pat).ok();
                let mut matched: Vec<String> = result
                    .iter()
                    .filter(|n| {
                        let g = glob
                            .as_ref()
                            .map(|g| g.matches_with(n, glob::MatchOptions { case_sensitive: false, ..Default::default() }))
                            .unwrap_or(false);
                        g || n.contains(pat)
                    })
                    .cloned()
                    .collect();
                if matched.is_empty() {
                    if let Ok(rx) = Regex::new(pat) {
                        matched = result.iter().filter(|n| rx.is_match(n)).cloned().collect();
                    }
                }
                result = matched;
            }
        }
        result
    }

    pub fn all_names(&self) -> Vec<String> {
        self.live.keys().cloned().collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.live.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.live
            .get(name)
            .map(|t| t.shape().to_vec())
            .ok_or_else(|| WeightXError::UnknownTensor(name.to_string()))
    }

    pub fn dtype(&self, name: &str) -> Result<String> {
        self.dtypes
            .get(name)
            .cloned()
            .ok_or_else(|| WeightXError::UnknownTensor(name.to_string()))
    }

    pub fn total_params(&self) -> usize {
        self.live.values().map(|t| t.len()).sum()
    }

    /// Monotonic mutation counter (used by the torch sidecar to sync weights).
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn get_readonly(&self, name: &str) -> Result<&Tensor> {
        self.live
            .get(name)
            .ok_or_else(|| WeightXError::UnknownTensor(name.to_string()))
    }

    pub fn get_original(&self, name: &str) -> Result<&Tensor> {
        if let Some(t) = self.original.get(name) {
            Ok(t)
        } else {
            self.get_readonly(name)
        }
    }

    pub fn is_modified(&self, name: &str) -> bool {
        self.original.contains_key(name)
    }

    pub fn list_tensors_table(&self, pattern: Option<&str>, layer_idx: Option<i64>) -> Vec<Value> {
        self.names(pattern, layer_idx)
            .into_iter()
            .map(|name| {
                let arr = &self.live[&name];
                let l2 = arr.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
                json!({
                    "name": name,
                    "shape": arr.shape().iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"),
                    "dtype": self.dtypes[&name],
                    "numel": arr.len(),
                    "l2_norm": (l2 * 1e4).round() / 1e4,
                    "modified": self.original.contains_key(&name),
                })
            })
            .collect()
    }

    pub fn describe_tensor(&self, name: &str, top_k: usize) -> Result<Value> {
        let arr = self.get_readonly(name)?;
        let flat: Vec<f64> = arr.iter().map(|v| *v as f64).collect();
        let n = flat.len();
        if n == 0 {
            return Ok(json!({"name": name, "shape": arr.shape(), "numel": 0}));
        }
        let mean = linalg::mean(&flat);
        let std = linalg::std_pop(&flat);
        let lo = flat.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let abs_mean = flat.iter().map(|v| v.abs()).sum::<f64>() / n as f64;
        let l2_norm = flat.iter().map(|v| v * v).sum::<f64>().sqrt();
        let frac_zero = flat.iter().filter(|&&v| v == 0.0).count() as f64 / n as f64;
        let qs: Vec<f64> = [0.01, 0.25, 0.5, 0.75, 0.99]
            .iter()
            .map(|&q| (quantile(&flat, q) * 1e5).round() / 1e5)
            .collect();
        let nonfinite = flat.iter().filter(|v| !v.is_finite()).count();

        let r6 = |v: f64| (v * 1e6).round() / 1e6;
        let r5 = |v: f64| (v * 1e5).round() / 1e5;
        let mut info = json!({
            "name": name,
            "shape": arr.shape(),
            "dtype": self.dtypes[name],
            "numel": n,
            "mean": r6(mean),
            "std": r6(std),
            "min": r6(lo),
            "max": r6(hi),
            "abs_mean": r6(abs_mean),
            "l2_norm": r5(l2_norm),
            "zeros_pct": (frac_zero * 100.0 * 1e3).round() / 1e3,
            "nonfinite_count": nonfinite,
            "quantiles_1_25_50_75_99": qs,
        });

        if arr.ndim() >= 2 {
            let m2d = tensor::as_2d(arr).mapv(|v| v as f64);
            let rn = linalg::row_norms(&m2d);
            let cn = linalg::col_norms(&m2d);
            let r_med = median(&rn);
            let c_med = median(&cn);
            let r_order = linalg::argsort_desc(&rn);
            let c_order = linalg::argsort_desc(&cn);
            info["row_norm_median"] = json!(r5(r_med));
            info["col_norm_median"] = json!(r5(c_med));
            info["largest_rows"] = json!(r_order
                .iter()
                .take(top_k)
                .map(|&i| json!({
                    "idx": i,
                    "l2": r5(rn[i]),
                    "ratio_to_median": ((rn[i] / r_med.max(1e-12)) * 1e3).round() / 1e3,
                }))
                .collect::<Vec<_>>());
            info["smallest_rows"] = json!(r_order
                .iter()
                .rev()
                .take(top_k.min(4))
                .map(|&i| json!({"idx": i, "l2": r5(rn[i])}))
                .collect::<Vec<_>>());
            info["largest_cols"] = json!(c_order
                .iter()
                .take(top_k.min(4))
                .map(|&i| json!({"idx": i, "l2": r5(cn[i])}))
                .collect::<Vec<_>>());

            let min_dim = m2d.nrows().min(m2d.ncols());
            if min_dim <= 2048 && m2d.len() <= 64_000_000 {
                let sv = linalg::singular_values(&m2d);
                let total_e: f64 = sv.iter().map(|v| v * v).sum();
                let mut cum = 0.0;
                let mut rank_90 = sv.len();
                for (i, v) in sv.iter().enumerate() {
                    cum += v * v;
                    if cum / total_e.max(1e-30) >= 0.90 {
                        rank_90 = i + 1;
                        break;
                    }
                }
                info["top_singular_values"] = json!(sv
                    .iter()
                    .take(top_k)
                    .map(|v| (v * 1e4).round() / 1e4)
                    .collect::<Vec<_>>());
                info["rank_90pct_energy"] = json!(format!("{}/{}", rank_90, sv.len()));
            }
        }
        Ok(info)
    }

    pub fn read_slice(
        &self,
        name: &str,
        index_expr: Option<&str>,
        max_elements: usize,
        precision: usize,
    ) -> Result<Value> {
        let arr = self.get_readonly(name)?;
        let items = parse_safe_slice(index_expr)?;
        let sliced = apply_slice(arr, &items)?;
        let flat: Vec<f32> = sliced.iter().copied().collect();
        let truncated = flat.len() > max_elements;
        let text = if truncated {
            flat[..max_elements]
                .iter()
                .map(|v| crate::toon::fmt_g(*v as f64, precision))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            format_array(&sliced, precision)
        };
        Ok(json!({
            "name": name,
            "full_shape": arr.shape(),
            "slice_expr": index_expr.unwrap_or(":"),
            "slice_shape": sliced.shape(),
            "slice_numel": sliced.len(),
            "truncated": truncated,
            "values": text,
        }))
    }

    fn ensure_snapshot(&mut self, name: &str) -> Result<()> {
        if !self.live.contains_key(name) {
            return Err(WeightXError::UnknownTensor(name.to_string()));
        }
        if !self.original.contains_key(name) {
            let cur = self.live[name].clone();
            self.original.insert(name.to_string(), cur);
        }
        Ok(())
    }

    pub fn set_tensor(
        &mut self,
        name: &str,
        new_value: &Tensor,
        apply_fisher_guard: bool,
        allow_outlier_repair: bool,
    ) -> Result<Option<FisherProtectionReport>> {
        self.ensure_snapshot(name)?;
        let cur_shape = self.live[name].shape().to_vec();
        let mut arr = new_value.clone();
        if arr.shape() != cur_shape.as_slice() {
            arr = broadcast_to(&arr, &cur_shape)?;
        }
        let mut report = None;
        if apply_fisher_guard {
            if let Some(guard) = self.fisher_guard.as_mut() {
                if guard.enabled {
                    let orig = self.original[name].clone();
                    let (protected, rep) =
                        guard.protect_update(name, &orig, &arr, allow_outlier_repair);
                    arr = protected;
                    report = Some(rep);
                }
            }
        }
        self.live.insert(name.to_string(), arr);
        self.revision += 1;
        Ok(report)
    }

    pub fn diff(&self) -> WorkspaceDiffSummary {
        let mut records = Vec::new();
        let mut changed_total = 0usize;
        for (name, orig) in &self.original {
            let cur = &self.live[name];
            let mut n_changed = 0usize;
            let mut l2_sq = 0.0f64;
            let mut max_d = 0.0f64;
            for (c, o) in cur.iter().zip(orig.iter()) {
                let d = (*c as f64) - (*o as f64);
                if d.abs() > 1e-12 {
                    n_changed += 1;
                }
                l2_sq += d * d;
                max_d = max_d.max(d.abs());
            }
            if n_changed == 0 {
                continue;
            }
            let base_l2 = orig.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
            records.push(TensorDiffRecord {
                name: name.clone(),
                numel: cur.len(),
                changed_count: n_changed,
                l2_delta: l2_sq.sqrt(),
                max_abs_delta: max_d,
                rel_delta: l2_sq.sqrt() / base_l2.max(1e-12),
                sha256_after: tensor::sha256_tensor(cur),
            });
            changed_total += n_changed;
        }
        WorkspaceDiffSummary {
            total_params: self.total_params(),
            changed_params: changed_total,
            tensors: records,
        }
    }

    pub fn checkpoint(&mut self, label: &str) {
        let mut snap = HashMap::new();
        for name in self.original.keys() {
            snap.insert(name.clone(), self.live[name].clone());
        }
        self.checkpoints.insert(label.to_string(), snap);
    }

    pub fn restore(&mut self, label: &str) -> Result<()> {
        let snap = self
            .checkpoints
            .get(label)
            .ok_or_else(|| {
                WeightXError::UnknownCheckpoint(
                    label.to_string(),
                    self.checkpoints.keys().cloned().collect(),
                )
            })?
            .clone();
        for (name, orig) in self.original.clone() {
            let val = snap.get(&name).unwrap_or(&orig);
            self.live.insert(name, val.clone());
        }
        self.revision += 1;
        Ok(())
    }

    pub fn revert_to_original(&mut self, name: Option<&str>) -> Result<()> {
        let targets: Vec<String> = match name {
            Some(n) => vec![n.to_string()],
            None => self.original.keys().cloned().collect(),
        };
        for n in targets {
            if let Some(orig) = self.original.remove(&n) {
                self.live.insert(n, orig);
            }
        }
        self.revision += 1;
        Ok(())
    }

    pub fn export_safetensors(&self, path: &Path, only_changed: bool) -> Result<PathBuf> {
        let tensors: Vec<(String, Tensor)> = if only_changed && !self.original.is_empty() {
            self.original
                .keys()
                .map(|k| (k.clone(), self.live[k].clone()))
                .collect()
        } else {
            self.live
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };
        tensor::save_safetensors(path, &tensors)?;
        Ok(path.to_path_buf())
    }

    /// Iterate over all live tensors (read-only).
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.live.iter()
    }
}

/// Assign `values` into the region of `t` selected by `items` (numpy `t[idx] = values`).
pub fn assign_slice(t: &mut Tensor, items: &[SliceItem], values: &Tensor) -> Result<()> {
    if items.is_empty() {
        if t.shape() != values.shape() {
            return Err(WeightXError::Value(format!(
                "Cannot assign shape {:?} into {:?}",
                values.shape(),
                t.shape()
            )));
        }
        t.assign(values);
        return Ok(());
    }
    let region = apply_slice(t, items)?;
    if region.len() != values.len() {
        return Err(WeightXError::Value(format!(
            "Slice region has {} elements, got {} values",
            region.len(),
            values.len()
        )));
    }
    let ndim = t.ndim();
    let n_ellipsis = items.iter().filter(|i| **i == SliceItem::Ellipsis).count();
    let concrete = items.iter().filter(|i| **i != SliceItem::Ellipsis).count();
    let mut expanded: Vec<SliceItem> = Vec::new();
    for it in items {
        if *it == SliceItem::Ellipsis {
            for _ in 0..(ndim.saturating_sub(concrete)) {
                expanded.push(SliceItem::All);
            }
        } else {
            expanded.push(it.clone());
        }
    }
    while expanded.len() < ndim {
        expanded.push(SliceItem::All);
    }
    let _ = n_ellipsis;
    let mut per_dim: Vec<Vec<usize>> = Vec::new();
    for (d, it) in expanded.iter().enumerate() {
        let (idxs, _) = dim_indices(it, t.shape()[d])?;
        per_dim.push(idxs);
    }
    let total: usize = per_dim.iter().map(|v| v.len()).product();
    let mut counters = vec![0usize; ndim];
    for k in 0..total {
        let idx: Vec<usize> = counters
            .iter()
            .enumerate()
            .map(|(d, &c)| per_dim[d][c])
            .collect();
        t[IxDyn(&idx)] = values.iter().nth(k).copied().unwrap_or(0.0);
        for d in (0..ndim).rev() {
            counters[d] += 1;
            if counters[d] < per_dim[d].len() {
                break;
            }
            counters[d] = 0;
        }
    }
    Ok(())
}

/// numpy-style broadcast of `arr` to `shape`.
pub fn broadcast_to(arr: &Tensor, shape: &[usize]) -> Result<Tensor> {
    let src = arr.shape();
    let nd = shape.len();
    if src.len() > nd {
        return Err(WeightXError::Value(format!(
            "Cannot broadcast shape {:?} to {:?}",
            src, shape
        )));
    }
    let mut padded = vec![1usize; nd - src.len()];
    padded.extend_from_slice(src);
    for (a, b) in padded.iter().zip(shape.iter()) {
        if a != b && *a != 1 {
            return Err(WeightXError::Value(format!(
                "Cannot broadcast shape {:?} to {:?}",
                src, shape
            )));
        }
    }
    let total: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total);
    let src_strides: Vec<usize> = {
        let mut s = vec![1usize; nd];
        for i in (0..nd.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * padded[i + 1];
        }
        // zero stride for broadcast dims
        for i in 0..nd {
            if padded[i] == 1 && shape[i] != 1 {
                s[i] = 0;
            }
        }
        s
    };
    let mut counters = vec![0usize; nd];
    for _ in 0..total {
        let mut flat_idx = 0usize;
        for d in 0..nd {
            flat_idx += counters[d] * src_strides[d];
        }
        data.push(arr.iter().nth(flat_idx).copied().unwrap_or(0.0));
        for d in (0..nd).rev() {
            counters[d] += 1;
            if counters[d] < shape[d] {
                break;
            }
            counters[d] = 0;
        }
    }
    Ok(tensor::from_vec(shape, data))
}

/// numpy.array2string-like formatting with fixed precision.
fn format_array(t: &Tensor, precision: usize) -> String {
    fn fmt_rec(t: &Tensor, precision: usize) -> String {
        if t.ndim() == 0 {
            return format!("{:.*}", precision, t.iter().next().copied().unwrap_or(0.0));
        }
        if t.ndim() == 1 {
            let inner = t
                .iter()
                .map(|v| format!("{:.*}", precision, v))
                .collect::<Vec<_>>()
                .join(" ");
            return format!("[{}]", inner);
        }
        let rows: Vec<String> = t
            .outer_iter()
            .map(|sub| fmt_rec(&sub.to_owned(), precision))
            .collect();
        format!("[{}]", rows.join("\n "))
    }
    fmt_rec(t, precision)
}