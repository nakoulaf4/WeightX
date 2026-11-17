// SPDX-License-Identifier: GPL-3.0-or-later
//! BrainSurgery-WX: declarative YAML plan compiler, assertion validator, executor.
//! Port of `weightx.declarative.plan_engine` with atomic rollback and Fisher protection.

use serde_json::{json, Value};

use crate::error::{Result, WeightXError};
use crate::linalg::{self, matmul, median, transpose};
use crate::eamit::{self, BatchFactSpec, EamitAlignConfig};
use crate::tensor::{self, Tensor};
use crate::workspace::{assign_slice, parse_safe_slice, SliceItem, WeightWorkspace};

pub fn plan_err(msg: impl Into<String>) -> WeightXError {
    WeightXError::PlanValidation(msg.into())
}

#[derive(Debug, Clone)]
pub struct StepExecutionResult {
    pub step_index: usize,
    pub op_name: String,
    pub matched_tensors: usize,
    pub status: String,
    pub detail: String,
}

impl StepExecutionResult {
    pub fn to_value(&self) -> Value {
        json!({
            "step": self.step_index,
            "op": self.op_name,
            "matched": self.matched_tensors,
            "status": self.status,
            "detail": self.detail,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PlanExecutionReport {
    pub success: bool,
    pub dry_run: bool,
    pub assertions_verified: usize,
    pub mutations_executed: usize,
    pub steps: Vec<StepExecutionResult>,
    pub diff_summary: Value,
    pub error: Option<String>,
}

impl PlanExecutionReport {
    pub fn to_value(&self) -> Value {
        json!({
            "success": self.success,
            "dry_run": self.dry_run,
            "assertions_verified": self.assertions_verified,
            "mutations_executed": self.mutations_executed,
            "error": self.error,
            "steps": self.steps.iter().map(|s| s.to_value()).collect::<Vec<_>>(),
            "diff_summary": self.diff_summary,
        })
    }
    pub fn to_toon(&self) -> String {
        crate::toon::dumps_toon(&self.to_value())
    }
}

const FORBIDDEN_FILESYSTEM_OPS: &[&str] = &["load", "save", "execute"];

const MUTATION_OPS: &[&str] = &[
    "copy", "assign", "scale", "scale_", "fill", "fill_", "zeroes", "ones",
    "add", "add_", "subtract", "subtract_", "multiply", "multiply_",
    "clamp", "clamp_", "phlora_", "svd_low_rank_", "spectral_norm_cap_",
    "project_out_direction_", "abliterate_direction_", "lerp_", "slerp_",
    "ties_trim_", "prune_magnitude_", "rescale_outlier_rows_", "replace_nonfinite_",
    "rank_one_update_", "closed_form_ridge_", "eamit_batch_",
];

const READONLY_OPS: &[&str] = &["assert", "dump", "diff"];

// ---------------------------------------------------------------------------
// Tensor references
// ---------------------------------------------------------------------------

pub fn parse_tensor_ref(ref_str: &str) -> Result<(Option<String>, String, Vec<SliceItem>)> {
    if ref_str.trim().is_empty() {
        return Err(plan_err(format!(
            "Tensor reference must be a non-empty string, got {ref_str:?}"
        )));
    }
    let parts: Vec<&str> = ref_str.trim().split("::").collect();
    let mut alias: Option<String> = None;
    let mut pattern: String;
    let mut slice_tup: Vec<SliceItem> = Vec::new();

    match parts.len() {
        1 => pattern = parts[0].to_string(),
        2 => {
            if parts[1].trim().starts_with('[') {
                pattern = parts[0].to_string();
                slice_tup = parse_safe_slice(Some(parts[1].trim()))?;
            } else {
                alias = Some(parts[0].to_string());
                pattern = parts[1].to_string();
            }
        }
        3 => {
            alias = Some(parts[0].to_string());
            pattern = parts[1].to_string();
            slice_tup = parse_safe_slice(Some(parts[2].trim()))?;
        }
        _ => {
            return Err(plan_err(format!(
                "Invalid tensor reference format: {ref_str:?}"
            )))
        }
    }

    if slice_tup.is_empty() && pattern.contains('[') && pattern.ends_with(']') {
        let b_idx = pattern.find('[').unwrap();
        let candidate = &pattern[b_idx..];
        if let Ok(sl) = parse_safe_slice(Some(candidate)) {
            slice_tup = sl;
            pattern = pattern[..b_idx].to_string();
        }
    }
    Ok((alias, pattern, slice_tup))
}

fn resolve_targets(ws: &WeightWorkspace, ref_str: &str) -> Result<(Vec<String>, Vec<SliceItem>)> {
    let (_alias, pattern, slice_tup) = parse_tensor_ref(ref_str)?;
    let matched = ws.names(Some(&pattern), None);
    if matched.is_empty() {
        return Err(plan_err(format!(
            "Reference {ref_str:?} matched 0 tensors in workspace"
        )));
    }
    Ok((matched, slice_tup))
}

fn get_f64(payload: &Value, key: &str) -> Option<f64> {
    payload.get(key).and_then(|v| {
        v.as_f64()
            .or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
    })
}

fn get_str(payload: &Value, key: &str) -> Option<String> {
    payload.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
}

fn target_ref(payload: &Value) -> Result<String> {
    get_str(payload, "target")
        .or_else(|| get_str(payload, "from"))
        .or_else(|| get_str(payload, "of"))
        .ok_or_else(|| plan_err("Transform requires 'target' (or 'from'/'of')"))
}

fn compare_scalar(actual: f64, spec: &Value, label: &str) -> Result<()> {
    let eps = get_f64(spec, "eps").unwrap_or(1e-5);
    if let Some(expected) = get_f64(spec, "is") {
        if (actual - expected).abs() > eps {
            return Err(plan_err(format!(
                "Assertion failed ({label}): {actual:.6} != {expected:.6} (eps={eps})"
            )));
        }
    }
    if let Some(le) = get_f64(spec, "le") {
        if !(actual <= le + eps) {
            return Err(plan_err(format!("Assertion failed ({label}): {actual:.6} > le={le}")));
        }
    }
    if let Some(lt) = get_f64(spec, "lt") {
        if !(actual < lt) {
            return Err(plan_err(format!("Assertion failed ({label}): {actual:.6} >= lt={lt}")));
        }
    }
    if let Some(ge) = get_f64(spec, "ge") {
        if !(actual + eps >= ge) {
            return Err(plan_err(format!("Assertion failed ({label}): {actual:.6} < ge={ge}")));
        }
    }
    if let Some(gt) = get_f64(spec, "gt") {
        if !(actual > gt) {
            return Err(plan_err(format!("Assertion failed ({label}): {actual:.6} <= gt={gt}")));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

pub fn evaluate_assertion(ws: &WeightWorkspace, expr: &Value) -> Result<(usize, String)> {
    let obj = expr
        .as_object()
        .ok_or_else(|| plan_err(format!("Each assert expression must be a single-key mapping, got {expr}")))?;
    if obj.len() != 1 {
        return Err(plan_err(format!(
            "Each assert expression must be a single-key mapping, got {expr}"
        )));
    }
    let (op, payload) = obj.iter().next().unwrap();

    match op.as_str() {
        "all" => {
            let arr = payload
                .as_array()
                .filter(|a| !a.is_empty())
                .ok_or_else(|| plan_err("assert.all requires a non-empty list of assertions"))?;
            let mut total = 0;
            for sub in arr {
                let (c, _d) = evaluate_assertion(ws, sub)?;
                total += c;
            }
            Ok((total, format!("all({} checks passed)", arr.len())))
        }
        "any" => {
            let arr = payload
                .as_array()
                .filter(|a| !a.is_empty())
                .ok_or_else(|| plan_err("assert.any requires a non-empty list of assertions"))?;
            let mut errors = Vec::new();
            for sub in arr {
                match evaluate_assertion(ws, sub) {
                    Ok((c, d)) => return Ok((c, format!("any({d})"))),
                    Err(e) => errors.push(e.to_string()),
                }
            }
            Err(plan_err(format!(
                "assert.any failed all branches: {}",
                errors.join("; ")
            )))
        }
        "not" => {
            if !payload.is_object() {
                return Err(plan_err("assert.not requires a nested assertion mapping"));
            }
            match evaluate_assertion(ws, payload) {
                Err(_) => Ok((1, "not(inner failed as expected)".to_string())),
                Ok(_) => Err(plan_err(format!(
                    "assert.not failed: inner assertion {payload} succeeded"
                ))),
            }
        }
        "exists" => {
            let ref_str = payload
                .as_str()
                .map(|s| s.to_string())
                .or_else(|| get_str(payload, "of"))
                .unwrap_or_default();
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            Ok((matched.len(), format!("exists({ref_str} -> {} tensors)", matched.len())))
        }
        "count" => {
            let ref_str = get_str(payload, "of").ok_or_else(|| plan_err("count requires 'of'"))?;
            let (_a, pattern, _s) = parse_tensor_ref(&ref_str)?;
            let matched = ws.names(Some(&pattern), None);
            compare_scalar(matched.len() as f64, payload, &format!("count({ref_str})"))?;
            Ok((matched.len(), format!("count({ref_str}) == {}", matched.len())))
        }
        "shape" => {
            let ref_str = get_str(payload, "of").ok_or_else(|| plan_err("shape requires 'of'"))?;
            let expected: Vec<usize> = payload
                .get("is")
                .and_then(|v| v.as_array())
                .ok_or_else(|| plan_err("shape requires 'is' list"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
                        .unwrap_or(0) as usize
                })
                .collect();
            let (matched, slice_tup) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let arr = ws.get_readonly(name)?;
                let sub = crate::workspace::apply_slice(arr, &slice_tup)?;
                if sub.shape() != expected.as_slice() {
                    return Err(plan_err(format!(
                        "Shape assertion failed for {name:?}: expected {expected:?}, got {:?}",
                        sub.shape()
                    )));
                }
            }
            Ok((matched.len(), format!("shape({ref_str}) == {expected:?}")))
        }
        "dimensions" => {
            let ref_str = get_str(payload, "of").ok_or_else(|| plan_err("dimensions requires 'of'"))?;
            let (matched, slice_tup) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let arr = ws.get_readonly(name)?;
                let sub = crate::workspace::apply_slice(arr, &slice_tup)?;
                compare_scalar(sub.ndim() as f64, payload, &format!("dimensions({name})"))?;
            }
            Ok((matched.len(), format!("dimensions({ref_str}) verified")))
        }
        "dtype" => {
            let ref_str = get_str(payload, "of").ok_or_else(|| plan_err("dtype requires 'of'"))?;
            let expected = get_str(payload, "is")
                .unwrap_or_default()
                .replace("torch.", "")
                .to_lowercase();
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let actual = ws.dtype(name)?.replace("torch.", "").to_lowercase();
                let ok = actual == expected
                    || (matches!(expected.as_str(), "float" | "float32" | "fp32")
                        && matches!(actual.as_str(), "float" | "float32"));
                if !ok {
                    return Err(plan_err(format!(
                        "Dtype assertion failed for {name:?}: expected {expected:?}, got {actual:?}"
                    )));
                }
            }
            Ok((matched.len(), format!("dtype({ref_str}) == {expected}")))
        }
        "equal" => {
            let left = get_str(payload, "left").ok_or_else(|| plan_err("equal requires 'left'"))?;
            let right = get_str(payload, "right").ok_or_else(|| plan_err("equal requires 'right'"))?;
            let eps = get_f64(payload, "eps").unwrap_or(1e-5);
            let (ln, ls) = resolve_targets(ws, &left)?;
            let (rn, rs) = resolve_targets(ws, &right)?;
            if ln.len() != rn.len() {
                return Err(plan_err(format!(
                    "assert.equal matched {} left vs {} right tensors",
                    ln.len(),
                    rn.len()
                )));
            }
            for (l, r) in ln.iter().zip(rn.iter()) {
                let la = crate::workspace::apply_slice(ws.get_readonly(l)?, &ls)?;
                let ra = crate::workspace::apply_slice(ws.get_readonly(r)?, &rs)?;
                if la.shape() != ra.shape() {
                    return Err(plan_err(format!(
                        "assert.equal shape mismatch: {l} {:?} vs {r} {:?}",
                        la.shape(),
                        ra.shape()
                    )));
                }
                let max_diff = la
                    .iter()
                    .zip(ra.iter())
                    .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
                    .fold(0.0f64, f64::max);
                if max_diff > eps {
                    return Err(plan_err(format!(
                        "assert.equal failed for {l} vs {r}: max_abs_diff={max_diff:.6} > eps={eps}"
                    )));
                }
            }
            Ok((ln.len(), format!("equal({left}, {right})")))
        }
        "iszero" => {
            let (ref_str, eps) = if let Some(s) = payload.as_str() {
                (s.to_string(), 1e-6)
            } else {
                (
                    get_str(payload, "of").ok_or_else(|| plan_err("iszero requires 'of'"))?,
                    get_f64(payload, "eps").unwrap_or(1e-6),
                )
            };
            let (matched, slice_tup) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let sub = crate::workspace::apply_slice(ws.get_readonly(name)?, &slice_tup)?;
                let mx = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;
                if mx > eps {
                    return Err(plan_err(format!(
                        "assert.iszero failed for {name:?}: max_abs={mx:.6} > {eps}"
                    )));
                }
            }
            Ok((matched.len(), format!("iszero({ref_str})")))
        }
        "finite" => {
            let ref_str = payload
                .as_str()
                .map(|s| s.to_string())
                .or_else(|| get_str(payload, "of"))
                .unwrap_or_else(|| "*".to_string());
            let (matched, slice_tup) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let sub = crate::workspace::apply_slice(ws.get_readonly(name)?, &slice_tup)?;
                let bad = sub.iter().filter(|v| !v.is_finite()).count();
                if bad > 0 {
                    return Err(plan_err(format!(
                        "assert.finite failed for {name:?}: {bad} NaN/Inf elements found"
                    )));
                }
            }
            Ok((matched.len(), format!("finite({ref_str})")))
        }
        "norm" => {
            let ref_str = get_str(payload, "of").ok_or_else(|| plan_err("norm requires 'of'"))?;
            let ord_kind = get_str(payload, "ord").unwrap_or_else(|| "l2".to_string()).to_lowercase();
            let (matched, slice_tup) = resolve_targets(ws, &ref_str)?;
            let mut last_val = 0.0;
            for name in &matched {
                let sub = crate::workspace::apply_slice(ws.get_readonly(name)?, &slice_tup)?;
                let f64s: Vec<f64> = sub.iter().map(|v| *v as f64).collect();
                let val = match ord_kind.as_str() {
                    "l2" | "fro" => f64s.iter().map(|v| v * v).sum::<f64>().sqrt(),
                    "max" | "inf" => f64s.iter().map(|v| v.abs()).fold(0.0, f64::max),
                    "l1" => f64s.iter().map(|v| v.abs()).sum(),
                    "spectral" => {
                        let m2d = linalg::to_f64_2d(&sub);
                        linalg::singular_values(&m2d).first().copied().unwrap_or(0.0)
                    }
                    other => return Err(plan_err(format!("Unsupported norm ord={other:?}"))),
                };
                compare_scalar(val, payload, &format!("norm({name}, ord={ord_kind})"))?;
                last_val = val;
            }
            Ok((matched.len(), format!("norm({ref_str}, {ord_kind}={last_val:.4})")))
        }
        "rel_delta" => {
            let ref_str = get_str(payload, "of").ok_or_else(|| plan_err("rel_delta requires 'of'"))?;
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let cur: Vec<f64> = ws.get_readonly(name)?.iter().map(|v| *v as f64).collect();
                let orig: Vec<f64> = ws.get_original(name)?.iter().map(|v| *v as f64).collect();
                let num = cur.iter().zip(orig.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
                let den = orig.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                compare_scalar(num / den, payload, &format!("rel_delta({name})"))?;
            }
            Ok((matched.len(), format!("rel_delta({ref_str}) verified")))
        }
        other => Err(plan_err(format!("Unknown assertion operator: {other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// Compilation & validation
// ---------------------------------------------------------------------------

pub fn compile_and_validate_plan(
    plan: &Value,
    require_assertions: bool,
    allow_filesystem_ops: bool,
) -> Result<Vec<(String, Value)>> {
    let raw_transforms: &Vec<Value> = if let Some(arr) = plan.as_array() {
        arr
    } else if let Some(obj) = plan.as_object() {
        if !allow_filesystem_ops {
            if let Some(out) = obj.get("output") {
                if !out.is_null() {
                    return Err(plan_err(
                        "Security Policy Violation: LLM surgery plans cannot specify filesystem 'output' paths",
                    ));
                }
            }
        }
        obj.get("transforms")
            .and_then(|v| v.as_array())
            .ok_or_else(|| plan_err("YAML plan must contain a non-empty 'transforms' list"))?
    } else {
        return Err(plan_err(
            "YAML plan must be a mapping with 'transforms' or a list of transforms",
        ));
    };

    if raw_transforms.is_empty() {
        return Err(plan_err("YAML plan must contain a non-empty 'transforms' list"));
    }

    let mut steps = Vec::new();
    let mut assertion_count = 0;
    let mut mutation_count = 0;

    for (idx, entry) in raw_transforms.iter().enumerate() {
        let obj = entry.as_object().filter(|o| o.len() == 1).ok_or_else(|| {
            plan_err(format!(
                "Transform #{idx} must be a single-key mapping (e.g. `- assert: ...`), got {entry}"
            ))
        })?;
        let (op_name, payload) = obj.iter().next().unwrap();
        if !allow_filesystem_ops && FORBIDDEN_FILESYSTEM_OPS.contains(&op_name.as_str()) {
            return Err(plan_err(format!(
                "Security Policy Violation: operation {op_name:?} at step #{idx} is forbidden in isolated mode"
            )));
        }
        if !MUTATION_OPS.contains(&op_name.as_str()) && !READONLY_OPS.contains(&op_name.as_str()) {
            let mut supported: Vec<&str> = MUTATION_OPS.to_vec();
            supported.extend(READONLY_OPS);
            supported.sort();
            return Err(plan_err(format!(
                "Unknown transform operation {op_name:?} at step #{idx}. Supported: {supported:?}"
            )));
        }
        if op_name == "assert" {
            assertion_count += 1;
        } else if MUTATION_OPS.contains(&op_name.as_str()) {
            mutation_count += 1;
        }
        steps.push((op_name.clone(), payload.clone()));
    }

    if require_assertions && mutation_count > 0 && assertion_count == 0 {
        return Err(plan_err(
            "Invariant Policy Violation: every mutation plan must contain at least one `assert` step (validating shape, dtype, norm, or finite invariants).",
        ));
    }
    Ok(steps)
}

// ---------------------------------------------------------------------------
// Mutation transforms
// ---------------------------------------------------------------------------

fn to_f64_dyn(t: &Tensor) -> ndarray::ArrayD<f64> {
    t.mapv(|v| v as f64)
}

fn from_f64_dyn(t: &ndarray::ArrayD<f64>) -> Tensor {
    t.mapv(|v| v as f32)
}

fn set_f64(
    ws: &mut WeightWorkspace,
    name: &str,
    val: &ndarray::ArrayD<f64>,
    fisher: bool,
    outlier_repair: bool,
) -> Result<()> {
    ws.set_tensor(name, &from_f64_dyn(val), fisher, outlier_repair)?;
    Ok(())
}

pub fn apply_mutation_step(
    ws: &mut WeightWorkspace,
    op_name: &str,
    payload: &Value,
    fisher: bool,
) -> Result<(usize, String)> {
    if !payload.is_object() {
        return Err(plan_err(format!(
            "Transform {op_name:?} payload must be a mapping, got {payload}"
        )));
    }

    match op_name {
        "scale" | "scale_" => {
            let ref_str = target_ref(payload)?;
            let dest_ref = if op_name == "scale" {
                get_str(payload, "to").unwrap_or(ref_str.clone())
            } else {
                ref_str.clone()
            };
            let factor = get_f64(payload, "by")
                .or_else(|| get_f64(payload, "factor"))
                .unwrap_or(1.0);
            let (src_names, src_slice) = resolve_targets(ws, &ref_str)?;
            let (dst_names, dst_slice) = resolve_targets(ws, &dest_ref)?;
            for (s_name, d_name) in src_names.iter().zip(dst_names.iter()) {
                let s_arr = ws.get_readonly(s_name)?.clone();
                let s_sub = crate::workspace::apply_slice(&s_arr, &src_slice)?;
                let scaled = s_sub.mapv(|v| v * factor as f32);
                let mut out = ws.get_readonly(d_name)?.clone();
                if !dst_slice.is_empty() {
                    assign_slice(&mut out, &dst_slice, &scaled)?;
                } else {
                    out = scaled;
                }
                ws.set_tensor(d_name, &out, fisher, true)?;
            }
            Ok((dst_names.len(), format!("{op_name}(by={factor}) on {} tensor(s)", dst_names.len())))
        }
        "fill" | "fill_" | "zeroes" | "ones" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "to"))
                .or_else(|| get_str(payload, "of"))
                .ok_or_else(|| plan_err("fill requires 'target'"))?;
            let (matched, slice_tup) = resolve_targets(ws, &ref_str)?;
            let val = match op_name {
                "zeroes" => 0.0f32,
                "ones" => 1.0f32,
                _ => get_f64(payload, "value").unwrap_or(0.0) as f32,
            };
            for name in &matched {
                let mut out = ws.get_readonly(name)?.clone();
                if !slice_tup.is_empty() {
                    let region = crate::workspace::apply_slice(&out, &slice_tup)?;
                    let filled = tensor::full(region.shape(), val);
                    assign_slice(&mut out, &slice_tup, &filled)?;
                } else {
                    out.fill(val);
                }
                ws.set_tensor(name, &out, fisher, true)?;
            }
            Ok((matched.len(), format!("{op_name}(value={val}) on {} tensor(s)", matched.len())))
        }
        "clamp" | "clamp_" => {
            let ref_str = target_ref(payload)?;
            let lo = get_f64(payload, "min");
            let hi = get_f64(payload, "max");
            let (matched, slice_tup) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let mut out = ws.get_readonly(name)?.clone();
                let sub = crate::workspace::apply_slice(&out, &slice_tup)?;
                let clamped = sub.mapv(|v| {
                    let mut x = v;
                    if let Some(l) = lo {
                        x = x.max(l as f32);
                    }
                    if let Some(h) = hi {
                        x = x.min(h as f32);
                    }
                    x
                });
                if !slice_tup.is_empty() {
                    assign_slice(&mut out, &slice_tup, &clamped)?;
                } else {
                    out = clamped;
                }
                ws.set_tensor(name, &out, fisher, true)?;
            }
            Ok((matched.len(), format!("{op_name}(min={lo:?}, max={hi:?}) on {} tensor(s)", matched.len())))
        }
        "add" | "add_" | "subtract" | "subtract_" | "multiply" | "multiply_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "to"))
                .ok_or_else(|| plan_err("arith op requires 'target'"))?;
            let (matched, slice_tup) = resolve_targets(ws, &ref_str)?;
            let scalar_val = payload.get("value").or_else(|| payload.get("by"));
            let from_ref = get_str(payload, "from").or_else(|| get_str(payload, "from_b"));
            for name in &matched {
                let mut out = ws.get_readonly(name)?.clone();
                let lhs = crate::workspace::apply_slice(&out, &slice_tup)?;
                let rhs: Tensor = if let Some(sv) = scalar_val {
                    if !sv.is_string() {
                        if let Some(f) = sv.as_f64() {
                            tensor::full(lhs.shape(), f as f32)
                        } else if let Some(arr) = sv.as_array() {
                            let vals: Vec<f32> = arr.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
                            let t = tensor::from_vec(&[vals.len()], vals);
                            crate::workspace::broadcast_to(&t, lhs.shape())?
                        } else {
                            return Err(plan_err(format!("{op_name} requires numeric 'value' or 'from'")));
                        }
                    } else if let Some(fr) = &from_ref {
                        let (f_names, f_slice) = resolve_targets(ws, fr)?;
                        crate::workspace::apply_slice(ws.get_readonly(&f_names[0])?, &f_slice)?
                    } else {
                        return Err(plan_err(format!("{op_name} requires 'value' or 'from'")));
                    }
                } else if let Some(fr) = &from_ref {
                    let (f_names, f_slice) = resolve_targets(ws, fr)?;
                    crate::workspace::apply_slice(ws.get_readonly(&f_names[0])?, &f_slice)?
                } else {
                    return Err(plan_err(format!("{op_name} requires 'value' or 'from'")));
                };
                let rhs_b = crate::workspace::broadcast_to(&rhs, lhs.shape())?;
                let res = match op_name {
                    o if o.starts_with("add") => &lhs + &rhs_b,
                    o if o.starts_with("subtract") => &lhs - &rhs_b,
                    _ => &lhs * &rhs_b,
                };
                if !slice_tup.is_empty() {
                    assign_slice(&mut out, &slice_tup, &res)?;
                } else {
                    out = res;
                }
                ws.set_tensor(name, &out, fisher, true)?;
            }
            Ok((matched.len(), format!("{op_name} on {} tensor(s)", matched.len())))
        }
        "copy" | "assign" => {
            let from = get_str(payload, "from").ok_or_else(|| plan_err("copy requires 'from'"))?;
            let to = get_str(payload, "to").ok_or_else(|| plan_err("copy requires 'to'"))?;
            let (src_names, src_slice) = resolve_targets(ws, &from)?;
            let (dst_names, dst_slice) = resolve_targets(ws, &to)?;
            for (s_name, d_name) in src_names.iter().zip(dst_names.iter()) {
                let s_arr = ws.get_readonly(s_name)?.clone();
                let s_sub = crate::workspace::apply_slice(&s_arr, &src_slice)?;
                let mut out = ws.get_readonly(d_name)?.clone();
                if !dst_slice.is_empty() {
                    assign_slice(&mut out, &dst_slice, &s_sub)?;
                } else {
                    out = s_sub;
                }
                ws.set_tensor(d_name, &out, fisher, true)?;
            }
            Ok((dst_names.len(), format!("{op_name}({from} -> {to})")))
        }
        "phlora_" | "svd_low_rank_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "of"))
                .ok_or_else(|| plan_err("svd op requires 'target'"))?;
            let rank = get_f64(payload, "rank");
            let energy_fraction = get_f64(payload, "energy_fraction");
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let arr = to_f64_dyn(ws.get_readonly(name)?);
                if arr.ndim() != 2 {
                    return Err(plan_err(format!(
                        "{op_name} requires 2D tensor, got {name} with shape {:?}",
                        arr.shape()
                    )));
                }
                let m2d = linalg::to_f64_2d(ws.get_readonly(name)?);
                let (u, s, vh) = linalg::svd_full(&m2d);
                let eff_rank = if let Some(ef) = energy_fraction {
                    let total: f64 = s.iter().map(|v| v * v).sum();
                    let mut cum = 0.0;
                    let mut r = s.len();
                    for (i, v) in s.iter().enumerate() {
                        cum += v * v;
                        if cum / total.max(1e-12) >= ef {
                            r = i + 1;
                            break;
                        }
                    }
                    r.clamp(1, s.len())
                } else {
                    (rank.map(|r| r as usize).unwrap_or(s.len() / 2)).max(1)
                };
                if eff_rank < m2d.nrows().min(m2d.ncols()) {
                    let u_k = u.slice(ndarray::s![.., ..eff_rank]).to_owned();
                    let vh_k = vh.slice(ndarray::s![..eff_rank, ..]).to_owned();
                    let s_k = ndarray::Array2::from_shape_fn((eff_rank, eff_rank), |(i, j)| {
                        if i == j { s[i] } else { 0.0 }
                    });
                    let approx = matmul(&matmul(&u_k, &s_k), &vh_k);
                    let approx_t = approx.into_dyn().into_shape_with_order(arr.shape()).unwrap().to_owned();
                    set_f64(ws, name, &approx_t, fisher, true)?;
                }
            }
            Ok((matched.len(), format!("{op_name}(rank={:?}) on {} tensor(s)", rank.or(energy_fraction), matched.len())))
        }
        "spectral_norm_cap_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "of"))
                .ok_or_else(|| plan_err("spectral_norm_cap_ requires 'target'"))?;
            let max_sigma = get_f64(payload, "max_sigma");
            let max_median_ratio = get_f64(payload, "max_median_ratio").unwrap_or(3.0);
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            let mut capped_count = 0;
            for name in &matched {
                let t = ws.get_readonly(name)?;
                if t.ndim() != 2 {
                    continue;
                }
                let m2d = linalg::to_f64_2d(t);
                let (u, s, vh) = linalg::svd_full(&m2d);
                let med_s = if s.is_empty() { 1.0 } else { median(&s) };
                let cap = max_sigma.unwrap_or((med_s * max_median_ratio).max(1e-6));
                if s.first().copied().unwrap_or(0.0) > cap {
                    let s_capped: Vec<f64> = s.iter().map(|v| v.min(cap)).collect();
                    let sm = ndarray::Array2::from_shape_fn((s.len(), s.len()), |(i, j)| {
                        if i == j { s_capped[i] } else { 0.0 }
                    });
                    let recon = matmul(&matmul(&u, &sm), &vh);
                    let recon_t = recon.into_dyn().into_shape_with_order(t.shape()).unwrap().to_owned();
                    set_f64(ws, name, &recon_t, fisher, true)?;
                    capped_count += 1;
                }
            }
            Ok((matched.len(), format!("spectral_norm_cap_(capped {capped_count}/{} tensor(s))", matched.len())))
        }
        "project_out_direction_" | "abliterate_direction_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "of"))
                .ok_or_else(|| plan_err("projection op requires 'target'"))?;
            let alpha = get_f64(payload, "alpha").unwrap_or(1.0);
            let direction_raw = payload.get("direction");
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let t = ws.get_readonly(name)?;
                if t.ndim() != 2 {
                    return Err(plan_err(format!("{op_name} requires 2D tensor, got {name}")));
                }
                let arr = linalg::to_f64_2d(t);
                let d: Vec<f64> = if let Some(dr) = direction_raw {
                    eamit::json_f64_vec(Some(dr))?
                } else {
                    let (u, _s, _vh) = linalg::svd_full(&arr);
                    u.column(0).to_vec()
                };
                let norm_d = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm_d > 1e-12 {
                    let d_hat: Vec<f64> = d.iter().map(|v| v / norm_d).collect();
                    let (rows, cols) = (arr.nrows(), arr.ncols());
                    let proj = if d_hat.len() == rows {
                        // outer(d_hat, d_hat @ arr)
                        let dh_arr = ndarray::Array2::from_shape_vec((rows, 1), d_hat.clone()).unwrap();
                        let dh_t = transpose(&dh_arr);
                        let inner = matmul(&dh_t, &arr); // 1 x cols
                        matmul(&dh_arr, &inner)
                    } else if d_hat.len() == cols {
                        let dh_arr = ndarray::Array2::from_shape_vec((cols, 1), d_hat.clone()).unwrap();
                        let inner = matmul(&arr, &dh_arr); // rows x 1
                        matmul(&inner, &transpose(&dh_arr))
                    } else {
                        return Err(plan_err(format!(
                            "{op_name} direction dim {} does not match {name} shape {:?}",
                            d_hat.len(),
                            arr.shape()
                        )));
                    };
                    let updated = &arr - proj.mapv(|v| v * alpha);
                    let updated_t = updated.into_dyn().into_shape_with_order(t.shape()).unwrap().to_owned();
                    set_f64(ws, name, &updated_t, fisher, true)?;
                }
            }
            Ok((matched.len(), format!("{op_name}(alpha={alpha}) on {} tensor(s)", matched.len())))
        }
        "lerp_" | "slerp_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "to"))
                .ok_or_else(|| plan_err("lerp requires 'target'"))?;
            let other_ref = get_str(payload, "from");
            let t_val = get_f64(payload, "t").unwrap_or(0.5);
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            let other_names = if let Some(or) = &other_ref {
                resolve_targets(ws, or)?.0
            } else {
                matched.clone()
            };
            for (name, o_name) in matched.iter().zip(other_names.iter()) {
                let a = to_f64_dyn(ws.get_readonly(name)?);
                let b = if other_ref.is_some() {
                    to_f64_dyn(ws.get_readonly(o_name)?)
                } else {
                    to_f64_dyn(ws.get_original(name)?)
                };
                let res = if op_name == "lerp_" {
                    a.mapv(|v| v * (1.0 - t_val)) + b.mapv(|v| v * t_val)
                } else {
                    let na = a.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if na < 1e-12 || nb < 1e-12 {
                        a.mapv(|v| v * (1.0 - t_val)) + b.mapv(|v| v * t_val)
                    } else {
                        let cos_omega = a
                            .iter()
                            .zip(b.iter())
                            .map(|(x, y)| (x / na) * (y / nb))
                            .sum::<f64>()
                            .clamp(-1.0, 1.0);
                        let omega = cos_omega.acos();
                        if omega.abs() < 1e-6 {
                            a.mapv(|v| v * (1.0 - t_val)) + b.mapv(|v| v * t_val)
                        } else {
                            let so = omega.sin();
                            let c1 = ((1.0 - t_val) * omega).sin() / so;
                            let c2 = (t_val * omega).sin() / so;
                            a.mapv(|v| v * c1) + b.mapv(|v| v * c2)
                        }
                    }
                };
                set_f64(ws, name, &res, fisher, true)?;
            }
            Ok((matched.len(), format!("{op_name}(t={t_val}) on {} tensor(s)", matched.len())))
        }
        "ties_trim_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "of"))
                .ok_or_else(|| plan_err("ties_trim_ requires 'target'"))?;
            let keep_top_fraction = get_f64(payload, "keep_top_fraction").unwrap_or(0.2);
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let cur = to_f64_dyn(ws.get_readonly(name)?);
                let orig = to_f64_dyn(ws.get_original(name)?);
                let delta = &cur - &orig;
                let flat: Vec<f64> = delta.iter().copied().collect();
                let k_keep = ((keep_top_fraction * flat.len() as f64) as usize).max(1);
                if k_keep < flat.len() {
                    let mut abs: Vec<f64> = flat.iter().map(|v| v.abs()).collect();
                    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let thresh = abs[abs.len() - k_keep];
                    let trimmed = orig.iter().zip(delta.iter()).map(|(o, d)| {
                        if d.abs() >= thresh { o + d } else { *o }
                    });
                    let res = ndarray::ArrayD::from_shape_vec(cur.raw_dim(), trimmed.collect()).unwrap();
                    set_f64(ws, name, &res, fisher, true)?;
                }
            }
            Ok((matched.len(), format!("ties_trim_(keep_top_fraction={keep_top_fraction}) on {} tensor(s)", matched.len())))
        }
        "prune_magnitude_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "of"))
                .ok_or_else(|| plan_err("prune_magnitude_ requires 'target'"))?;
            let fraction = get_f64(payload, "fraction")
                .ok_or_else(|| plan_err("prune_magnitude_ requires 'fraction'"))?;
            let (matched, slice_tup) = resolve_targets(ws, &ref_str)?;
            for name in &matched {
                let mut out = ws.get_readonly(name)?.clone();
                let region = crate::workspace::apply_slice(&out, &slice_tup)?;
                let mut flat: Vec<f32> = region.iter().copied().collect();
                let k = (fraction * flat.len() as f64) as usize;
                if k > 0 {
                    let mut order: Vec<usize> = (0..flat.len()).collect();
                    order.sort_by(|&a, &b| flat[a].abs().partial_cmp(&flat[b].abs()).unwrap());
                    for &i in order.iter().take(k) {
                        flat[i] = 0.0;
                    }
                    let pruned = tensor::from_vec(region.shape(), flat);
                    if !slice_tup.is_empty() {
                        assign_slice(&mut out, &slice_tup, &pruned)?;
                    } else {
                        out = pruned;
                    }
                    ws.set_tensor(name, &out, fisher, true)?;
                }
            }
            Ok((matched.len(), format!("prune_magnitude_(fraction={fraction}) on {} tensor(s)", matched.len())))
        }
        "rescale_outlier_rows_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "of"))
                .ok_or_else(|| plan_err("rescale_outlier_rows_ requires 'target'"))?;
            let min_ratio = get_f64(payload, "min_ratio").unwrap_or(2.5);
            let explicit_rows = payload.get("rows").and_then(|v| v.as_array()).map(|a| {
                a.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>()
            });
            let also_zero = payload
                .get("zero_outliers")
                .and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s.eq_ignore_ascii_case("true"))))
                .unwrap_or(false);
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            let mut rescaled_rows_total = 0;
            for name in &matched {
                let t = ws.get_readonly(name)?;
                if t.ndim() < 2 {
                    continue;
                }
                let mut m2d = linalg::to_f64_2d(t);
                let rn = linalg::row_norms(&m2d);
                let med = median(&rn);
                if med <= 1e-12 {
                    continue;
                }
                let rows_to_fix: Vec<usize> = if let Some(rows) = &explicit_rows {
                    rows.iter()
                        .filter(|&&r| r >= 0 && (r as usize) < m2d.nrows())
                        .map(|&r| r as usize)
                        .collect()
                } else {
                    rn.iter()
                        .enumerate()
                        .filter(|(_, &v)| v >= min_ratio * med)
                        .map(|(i, _)| i)
                        .collect()
                };
                for &r in &rows_to_fix {
                    if also_zero {
                        for j in 0..m2d.ncols() {
                            m2d[[r, j]] = 0.0;
                        }
                    } else if rn[r] > 0.0 {
                        let f = med / rn[r];
                        for j in 0..m2d.ncols() {
                            m2d[[r, j]] *= f;
                        }
                    }
                    rescaled_rows_total += 1;
                }
                let res = m2d.into_dyn().into_shape_with_order(t.shape()).unwrap().to_owned();
                set_f64(ws, name, &res, fisher, true)?;

                if name.ends_with(".weight") {
                    let bias_name = format!("{}.bias", &name[..name.len() - ".weight".len()]);
                    if ws.contains(&bias_name) {
                        let b_t = ws.get_readonly(&bias_name)?;
                        if b_t.ndim() == 1 && b_t.shape()[0] == rn.len() {
                            let mut b_arr: Vec<f64> = b_t.iter().map(|v| *v as f64).collect();
                            for &r in &rows_to_fix {
                                if also_zero {
                                    b_arr[r] = 0.0;
                                } else if rn[r] > 0.0 {
                                    b_arr[r] *= med / rn[r];
                                }
                            }
                            let b_new = tensor::from_vec(b_t.shape(), b_arr.iter().map(|v| *v as f32).collect());
                            ws.set_tensor(&bias_name, &b_new, fisher, true)?;
                        }
                    }
                }
            }
            Ok((matched.len(), format!("rescale_outlier_rows_(fixed {rescaled_rows_total} row(s))")))
        }
        "replace_nonfinite_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "of"))
                .unwrap_or_else(|| "*".to_string());
            let val = get_f64(payload, "value").unwrap_or(0.0) as f32;
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            let mut fixed = 0;
            for name in &matched {
                let mut arr = ws.get_readonly(name)?.clone();
                let bad = arr.iter().filter(|v| !v.is_finite()).count();
                if bad > 0 {
                    fixed += bad;
                    arr.mapv_inplace(|v| if v.is_finite() { v } else { val });
                    ws.set_tensor(name, &arr, fisher, true)?;
                }
            }
            Ok((matched.len(), format!("replace_nonfinite_(fixed {fixed} elements)")))
        }
        "rank_one_update_" => {
            let ref_str = get_str(payload, "target")
                .or_else(|| get_str(payload, "of"))
                .ok_or_else(|| plan_err("rank_one_update_ requires 'target'"))?;
            let (matched, _) = resolve_targets(ws, &ref_str)?;
            let u = eamit::json_f64_vec(payload.get("u"))?;
            let v = eamit::json_f64_vec(payload.get("v"))?;
            let alpha = get_f64(payload, "alpha").unwrap_or(1.0);
            let u_mat = ndarray::Array2::from_shape_vec((u.len(), 1), u).unwrap();
            let v_mat = ndarray::Array2::from_shape_vec((1, v.len()), v).unwrap();
            let uv = matmul(&u_mat, &v_mat);
            for name in &matched {
                let t = ws.get_readonly(name)?;
                let arr = linalg::to_f64_2d(t);
                if arr.nrows() != uv.nrows() || arr.ncols() != uv.ncols() {
                    return Err(plan_err(format!(
                        "rank_one_update_ shape mismatch for {name}: {:?} vs ({}, {})",
                        arr.shape(),
                        uv.nrows(),
                        uv.ncols()
                    )));
                }
                let updated = &arr + uv.mapv(|x| x * alpha);
                let updated_t = updated.into_dyn().into_shape_with_order(t.shape()).unwrap().to_owned();
                set_f64(ws, name, &updated_t, fisher, true)?;
            }
            Ok((matched.len(), format!("rank_one_update_(alpha={alpha})")))
        }
        "closed_form_ridge_" => {
            let target = get_str(payload, "target")
                .ok_or_else(|| plan_err("closed_form_ridge_ requires 'target'"))?;
            let keys = eamit::json_f64_matrix(payload.get("keys").ok_or_else(|| plan_err("keys required"))?)?;
            let values = eamit::json_f64_matrix(payload.get("values").ok_or_else(|| plan_err("values required"))?)?;
            let lam = get_f64(payload, "lambda_reg").unwrap_or(1e-2);
            let info = eamit::apply_closed_form_ridge(ws, &target, &keys, &values, lam, fisher)?;
            Ok((1, format!("closed_form_ridge_({target}, post_mse={})", info["post_mse"])))
        }
        "eamit_batch_" => {
            let raw_targets = payload
                .get("targets")
                .filter(|v| !matches!(v, Value::Array(a) if a.is_empty()))
                .or_else(|| payload.get("target"));
            let targets: Vec<String> = match raw_targets {
                Some(Value::String(s)) => resolve_targets(ws, s)?.0,
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect(),
                _ => return Err(plan_err("eamit_batch_ requires 'targets'")),
            };
            let raw_facts = payload
                .get("facts")
                .and_then(|v| v.as_array())
                .ok_or_else(|| plan_err("eamit_batch_ requires 'facts'"))?;
            let facts: Vec<BatchFactSpec> = eamit::facts_from_json(raw_facts)?;
            let lam = get_f64(payload, "lambda_cov").unwrap_or(1e-2);
            let mut align_cfg = EamitAlignConfig::default();
            if let Some(v) = get_f64(payload, "lambda_kl") {
                align_cfg.lambda_kl = v;
            }
            if let Some(v) = get_f64(payload, "lambda_mse") {
                align_cfg.lambda_mse = v;
            }
            if let Some(v) = payload.get("top_m").and_then(|x| x.as_u64()) {
                align_cfg.top_m = v as usize;
            }
            if let Some(v) = payload.get("align_steps").and_then(|x| x.as_u64()) {
                align_cfg.align_steps = v as usize;
            }
            let res = eamit::apply_eamit_batch_with_config(ws, &targets, &facts, lam, fisher, &align_cfg)?;
            Ok((targets.len(), format!(
                "eamit_batch_({} facts across {} layers, mse {:.4}->{:.4}, kl {:.4}->{:.4})",
                res.num_facts, targets.len(), res.pre_edit_mse, res.post_edit_mse,
                res.pre_alignment_kl, res.post_alignment_kl
            )))
        }
        other => Err(plan_err(format!("Unsupported mutation op: {other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

pub fn execute_yaml_plan(
    ws: &mut WeightWorkspace,
    plan_yaml: &str,
    dry_run: bool,
    require_assertions: bool,
    apply_fisher_guard: bool,
) -> PlanExecutionReport {
    let parsed: std::result::Result<Value, _> = serde_yaml::from_str(plan_yaml);
    let plan_value = match parsed {
        Ok(v) => v,
        Err(e) => {
            return PlanExecutionReport {
                success: false,
                dry_run,
                assertions_verified: 0,
                mutations_executed: 0,
                steps: vec![],
                diff_summary: Value::Null,
                error: Some(format!("PlanValidationError: Invalid YAML syntax in surgery plan: {e}")),
            };
        }
    };
    execute_yaml_plan_value(ws, &plan_value, dry_run, require_assertions, apply_fisher_guard)
}

pub fn execute_yaml_plan_value(
    ws: &mut WeightWorkspace,
    plan_value: &Value,
    dry_run: bool,
    require_assertions: bool,
    apply_fisher_guard: bool,
) -> PlanExecutionReport {
    let compiled = compile_and_validate_plan(plan_value, require_assertions, false);
    let compiled_steps = match compiled {
        Ok(s) => s,
        Err(e) => {
            return PlanExecutionReport {
                success: false,
                dry_run,
                assertions_verified: 0,
                mutations_executed: 0,
                steps: vec![],
                diff_summary: Value::Null,
                error: Some(e.to_string()),
            };
        }
    };

    let ckpt_label = "__pre_yaml_plan__";
    ws.checkpoint(ckpt_label);

    let mut step_results = Vec::new();
    let mut assertions_verified = 0;
    let mut mutations_executed = 0;

    let mut run = || -> Result<()> {
        for (idx, (op_name, payload)) in compiled_steps.iter().enumerate() {
            if op_name == "assert" {
                let (matched_count, detail) = evaluate_assertion(ws, payload)?;
                assertions_verified += 1;
                step_results.push(StepExecutionResult {
                    step_index: idx,
                    op_name: "assert".to_string(),
                    matched_tensors: matched_count,
                    status: "verified".to_string(),
                    detail,
                });
            } else if op_name == "dump" || op_name == "diff" {
                step_results.push(StepExecutionResult {
                    step_index: idx,
                    op_name: op_name.clone(),
                    matched_tensors: ws.all_names().len(),
                    status: "ok".to_string(),
                    detail: format!("{op_name} inspected"),
                });
            } else {
                let (matched_count, detail) = apply_mutation_step(ws, op_name, payload, apply_fisher_guard)?;
                mutations_executed += 1;
                step_results.push(StepExecutionResult {
                    step_index: idx,
                    op_name: op_name.clone(),
                    matched_tensors: matched_count,
                    status: "applied".to_string(),
                    detail,
                });
            }
        }
        Ok(())
    };

    match run() {
        Ok(()) => {
            let diff_dict = ws.diff().to_value();
            if dry_run {
                let _ = ws.restore(ckpt_label);
            }
            PlanExecutionReport {
                success: true,
                dry_run,
                assertions_verified,
                mutations_executed,
                steps: step_results,
                diff_summary: diff_dict,
                error: None,
            }
        }
        Err(exc) => {
            let _ = ws.restore(ckpt_label);
            PlanExecutionReport {
                success: false,
                dry_run,
                assertions_verified,
                mutations_executed,
                steps: step_results,
                diff_summary: ws.diff().to_value(),
                error: Some(exc.to_string()),
            }
        }
    }
}

