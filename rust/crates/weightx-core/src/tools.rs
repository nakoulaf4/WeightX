// SPDX-License-Identifier: GPL-3.0-or-later
//! Isolated tool layer: JSON-Schema tool catalog exposed to the LLM brain.
//! Rust port of `weightx.tools.registry` (definitions + TOON/JSON catalogs).
//! Invocation dispatch lives in [`crate::harness::WeightXHarness::invoke_tool`].

use serde_json::{json, Value};

use crate::toon;

#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub layer: String,
    pub parallel_safe: bool,
}

impl ToolDefinition {
    pub fn to_openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }

    pub fn to_anthropic_schema(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        })
    }
}

fn schema(props: Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": false,
    })
}

/// The full L1 + L2 + L3 + Verification tool suite (11 tools).
#[derive(Debug, Clone)]
pub struct ToolCatalog {
    pub tools: Vec<ToolDefinition>,
    pub use_toon: bool,
}

impl Default for ToolCatalog {
    fn default() -> Self {
        Self::new(true)
    }
}

impl ToolCatalog {
    pub fn new(use_toon: bool) -> Self {
        let t = |name: &str, desc: &str, layer: &str, psafe: bool, params: Value| ToolDefinition {
            name: name.into(),
            description: desc.into(),
            parameters: params,
            layer: layer.into(),
            parallel_safe: psafe,
        };
        let tools = vec![
            t(
                "l1_autoarch_inspect",
                "Inspect the target model's architecture (family, d_model, layer count, tensor roles q/k/v/o/gate/up/down/norm/embed, and calibration NLL head disambiguation).",
                "L1",
                true,
                schema(json!({
                    "resolve_with_calibration": {"type": "boolean", "description": "Whether to resolve head/RoPE ambiguities using calibration data."},
                    "include_tensor_table": {"type": "boolean", "description": "Whether to include the full per-tensor role table."}
                }), &[]),
            ),
            t(
                "l2_unilayer_profile",
                "Profile all model layers across 13 metrics (effective_rank, stable_rank, block_influence, fisher_information, neuron_outlier_ratio, etc.) and identify redundant, bottleneck, and anomalous outlier layers.",
                "L2",
                true,
                schema(json!({
                    "outlier_row_threshold": {"type": "number", "description": "Row L2 norm ratio to median above which a neuron is flagged as an outlier (default 2.5)."},
                    "include_full_table": {"type": "boolean", "description": "Whether to include the full 13-metric table per layer."}
                }), &[]),
            ),
            t(
                "l2_knowledge_map",
                "Run Logit Lens and Domain Attention Probing to compute the Knowledge Map and Gini Knowledge Concentration Index (KCI) across factual, syntactic, and reasoning domains.",
                "L2",
                true,
                schema(json!({}), &[]),
            ),
            t(
                "l3_list_tensors",
                "List parameter tensors in the model matching an optional glob/regex pattern or layer index.",
                "L3",
                true,
                schema(json!({
                    "pattern": {"type": "string", "description": "Optional substring, glob, or regex pattern (e.g. '*layers.2*')."},
                    "layer_idx": {"type": "integer", "description": "Optional integer layer index to filter by."}
                }), &[]),
            ),
            t(
                "l3_describe_tensor",
                "Compute statistical moments, quantiles, largest/smallest row & column L2 norms (outlier neurons), and leading SVD singular values for one tensor.",
                "L3",
                true,
                schema(json!({
                    "name": {"type": "string", "description": "Exact tensor name to describe."},
                    "top_k": {"type": "integer", "description": "Number of largest/smallest row norms and singular values to report (default 8)."}
                }), &["name"]),
            ),
            t(
                "l3_read_tensor_slice",
                "Read a multi-dimensional slice of a weight tensor (e.g. '0:8, :16' or '185:186, :') without overflowing context.",
                "L3",
                true,
                schema(json!({
                    "name": {"type": "string", "description": "Exact tensor name."},
                    "index": {"type": "string", "description": "NumPy-style slice string such as '0:8, :16'."},
                    "max_elements": {"type": "integer", "description": "Maximum number of scalar elements to return (default 256)."}
                }), &["name"]),
            ),
            t(
                "l3_apply_yaml_plan",
                "Validate and apply a declarative BrainSurgery-WX YAML plan. The plan MUST include `assert` steps (shape, dtype, norm, finite) and can use transforms like `rescale_outlier_rows_`, `scale_`, `fill_`, `clamp_`, `phlora_`, `closed_form_ridge_`, or `eamit_batch_`.",
                "L3",
                false,
                schema(json!({
                    "plan_yaml": {"type": "string", "description": "YAML text containing `transforms: [...]` with assertions and weight edits."},
                    "dry_run": {"type": "boolean", "description": "If true, validates and simulates the plan then rolls back immediately."}
                }), &["plan_yaml"]),
            ),
            t(
                "l3_batch_edit_eamit",
                "Apply EAMIT (Embedding Alignment Model Editing in Transformers) simultaneous batch associative memory editing across one or more 2D layer matrices with KL + top-M similarity alignment and IWCKEdit Fisher regularization.",
                "L3",
                false,
                schema(json!({
                    "target_tensors": {"type": "array", "items": {"type": "string"}, "description": "List of 2D weight tensor names to distribute the batch update across."},
                    "facts": {"type": "array", "items": {"type": "object"}, "description": "List of `{id, key, value}` fact objects where `key` has length d_in and `value` has length d_out."},
                    "lambda_cov": {"type": "number", "description": "Covariance regularization strength (default 0.01)."},
                    "lambda_kl": {"type": "number", "description": "Key-residual KL alignment regularization weight (default 0.05)."},
                    "lambda_mse": {"type": "number", "description": "Top-M cosine-similarity MSE alignment weight (default 0.25)."},
                    "top_m": {"type": "integer", "description": "Number of top key-similar neighbors per item for MSE alignment (default 8)."},
                    "align_steps": {"type": "integer", "description": "Number of embedding alignment optimization iterations (default 20)."}
                }), &["target_tensors", "facts"]),
            ),
            t(
                "l3_run_sandbox",
                "Run constrained NumPy/math analysis or surgical weight edits in a scanner-guarded, process-isolated sandbox (`W` holds writable NumPy arrays matching `tensor_pattern`).",
                "L3",
                false,
                schema(json!({
                    "code": {"type": "string", "description": "Python code using `W` (dict of NumPy arrays), `np`, and `math`. No OS/network/file imports allowed."},
                    "tensor_pattern": {"type": "string", "description": "Glob or substring pattern restricting which tensors are mounted into `W`."}
                }), &["code"]),
            ),
            t(
                "l3_practice_probe",
                "Run a budgeted evaluation on the visible practice split to check whether a localized edit improved behavior before committing.",
                "Execution",
                false,
                schema(json!({}), &[]),
            ),
            t(
                "l3_verify_and_commit",
                "Run the Vespasian Verification & Rollback engine on the hidden held-out test suite. Automatically commits if held-out score improves and KL/health contracts pass, or automatically rolls back to the best checkpoint on regression.",
                "Verification",
                false,
                schema(json!({
                    "hypothesis": {"type": "string", "description": "Concise explanation of the localized root cause and the weight edit applied."}
                }), &[]),
            ),
        ];
        Self { tools, use_toon }
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.iter().find(|t| t.name == name)
    }

    pub fn to_openai_tools(&self) -> Vec<Value> {
        self.tools.iter().map(|t| t.to_openai_schema()).collect()
    }

    pub fn to_anthropic_tools(&self) -> Vec<Value> {
        self.tools.iter().map(|t| t.to_anthropic_schema()).collect()
    }

    /// Concise TOON tool catalog for CLI agents and text-mode LLMs.
    pub fn to_prompt_catalog(&self) -> String {
        let rows: Vec<Value> = self
            .tools
            .iter()
            .map(|t| {
                let required: Vec<String> = t.parameters["required"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                json!({
                    "name": t.name,
                    "layer": t.layer,
                    "description": t.description,
                    "required_args": required.join(","),
                })
            })
            .collect();
        toon::dumps_toon(&json!({"available_tools": rows}))
    }
}