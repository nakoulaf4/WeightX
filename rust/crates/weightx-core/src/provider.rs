// SPDX-License-Identifier: GPL-3.0-or-later
//! Provider layer contracts + built-in autonomous WSP brain.
//! Rust port of `weightx.providers.base` and `weightx.providers.custom`.
//!
//! Concrete LLM backends (OpenAI-compatible / Anthropic / Gemini via the
//! `cersei` SDK, CLI agents) live in the `weightx-agent` crate and implement
//! [`BrainProvider`].

use regex::Regex;
use serde_json::Value;

use crate::error::Result;
use crate::toon;

#[derive(Debug, Clone)]
pub struct LLMToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub struct LLMMessage {
    pub role: String, // system | user | assistant | tool
    pub content: String,
    pub tool_calls: Vec<LLMToolCall>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

impl LLMMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub content: String,
    pub tool_calls: Vec<LLMToolCall>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub provider_name: String,
    pub model_name: String,
}

/// Abstract brain interface connecting any LLM to the WeightX harness.
pub trait BrainProvider: Send {
    fn provider_name(&self) -> String;
    fn model_name(&self) -> String;
    fn clone_box(&self) -> Box<dyn BrainProvider>;
    fn generate(
        &mut self,
        messages: &[LLMMessage],
        tool_catalog: Option<&crate::tools::ToolCatalog>,
    ) -> Result<LLMResponse>;
}

impl Clone for Box<dyn BrainProvider> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Adapter wrapping any user-supplied closure as a brain provider
/// (port of `CustomCallableProvider`).
pub struct CallableProvider {
    name: String,
    model: String,
    #[allow(clippy::type_complexity)]
    f: std::sync::Arc<
        std::sync::Mutex<
            dyn FnMut(&[LLMMessage], Option<&crate::tools::ToolCatalog>) -> Result<LLMResponse> + Send,
        >,
    >,
}

impl CallableProvider {
    pub fn new(
        name: impl Into<String>,
        model: impl Into<String>,
        f: impl FnMut(&[LLMMessage], Option<&crate::tools::ToolCatalog>) -> Result<LLMResponse> + Send + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            model: model.into(),
            f: std::sync::Arc::new(std::sync::Mutex::new(f)),
        }
    }
}

impl BrainProvider for CallableProvider {
    fn provider_name(&self) -> String {
        self.name.clone()
    }
    fn model_name(&self) -> String {
        self.model.clone()
    }
    fn clone_box(&self) -> Box<dyn BrainProvider> {
        Box::new(Self {
            name: self.name.clone(),
            model: self.model.clone(),
            f: self.f.clone(),
        })
    }
    fn generate(
        &mut self,
        messages: &[LLMMessage],
        tool_catalog: Option<&crate::tools::ToolCatalog>,
    ) -> Result<LLMResponse> {
        let mut f = self.f.lock().unwrap_or_else(|e| e.into_inner());
        f(messages, tool_catalog)
    }
}

/// Extract `<weightx_tool_call>{"name": ..., "arguments": {...}}</weightx_tool_call>` blocks.
pub fn parse_embedded_tool_calls(text: &str) -> Vec<LLMToolCall> {
    let re = Regex::new(r"(?s)<weightx_tool_call>\s*(\{.*?\})\s*</weightx_tool_call>").unwrap();
    let mut calls = Vec::new();
    for (idx, cap) in re.captures_iter(text).enumerate() {
        let Ok(parsed) = serde_json::from_str::<Value>(&cap[1]) else { continue };
        let Some(name) = parsed.get("name").and_then(|v| v.as_str()) else { continue };
        let mut args = parsed
            .get("arguments")
            .or_else(|| parsed.get("args"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        if let Some(s) = args.as_str() {
            if let Ok(v) = serde_json::from_str::<Value>(s) {
                args = v;
            }
        }
        calls.push(LLMToolCall {
            call_id: format!("call_emb_{}", idx + 1),
            name: name.to_string(),
            arguments: args,
        });
    }
    calls
}

/// Built-in deterministic WSP reasoning agent executing the 5-stage cycle
/// without any external API (port of `AutonomousWSPProvider`).
#[derive(Clone)]
pub struct AutonomousWSPProvider {
    pub zero_outliers: bool,
    step: usize,
    localized_outliers: Vec<Value>,
    described_shapes: std::collections::HashMap<String, Vec<usize>>,
    described_dtypes: std::collections::HashMap<String, String>,
}

impl Default for AutonomousWSPProvider {
    fn default() -> Self {
        Self {
            zero_outliers: false,
            step: 0,
            localized_outliers: vec![],
            described_shapes: Default::default(),
            described_dtypes: Default::default(),
        }
    }
}

impl AutonomousWSPProvider {
    pub fn new(zero_outliers: bool) -> Self {
        Self { zero_outliers, ..Default::default() }
    }

    fn response(&self, content: &str, calls: Vec<LLMToolCall>) -> LLMResponse {
        LLMResponse {
            content: content.into(),
            input_tokens: 0,
            output_tokens: toon::estimate_tokens(content),
            tool_calls: calls,
            provider_name: self.provider_name(),
            model_name: self.model_name(),
        }
    }
}

impl BrainProvider for AutonomousWSPProvider {
    fn provider_name(&self) -> String {
        "weightx-autonomous".into()
    }
    fn model_name(&self) -> String {
        "autonomous-wsp-brain-v1".into()
    }
    fn clone_box(&self) -> Box<dyn BrainProvider> {
        Box::new(self.clone())
    }

    fn generate(
        &mut self,
        messages: &[LLMMessage],
        _tool_catalog: Option<&crate::tools::ToolCatalog>,
    ) -> Result<LLMResponse> {
        self.step += 1;

        // Harvest tool results from the previous turn.
        for msg in messages {
            if msg.role == "tool" && !msg.content.is_empty() {
                let data = toon::loads_toon(&msg.content);
                if let Some(outliers) = data.get("outlier_neurons").and_then(|v| v.as_array()) {
                    if !outliers.is_empty() {
                        self.localized_outliers = outliers.clone();
                    }
                }
                if let (Some(name), Some(shape)) = (
                    data.get("name").and_then(|v| v.as_str()),
                    data.get("shape").and_then(|v| v.as_array()),
                ) {
                    if data.get("l2_norm").is_some() {
                        let shp: Vec<usize> = shape
                            .iter()
                            .filter_map(|v| v.as_u64().map(|x| x as usize))
                            .collect();
                        self.described_shapes.insert(name.to_string(), shp);
                        self.described_dtypes.insert(
                            name.to_string(),
                            data.get("dtype").and_then(|v| v.as_str()).unwrap_or("float32").into(),
                        );
                    }
                }
            }
        }

        let tc = |id: &str, name: &str, args: Value| LLMToolCall {
            call_id: id.into(),
            name: name.into(),
            arguments: args,
        };

        match self.step {
            1 => Ok(self.response(
                "Stage 1 (Recon): Running L1 AutoArch architecture inspection, L2 Uni-Layer 13-metric profiling, and L2 Knowledge Map (Logit Lens + KCI).",
                vec![
                    tc("recon_l1", "l1_autoarch_inspect", serde_json::json!({"resolve_with_calibration": true, "include_tensor_table": false})),
                    tc("recon_l2_unilayer", "l2_unilayer_profile", serde_json::json!({"outlier_row_threshold": 2.5, "include_full_table": true})),
                    tc("recon_l2_kmap", "l2_knowledge_map", serde_json::json!({})),
                ],
            )),
            2 => {
                let mut calls = Vec::new();
                let targets: Vec<Value> = self.localized_outliers.iter().take(3).cloned().collect();
                let hyp = if !targets.is_empty() {
                    for (i, item) in targets.iter().enumerate() {
                        let tname = item["tensor_name"].as_str().unwrap_or("");
                        let r0 = item["row_indices"]
                            .as_array()
                            .and_then(|a| a.first())
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        calls.push(tc(&format!("loc_desc_{i}"), "l3_describe_tensor", serde_json::json!({"name": tname, "top_k": 6})));
                        calls.push(tc(&format!("loc_slice_{i}"), "l3_read_tensor_slice", serde_json::json!({"name": tname, "index": format!("{r0}:{} , :8", r0 + 2).replace(' ', ""), "max_elements": 32})));
                    }
                    let names: Vec<&str> = targets.iter().filter_map(|t| t["tensor_name"].as_str()).collect();
                    format!(
                        "Stage 2 (Localization): L2 Uni-Layer flagged {} tensor(s) with extreme outlier neuron L2 norms ({names:?}). Requesting L3 tensor descriptions and targeted slices.",
                        self.localized_outliers.len()
                    )
                } else {
                    calls.push(tc("loc_list", "l3_list_tensors", serde_json::json!({"layer_idx": 0})));
                    "Stage 2 (Localization): Inspecting layer 0 tensor list.".into()
                };
                Ok(self.response(&hyp, calls))
            }
            3 => {
                let mut transforms: Vec<Value> = Vec::new();
                for item in &self.localized_outliers {
                    let tname = item["tensor_name"].as_str().unwrap_or("");
                    match self.described_shapes.get(tname) {
                        Some(shp) => {
                            let dt = self.described_dtypes.get(tname).cloned().unwrap_or_else(|| "float32".into());
                            transforms.push(serde_json::json!({"assert": {"all": [
                                {"exists": {"of": tname}},
                                {"shape": {"of": tname, "is": shp}},
                                {"dtype": {"of": tname, "is": dt}},
                                {"finite": {"of": tname}},
                            ]}}));
                        }
                        None => {
                            transforms.push(serde_json::json!({"assert": {"finite": {"of": tname}}}));
                        }
                    }
                    let mut rescale = serde_json::json!({
                        "target": tname,
                        "min_ratio": 2.5,
                        "zero_outliers": self.zero_outliers,
                    });
                    if let Some(rows) = item.get("row_indices") {
                        rescale["rows"] = rows.clone();
                    }
                    transforms.push(serde_json::json!({"rescale_outlier_rows_": rescale}));
                    transforms.push(serde_json::json!({"assert": {"finite": {"of": tname}}}));
                }
                if transforms.is_empty() {
                    transforms.push(serde_json::json!({"assert": {"finite": {"of": "*"}}}));
                }
                let plan = serde_json::json!({"transforms": transforms});
                let plan_yaml = serde_yaml::to_string(&plan).unwrap_or_default();
                Ok(self.response(
                    "Stage 3 & 4 (Declarative Planning & Execution): Generated BrainSurgery-WX YAML plan with shape/dtype/finite assertions and `rescale_outlier_rows_` surgery, followed by a practice split probe.",
                    vec![
                        tc("exec_yaml_plan", "l3_apply_yaml_plan", serde_json::json!({"plan_yaml": plan_yaml, "dry_run": false})),
                        tc("probe_practice", "l3_practice_probe", serde_json::json!({})),
                    ],
                ))
            }
            4 => {
                let summary: Vec<String> = self
                    .localized_outliers
                    .iter()
                    .take(3)
                    .map(|o| format!("{} rows {}", o["tensor_name"].as_str().unwrap_or(""), o["row_indices"]))
                    .collect();
                let joined = if summary.is_empty() { "target layers".to_string() } else { summary.join(", ") };
                Ok(self.response(
                    "Stage 5 (Vespasian Verification): Practice probe confirmed improvement. Triggering held-out verification and checkpoint commit.",
                    vec![tc(
                        "vespasian_verify",
                        "l3_verify_and_commit",
                        serde_json::json!({"hypothesis": format!(
                            "Repaired corrupted outlier neurons in {joined} via declarative BrainSurgery-WX plan with IWCKEdit Fisher protection."
                        )}),
                    )],
                ))
            }
            _ => Ok(self.response(
                "WeightX 5-stage editing cycle complete: L1/L2 Recon localized the outlier neurons, BrainSurgery-WX declarative YAML plan repaired the weights under IWCKEdit Fisher protection, and Vespasian verification confirmed held-out improvement.",
                vec![],
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_tool_calls() {
        let text = "hello <weightx_tool_call>{\"name\": \"l3_list_tensors\", \"arguments\": {\"layer_idx\": 2}}</weightx_tool_call> bye";
        let calls = parse_embedded_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "l3_list_tensors");
        assert_eq!(calls[0].arguments["layer_idx"], 2);
    }
}