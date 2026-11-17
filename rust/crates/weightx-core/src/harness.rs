// SPDX-License-Identifier: GPL-3.0-or-later
//! WeightX Harness Orchestrator — Rust port of `weightx.harness`.
//!
//! Coordinates the three isolated layers:
//!   1. Provider layer ([`BrainProvider`] — cersei-backed LLMs in `weightx-agent`,
//!      CLI agents, or the built-in autonomous brain),
//!   2. Tool layer ([`ToolCatalog`] + [`WeightXHarness::invoke_tool`]),
//!   3. Execution & verification layer (workspace, Fisher guard, sandbox, Vespasian).

use std::sync::Arc;

use ndarray::Array2;
use serde_json::{json, Value};

use crate::error::{Result, WeightXError};
use crate::fisher::FisherGuard;
use crate::l1;
use crate::l2;
use crate::eamit::{self, EamitAlignConfig};
use crate::model::ModelArch;
use crate::plan;
use crate::provider::{AutonomousWSPProvider, BrainProvider, LLMMessage};
use crate::sandbox::IsolatedSandbox;
use crate::tensor::Tensor;
use crate::tools::ToolCatalog;
use crate::toon;
use crate::vespasian::{EvaluationSplit, VespasianEngine, VespasianIterationResult};
use crate::workspace::{WeightWorkspace, WorkspaceDiffSummary};

pub const SYSTEM_PROMPT_WEIGHTX: &str = "You are a Weight-Space Programming (WSP) agent operating inside the WeightX Harness.
You do not have direct access to the filesystem, raw model objects, or held-out evaluation tests.
Follow the 5-stage WeightX editing cycle strictly:
  1. RECON: Call `l1_autoarch_inspect`, `l2_unilayer_profile`, and `l2_knowledge_map` to obtain a compact TOON profile of the model architecture, 13 layer metrics, and Knowledge Concentration Index (KCI).
  2. LOCALIZATION: Form a precise hypothesis (e.g., anomalous outlier neurons or localized factual projections in specific layers) and inspect only the relevant tensors via `l3_describe_tensor` and `l3_read_tensor_slice`.
  3. PLANNING: Construct a declarative BrainSurgery-WX YAML plan (`transforms: [...]`) that includes `assert` steps (`exists`, `shape`, `dtype`, `finite`, `norm`) before and after mutations.
  4. EXECUTION: Apply the YAML plan via `l3_apply_yaml_plan` (or `l3_batch_edit_eamit` for batch factual edits) and check the visible practice split via `l3_practice_probe`.
  5. VERIFICATION & ROLLBACK (Vespasian): Call `l3_verify_and_commit` with your hypothesis so the harness evaluates the edit on the hidden held-out test split and either commits the new best checkpoint or automatically rolls back regressions.
";

#[derive(Debug, Clone)]
pub struct HarnessRunResult {
    pub goal: String,
    pub provider_name: String,
    pub model_name: String,
    pub turns_executed: usize,
    pub tool_calls_executed: usize,
    pub baseline_held_out_score: f64,
    pub final_best_held_out_score: f64,
    pub net_improvement: f64,
    pub final_assistant_summary: String,
    pub diff_summary: WorkspaceDiffSummary,
    pub vespasian_history: Vec<VespasianIterationResult>,
    pub tool_call_log: Vec<Value>,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
}

impl HarnessRunResult {
    pub fn to_value(&self) -> Value {
        let r5 = |v: f64| (v * 1e5).round() / 1e5;
        json!({
            "goal": self.goal,
            "provider": self.provider_name,
            "llm_model": self.model_name,
            "turns_executed": self.turns_executed,
            "tool_calls_executed": self.tool_calls_executed,
            "baseline_held_out_score": r5(self.baseline_held_out_score),
            "final_best_held_out_score": r5(self.final_best_held_out_score),
            "net_improvement": r5(self.net_improvement),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "final_assistant_summary": self.final_assistant_summary,
            "diff_summary": self.diff_summary.to_value(),
            "vespasian_history": self.vespasian_history.iter().map(|v| v.to_value()).collect::<Vec<_>>(),
            "tool_call_log": self.tool_call_log,
        })
    }
    pub fn to_toon(&self) -> String {
        toon::dumps_toon(&self.to_value())
    }
}

pub struct HarnessOptions {
    pub practice_split: Option<EvaluationSplit>,
    pub held_out_split: Option<EvaluationSplit>,
    pub calibration_input_ids: Option<Array2<i64>>,
    pub tokenizer: Option<tokenizers::Tokenizer>,
    pub bos_token_id: Option<i64>,
    pub dtypes: std::collections::HashMap<String, String>,
    pub fisher_damping_alpha: f64,
    pub fisher_freeze_top_fraction: f64,
    pub improvement_threshold: f64,
    pub max_kl_drift: f64,
    pub max_changed_fraction: f64,
    pub practice_budget: usize,
    pub use_toon: bool,
}

impl Default for HarnessOptions {
    fn default() -> Self {
        Self {
            practice_split: None,
            held_out_split: None,
            calibration_input_ids: None,
            tokenizer: None,
            bos_token_id: None,
            dtypes: std::collections::HashMap::new(),
            fisher_damping_alpha: 0.25,
            fisher_freeze_top_fraction: 0.05,
            improvement_threshold: 1e-4,
            max_kl_drift: 0.50,
            max_changed_fraction: 0.25,
            practice_budget: 10,
            use_toon: true,
        }
    }
}

pub struct WeightXHarness {
    pub workspace: WeightWorkspace,
    pub vespasian: VespasianEngine,
    pub sandbox: IsolatedSandbox,
    pub arch: Option<Arc<dyn ModelArch>>,
    pub calib_ids: Option<Array2<i64>>,
    pub tokenizer: Option<tokenizers::Tokenizer>,
    pub bos_token_id: Option<i64>,
    pub tools: ToolCatalog,
    pub provider: Box<dyn BrainProvider>,
    pub call_log: Vec<Value>,
}

impl WeightXHarness {
    pub fn new(
        tensors: Vec<(String, Tensor)>,
        arch: Option<Arc<dyn ModelArch>>,
        provider: Option<Box<dyn BrainProvider>>,
        opts: HarnessOptions,
    ) -> Self {
        let guard = FisherGuard::new(
            Default::default(),
            opts.fisher_damping_alpha,
            opts.fisher_freeze_top_fraction,
            2.5,
            true,
        );
        let mut workspace = WeightWorkspace::from_tensors_with_dtypes(tensors, opts.dtypes.clone(), Some(guard));
        workspace.arch = arch.clone();
        // Empirical diagonal Fisher from calibration batches (as in the Python harness).
        if let (Some(arch), Some(ids)) = (&arch, &opts.calibration_input_ids) {
            if let Ok(fmap) = crate::fisher::compute_empirical_fisher(&workspace, arch.as_ref(), &[ids.clone()]) {
                if let Some(g) = workspace.fisher_guard.as_mut() {
                    g.fisher_map = fmap;
                }
            }
        }
        let arch_ref: Option<&dyn ModelArch> = arch.as_deref();
        let vespasian = VespasianEngine::new(
            &mut workspace,
            arch_ref,
            opts.practice_split,
            opts.held_out_split,
            opts.improvement_threshold,
            opts.max_kl_drift,
            opts.max_changed_fraction,
            opts.practice_budget,
        );
        Self {
            workspace,
            vespasian,
            sandbox: IsolatedSandbox::default(),
            arch,
            calib_ids: opts.calibration_input_ids,
            tokenizer: opts.tokenizer,
            bos_token_id: opts.bos_token_id,
            tools: ToolCatalog::new(opts.use_toon),
            provider: provider.unwrap_or_else(|| Box::new(AutonomousWSPProvider::default())),
            call_log: Vec::new(),
        }
    }

    fn arch_ref(&self) -> Option<&dyn ModelArch> {
        self.arch.as_deref()
    }

    /// Invoke an isolated tool safely; result serialized in TOON (or JSON).
    pub fn invoke_tool(&mut self, name: &str, arguments: &Value) -> String {
        let args = match arguments {
            Value::Null => json!({}),
            Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({})),
            v => v.clone(),
        };
        let result = self.dispatch_tool(name, &args);
        match result {
            Ok(raw) => {
                self.call_log.push(json!({"tool": name, "args": args, "status": "ok"}));
                if let Some(s) = raw.as_str() {
                    return s.to_string();
                }
                if self.tools.use_toon {
                    toon::dumps_toon(&raw)
                } else {
                    serde_json::to_string_pretty(&raw).unwrap_or_default()
                }
            }
            Err(e) => {
                let msg = e.to_string();
                self.call_log.push(json!({"tool": name, "args": args, "status": "error", "error": msg}));
                toon::dumps_toon(&json!({"status": "error", "tool": name, "error": msg}))
            }
        }
    }

    fn dispatch_tool(&mut self, name: &str, args: &Value) -> Result<Value> {
        if self.tools.get(name).is_none() {
            return Err(WeightXError::Value(format!(
                "Unknown tool {name:?}. Available: {:?}",
                self.tools.names()
            )));
        }
        let get_str = |k: &str| args.get(k).and_then(|v| v.as_str()).map(String::from);
        let get_bool = |k: &str, d: bool| {
            args.get(k)
                .and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s.eq_ignore_ascii_case("true"))))
                .unwrap_or(d)
        };
        let get_f64 = |k: &str, d: f64| {
            args.get(k)
                .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok())))
                .unwrap_or(d)
        };
        let get_usize = |k: &str, d: usize| {
            args.get(k)
                .and_then(|v| {
                    v.as_u64()
                        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
                        .map(|x| x as usize)
                })
                .unwrap_or(d)
        };

        match name {
            "l1_autoarch_inspect" => {
                let mut rep = l1::inspect_workspace(&self.workspace);
                if get_bool("resolve_with_calibration", true) {
                    let base_nll = match (&self.arch, &self.calib_ids) {
                        (Some(arch), Some(ids)) => {
                            let seqs: Vec<Vec<i64>> = ids.rows().into_iter().map(|r| r.to_vec()).collect();
                            crate::domain::token_nll(&self.workspace, arch.as_ref(), &seqs)
                                .map(|(n, _)| n)
                                .unwrap_or(0.0)
                        }
                        _ => 0.0,
                    };
                    let d_model = rep.d_model.unwrap_or(64);
                    let calib = l1::resolve_ambiguities_with_calibration(
                        &self.workspace,
                        &rep.heads_candidates,
                        d_model,
                        base_nll,
                        None,
                    );
                    if calib.identifiable {
                        rep.ambiguities.retain(|a| !a.starts_with("heads/head_dim ambiguous"));
                    }
                    rep.calibration_resolution = Some(calib);
                }
                Ok(rep.to_value(get_bool("include_tensor_table", false)))
            }
            "l2_unilayer_profile" => {
                let rep = l2::profile_model_layers(
                    &self.workspace,
                    self.arch_ref(),
                    self.calib_ids.as_ref(),
                    "target_model",
                    get_f64("outlier_row_threshold", 2.5),
                );
                Ok(rep.to_value(get_bool("include_full_table", true)))
            }
            "l2_knowledge_map" => {
                let probes = self
                    .tokenizer
                    .as_ref()
                    .map(|t| l2::knowledge_probes_from_tokenizer(t, self.bos_token_id));
                let rep = l2::run_knowledge_map(&self.workspace, self.arch_ref(), probes.as_deref(), None);
                Ok(rep.to_value())
            }
            "l3_list_tensors" => {
                let layer_idx = args.get("layer_idx").and_then(|v| v.as_i64());
                let rows = self.workspace.list_tensors_table(get_str("pattern").as_deref(), layer_idx);
                Ok(json!({"matched_count": rows.len(), "tensors": rows}))
            }
            "l3_describe_tensor" => {
                let tname = get_str("name").ok_or_else(|| WeightXError::Value("name is required".into()))?;
                self.workspace.describe_tensor(&tname, get_usize("top_k", 8))
            }
            "l3_read_tensor_slice" => {
                let tname = get_str("name").ok_or_else(|| WeightXError::Value("name is required".into()))?;
                self.workspace.read_slice(
                    &tname,
                    get_str("index").as_deref(),
                    get_usize("max_elements", 256),
                    6,
                )
            }
            "l3_apply_yaml_plan" => {
                let yaml = get_str("plan_yaml").ok_or_else(|| WeightXError::Value("plan_yaml is required".into()))?;
                let rep = plan::execute_yaml_plan(
                    &mut self.workspace,
                    &yaml,
                    get_bool("dry_run", false),
                    true,
                    true,
                );
                Ok(rep.to_value())
            }
            "l3_batch_edit_eamit" => {
                let targets: Vec<String> = args["target_tensors"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let facts = eamit::facts_from_json(args["facts"].as_array().map(|v| v.as_slice()).unwrap_or(&[]))?;
                let mut align_cfg = EamitAlignConfig::default();
                if let Some(v) = args.get("lambda_kl").and_then(|x| x.as_f64()) {
                    align_cfg.lambda_kl = v;
                }
                if let Some(v) = args.get("lambda_mse").and_then(|x| x.as_f64()) {
                    align_cfg.lambda_mse = v;
                }
                if let Some(v) = args.get("top_m").and_then(|x| x.as_u64()) {
                    align_cfg.top_m = v as usize;
                }
                if let Some(v) = args.get("align_steps").and_then(|x| x.as_u64()) {
                    align_cfg.align_steps = v as usize;
                }
                let res = eamit::apply_eamit_batch_with_config(
                    &mut self.workspace,
                    &targets,
                    &facts,
                    get_f64("lambda_cov", 1e-2),
                    true,
                    &align_cfg,
                )?;
                Ok(res.to_value())
            }
            "l3_run_sandbox" => {
                let code = get_str("code").ok_or_else(|| WeightXError::Value("code is required".into()))?;
                let res = self.sandbox.run(&mut self.workspace, &code, get_str("tensor_pattern").as_deref());
                Ok(res.to_value())
            }
            "l3_practice_probe" => {
                let Self { vespasian, workspace, arch, .. } = self;
                Ok(vespasian.run_practice_probe(workspace, arch.as_deref()))
            }
            "l3_verify_and_commit" => {
                let hypothesis = get_str("hypothesis").unwrap_or_default();
                let Self { vespasian, workspace, arch, .. } = self;
                let res = vespasian.verify_and_decide(workspace, arch.as_deref(), &hypothesis);
                Ok(res.to_value())
            }
            _ => Err(WeightXError::Value(format!("Tool {name} not dispatched"))),
        }
    }

    /// Multi-turn LLM-driven Recon -> Localize -> Plan -> Execute -> Verify loop.
    pub fn run_editing_cycle(&mut self, goal: &str, max_turns: usize) -> Result<HarnessRunResult> {
        let mut messages = vec![
            LLMMessage::new("system", SYSTEM_PROMPT_WEIGHTX),
            LLMMessage::new("user", format!("Target WSP Goal: {goal}")),
        ];
        let mut total_in = 0usize;
        let mut total_out = 0usize;
        let mut turns = 0usize;
        let mut tool_calls_count = 0usize;
        let mut final_summary = String::new();

        for _ in 0..max_turns {
            turns += 1;
            let resp = self.provider.generate(&messages, Some(&self.tools))?;
            total_in += resp.input_tokens;
            total_out += resp.output_tokens;

            let mut assistant = LLMMessage::new("assistant", resp.content.clone());
            assistant.tool_calls = resp.tool_calls.clone();
            messages.push(assistant);

            if resp.tool_calls.is_empty() {
                final_summary = resp.content;
                break;
            }
            for tc in &resp.tool_calls {
                tool_calls_count += 1;
                let tool_out = self.invoke_tool(&tc.name, &tc.arguments);
                let mut m = LLMMessage::new("tool", tool_out);
                m.tool_call_id = Some(tc.call_id.clone());
                m.name = Some(tc.name.clone());
                messages.push(m);
            }
            final_summary = resp.content;
        }

        // Ensure final state holds the Vespasian best verified checkpoint.
        let _ = self.workspace.restore("vespasian_best");

        Ok(HarnessRunResult {
            goal: goal.to_string(),
            provider_name: self.provider.provider_name(),
            model_name: self.provider.model_name(),
            turns_executed: turns,
            tool_calls_executed: tool_calls_count,
            baseline_held_out_score: self.vespasian.baseline_score,
            final_best_held_out_score: self.vespasian.best_score,
            net_improvement: self.vespasian.best_score - self.vespasian.baseline_score,
            final_assistant_summary: final_summary,
            diff_summary: self.workspace.diff(),
            vespasian_history: self.vespasian.history.clone(),
            tool_call_log: self.call_log.clone(),
            total_input_tokens: total_in,
            total_output_tokens: total_out,
        })
    }
}