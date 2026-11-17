// SPDX-License-Identifier: GPL-3.0-or-later
//! WeightX StudioSession — shared backend for the interactive CLI (`wx`) and Desktop app.
//! Rust port of `weightx.studio`.

use std::path::Path;
use std::sync::Arc;

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::catalog::{self, LoadedModel, ModelCatalog, PromptSuggestion};
use crate::corpora;
use crate::domain::{self, DomainCalibOptions};
use crate::error::{Result, WeightXError};
use crate::harness::{HarnessOptions, WeightXHarness};
use crate::l1;
use crate::l2;
use crate::eamit::{self, BatchFactSpec};
use crate::model::minillama;
use crate::plan;
use crate::provider::{AutonomousWSPProvider, BrainProvider};
use crate::skills::{self, SkillExecutionResult, WSP_SKILLS};
use crate::toon;
use crate::vespasian::EvaluationSplit;

pub const PROVIDER_KINDS: &[&str] = &[
    "autonomous (встроенный, без API)",
    "ollama",
    "lm-studio",
    "openai",
    "openrouter",
    "vllm",
    "anthropic",
    "cli:claude-code",
    "cli:codex",
    "cli:opencode",
    "cli:antigravity",
    "cli:dsh",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudioConfig {
    pub provider_kind: String,
    pub provider_model: String,
    pub api_key: String,
    pub base_url: String,
    pub model_id: String,
    pub history: Vec<Value>,
}

impl Default for StudioConfig {
    fn default() -> Self {
        Self {
            provider_kind: PROVIDER_KINDS[0].to_string(),
            provider_model: String::new(),
            api_key: String::new(),
            base_url: String::new(),
            model_id: "minillama-demo".to_string(),
            history: Vec::new(),
        }
    }
}

impl StudioConfig {
    pub fn config_path() -> std::path::PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".weightx")
            .join("studio_config.json")
    }

    pub fn load() -> Self {
        let p = Self::config_path();
        std::fs::read_to_string(&p)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let p = Self::config_path();
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut data = serde_json::to_value(self).unwrap_or_else(|_| json!({}));
        if let Some(h) = data.get_mut("history").and_then(|v| v.as_array_mut()) {
            let n = h.len();
            if n > 50 {
                h.drain(0..n - 50);
            }
        }
        let _ = std::fs::write(&p, serde_json::to_string_pretty(&data).unwrap_or_default());
    }
}

/// Provider selection (kind + credentials) used to (re)build a brain.
#[derive(Debug, Clone, Default)]
pub struct ProviderSpec {
    pub kind: String,
    pub model: String,
    pub api_key: String,
    pub base_url: String,
}

pub type ProviderFactory =
    Arc<dyn Fn(&ProviderSpec) -> Result<Box<dyn BrainProvider>> + Send + Sync>;

/// Core-only factory: supports the built-in autonomous brain. Concrete LLM
/// backends (cersei) are registered by the `weightx-agent` crate.
pub fn default_provider_factory() -> ProviderFactory {
    Arc::new(|spec: &ProviderSpec| {
        let kind = spec.kind.split(' ').next().unwrap_or("autonomous");
        if kind == "autonomous" {
            Ok(Box::new(AutonomousWSPProvider::default()))
        } else {
            Err(WeightXError::Value(format!(
                "Provider {kind:?} requires the weightx-agent crate (cersei backend)"
            )))
        }
    })
}

pub struct StudioSession {
    pub config: StudioConfig,
    pub catalog: ModelCatalog,
    pub loaded: Option<LoadedModel>,
    pub harness: Option<WeightXHarness>,
    pub calib_ids: Option<Array2<i64>>,
    pub prac_seqs: Vec<Vec<i64>>,
    pub held_seqs: Vec<Vec<i64>>,
    pub ctrl_seqs: Vec<Vec<i64>>,
    pub provider: Box<dyn BrainProvider>,
    pub factory: ProviderFactory,
    pub log: Arc<dyn Fn(&str) + Send + Sync>,
}

impl Default for StudioSession {
    fn default() -> Self {
        Self::new()
    }
}

impl StudioSession {
    pub fn new() -> Self {
        Self {
            config: StudioConfig::load(),
            catalog: ModelCatalog::new(),
            loaded: None,
            harness: None,
            calib_ids: None,
            prac_seqs: Vec::new(),
            held_seqs: Vec::new(),
            ctrl_seqs: Vec::new(),
            provider: Box::new(AutonomousWSPProvider::default()),
            factory: default_provider_factory(),
            log: Arc::new(|msg: &str| println!("{msg}")),
        }
    }

    pub fn set_log(&mut self, log: Arc<dyn Fn(&str) + Send + Sync>) {
        self.log = log;
    }

    pub fn suggestions(&self) -> Vec<PromptSuggestion> {
        catalog::codex_prompt_suggestions()
    }

    pub fn skills(&self) -> &'static [skills::WeightXSkill] {
        WSP_SKILLS
    }

    pub fn provider_spec(&self) -> ProviderSpec {
        ProviderSpec {
            kind: self.config.provider_kind.clone(),
            model: self.config.provider_model.clone(),
            api_key: self.config.api_key.clone(),
            base_url: self.config.base_url.clone(),
        }
    }

    pub fn set_provider(
        &mut self,
        kind: &str,
        model: &str,
        api_key: &str,
        base_url: &str,
    ) -> Result<String> {
        let spec = ProviderSpec {
            kind: kind.to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        };
        let provider = (self.factory)(&spec)?;
        self.config.provider_kind = kind.to_string();
        self.config.provider_model = model.to_string();
        self.config.api_key = api_key.to_string();
        self.config.base_url = base_url.to_string();
        self.config.save();
        if let Some(h) = &mut self.harness {
            h.provider = provider.clone_box();
        }
        self.provider = provider;
        Ok(format!("Мозг (provider) установлен: {kind} {model}").trim().to_string())
    }

    pub fn run_skill(&mut self, skill_query: &str, user_args: &str) -> Result<SkillExecutionResult> {
        skills::execute_skill(self, skill_query, user_args)
    }

    // ------------------------------------------------------------------ setup
    pub fn load_model(&mut self, model_id: &str) -> Result<String> {
        (self.log)(&format!("Загрузка модели «{model_id}»…"));
        let mut opts = HarnessOptions {
            fisher_damping_alpha: 0.15,
            ..Default::default()
        };

        let mut loaded: LoadedModel;
        if model_id == "minillama-demo" {
            let bundle = minillama::build_corrupted_model_and_splits(2026)?;
            let entry = self
                .catalog
                .get("minillama-demo")
                .ok_or_else(|| WeightXError::Value("minillama-demo entry missing".into()))?;
            opts.practice_split = Some(bundle.practice);
            opts.held_out_split = Some(bundle.held_out);
            opts.calibration_input_ids = Some(bundle.calib_ids.clone());
            opts.improvement_threshold = 0.01;
            opts.max_kl_drift = 2.0;
            self.calib_ids = Some(bundle.calib_ids);
            self.prac_seqs.clear();
            self.held_seqs.clear();
            self.ctrl_seqs.clear();
            loaded = LoadedModel {
                arch: Some(Arc::new(bundle.model)),
                tensors: bundle.tensors,
                tokenizer: None,
                bos_token_id: None,
                entry,
                dtypes: Default::default(),
            };
        } else {
            let mut lm = self.catalog.load_model(model_id)?;
            opts.tokenizer = lm.tokenizer.clone();
            opts.bos_token_id = lm.bos_token_id;
            if let Some(tok) = &lm.tokenizer {
                let prac = catalog::tokenize_texts(tok, lm.bos_token_id, corpora::WSP_PRACTICE)?;
                let held = catalog::tokenize_texts(tok, lm.bos_token_id, corpora::WSP_HELD_OUT)?;
                let ctrl = catalog::tokenize_texts(tok, lm.bos_token_id, corpora::GENERAL_CONTROL)?;
                let ctrl_ref = stack_min_len(&ctrl);
                opts.practice_split = Some(EvaluationSplit::seq_nll("practice", prac.clone(), None));
                opts.held_out_split = Some(EvaluationSplit::seq_nll(
                    "held_out_hidden",
                    held.clone(),
                    ctrl_ref,
                ));
                let calib = stack_min_len(&prac.iter().take(6).cloned().collect::<Vec<_>>());
                opts.calibration_input_ids = calib.clone();
                opts.improvement_threshold = 1e-4;
                opts.max_kl_drift = 0.05;
                self.calib_ids = calib;
                self.prac_seqs = prac;
                self.held_seqs = held;
                self.ctrl_seqs = ctrl;
            } else {
                let (prac, held, calib) = synthetic_splits(lm.arch.as_deref(), &lm.tensors);
                opts.practice_split = Some(prac);
                opts.held_out_split = Some(held);
                opts.calibration_input_ids = Some(calib.clone());
                opts.improvement_threshold = 0.01;
                opts.max_kl_drift = 2.0;
                self.calib_ids = Some(calib);
                self.prac_seqs.clear();
                self.held_seqs.clear();
                self.ctrl_seqs.clear();
            }
            loaded = lm;
        }

        let n_params: usize = loaded.tensors.iter().map(|(_, t)| t.len()).sum();
        let display = loaded.entry.display_name.clone();
        let arch = loaded.arch.clone();
        opts.dtypes = loaded.dtypes.clone();
        let tensors = std::mem::take(&mut loaded.tensors);
        let harness = WeightXHarness::new(
            tensors,
            arch,
            Some(self.provider.clone_box()),
            opts,
        );
        let baseline = harness.vespasian.baseline_score;
        self.loaded = Some(loaded);
        self.harness = Some(harness);
        self.config.model_id = model_id.to_string();
        self.config.save();
        Ok(format!(
            "Модель загружена: {display} | параметров: {n_params} | базовый held-out score: {baseline:.5}"
        ))
    }

    pub fn require_model(&mut self) -> Result<&mut WeightXHarness> {
        if self.harness.is_none() {
            let id = if self.config.model_id.is_empty() {
                "minillama-demo".to_string()
            } else {
                self.config.model_id.clone()
            };
            let msg = self.load_model(&id)?;
            (self.log)(&msg);
        }
        Ok(self.harness.as_mut().unwrap())
    }

    pub fn arch(&self) -> Option<&dyn crate::model::ModelArch> {
        self.loaded.as_ref().and_then(|l| l.arch.as_deref())
    }

    pub fn has_tokenizer(&self) -> bool {
        self.loaded.as_ref().map(|l| l.tokenizer.is_some()).unwrap_or(false)
    }

    pub fn mlp_out_tensors(&mut self) -> Result<Vec<String>> {
        let h = self.require_model()?;
        let names: Vec<String> = h
            .workspace
            .all_names()
            .into_iter()
            .filter(|n| h.workspace.shape(n).map(|s| s.len() == 2).unwrap_or(false))
            .collect();
        Ok(names
            .into_iter()
            .filter(|n| {
                ["down_proj", "mlp.c_proj", "fc2", "dense_4h_to_h"]
                    .iter()
                    .any(|k| n.contains(k))
            })
            .collect())
    }

    // --------------------------------------------------------------- actions
    pub fn run_action(&mut self, action_tag: &str, prompt_text: &str) -> String {
        self.config.history.push(json!({
            "action": action_tag,
            "prompt": prompt_text.chars().take(500).collect::<String>(),
        }));
        self.config.save();
        let result = self.run_action_inner(action_tag, prompt_text);
        match result {
            Ok(s) => s,
            Err(e) => format!("Ошибка: {e}"),
        }
    }

    fn run_action_inner(&mut self, action_tag: &str, prompt_text: &str) -> Result<String> {
        if let Some(sk_id) = action_tag.strip_prefix("skill:") {
            return Ok(self.run_skill(sk_id, prompt_text)?.to_markdown());
        }
        let sk = skills::resolve_skill(action_tag).or_else(|| skills::resolve_skill(prompt_text));
        if let Some(sk) = sk {
            if ["chat", "skill", sk.skill_id, &format!("/{}", sk.skill_id)]
                .iter()
                .any(|t| *t == action_tag)
            {
                return Ok(self.run_skill(sk.skill_id, prompt_text)?.to_markdown());
            }
        }
        match action_tag {
            "recon_profile" | "recon" => self.act_recon_profile(),
            "knowledge_map" => self.act_knowledge_map(),
            "chat" | "full_cycle" => self.act_chat(prompt_text),
            "auto_improve" => self.act_auto_improve(),
            "eamit_demo" => self.act_eamit_demo(),
            "abliterate" => self.act_abliterate(),
            "rollback_test" => self.act_rollback_test(),
            _ => {
                if let Some(sk) = sk {
                    Ok(self.run_skill(sk.skill_id, prompt_text)?.to_markdown())
                } else {
                    self.act_chat(prompt_text)
                }
            }
        }
    }

    pub fn act_recon_profile(&mut self) -> Result<String> {
        let arch = self.loaded.as_ref().and_then(|l| l.arch.clone());
        let calib = self.calib_ids.clone();
        let h = self.require_model()?;
        let rep = l1::inspect_workspace(&h.workspace);
        let prof = l2::profile_model_layers(&h.workspace, arch.as_deref(), calib.as_ref(), "target_model", 2.5);
        Ok(format!(
            "## L1 AutoArch\n{}\n\n## L2 Uni-Layer\n{}",
            rep.to_toon(false),
            prof.to_toon(false)
        ))
    }

    pub fn act_knowledge_map(&mut self) -> Result<String> {
        let arch = self.loaded.as_ref().and_then(|l| l.arch.clone());
        let probes = self
            .loaded
            .as_ref()
            .and_then(|l| l.tokenizer.as_ref())
            .map(|t| l2::knowledge_probes_from_tokenizer(t, self.loaded.as_ref().and_then(|l| l.bos_token_id)));
        let h = self.require_model()?;
        let rep = l2::run_knowledge_map(&h.workspace, arch.as_deref(), probes.as_deref(), None);
        Ok(rep.to_toon())
    }

    pub fn act_chat(&mut self, prompt: &str) -> Result<String> {
        let provider = self.provider.clone_box();
        let h = self.require_model()?;
        h.provider = provider;
        let goal = if prompt.is_empty() {
            "Diagnose and repair the target model."
        } else {
            prompt
        };
        let res = h.run_editing_cycle(goal, 8)?;
        Ok(res.to_toon())
    }

    pub fn act_auto_improve(&mut self) -> Result<String> {
        if !self.has_tokenizer() {
            return self.act_chat("Localize and repair outlier neurons, then verify with Vespasian.");
        }
        let arch = self.loaded.as_ref().and_then(|l| l.arch.clone());
        let Some(arch) = arch else {
            return Err(WeightXError::Value("model has no runtime architecture".into()));
        };
        let prac = self.prac_seqs.clone();
        let ctrl = self.ctrl_seqs.clone();
        let log = self.log.clone();
        let h = self.require_model()?;
        let mut lines = Vec::new();
        for it in 0..3 {
            let opts = DomainCalibOptions {
                iterations: 1,
                max_control_nll_increase: 0.01,
                apply_fisher_guard: false,
                ..Default::default()
            };
            let res = domain::eamit_domain_calibration(&mut h.workspace, arch.as_ref(), &prac, &ctrl, &opts)?;
            let last = res.steps.last();
            if last.map(|s| s.chosen_rel_step == 0.0).unwrap_or(true) {
                lines.push(format!("итерация {}: безопасный шаг не найден — стоп", it + 1));
                break;
            }
            let s = last.unwrap();
            let v = h.vespasian.verify_and_decide(
                &mut h.workspace,
                Some(arch.as_ref()),
                &format!("domain calibration {}", it + 1),
            );
            let line = format!(
                "итерация {}: practice NLL {:.4}→{:.4}; Vespasian: {}",
                it + 1, s.practice_nll_before, s.practice_nll_after, v.verdict_reason
            );
            log(&line);
            lines.push(line);
            if !v.accepted {
                break;
            }
        }
        lines.push(format!(
            "Итоговый held-out score: {:.5} (база {:.5})",
            h.vespasian.best_score, h.vespasian.baseline_score
        ));
        Ok(lines.join("\n"))
    }

    pub fn act_eamit_demo(&mut self) -> Result<String> {
        let arch = self.loaded.as_ref().and_then(|l| l.arch.clone());
        let targets_all = self.mlp_out_tensors()?;
        let targets: Vec<String> = targets_all
            .iter()
            .rev()
            .take(3)
            .rev()
            .cloned()
            .collect();
        if targets.is_empty() {
            return Err(WeightXError::Value("no MLP output tensors found".into()));
        }
        let h = self.require_model()?;
        let shape = h.workspace.shape(&targets[0])?;
        let facts = synthetic_facts(7, 32, shape[1], shape[0], 0.05);
        let res = eamit::apply_eamit_batch(&mut h.workspace, &targets, &facts, 1e-2, true)?;
        let v = h.vespasian.verify_and_decide(
            &mut h.workspace,
            arch.as_deref(),
            "EAMIT synthetic batch",
        );
        Ok(format!(
            "## EAMIT\n{}\n\n## Vespasian\n{}",
            toon::dumps_toon(&res.to_value()),
            v.verdict_reason
        ))
    }

    pub fn act_abliterate(&mut self) -> Result<String> {
        let targets = self.mlp_out_tensors()?;
        let t = targets
            .last()
            .ok_or_else(|| WeightXError::Value("no MLP output tensors found".into()))?
            .clone();
        let h = self.require_model()?;
        let sh = h.workspace.shape(&t)?;
        let plan = format!(
            "transforms:\n  - assert: {{ shape: {{ of: '{t}', is: {sh:?} }} }}\n  - project_out_direction_: {{ target: '{t}', alpha: 0.3 }}\n  - assert: {{ finite: {{ of: '{t}' }} }}\n"
        );
        self.apply_yaml(&plan, true)
    }

    pub fn act_rollback_test(&mut self) -> Result<String> {
        let h = self.require_model()?;
        let two_d: Vec<String> = h
            .workspace
            .all_names()
            .into_iter()
            .filter(|n| h.workspace.shape(n).map(|s| s.len() == 2).unwrap_or(false))
            .collect();
        let t = two_d
            .get(1)
            .ok_or_else(|| WeightXError::Value("not enough 2D tensors".into()))?
            .clone();
        let plan = format!(
            "transforms:\n  - assert: {{ exists: {{ of: '{t}' }} }}\n  - scale_: {{ target: '{t}', by: -50.0 }}\n  - assert: {{ finite: {{ of: '{t}' }} }}\n"
        );
        self.apply_yaml(&plan, false)
    }

    pub fn apply_yaml(&mut self, plan: &str, fisher: bool) -> Result<String> {
        let arch = self.loaded.as_ref().and_then(|l| l.arch.clone());
        let h = self.require_model()?;
        let rep = plan::execute_yaml_plan(&mut h.workspace, plan, false, true, fisher);
        let mut out = format!("## План BrainSurgery-WX\n{}", toon::dumps_toon(&rep.to_value()));
        if rep.success {
            let v = h.vespasian.verify_and_decide(&mut h.workspace, arch.as_deref(), "user YAML plan");
            out += &format!("\n\n## Vespasian\n{}", v.to_toon());
        }
        Ok(out)
    }

    pub fn export_checkpoint(&mut self, path: &str) -> Result<String> {
        let h = self.require_model()?;
        let p = h.workspace.export_safetensors(Path::new(path), false)?;
        Ok(format!("Чекпоинт сохранён: {}", p.display()))
    }

    pub fn status(&mut self) -> String {
        let Some(h) = &self.harness else {
            return "Модель не загружена.".to_string();
        };
        let v = &h.vespasian;
        let d = h.workspace.diff();
        format!(
            "Модель: {} | мозг: {} {}\nHeld-out: база {:.5} → лучший {:.5} | итераций Vespasian: {}\nИзменено параметров: {} / {}",
            self.config.model_id,
            self.config.provider_kind,
            self.config.provider_model,
            v.baseline_score,
            v.best_score,
            v.iteration,
            d.changed_params,
            d.total_params
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Stack sequences truncated to the common minimum length into an [N, L] matrix.
pub fn stack_min_len(seqs: &[Vec<i64>]) -> Option<Array2<i64>> {
    let min_len = seqs.iter().map(|s| s.len()).min()?;
    if min_len == 0 || seqs.is_empty() {
        return None;
    }
    let mut data = Vec::with_capacity(seqs.len() * min_len);
    for s in seqs {
        data.extend_from_slice(&s[..min_len]);
    }
    Array2::from_shape_vec((seqs.len(), min_len), data).ok()
}

/// Random-token splits for checkpoints without a tokenizer (mirrors improver fallback).
pub fn synthetic_splits(
    arch: Option<&dyn crate::model::ModelArch>,
    tensors: &[(String, crate::tensor::Tensor)],
) -> (EvaluationSplit, EvaluationSplit, Array2<i64>) {
    use rand::Rng;
    use rand::SeedableRng;
    let vocab = arch
        .map(|a| a.config().vocab_size)
        .unwrap_or_else(|| {
            tensors
                .iter()
                .find(|(n, _)| n.contains("embed_tokens") || n.contains("wte"))
                .map(|(_, t)| t.shape()[0])
                .unwrap_or(128)
        })
        .max(12);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2026);
    let mut all_x = Vec::new();
    for _ in 0..32 {
        let row: Vec<i64> = (0..8).map(|_| rng.gen_range(2..(vocab as i64 - 2).max(3))).collect();
        all_x.push(row);
    }
    let mut all_y: Vec<i64> = vec![0; 32];
    if let Some(arch) = arch {
        let ws = crate::workspace::WeightWorkspace::from_tensors(tensors.to_vec(), None);
        if let Ok(ids) = Array2::from_shape_vec((32, 8), all_x.iter().flatten().copied().collect()) {
            if let Ok(out) = arch.forward(&ws, &ids, false) {
                let s = out.logits.shape()[1];
                for i in 0..32 {
                    let row = out.logits.slice(ndarray::s![i, s - 1, ..]);
                    let mut best = 0i64;
                    let mut bv = f32::NEG_INFINITY;
                    for (j, &v) in row.iter().enumerate() {
                        if v > bv {
                            bv = v;
                            best = j as i64;
                        }
                    }
                    all_y[i] = best;
                }
            }
        }
    }
    let x = Array2::from_shape_vec((32, 8), all_x.iter().flatten().copied().collect()).unwrap();
    let practice = EvaluationSplit::targets(
        "practice",
        x.slice(ndarray::s![..16, ..]).to_owned(),
        all_y[..16].to_vec(),
        Some(x.slice(ndarray::s![..8, ..]).to_owned()),
    );
    let held = EvaluationSplit::targets(
        "held_out",
        x.slice(ndarray::s![16.., ..]).to_owned(),
        all_y[16..].to_vec(),
        Some(x.slice(ndarray::s![16..24, ..]).to_owned()),
    );
    let calib = x.slice(ndarray::s![..6, ..]).to_owned();
    (practice, held, calib)
}

/// Deterministic synthetic EAMIT facts (unit keys, small random values).
pub fn synthetic_facts(seed: u64, n: usize, d_in: usize, d_out: usize, val_scale: f64) -> Vec<BatchFactSpec> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut facts = Vec::new();
    for i in 0..n {
        let mut key: Vec<f64> = (0..d_in).map(|_| StandardNormal.sample(&mut rng)).collect();
        let norm = key.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-8;
        for v in &mut key {
            *v /= norm;
        }
        let val: Vec<f64> = (0..d_out)
            .map(|_| -> f64 { StandardNormal.sample(&mut rng) })
            .map(|v| v * val_scale)
            .collect();
        facts.push(BatchFactSpec::new(format!("fact_{i}"), key, val));
    }
    facts
}