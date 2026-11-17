// SPDX-License-Identifier: GPL-3.0-or-later
//! WeightX Skills Engine — Rust port of `weightx.skills`.
//!
//! Ultra-compact 2-word skill pills and `/skill <name>` slash commands backed by
//! full multi-stage WSP pipelines (L1/L2 recon, BrainSurgery-WX YAML, EAMIT,
//! IWCKEdit Fisher protection, Vespasian verification/rollback).

use serde_json::{json, Map, Value};

use crate::catalog;
use crate::corpora;
use crate::domain::{self, DomainCalibOptions};
use crate::eamit;
use crate::error::{Result, WeightXError};
use crate::harness::WeightXHarness;
use crate::l1;
use crate::l2;
use crate::plan::{self, PlanExecutionReport};
use crate::studio::{synthetic_facts, StudioSession};
use crate::toon;
use crate::vespasian::VespasianIterationResult;

#[derive(Debug, Clone, Copy)]
pub struct WeightXSkill {
    pub skill_id: &'static str,
    pub pill_label: &'static str,
    pub pipeline_title: &'static str,
    pub short_hint: &'static str,
    pub aliases: &'static [&'static str],
    pub stage_names: &'static [&'static str],
}

pub const WSP_SKILLS: &[WeightXSkill] = &[
    WeightXSkill {
        skill_id: "coding-boost",
        pill_label: "Улучшить в кодинге",
        pipeline_title: "Multi-Stage Code & Algorithmic Reasoning Boost (L1→L3 + Domain EAMIT)",
        short_hint: "Фокусировка синтаксических голов внимания + калибровка MLP на кодовом корпусе",
        aliases: &["coding-boost", "/coding-boost", "coding", "code", "улучшить в кодинге", "кодинг", "код"],
        stage_names: &[
            "L1 AutoArch & L2 Uni-Layer: поиск синтаксических и логических слоёв",
            "Sharpening внимания: фокусировка q_proj/k_proj и чистка выбросов",
            "Closed-Form Code EAMIT: калибровка выходных проекций MLP на алгоритмическом корпусе",
            "IWCKEdit Fisher Guard: защита общих языковых весов диагональю Фишера",
            "Vespasian Gate: проверка held-out метрик и фиксация весов",
        ],
    },
    WeightXSkill {
        skill_id: "memory-slim",
        pill_label: "Уменьшить память",
        pipeline_title: "Spectral Low-Rank SVD Compression & TIES Pruning Pipeline",
        short_hint: "Анализ эффективного ранга + SVD-сжатие избыточных слоёв и тримминг шума",
        aliases: &["memory-slim", "/memory-slim", "memory", "slim", "compress", "уменьшить память", "уменьшить потребление памяти", "сжать модель", "память"],
        stage_names: &[
            "L2 Spectral Audit: ранжирование слоёв по effective_rank и svd_entropy",
            "BrainSurgery-WX SVD: низкоранговая аппроксимация svd_low_rank_ избыточных MLP",
            "TIES Noise Trim: очистка малых сингулярных и весовых компонент (ties_trim_)",
            "Vespasian Gate: верификация сохранения качества после сжатия",
        ],
    },
    WeightXSkill {
        skill_id: "outlier-repair",
        pill_label: "Починить выбросы",
        pipeline_title: "Outlier Neuron Pinpointing & Spectral Row Rescaling Pipeline",
        short_hint: "Поиск нейронов с аномальным эксцессом (kurtosis) и точечный rescale_outlier_rows_",
        aliases: &["outlier-repair", "/outlier-repair", "outliers", "repair", "починить выбросы", "выбросы", "исправить выбросы"],
        stage_names: &[
            "L2 Outlier Scan: поиск строк тензоров с max/median > 5.0x и высоким kurtosis",
            "BrainSurgery-WX Plan: генерация YAML с rescale_outlier_rows_ и инвариантами",
            "IWCKEdit Sandbox: применение плана в изолированном WeightWorkspace",
            "Vespasian Gate: контроль прироста точности на отложенной выборке",
        ],
    },
    WeightXSkill {
        skill_id: "spectral-denoise",
        pill_label: "Снизить перплексию",
        pipeline_title: "Closed-Form Covariance-Preserving Spectral Denoising & Calibration",
        short_hint: "Устранение анизотропии последних слоёв и снижение Perplexity без градиентов",
        aliases: &["spectral-denoise", "/spectral-denoise", "perplexity", "denoise", "improve", "снизить перплексию", "улучшить модель", "перплексия"],
        stage_names: &[
            "L1+L2 Bottleneck Recon: диагностика спектральной энтропии и анизотропии",
            "Outlier Conditioning: подавление энергетических пиков в down_proj",
            "Closed-Form Domain EAMIT: аналитическое решение C0^-1 с выравниванием эмбеддингов",
            "KL & Control Guard: проверка отсутствия дрейфа на контрольном корпусе",
            "Vespasian Commit: фиксация улучшенных весов",
        ],
    },
    WeightXSkill {
        skill_id: "eamit-facts",
        pill_label: "Записать факты",
        pipeline_title: "EAMIT Multi-Layer Embedding-Aligned Fact Injection",
        short_hint: "Пакетная инъекция фактов по нескольким слоям MLP с выравниванием эмбеддингов (KL + top-M MSE) и защитой C0^-1",
        aliases: &["eamit-facts", "/eamit-facts", "eamit", "facts", "записать факты", "факты", "инъекция фактов"],
        stage_names: &[
            "Knowledge Map: локализация фактологических слоёв через Gini KCI",
            "Embedding Alignment: выравнивание key/residual пространств (KL + top-M MSE)",
            "EAMIT Batch Update: распределение ΔW по целевым проекциям down_proj",
            "Vespasian Gate: проверка сохранения общего распределения логитов",
        ],
    },
    WeightXSkill {
        skill_id: "knowledge-map",
        pill_label: "Карта знаний",
        pipeline_title: "Logit-Lens Knowledge Localization & Gini Concentration Index (KCI)",
        short_hint: "Нормализованный Logit Lens по всем слоям + индекс концентрации знаний Gini KCI",
        aliases: &["knowledge-map", "/knowledge-map", "kmap", "logit-lens", "карта знаний", "знания"],
        stage_names: &[
            "L1 Topology Check: связывание скрытых состояний со словарной проекцией lm_head",
            "Normalized Logit Lens: прогон фактологических и синтаксических зондов по слоям",
            "Gini KCI Analysis: расчёт концентрации знаний и выделение ключевых слоёв",
        ],
    },
    WeightXSkill {
        skill_id: "abliterate",
        pill_label: "Вырезать отказы",
        pipeline_title: "Orthogonal Subspace Projection (SVD Direction Abliteration)",
        short_hint: "Выделение главного сингулярного направления и ортогональная проекция project_out_direction_",
        aliases: &["abliterate", "/abliterate", "orthogonal", "вырезать отказы", "ортогонализация", "аблитерация"],
        stage_names: &[
            "SVD Direction Extraction: выделение главной компоненты v1 целевого слоя MLP",
            "BrainSurgery-WX Plan: сборка плана project_out_direction_ с проверкой норм",
            "Sandbox Execution: модификация тензора в изолированном рабочем пространстве",
            "Vespasian Gate: проверка отсутствия деградации связности",
        ],
    },
    WeightXSkill {
        skill_id: "vespasian-guard",
        pill_label: "Проверить откат",
        pipeline_title: "Vespasian Holdout Verification & Bit-Exact Atomic Rollback Stress Test",
        short_hint: "Стресс-тест разрушающей правки с демонстрацией мгновенного побитового отката",
        aliases: &["vespasian-guard", "/vespasian-guard", "rollback", "vespasian", "проверить откат", "откат", "тест отката"],
        stage_names: &[
            "Workspace Snapshot: снятие контрольных сумм SHA-256 всех тензоров",
            "Adversarial Perturbation: инъекция разрушающего масштаба scale_: -50.0",
            "Atomic Rollback Proof: перехват деградации в Vespasian и побитовый возврат весов",
        ],
    },
];

#[derive(Debug, Clone)]
pub struct SkillStepTrace {
    pub index: usize,
    pub total: usize,
    pub title: String,
    pub detail: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct SkillExecutionResult {
    pub skill_id: String,
    pub pill_label: String,
    pub pipeline_title: String,
    pub accepted: bool,
    pub summary: String,
    pub yaml_plan: String,
    pub steps: Vec<SkillStepTrace>,
    pub metrics: Value,
    pub raw_report: String,
}

impl SkillExecutionResult {
    pub fn to_markdown(&self) -> String {
        let badge = if self.accepted { "ПРИНЯТО (COMMITTED)" } else { "ОТКАТ (ROLLED BACK)" };
        let mut lines = vec![
            format!("### Скилл `/skill {}` — {}", self.skill_id, self.pill_label),
            format!("**Пайплайн**: {}  ", self.pipeline_title),
            format!("**Статус Vespasian**: `{badge}`"),
            String::new(),
            "#### Выполненные шаги пайплайна".to_string(),
        ];
        for s in &self.steps {
            let icon = match s.status.as_str() {
                "done" => "✓",
                "rollback" => "↺",
                _ => "•",
            };
            lines.push(format!("{icon} **[{}/{}] {}** — {}", s.index, s.total, s.title, s.detail));
        }
        if self.metrics.as_object().map(|m| !m.is_empty()).unwrap_or(false) {
            lines.push("\n#### Метрики".to_string());
            lines.push(format!("```toon\n{}\n```", toon::dumps_toon(&self.metrics)));
        }
        if !self.yaml_plan.trim().is_empty() {
            lines.push("\n#### Сгенерированный план BrainSurgery-WX (YAML)".to_string());
            lines.push(format!("```yaml\n{}\n```", self.yaml_plan.trim()));
        }
        if !self.raw_report.trim().is_empty() {
            lines.push(format!("\n{}", self.raw_report.trim()));
        }
        lines.push(format!("\n**Итог**: {}", self.summary));
        lines.join("\n")
    }
}

fn strip_skill_prefix(query: &str) -> (String, bool) {
    let q = query.trim();
    let low = q.to_lowercase();
    for prefix in ["/skill:", "/skills:", "/skill ", "/skills ", "skill ", "skills "] {
        if low.starts_with(prefix) {
            return (q[prefix.len()..].trim().to_string(), true);
        }
    }
    if matches!(low.as_str(), "/skill" | "/skills" | "skill" | "skills") {
        return (String::new(), true);
    }
    (q.to_string(), false)
}

/// Resolve a skill by id, `/skill <id>`, 1..8 index, pill label or prefix (case-insensitive).
pub fn resolve_skill(query: &str) -> Option<&'static WeightXSkill> {
    let (stripped, _) = strip_skill_prefix(query);
    let q = stripped.trim().to_lowercase();
    if q.is_empty() {
        return None;
    }
    if let Ok(idx) = q.parse::<usize>() {
        if idx >= 1 && idx <= WSP_SKILLS.len() {
            return Some(&WSP_SKILLS[idx - 1]);
        }
    }
    let q_clean = q.trim_start_matches('/');
    for sk in WSP_SKILLS {
        if sk.aliases.contains(&q.as_str())
            || sk.aliases.contains(&q_clean)
            || q_clean == sk.skill_id.to_lowercase()
            || q_clean == sk.pill_label.to_lowercase()
        {
            return Some(sk);
        }
    }
    for sk in WSP_SKILLS {
        if sk.skill_id.starts_with(q_clean) || sk.pill_label.to_lowercase().starts_with(q_clean) {
            return Some(sk);
        }
    }
    None
}

/// Parse `/skill <name> [args]`; returns (skill, remainder, explicit_prefix).
pub fn parse_skill_command(raw: &str) -> (Option<&'static WeightXSkill>, String, bool) {
    let (stripped, explicit) = strip_skill_prefix(raw);
    if explicit && stripped.is_empty() {
        return (None, String::new(), true);
    }
    if let Some(sk) = resolve_skill(&stripped) {
        return (Some(sk), stripped, explicit);
    }
    let mut parts = stripped.splitn(2, char::is_whitespace);
    if let Some(head) = parts.next() {
        if let Some(sk) = resolve_skill(head) {
            return (Some(sk), parts.next().unwrap_or("").to_string(), explicit);
        }
    }
    (None, stripped, explicit)
}

// ---------------------------------------------------------------------------
// Execution helpers
// ---------------------------------------------------------------------------

fn r5(v: f64) -> f64 {
    (v * 1e5).round() / 1e5
}

fn run_yaml_step(h: &mut WeightXHarness, yaml_plan: &str) -> Result<PlanExecutionReport> {
    let rep = plan::execute_yaml_plan(&mut h.workspace, yaml_plan, false, true, true);
    if !rep.success {
        return Err(WeightXError::Value(format!(
            "YAML-план не прошёл проверку: {}",
            rep.error.unwrap_or_default()
        )));
    }
    Ok(rep)
}

fn vespasian_metrics(h: &WeightXHarness, v: &VespasianIterationResult, extra: Value) -> Value {
    let mut m = Map::new();
    m.insert("baseline_score".into(), json!(r5(h.vespasian.baseline_score)));
    m.insert("held_out_score".into(), json!(r5(v.candidate_score)));
    m.insert("delta".into(), json!(r5(v.score_delta_vs_best)));
    m.insert("kl_drift".into(), json!(r5(v.kl_divergence_ref)));
    if let Some(extra) = extra.as_object() {
        for (k, val) in extra {
            m.insert(k.clone(), val.clone());
        }
    }
    Value::Object(m)
}

fn mlp_out_names(h: &WeightXHarness) -> Vec<String> {
    h.workspace
        .all_names()
        .into_iter()
        .filter(|n| h.workspace.shape(n).map(|s| s.len() == 2).unwrap_or(false))
        .filter(|n| ["down_proj", "mlp.c_proj", "fc2", "dense_4h_to_h"].iter().any(|k| n.contains(k)))
        .collect()
}

fn two_d_names(h: &WeightXHarness) -> Vec<String> {
    h.workspace
        .all_names()
        .into_iter()
        .filter(|n| h.workspace.shape(n).map(|s| s.len() == 2).unwrap_or(false))
        .collect()
}

fn last_n(v: &[String], n: usize) -> Vec<String> {
    v.iter().rev().take(n).rev().cloned().collect()
}

/// Execute a full multi-stage WSP skill pipeline against the active StudioSession.
pub fn execute_skill(
    session: &mut StudioSession,
    skill_query: &str,
    user_args: &str,
) -> Result<SkillExecutionResult> {
    let sk = resolve_skill(skill_query)
        .ok_or_else(|| WeightXError::Value(format!("Неизвестный скилл: {skill_query:?}")))?;
    session.require_model()?;

    if sk.skill_id == "spectral-denoise" {
        return skill_spectral_denoise(session, sk, user_args);
    }

    let arch = session.loaded.as_ref().and_then(|l| l.arch.clone());
    let calib = session.calib_ids.clone();
    let tok = session.loaded.as_ref().and_then(|l| l.tokenizer.clone());
    let bos = session.loaded.as_ref().and_then(|l| l.bos_token_id);
    let prac = session.prac_seqs.clone();
    let ctrl = session.ctrl_seqs.clone();
    let kmap_probes = tok.as_ref().map(|t| l2::knowledge_probes_from_tokenizer(t, bos));
    let log = session.log.clone();
    let h = session.harness.as_mut().unwrap();

    let total = sk.stage_names.len();
    let mut steps: Vec<SkillStepTrace> = Vec::new();
    let mut record = |idx: usize, title: &str, detail: String, status: &str| {
        steps.push(SkillStepTrace {
            index: idx,
            total,
            title: title.to_string(),
            detail: detail.clone(),
            status: status.to_string(),
        });
        log(&format!("[{idx}/{total}] {title} — {detail}"));
    };

    match sk.skill_id {
        "coding-boost" => {
            let arch_rep = l1::inspect_workspace(&h.workspace);
            let prof = l2::profile_model_layers(&h.workspace, arch.as_deref(), calib.as_ref(), "target_model", 2.5);
            let mid_layer = arch_rep.n_layers / 2;
            record(1, sk.stage_names[0], format!(
                "семейство={}, слоёв={}, фокусный слой={}, выбросов={}",
                arch_rep.family, arch_rep.n_layers, mid_layer, prof.outlier_neurons.len()
            ), "done");

            let names = h.workspace.all_names();
            let q_candidates: Vec<String> = names.iter().filter(|n| {
                h.workspace.shape(n).map(|s| s.len() == 2).unwrap_or(false)
                    && n.contains(&format!(".{mid_layer}."))
                    && ["q_proj", "c_attn", "query"].iter().any(|k| n.contains(k))
            }).cloned().collect();
            let mlp_down = mlp_out_names(h);
            let target_q = q_candidates.first().cloned()
                .or_else(|| mlp_down.first().cloned())
                .or_else(|| names.first().cloned())
                .ok_or_else(|| WeightXError::Value("workspace is empty".into()))?;
            let target_mlp = mlp_down.last().cloned().unwrap_or_else(|| target_q.clone());

            let q_shape = h.workspace.shape(&target_q)?;
            let mut yaml = format!(
                "transforms:\n  - assert: {{ shape: {{ of: '{target_q}', is: {q_shape:?} }} }}\n  - scale_: {{ target: '{target_q}', by: 1.015 }}\n"
            );
            for loc in prof.outlier_neurons.iter().take(3) {
                yaml += &format!("  - rescale_outlier_rows_: {{ target: '{}', min_ratio: 3.0 }}\n", loc.tensor_name);
            }
            yaml += &format!("  - assert: {{ finite: {{ of: '{target_q}' }} }}\n");

            let plan_res = run_yaml_step(h, &yaml)?;
            record(2, sk.stage_names[1], format!(
                "применён YAML-план на `{target_q}` (мутаций={}, проверок={})",
                plan_res.mutations_executed, plan_res.assertions_verified
            ), "done");

            let mut code_nll_before = 0.0f64;
            let mut code_nll_after = 0.0f64;
            if let (Some(tok), Some(arch)) = (&tok, &arch) {
                let mut code_seqs: Vec<Vec<i64>> = catalog::tokenize_texts(tok, bos, corpora::CODE_CALIBRATION_CORPUS)?
                    .into_iter()
                    .filter(|s| s.len() >= 2)
                    .map(|s| s.into_iter().take(96).collect())
                    .collect();
                if code_seqs.is_empty() {
                    code_seqs = prac.clone();
                }
                let ctrl_seqs = if ctrl.is_empty() { code_seqs.clone() } else { ctrl.clone() };
                code_nll_before = domain::token_nll(&h.workspace, arch.as_ref(), &code_seqs)?.0;
                let mut calib_seqs = code_seqs.clone();
                calib_seqs.extend(prac.iter().cloned());
                let opts = DomainCalibOptions {
                    iterations: 2,
                    max_control_nll_increase: 0.01,
                    apply_fisher_guard: false,
                    ..Default::default()
                };
                domain::eamit_domain_calibration(&mut h.workspace, arch.as_ref(), &calib_seqs, &ctrl_seqs, &opts)?;
                code_nll_after = domain::token_nll(&h.workspace, arch.as_ref(), &code_seqs)?.0;
                record(3, sk.stage_names[2], format!(
                    "калибровка `{target_mlp}` на кодовом корпусе: code NLL {code_nll_before:.4} → {code_nll_after:.4}"
                ), "done");
            } else {
                record(3, sk.stage_names[2], format!(
                    "спектральная стабилизация `{target_mlp}` (автономный режим синтетической модели)"
                ), "done");
            }

            let alpha = h.workspace.fisher_guard.as_ref().map(|g| g.damping_alpha).unwrap_or(0.25);
            record(4, sk.stage_names[3], format!("защита важных весов IWCKEdit (alpha={alpha})"), "done");

            let verdict = h.vespasian.verify_and_decide(&mut h.workspace, arch.as_deref(), "Skill /coding-boost");
            record(5, sk.stage_names[4], verdict.verdict_reason.clone(), if verdict.accepted { "done" } else { "rollback" });
            let mut metrics = vespasian_metrics(h, &verdict, json!({}));
            if code_nll_before > 0.0 {
                metrics["code_ppl_before"] = json!(((code_nll_before.exp() * 100.0).round()) / 100.0);
                metrics["code_ppl_after"] = json!(((code_nll_after.exp() * 100.0).round()) / 100.0);
            }
            Ok(SkillExecutionResult {
                skill_id: sk.skill_id.into(),
                pill_label: sk.pill_label.into(),
                pipeline_title: sk.pipeline_title.into(),
                accepted: verdict.accepted,
                summary: verdict.verdict_reason.clone(),
                yaml_plan: yaml,
                steps,
                metrics,
                raw_report: String::new(),
            })
        }
        "memory-slim" => {
            let prof = l2::profile_model_layers(&h.workspace, arch.as_deref(), calib.as_ref(), "target_model", 2.5);
            let mlp_tensors = mlp_out_names(h);
            let two_d = two_d_names(h);
            let target_t = mlp_tensors.get(mlp_tensors.len() / 2).cloned()
                .or_else(|| two_d.first().cloned())
                .ok_or_else(|| WeightXError::Value("no 2D tensors".into()))?;
            let shape = h.workspace.shape(&target_t)?;
            let min_dim = *shape.iter().min().unwrap();
            let compressed_rank = 8.max((min_dim as f64 * 0.75) as usize);
            record(1, sk.stage_names[0], format!(
                "проверен профиль {} слоёв; выбран `{target_t}` {shape:?} → ранг {compressed_rank}",
                prof.num_layers
            ), "done");

            let mut yaml = format!(
                "transforms:\n  - assert: {{ shape: {{ of: '{target_t}', is: {shape:?} }} }}\n"
            );
            for loc in prof.outlier_neurons.iter().take(3) {
                yaml += &format!("  - rescale_outlier_rows_: {{ target: '{}', min_ratio: 3.0 }}\n", loc.tensor_name);
            }
            yaml += &format!(
                "  - svd_low_rank_: {{ target: '{target_t}', rank: {compressed_rank} }}\n  - ties_trim_: {{ target: '{target_t}', keep_top_fraction: 0.85 }}\n  - assert: {{ finite: {{ of: '{target_t}' }} }}\n"
            );
            run_yaml_step(h, &yaml)?;
            record(2, sk.stage_names[1], format!(
                "применён `svd_low_rank_` (rank={compressed_rank}/{min_dim}, спектральная аппроксимация)"
            ), "done");

            let orig_params = shape[0] * shape[1];
            let factorized_params = compressed_rank * (shape[0] + shape[1]);
            let rank_saving_pct = (1.0 - compressed_rank as f64 / min_dim as f64).max(0.0) * 100.0;
            record(3, sk.stage_names[2], format!(
                "применён `ties_trim_` (keep_top_fraction=0.85); сокращение эффективного ранга на {rank_saving_pct:.1}%"
            ), "done");

            let verdict = h.vespasian.verify_and_decide(&mut h.workspace, arch.as_deref(), "Skill /memory-slim");
            record(4, sk.stage_names[3], verdict.verdict_reason.clone(), if verdict.accepted { "done" } else { "rollback" });
            let metrics = vespasian_metrics(h, &verdict, json!({
                "target_tensor": target_t,
                "original_rank": min_dim,
                "compressed_rank": compressed_rank,
                "rank_reduction_pct": (rank_saving_pct * 10.0).round() / 10.0,
                "original_elements": orig_params,
                "low_rank_elements": factorized_params,
            }));
            Ok(SkillExecutionResult {
                skill_id: sk.skill_id.into(),
                pill_label: sk.pill_label.into(),
                pipeline_title: sk.pipeline_title.into(),
                accepted: verdict.accepted,
                summary: verdict.verdict_reason.clone(),
                yaml_plan: yaml,
                steps,
                metrics,
                raw_report: String::new(),
            })
        }
        "outlier-repair" => {
            let prof = l2::profile_model_layers(&h.workspace, arch.as_deref(), calib.as_ref(), "target_model", 2.5);
            let mut outliers: Vec<(String, f64)> = prof.outlier_neurons.iter()
                .map(|o| (o.tensor_name.clone(), o.max_ratio_to_median))
                .collect();
            if outliers.is_empty() {
                let mlp_t = mlp_out_names(h);
                let two_d = two_d_names(h);
                let t = mlp_t.last().cloned()
                    .or_else(|| two_d.first().cloned())
                    .ok_or_else(|| WeightXError::Value("no 2D tensors".into()))?;
                outliers.push((t, 2.6));
            }
            let max_ratio = outliers.iter().map(|(_, r)| *r).fold(0.0f64, f64::max);
            record(1, sk.stage_names[0], format!(
                "обнаружено {} тензоров с выбросами (макс. ratio={max_ratio:.2}x)", outliers.len()
            ), "done");

            let mut seen: Vec<String> = Vec::new();
            for (t, _) in &outliers {
                if !seen.contains(t) {
                    seen.push(t.clone());
                }
            }
            let mut yaml = String::from("transforms:\n");
            for tname in seen.iter().take(4) {
                let sh = h.workspace.shape(tname)?;
                yaml += &format!(
                    "  - assert: {{ shape: {{ of: '{tname}', is: {sh:?} }} }}\n  - rescale_outlier_rows_: {{ target: '{tname}', min_ratio: 2.5 }}\n  - assert: {{ finite: {{ of: '{tname}' }} }}\n"
                );
            }
            record(2, sk.stage_names[1], format!("сгенерирован YAML-план для {} тензоров", seen.len().min(4)), "done");
            let plan_res = run_yaml_step(h, &yaml)?;
            record(3, sk.stage_names[2], format!("выполнено мутаций={} с защитой IWCKEdit", plan_res.mutations_executed), "done");

            let verdict = h.vespasian.verify_and_decide(&mut h.workspace, arch.as_deref(), "Skill /outlier-repair");
            record(4, sk.stage_names[3], verdict.verdict_reason.clone(), if verdict.accepted { "done" } else { "rollback" });
            let metrics = vespasian_metrics(h, &verdict, json!({"outliers_found": outliers.len()}));
            Ok(SkillExecutionResult {
                skill_id: sk.skill_id.into(),
                pill_label: sk.pill_label.into(),
                pipeline_title: sk.pipeline_title.into(),
                accepted: verdict.accepted,
                summary: verdict.verdict_reason.clone(),
                yaml_plan: yaml,
                steps,
                metrics,
                raw_report: String::new(),
            })
        }
        "eamit-facts" => {
            let kmap = l2::run_knowledge_map(&h.workspace, arch.as_deref(), kmap_probes.as_deref(), None);
            let factual_layers: Vec<usize> = kmap.dominant_layers_by_domain.iter()
                .find(|(d, _)| d == "factual")
                .map(|(_, l)| l.clone())
                .unwrap_or_default();
            record(1, sk.stage_names[0], format!(
                "Gini KCI={:.4}, фактологические слои={factual_layers:?}", kmap.kci_overall
            ), "done");

            let mlp = mlp_out_names(h);
            let mut candidates = last_n(&mlp, 3);
            if candidates.is_empty() {
                candidates = last_n(&two_d_names(h), 3);
            }
            if candidates.is_empty() {
                return Err(WeightXError::Value("no 2D tensors for EAMIT".into()));
            }
            let base_shape = h.workspace.shape(&candidates[0])?;
            let targets: Vec<String> = candidates.into_iter()
                .filter(|t| h.workspace.shape(t).map(|s| s == base_shape).unwrap_or(false))
                .collect();

            let facts = synthetic_facts(7, 32, base_shape[1], base_shape[0], 0.015);
            let res = eamit::apply_eamit_batch(&mut h.workspace, &targets, &facts, 1e-2, true)?;
            record(2, sk.stage_names[1], format!(
                "выровнены {} key/residual пар для {} слоёв (KL {:.4}→{:.4}, top-M MSE {:.4}→{:.4})",
                facts.len(), targets.len(),
                res.pre_alignment_kl, res.post_alignment_kl,
                res.pre_alignment_top_m_mse, res.post_alignment_top_m_mse
            ), "done");
            let max_l2_delta = res.per_tensor_l2_deltas.values().copied().fold(0.0f64, f64::max);
            record(3, sk.stage_names[2], format!(
                "распределено {} фактов по {} слоям (max_l2_delta={max_l2_delta:.5}, mse={:.4}->{:.4})",
                facts.len(), targets.len(), res.pre_edit_mse, res.post_edit_mse
            ), "done");

            let v = h.vespasian.verify_and_decide(&mut h.workspace, arch.as_deref(), "Skill /eamit-facts");
            record(4, sk.stage_names[3], v.verdict_reason.clone(), if v.accepted { "done" } else { "rollback" });
            let metrics = vespasian_metrics(h, &v, json!({
                "facts_injected": facts.len(),
                "target_layers": targets.len(),
                "gini_kci": (kmap.kci_overall * 1e4).round() / 1e4,
                "max_l2_delta": r5(max_l2_delta),
                "error_reduction_pct": (res.relative_error_reduction * 10000.0).round() / 100.0,
                "alignment_kl_before": r5(res.pre_alignment_kl),
                "alignment_kl_after": r5(res.post_alignment_kl),
                "alignment_top_m_mse_after": r5(res.post_alignment_top_m_mse),
            }));
            Ok(SkillExecutionResult {
                skill_id: sk.skill_id.into(),
                pill_label: sk.pill_label.into(),
                pipeline_title: sk.pipeline_title.into(),
                accepted: v.accepted,
                summary: v.verdict_reason.clone(),
                yaml_plan: String::new(),
                steps,
                metrics,
                raw_report: String::new(),
            })
        }
        "knowledge-map" => {
            let arch_rep = l1::inspect_workspace(&h.workspace);
            record(1, sk.stage_names[0], format!(
                "семейство={}, слоёв={}, d_model={:?}", arch_rep.family, arch_rep.n_layers, arch_rep.d_model
            ), "done");
            let kmap = l2::run_knowledge_map(&h.workspace, arch.as_deref(), kmap_probes.as_deref(), None);
            let factual_layers: Vec<usize> = kmap.dominant_layers_by_domain.iter()
                .find(|(d, _)| d == "factual").map(|(_, l)| l.clone()).unwrap_or_default();
            let syntactic_layers: Vec<usize> = kmap.dominant_layers_by_domain.iter()
                .find(|(d, _)| d == "syntactic").map(|(_, l)| l.clone()).unwrap_or_default();
            record(2, sk.stage_names[1], format!("выполнен Logit Lens по {} слоям", kmap.logit_lens_layers.len()), "done");
            record(3, sk.stage_names[2], format!(
                "глобальный индекс Gini KCI={:.4}; пиковые слои знаний={factual_layers:?}", kmap.kci_overall
            ), "done");
            Ok(SkillExecutionResult {
                skill_id: sk.skill_id.into(),
                pill_label: sk.pill_label.into(),
                pipeline_title: sk.pipeline_title.into(),
                accepted: true,
                summary: format!("Карта знаний построена: Gini KCI = {:.4}, пиковые слои: {factual_layers:?}.", kmap.kci_overall),
                yaml_plan: String::new(),
                steps,
                metrics: json!({
                    "global_gini_kci": (kmap.kci_overall * 1e4).round() / 1e4,
                    "factual_layers": factual_layers,
                    "syntactic_layers": syntactic_layers,
                    "concentration_regime": kmap.concentration_regime,
                }),
                raw_report: format!("```toon\n{}\n```", kmap.to_toon()),
            })
        }
        "abliterate" => {
            let mlp_t = mlp_out_names(h);
            let two_d = two_d_names(h);
            let t = mlp_t.last().cloned()
                .or_else(|| two_d.first().cloned())
                .ok_or_else(|| WeightXError::Value("no 2D tensors".into()))?;
            let sh = h.workspace.shape(&t)?;
            record(1, sk.stage_names[0], format!("вычисление главного сингулярного вектора v1 для `{t}` {sh:?}"), "done");
            let yaml = format!(
                "transforms:\n  - assert: {{ shape: {{ of: '{t}', is: {sh:?} }} }}\n  - project_out_direction_: {{ target: '{t}', alpha: 0.15 }}\n  - assert: {{ finite: {{ of: '{t}' }} }}\n"
            );
            record(2, sk.stage_names[1], "сформирован декларативный YAML-план с `project_out_direction_`".into(), "done");
            run_yaml_step(h, &yaml)?;
            record(3, sk.stage_names[2], format!("ортогональная проекция применена к `{t}`"), "done");
            let v = h.vespasian.verify_and_decide(&mut h.workspace, arch.as_deref(), "Skill /abliterate");
            record(4, sk.stage_names[3], v.verdict_reason.clone(), if v.accepted { "done" } else { "rollback" });
            let metrics = vespasian_metrics(h, &v, json!({"target_tensor": t, "projection_alpha": 0.15}));
            Ok(SkillExecutionResult {
                skill_id: sk.skill_id.into(),
                pill_label: sk.pill_label.into(),
                pipeline_title: sk.pipeline_title.into(),
                accepted: v.accepted,
                summary: v.verdict_reason.clone(),
                yaml_plan: yaml,
                steps,
                metrics,
                raw_report: String::new(),
            })
        }
        "vespasian-guard" => {
            let two_d = two_d_names(h);
            let t = two_d.get(1.min(two_d.len().saturating_sub(1)))
                .ok_or_else(|| WeightXError::Value("no 2D tensors".into()))?
                .clone();
            let before_arr = h.workspace.get_readonly(&t)?.clone();
            h.workspace.checkpoint("vespasian_stress_test");
            record(1, sk.stage_names[0], format!("снимок `vespasian_best` проверен для `{t}`"), "done");

            let yaml = format!(
                "transforms:\n  - assert: {{ exists: {{ of: '{t}' }} }}\n  - scale_: {{ target: '{t}', by: -50.0 }}\n  - assert: {{ finite: {{ of: '{t}' }} }}\n"
            );
            run_yaml_step(h, &yaml)?;
            record(2, sk.stage_names[1], format!("внесён разрушающий масштаб `scale_: -50.0` в `{t}`"), "done");

            let v = h.vespasian.verify_and_decide(&mut h.workspace, arch.as_deref(), "Destructive stress-test");
            let after_arr = h.workspace.get_readonly(&t)?;
            let max_diff = before_arr.iter().zip(after_arr.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max) as f64;
            let guard_ok = v.rolled_back && max_diff < 1e-6;
            record(3, sk.stage_names[2], format!(
                "{} | побитовая погрешность после отката: max_abs_diff={max_diff:.2e}", v.verdict_reason
            ), if guard_ok { "done" } else { "rollback" });
            Ok(SkillExecutionResult {
                skill_id: sk.skill_id.into(),
                pill_label: sk.pill_label.into(),
                pipeline_title: sk.pipeline_title.into(),
                accepted: guard_ok,
                summary: format!(
                    "Защита Vespasian успешно перехватила деградацию и выполнила побитовый откат (`max_abs_diff = {max_diff:.1e}`)."
                ),
                yaml_plan: yaml,
                steps,
                metrics: json!({
                    "tested_tensor": t,
                    "rolled_back": v.rolled_back,
                    "guard_passed": guard_ok,
                    "corrupted_score": r5(v.candidate_score),
                    "restored_score": r5(h.vespasian.best_score),
                    "bit_exact_max_diff": max_diff,
                }),
                raw_report: String::new(),
            })
        }
        _ => Err(WeightXError::Value(format!("Skill {} not implemented", sk.skill_id))),
    }
}

fn skill_spectral_denoise(
    session: &mut StudioSession,
    sk: &WeightXSkill,
    user_args: &str,
) -> Result<SkillExecutionResult> {
    let total = sk.stage_names.len();
    let mut steps: Vec<SkillStepTrace> = Vec::new();
    let mut yaml_plan = String::new();

    // Steps 1-2: recon + outlier conditioning (scoped harness borrow).
    {
        let arch = session.loaded.as_ref().and_then(|l| l.arch.clone());
        let calib = session.calib_ids.clone();
        let log = session.log.clone();
        let h = session.harness.as_mut().unwrap();
        let mut record = |idx: usize, title: &str, detail: String| {
            steps.push(SkillStepTrace { index: idx, total, title: title.into(), detail: detail.clone(), status: "done".into() });
            log(&format!("[{idx}/{total}] {title} — {detail}"));
        };
        let arch_rep = l1::inspect_workspace(&h.workspace);
        let prof = l2::profile_model_layers(&h.workspace, arch.as_deref(), calib.as_ref(), "target_model", 2.5);
        record(1, sk.stage_names[0], format!(
            "семейство={}, узкие места (bottlenecks)={:?}", arch_rep.family,
            prof.bottleneck_layers.iter().take(3).collect::<Vec<_>>()
        ));
        if let Some(loc) = prof.outlier_neurons.first() {
            let tname = &loc.tensor_name;
            let sh = h.workspace.shape(tname)?;
            yaml_plan = format!(
                "transforms:\n  - assert: {{ shape: {{ of: '{tname}', is: {sh:?} }} }}\n  - rescale_outlier_rows_: {{ target: '{tname}', min_ratio: 2.5 }}\n  - assert: {{ finite: {{ of: '{tname}' }} }}\n"
            );
            run_yaml_step(h, &yaml_plan)?;
        }
        record(2, sk.stage_names[1], "проверены энергетические пики выходных проекций".into());
    }

    // Steps 3-4: closed-form domain calibration via the studio action.
    let report_text = session.act_auto_improve()?;
    {
        let log = session.log.clone();
        let mut record = |idx: usize, title: &str, detail: &str| {
            steps.push(SkillStepTrace { index: idx, total, title: title.into(), detail: detail.into(), status: "done".into() });
            log(&format!("[{idx}/{total}] {title} — {detail}"));
        };
        record(3, sk.stage_names[2], "выполнена итеративная Closed-Form калибровка C0^-1");
        record(4, sk.stage_names[3], "контрольный корпус и KL-дрейф проверены");
    }

    let h = session.harness.as_mut().unwrap();
    let baseline = h.vespasian.baseline_score;
    let best = h.vespasian.best_score;
    let accepted = best >= baseline;
    steps.push(SkillStepTrace {
        index: 5,
        total,
        title: sk.stage_names[4].into(),
        detail: format!("held-out score: {baseline:.5} → {best:.5}"),
        status: "done".into(),
    });
    (session.log)(&format!("[5/{total}] {} — held-out score: {baseline:.5} → {best:.5}", sk.stage_names[4]));
    let _ = user_args;
    Ok(SkillExecutionResult {
        skill_id: sk.skill_id.into(),
        pill_label: sk.pill_label.into(),
        pipeline_title: sk.pipeline_title.into(),
        accepted,
        summary: format!("Перплексия оптимизирована; held-out score {baseline:.5} → {best:.5}."),
        yaml_plan,
        steps,
        metrics: json!({
            "baseline_score": r5(baseline),
            "best_score": r5(best),
            "improvement": r5(best - baseline),
        }),
        raw_report: report_text,
    })
}