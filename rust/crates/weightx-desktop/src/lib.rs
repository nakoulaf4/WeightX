// SPDX-License-Identifier: GPL-3.0-or-later
//! WeightX Desktop (Tauri 2) — functional GUI over the weightx-core StudioSession.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tauri::{Emitter, Manager, State};
use weightx_core::studio::StudioSession;

struct AppState {
    session: Mutex<StudioSession>,
}

type CmdResult = Result<String, String>;

fn lock<'a>(state: &'a State<AppState>) -> std::sync::MutexGuard<'a, StudioSession> {
    state.session.lock().unwrap_or_else(|e| e.into_inner())
}

#[tauri::command(async)]
fn provider_kinds() -> Vec<String> {
    weightx_core::studio::PROVIDER_KINDS.iter().map(|s| s.to_string()).collect()
}

#[tauri::command(async)]
fn skills() -> Value {
    Value::Array(
        weightx_core::skills::WSP_SKILLS
            .iter()
            .enumerate()
            .map(|(i, sk)| {
                json!({
                    "idx": i + 1,
                    "skill_id": sk.skill_id,
                    "pill_label": sk.pill_label,
                    "pipeline_title": sk.pipeline_title,
                    "short_hint": sk.short_hint,
                    "stages": sk.stage_names,
                })
            })
            .collect(),
    )
}

#[tauri::command(async)]
fn suggestions() -> Value {
    Value::Array(
        weightx_core::catalog::codex_prompt_suggestions()
            .iter()
            .map(|s| s.to_value())
            .collect(),
    )
}

#[tauri::command(async)]
fn catalog(state: State<AppState>) -> Value {
    let mut s = lock(&state);
    Value::Array(s.catalog.list_entries().iter().map(|e| e.to_value()).collect())
}

/// Live Hugging Face Hub browse/search with real API filters, sort and cursor
/// pagination (native HTTP, no webview CORS limits). max_params silently limits
/// results to models that fit the host computer.
#[tauri::command(async)]
fn hf_search(query: String, limit: usize, cursor: String, task: String, license: String, sort: String, max_params: i64) -> Result<Value, String> {
    weightx_core::catalog::hf_search(&query, limit, &cursor, &task, &license, &sort, max_params).map_err(|e| e.to_string())
}

/// Real host resources: RAM and free disk space on the HF cache drive.
#[tauri::command(async)]
fn system_resources() -> Value {
    weightx_core::catalog::system_resources()
}

/// Real org/user avatar for a model namespace (cached in the core).
#[tauri::command(async)]
fn hf_avatar(org: String) -> Result<String, String> {
    weightx_core::catalog::hf_org_avatar(&org).map_err(|e| e.to_string())
}

/// Full model card: HF metadata + real README.md + avatar.
#[tauri::command(async)]
fn hf_model_card(repo: String) -> Result<Value, String> {
    weightx_core::catalog::hf_model_card(&repo).map_err(|e| e.to_string())
}

/// Open a Hugging Face model page or the cersei SDK repo in the system browser.
#[tauri::command(async)]
fn open_external(url: String) -> Result<(), String> {
    let allowed = url.starts_with("https://huggingface.co/") || url.starts_with("https://github.com/pacifio/cersei");
    if !allowed {
        return Err("разрешены только ссылки Hugging Face и репозиторий cersei".into());
    }
    #[cfg(windows)]
    std::process::Command::new("cmd")
        .args(["/C", "start", "", &url])
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Facts about the real LLM provider stack (shown in About).
/// API providers run through the cersei SDK (crates.io cersei-provider /
/// cersei-types, repository github.com/pacifio/cersei, pinned in Cargo.toml).
#[tauri::command(async)]
fn provider_stack() -> Value {
    json!({
        "sdk": "cersei",
        "version": "0.2.6",
        "repository": "https://github.com/pacifio/cersei",
        "api_kinds": ["openai", "openrouter", "ollama", "lm-studio", "vllm", "anthropic"],
        "cli_kinds": ["claude-code", "codex", "opencode", "antigravity", "dsh"],
        "detail": "Все API-провайдеры работают через cersei SDK: CompletionRequest → Provider::complete_blocking (cersei-provider + cersei-types). CLI-агенты подключаются напрямую через свои бинарники.",
    })
}

#[tauri::command(async)]
fn load_model(state: State<AppState>, model_id: String) -> CmdResult {
    let mut s = lock(&state);
    s.load_model(&model_id).map_err(|e| e.to_string())
}

/// Structured snapshot of the loaded model for the titlebar/statusbar/views.
#[tauri::command(async)]
fn model_info(state: State<AppState>) -> Value {
    let s = lock(&state);
    let Some(h) = &s.harness else {
        return json!({
            "loaded": false,
            "model_id": s.config.model_id,
            "provider_kind": s.config.provider_kind,
            "provider_model": s.config.provider_model,
        });
    };
    let entry = s.loaded.as_ref().map(|l| l.entry.to_value()).unwrap_or(json!({}));
    let d = h.workspace.diff();
    json!({
        "loaded": true,
        "model_id": s.config.model_id,
        "entry": entry,
        "total_params": h.workspace.total_params(),
        "tensor_count": h.workspace.all_names().len(),
        "backend": s.loaded.as_ref().and_then(|l| l.arch.as_ref()).map(|a| a.backend_name()).unwrap_or("rust"),
        "has_tokenizer": s.has_tokenizer(),
        "provider_kind": s.config.provider_kind,
        "provider_model": s.config.provider_model,
        "vespasian": {
            "baseline": h.vespasian.baseline_score,
            "best": h.vespasian.best_score,
            "iteration": h.vespasian.iteration,
        },
        "pipeline": {
            "fisher_alpha": h.workspace.fisher_guard.as_ref().map(|g| g.damping_alpha).unwrap_or(0.15),
            "fisher_freeze": h.workspace.fisher_guard.as_ref().map(|g| g.freeze_top_fraction).unwrap_or(0.0),
            "improvement_threshold": h.vespasian.improvement_threshold,
            "max_kl_drift": h.vespasian.max_kl_drift,
            "max_changed_fraction": h.vespasian.max_changed_fraction,
            "practice_budget": h.vespasian.practice_budget,
        },
        "diff": { "changed_params": d.changed_params, "total_params": d.total_params },
    })
}

#[tauri::command(async)]
fn set_provider(
    state: State<AppState>,
    kind: String,
    model: String,
    api_key: String,
    base_url: String,
) -> CmdResult {
    let mut s = lock(&state);
    s.set_provider(&kind, &model, &api_key, &base_url).map_err(|e| e.to_string())
}

/// Translate raw transport/HTTP errors into clear, human Russian messages.
fn humanize_provider_error(k: &str, raw: &str) -> String {
    let low = raw.to_lowercase();
    let local_name = match k {
        "ollama" => "Ollama",
        "lm-studio" => "LM Studio",
        "vllm" => "vLLM",
        _ => "",
    };
    let refused = low.contains("10061") || low.contains("connection refused") || low.contains("отверг запрос");
    let timeout = low.contains("timed out") || low.contains("timeout");
    let dns = low.contains("dns error") || low.contains("resolve") || low.contains("getaddrinfo") || low.contains("name or service");
    if refused && !local_name.is_empty() {
        return format!("{local_name} не отвечает — вероятно, не запущена. Запустите {local_name} и повторите.");
    }
    if refused {
        return "Не удалось соединиться с API. Проверьте base_url и доступность сервера.".into();
    }
    if timeout {
        return "Провайдер не ответил вовремя. Проверьте, что он запущен, и повторите.".into();
    }
    if dns {
        return "Не удалось найти сервер. Проверьте base_url и подключение к интернету.".into();
    }
    if low.contains("401") || low.contains("unauthorized") {
        return "API key неверный или просрочен. Укажите корректный ключ.".into();
    }
    if low.contains("403") || low.contains("forbidden") {
        return "Доступ запрещён. Проверьте API key и права.".into();
    }
    if low.contains("404") {
        return "Эндпоинт не найден. Проверьте base_url.".into();
    }
    if low.contains("429") {
        return "Слишком много запросов (лимит API). Повторите позже.".into();
    }
    if low.contains("certificate") || low.contains("tls") {
        return "Ошибка защищённого соединения (TLS). Проверьте сеть и сертификаты.".into();
    }
    format!("Провайдер недоступен: {}", raw.chars().take(160).collect::<String>())
}

/// Run a CLI agent's real model-listing command with a timeout.
fn run_cli_models(binary: &str, args: &[&str]) -> Result<String, String> {
    let path = weightx_agent::cli_agent::which(binary)
        .ok_or_else(|| format!("CLI «{binary}» не найден в PATH. Установите его или добавьте в PATH."))?;
    let mut child = std::process::Command::new(&path)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("Не удалось запустить {binary}: {e}"))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(25);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return Err(format!("Команда «{binary} {}» завершилась с ошибкой.", args.join(" ")));
                }
                let out = child.wait_with_output().map_err(|e| e.to_string())?;
                return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
            }
            Ok(None) => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    return Err(format!("Команда «{binary} {}» не ответила за 25 секунд.", args.join(" ")));
                }
                std::thread::sleep(std::time::Duration::from_millis(120));
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// Real model lists from CLI agents: `opencode models` (provider/model lines)
/// and `agy models` (TAB-separated id/name lines).
fn cli_model_listing(agent: &str) -> Result<Value, String> {
    match agent {
        "opencode" => {
            let out = run_cli_models("opencode", &["models"])?;
            let models: Vec<String> = out
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.contains(char::is_whitespace) && l.contains('/'))
                .take(300)
                .map(String::from)
                .collect();
            Ok(json!({ "models": models, "source": "opencode models" }))
        }
        "antigravity" => {
            let out = run_cli_models("agy", &["models"])?;
            let models: Vec<String> = out
                .lines()
                .map(|l| l.split('\t').next().unwrap_or("").trim())
                .filter(|l| !l.is_empty() && !l.contains(char::is_whitespace))
                .take(300)
                .map(String::from)
                .collect();
            Ok(json!({ "models": models, "source": "agy models" }))
        }
        other => Ok(json!({ "models": [], "source": "", "note": format!("CLI «{other}» не отдаёт список моделей — укажите модель вручную.") })),
    }
}

/// Shared real-API model listing (ollama / lm-studio / openai / openrouter /
/// vllm / anthropic). Returns (models, source_url) or a human-readable error.
fn fetch_provider_models(k: &str, base: &str, key: &str) -> Result<(Vec<String>, String), String> {
    let base = base.trim().trim_end_matches('/');
    let key = key.trim();
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(8))
        .build();
    let (url, bearer, extra): (String, bool, Vec<(String, String)>) = match k {
        "ollama" => (
            format!("{}/api/tags", if base.is_empty() { "http://localhost:11434" } else { base }),
            false,
            vec![],
        ),
        "lm-studio" => (
            format!("{}/v1/models", if base.is_empty() { "http://localhost:1234" } else { base }),
            false,
            vec![],
        ),
        "vllm" => (
            format!("{}/v1/models", if base.is_empty() { "http://localhost:8000" } else { base }),
            !key.is_empty(),
            vec![],
        ),
        "openai" => (
            format!("{}/v1/models", if base.is_empty() { "https://api.openai.com" } else { base }),
            true,
            vec![],
        ),
        "openrouter" => ("https://openrouter.ai/api/v1/models".into(), !key.is_empty(), vec![]),
        "anthropic" => (
            "https://api.anthropic.com/v1/models".into(),
            false,
            vec![
                ("x-api-key".into(), key.to_string()),
                ("anthropic-version".into(), "2023-06-01".into()),
            ],
        ),
        _ => return Err("для этого провайдера список моделей не предоставляется — укажите модель вручную".into()),
    };
    if k == "anthropic" && key.is_empty() {
        return Err("нужен API key".into());
    }
    if k == "openai" && key.is_empty() && base.is_empty() {
        return Err("нужен API key или base_url".into());
    }
    let mut req = agent.get(&url).set("User-Agent", "WeightX/0.4.0");
    if bearer && !key.is_empty() {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    for (h, v) in &extra {
        if !v.is_empty() {
            req = req.set(h, v);
        }
    }
    let body = req
        .call()
        .map_err(|e| humanize_provider_error(k, &e.to_string()))?
        .into_string()
        .map_err(|e| humanize_provider_error(k, &e.to_string()))?;
    let data: Value = serde_json::from_str(&body).map_err(|_| "Провайдер вернул неожиданный ответ (не JSON). Проверьте base_url.".to_string())?;
    let models: Vec<String> = if k == "ollama" {
        data["models"]
            .as_array()
            .map(|a| a.iter().filter_map(|m| m["name"].as_str().map(String::from)).collect())
            .unwrap_or_default()
    } else {
        data["data"]
            .as_array()
            .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(String::from)).collect())
            .unwrap_or_default()
    };
    Ok((models, url))
}

/// Live model list from the provider's real API or CLI command. No hardcoded catalogs.
#[tauri::command(async)]
fn provider_models(kind: String, base_url: String, api_key: String) -> Result<Value, String> {
    let k = kind.split(' ').next().unwrap_or("").to_string();
    if let Some(agent) = k.strip_prefix("cli:") {
        return cli_model_listing(agent);
    }
    match fetch_provider_models(&k, &base_url, &api_key) {
        Ok((models, url)) => Ok(json!({ "models": models, "source": url })),
        Err(e) if e.starts_with("для этого провайдера") => Ok(json!({ "models": [], "source": "", "note": e })),
        Err(e) => Err(e),
    }
}

/// Real availability check before a provider is activated as the brain:
/// CLI kinds -> binary lookup on PATH; API kinds -> live endpoint ping.
#[tauri::command(async)]
fn provider_check(kind: String, base_url: String, api_key: String) -> Value {
    let k = kind.split(' ').next().unwrap_or("");
    if k == "autonomous" {
        return json!({ "ok": true, "detail": "встроенный runtime · всегда доступен" });
    }
    if let Some(agent) = k.strip_prefix("cli:") {
        let binary = match agent {
            "claude-code" => "claude",
            "codex" => "codex",
            "opencode" => "opencode",
            "antigravity" => "agy",
            "dsh" => "dsh",
            other => other,
        };
        return match weightx_agent::cli_agent::which(binary) {
            Some(p) => json!({ "ok": true, "detail": format!("CLI найден: {}", p.display()) }),
            None => json!({ "ok": false, "detail": format!("CLI «{binary}» не найден в PATH — установите его или проверьте PATH") }),
        };
    }
    match fetch_provider_models(k, &base_url, &api_key) {
        Ok((models, url)) => json!({ "ok": true, "detail": format!("{url} · моделей доступно: {}", models.len()) }),
        Err(e) => json!({ "ok": false, "detail": e }),
    }
}

#[tauri::command(async)]
fn run_skill(state: State<AppState>, skill_id: String, args: String) -> Result<Value, String> {
    let mut s = lock(&state);
    s.run_skill(&skill_id, &args)
        .map(|r| {
            json!({
                "markdown": r.to_markdown(),
                "yaml_plan": r.yaml_plan,
                "accepted": r.accepted,
                "skill_id": r.skill_id,
                "pill_label": r.pill_label,
                "pipeline_title": r.pipeline_title,
                "summary": r.summary,
                "metrics": r.metrics,
                "steps": r.steps.iter().map(|st| json!({
                    "index": st.index, "total": st.total,
                    "title": st.title, "detail": st.detail, "status": st.status,
                })).collect::<Vec<_>>(),
            })
        })
        .map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn tensor_names(state: State<AppState>) -> Result<Vec<String>, String> {
    let s = lock(&state);
    match &s.harness {
        Some(h) => Ok(h.workspace.all_names()),
        None => Err("Модель не загружена".into()),
    }
}

/// Full tensor table (name/shape/dtype/numel/l2_norm/modified) for the L3 view.
#[tauri::command(async)]
fn tensor_table(state: State<AppState>) -> Result<Value, String> {
    let s = lock(&state);
    let h = s.harness.as_ref().ok_or("Модель не загружена")?;
    Ok(Value::Array(h.workspace.list_tensors_table(None, None)))
}

#[tauri::command(async)]
fn describe_tensor(state: State<AppState>, name: String, top_k: usize) -> Result<Value, String> {
    let s = lock(&state);
    let h = s.harness.as_ref().ok_or("Модель не загружена")?;
    h.workspace.describe_tensor(&name, top_k).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn read_tensor_slice(
    state: State<AppState>,
    name: String,
    index: String,
    max_elements: usize,
) -> Result<Value, String> {
    let s = lock(&state);
    let h = s.harness.as_ref().ok_or("Модель не загружена")?;
    h.workspace
        .read_slice(&name, Some(&index), max_elements, 6)
        .map_err(|e| e.to_string())
}

/// Top-left numeric block of a tensor for the weight heatmap.
#[tauri::command(async)]
fn tensor_block(state: State<AppState>, name: String, size: usize) -> Result<Value, String> {
    let s = lock(&state);
    let h = s.harness.as_ref().ok_or("Модель не загружена")?;
    let t = h.workspace.get_readonly(&name).map_err(|e| e.to_string())?;
    let size = size.clamp(2, 32);
    let m2d = weightx_core::tensor::as_2d(t);
    let rows = m2d.nrows().min(size);
    let cols = m2d.ncols().min(size);
    let values: Vec<f64> = m2d
        .slice(ndarray::s![..rows, ..cols])
        .iter()
        .map(|v| *v as f64)
        .collect();
    Ok(json!({ "rows": rows, "cols": cols, "values": values }))
}

/// L1 AutoArch + L2 Uni-Layer profile + Knowledge Map as structured JSON.
#[tauri::command(async)]
fn diagnostics(state: State<AppState>) -> Result<Value, String> {
    let mut s = lock(&state);
    let arch = s.loaded.as_ref().and_then(|l| l.arch.clone());
    let calib = s.calib_ids.clone();
    let probes = s.loaded.as_ref().and_then(|l| l.tokenizer.as_ref()).map(|t| {
        weightx_core::l2::knowledge_probes_from_tokenizer(t, s.loaded.as_ref().and_then(|l| l.bos_token_id))
    });
    let model_id = s.config.model_id.clone();
    let h = s.harness.as_mut().ok_or("Модель не загружена")?;
    let rep = weightx_core::l1::inspect_workspace(&h.workspace);
    let prof = weightx_core::l2::profile_model_layers(
        &h.workspace,
        arch.as_deref(),
        calib.as_ref(),
        &model_id,
        2.5,
    );
    let knowledge = weightx_core::l2::run_knowledge_map(&h.workspace, arch.as_deref(), probes.as_deref(), None);
    Ok(json!({
        "l1": rep.to_value(false),
        "summary": prof.summary,
        "num_layers": prof.num_layers,
        "redundant": prof.redundant_layers,
        "bottlenecks": prof.bottleneck_layers,
        "outliers": prof.outlier_neurons.iter().map(|o| o.to_value()).collect::<Vec<_>>(),
        "layers": prof.per_layer_metrics,
        "knowledge": knowledge.to_value(),
    }))
}

/// Real Vespasian iteration history (committed / rolled-back checkpoints).
#[tauri::command(async)]
fn history(state: State<AppState>) -> Value {
    let s = lock(&state);
    let Some(h) = &s.harness else {
        return json!({ "model_id": s.config.model_id, "baseline": 0.0, "best": 0.0, "iterations": [] });
    };
    json!({
        "model_id": s.config.model_id,
        "baseline": h.vespasian.baseline_score,
        "best": h.vespasian.best_score,
        "iterations": h.vespasian.history.iter().enumerate().map(|(i, v)| json!({
            "id": format!("cp-{:03}", i + 1),
            "iteration": v.iteration,
            "accepted": v.accepted,
            "rolled_back": v.rolled_back,
            "hypothesis": v.hypothesis,
            "verdict": v.verdict_reason,
            "baseline_score": v.baseline_score,
            "previous_best": v.previous_best_score,
            "candidate_score": v.candidate_score,
            "best_after": v.best_score_after,
            "delta": v.score_delta_vs_best,
            "kl_drift": v.kl_divergence_ref,
            "changed_pct": v.changed_params_pct,
            "nonfinite": v.nonfinite_params,
        })).collect::<Vec<_>>(),
    })
}

/// Bit-exact restore of all modified tensors to the original loaded weights.
#[tauri::command(async)]
fn rollback_weights(state: State<AppState>) -> CmdResult {
    let mut s = lock(&state);
    let h = s.harness.as_mut().ok_or("Модель не загружена")?;
    h.workspace.revert_to_original(None).map_err(|e| e.to_string())?;
    let d = h.workspace.diff();
    Ok(format!(
        "Веса восстановлены к исходному состоянию (bit-exact). Изменено параметров: {} / {}",
        d.changed_params, d.total_params
    ))
}

#[tauri::command(async)]
fn run_action(state: State<AppState>, tag: String, prompt: String) -> String {
    let mut s = lock(&state);
    s.run_action(&tag, &prompt)
}

#[tauri::command(async)]
fn apply_yaml(state: State<AppState>, plan: String, fisher: bool) -> CmdResult {
    let mut s = lock(&state);
    s.apply_yaml(&plan, fisher).map_err(|e| e.to_string())
}

/// YAML surgery with explicit Fisher / Vespasian guard switches.
#[tauri::command(async)]
fn apply_yaml_ex(state: State<AppState>, plan: String, fisher: bool, verify: bool) -> CmdResult {
    let mut s = lock(&state);
    if verify {
        return s.apply_yaml(&plan, fisher).map_err(|e| e.to_string());
    }
    let h = s.harness.as_mut().ok_or("Модель не загружена")?;
    let rep = weightx_core::plan::execute_yaml_plan(&mut h.workspace, &plan, false, true, fisher);
    let mut out = format!(
        "## План BrainSurgery-WX\n{}",
        weightx_core::toon::dumps_toon(&rep.to_value())
    );
    if rep.success {
        out += "\n\n## Vespasian\nпропущен — auto-rollback выключен в настройках";
    }
    Ok(out)
}

/// Live tuning of the real Fisher guard + Vespasian engine parameters.
#[tauri::command(async)]
fn set_pipeline(
    state: State<AppState>,
    fisher_alpha: f64,
    fisher_freeze: f64,
    improvement_threshold: f64,
    max_kl_drift: f64,
    max_changed_fraction: f64,
    practice_budget: usize,
) -> CmdResult {
    let mut s = lock(&state);
    let h = s.harness.as_mut().ok_or("Модель не загружена")?;
    if let Some(fg) = &mut h.workspace.fisher_guard {
        fg.damping_alpha = fisher_alpha.clamp(0.0, 1.0);
        fg.freeze_top_fraction = fisher_freeze.clamp(0.0, 1.0);
    }
    h.vespasian.improvement_threshold = improvement_threshold.max(0.0);
    h.vespasian.max_kl_drift = max_kl_drift.max(0.0);
    h.vespasian.max_changed_fraction = max_changed_fraction.clamp(0.0, 1.0);
    h.vespasian.practice_budget = practice_budget.max(1);
    Ok("Параметры пайплайна применены к движку".into())
}

#[tauri::command(async)]
fn export_checkpoint(state: State<AppState>, path: String) -> CmdResult {
    let mut s = lock(&state);
    s.export_checkpoint(&path).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn status(state: State<AppState>) -> String {
    let mut s = lock(&state);
    s.status()
}

#[tauri::command(async)]
fn diff(state: State<AppState>) -> Value {
    let s = lock(&state);
    match &s.harness {
        Some(h) => h.workspace.diff().to_value(),
        None => json!({"error": "model not loaded"}),
    }
}

fn arg_flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

pub fn run() {
    let args: Vec<String> = std::env::args().collect();
    let smoke_ms: Option<u64> = arg_flag(&args, "--smoke-test-ms").and_then(|s| s.parse().ok());

    tauri::Builder::default()
        .setup(move |app| {
            let handle = app.handle().clone();
            let mut session = StudioSession::new();
            session.factory = weightx_agent::agent_factory();
            let spec = session.provider_spec();
            if let Ok(p) = weightx_agent::build_provider(&spec) {
                session.provider = p;
            }
            // No model is loaded at startup: the user picks one explicitly in the
            // catalog. Auto-load only for smoke tests or an explicit --model flag.
            let explicit_model = arg_flag(&args, "--model");
            let initial_model = explicit_model.clone().unwrap_or_else(|| {
                if session.config.model_id.is_empty() {
                    "minillama-demo".to_string()
                } else {
                    session.config.model_id.clone()
                }
            });
            let log_handle = handle.clone();
            session.set_log(Arc::new(move |msg: &str| {
                let _ = log_handle.emit("log", msg.to_string());
            }));
            app.manage(AppState { session: Mutex::new(session) });

            let load_handle = handle.clone();
            let model = initial_model.clone();
            let smoke = smoke_ms.is_some();
            let should_load = smoke || explicit_model.is_some();
            std::thread::spawn(move || {
                let state = load_handle.state::<AppState>();
                if should_load {
                    let result = state.session.lock().unwrap_or_else(|e| e.into_inner()).load_model(&model);
                    match result {
                        Ok(msg) => {
                            let _ = load_handle.emit("model-loaded", msg.clone());
                            if smoke {
                                println!("[smoke] {msg}");
                            }
                        }
                        Err(e) => {
                            let _ = load_handle.emit("model-error", e.to_string());
                            if smoke {
                                eprintln!("[smoke] model load failed: {e}");
                                load_handle.exit(1);
                            }
                        }
                    }
                }
                let _ = load_handle.emit("status-changed", ());
                if smoke {
                    let mut s = state.session.lock().unwrap_or_else(|e| e.into_inner());
                    match s.run_skill("vespasian-guard", "") {
                        Ok(res) => {
                            println!("[smoke] vespasian-guard accepted={}", res.accepted);
                            if !res.accepted {
                                load_handle.exit(1);
                            }
                        }
                        Err(e) => {
                            eprintln!("[smoke] skill failed: {e}");
                            load_handle.exit(1);
                        }
                    }
                    let recon = s.run_action("recon_profile", "");
                    println!("[smoke] recon_profile ok={}", recon.contains("L1 AutoArch"));
                    println!("[smoke] {}", s.status().replace('\n', " | "));
                    println!("[smoke] SMOKE OK");
                    load_handle.exit(0);
                }
            });

            if let Some(ms) = smoke_ms {
                let exit_handle = handle.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(ms));
                    eprintln!("[smoke] timeout after {ms}ms");
                    exit_handle.exit(1);
                });
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            provider_kinds,
            skills,
            suggestions,
            catalog,
            hf_search,
            hf_avatar,
            hf_model_card,
            open_external,
            system_resources,
            load_model,
            model_info,
            set_provider,
            provider_models,
            provider_check,
            provider_stack,
            run_skill,
            run_action,
            apply_yaml,
            apply_yaml_ex,
            set_pipeline,
            export_checkpoint,
            status,
            diff,
            history,
            rollback_weights,
            diagnostics,
            tensor_names,
            tensor_table,
            tensor_block,
            describe_tensor,
            read_tensor_slice
        ])
        .run(tauri::generate_context!())
        .expect("error while running WeightX desktop");
}