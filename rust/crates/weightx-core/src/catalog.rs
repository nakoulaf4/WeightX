// SPDX-License-Identifier: GPL-3.0-or-later
//! WeightX Model Catalog & prompt suggestions.
//! Rust port of `weightx.catalog` (built-in catalog, local/HF-cache discovery,
//! unified loading of MiniLlama / local safetensors / HuggingFace repos).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::{Result, WeightXError};
use crate::model::gpt2::Gpt2Model;
use crate::model::llama::LlamaModel;
use crate::model::minillama;
use crate::model::{ModelArch, ModelConfig};
use crate::tensor::{self, Tensor};
use crate::toon;

#[derive(Debug, Clone)]
pub struct ModelCatalogEntry {
    pub model_id: String,
    pub hf_repo_id: String,
    pub display_name: String,
    pub family: String,
    pub params_human: String,
    pub num_layers: usize,
    pub d_model: usize,
    pub vram_fp32_mb: usize,
    pub license_spdx: String,
    pub description: String,
    pub recommended_tasks: Vec<String>,
    pub is_builtin_synthetic: bool,
    pub local_cached: bool,
    pub local_path: Option<String>,
}

impl ModelCatalogEntry {
    pub fn to_value(&self) -> Value {
        json!({
            "model_id": self.model_id,
            "hf_repo_id": self.hf_repo_id,
            "display_name": self.display_name,
            "family": self.family,
            "params_human": self.params_human,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "vram_fp32_mb": self.vram_fp32_mb,
            "license_spdx": self.license_spdx,
            "description": self.description,
            "recommended_tasks": self.recommended_tasks,
            "is_builtin_synthetic": self.is_builtin_synthetic,
            "local_cached": self.local_cached,
            "local_path": self.local_path,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PromptSuggestion {
    pub prompt_id: String,
    pub category: String,
    pub icon: String,
    pub title: String,
    pub short_hint: String,
    pub prompt_text: String,
    pub target_tools: Vec<String>,
    pub action_tag: String,
}

impl PromptSuggestion {
    pub fn to_value(&self) -> Value {
        json!({
            "prompt_id": self.prompt_id,
            "category": self.category,
            "icon": self.icon,
            "title": self.title,
            "short_hint": self.short_hint,
            "prompt_text": self.prompt_text,
            "target_tools": self.target_tools,
            "action_tag": self.action_tag,
        })
    }
}

macro_rules! entry {
    ($id:expr, $repo:expr, $name:expr, $fam:expr, $params:expr, $layers:expr, $d:expr, $vram:expr, $lic:expr, $desc:expr, $tasks:expr) => {
        ModelCatalogEntry {
            model_id: $id.into(),
            hf_repo_id: $repo.into(),
            display_name: $name.into(),
            family: $fam.into(),
            params_human: $params.into(),
            num_layers: $layers,
            d_model: $d,
            vram_fp32_mb: $vram,
            license_spdx: $lic.into(),
            description: $desc.into(),
            recommended_tasks: $tasks.iter().map(|s| s.to_string()).collect(),
            is_builtin_synthetic: false,
            local_cached: false,
            local_path: None,
        }
    };
}

pub fn default_catalog_entries() -> Vec<ModelCatalogEntry> {
    let mut entries = vec![
        entry!("minillama-demo", "builtin/MiniLlama-4L-64D", "MiniLlama-4L (Встроенный стенд с инъекцией выбросов)", "llama", "215K", 4, 64, 4, "GPL-3.0-or-later",
            "Быстрая встроенная модель семейства Llama (RMSNorm + SwiGLU + GQA/MHA) для мгновенных локальных экспериментов и отладки YAML-планов.",
            ["outlier_repair", "eamit_batch", "vespasian_rollback"]),
        entry!("tiny-gpt2", "sshleifer/tiny-gpt2", "Tiny-GPT2 (HuggingFace Pretrained Causal LM)", "gpt2", "103K", 2, 2, 5, "MIT",
            "Ультракомпактная предобученная модель GPT-2 из HuggingFace Hub с полноценным BPE-токенизатором для мгновенного тестирования реального пайплайна.",
            ["spectral_denoise", "outlier_repair", "logit_lens"]),
        entry!("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M Base (HuggingFaceTB) — рекомендуемая цель улучшения", "llama", "135M", 30, 576, 540, "Apache-2.0",
            "Базовая (pretrained) SmolLM2-135M. Используется в бенчмарке WeightX: closed-form EAMIT доменная калибровка с проверкой на скрытом held-out.",
            ["domain_calibration", "unilayer_13_metrics", "knowledge_map"]),
        entry!("smollm2-135m-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M-Instruct (HuggingFaceTB)", "llama", "135M", 30, 576, 540, "Apache-2.0",
            "Современная компактная Llama-совместимая модель (30 слоев, GQA, SwiGLU). Идеальна для исследования глубоких бутылочных слоев и EAMIT на CPU/GPU.",
            ["unilayer_13_metrics", "knowledge_map", "eamit_batch", "spectral_denoise"]),
        entry!("smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2-360M-Instruct (HuggingFaceTB)", "llama", "360M", 32, 960, 1440, "Apache-2.0",
            "Сбалансированная модель 360M параметров с выраженной специализацией средних и поздних слоев.",
            ["knowledge_map", "abliteration", "eamit_batch"]),
        entry!("gpt2-124m", "openai-community/gpt2", "GPT-2 Base (124M, классический стенд ROME/EAMIT)", "gpt2", "124M", 12, 768, 500, "MIT",
            "Классическая 12-слойная трансформерная модель, используемая в экспериментах ROME, EAMIT и Logit Lens.",
            ["logit_lens", "eamit_batch", "outlier_repair"]),
        entry!("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct (24 слоя, сильная мультиязычность)", "qwen2", "494M", 24, 896, 1980, "Apache-2.0",
            "Высококачественная модель 0.5B параметров (поддерживает русский и английский языки), отличная цель для точечной хирургии весов.",
            ["unilayer_13_metrics", "knowledge_map", "eamit_batch", "abliteration"]),
        entry!("tinyllama-1.1b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B-Chat-v1.0 (22 слоя, архитектура Llama-2)", "llama", "1.1B", 22, 2048, 4400, "Apache-2.0",
            "Полноценная 1.1B модель архитектуры Llama-2 для масштабирования YAML-планов BrainSurgery-WX.",
            ["unilayer_13_metrics", "eamit_batch", "spectral_denoise"]),
    ];
    entries[0].is_builtin_synthetic = true;
    entries[0].local_cached = true;
    entries
}

macro_rules! suggestion {
    ($id:expr, $icon:expr, $title:expr, $hint:expr, $tools:expr) => {
        PromptSuggestion {
            prompt_id: $id.into(),
            category: format!("Скилл /skill {}", $id),
            icon: $icon.into(),
            title: $title.into(),
            short_hint: $hint.into(),
            prompt_text: format!("/skill {}", $id),
            target_tools: $tools.iter().map(|s| s.to_string()).collect(),
            action_tag: format!("skill:{}", $id),
        }
    };
}

pub fn codex_prompt_suggestions() -> Vec<PromptSuggestion> {
    vec![
        suggestion!("coding-boost", "⚡", "Улучшить в кодинге",
            "Пайплайн 5 шагов: L1/L2 -> фокусировка внимания -> Code EAMIT -> IWCKEdit -> Vespasian",
            ["l1_autoarch_inspect", "l2_unilayer_profile", "l3_apply_yaml_plan", "l3_verify_and_commit"]),
        suggestion!("memory-slim", "🪶", "Уменьшить память",
            "Пайплайн 4 шага: спектральный аудит ранга -> svd_low_rank_ -> ties_trim_ -> Vespasian",
            ["l2_unilayer_profile", "l3_apply_yaml_plan", "l3_verify_and_commit"]),
        suggestion!("outlier-repair", "🛠", "Починить выбросы",
            "Пайплайн 4 шага: поиск kurtosis-выбросов -> rescale_outlier_rows_ -> IWCKEdit -> Vespasian",
            ["l2_unilayer_profile", "l3_describe_tensor", "l3_apply_yaml_plan", "l3_verify_and_commit"]),
        suggestion!("spectral-denoise", "📉", "Снизить перплексию",
            "Пайплайн 5 шагов: аудит анизотропии -> Closed-Form Domain EAMIT -> контроль KL -> Vespasian",
            ["l3_describe_tensor", "l3_apply_yaml_plan", "l3_verify_and_commit"]),
        suggestion!("eamit-facts", "🧠", "Записать факты",
            "Пайплайн 4 шага: локализация KCI -> выравнивание эмбеддингов (KL + top-M MSE) -> EAMIT -> Vespasian",
            ["l3_batch_edit_eamit", "l3_verify_and_commit"]),
        suggestion!("knowledge-map", "🗺", "Карта знаний",
            "Пайплайн 3 шага: нормализованный Logit Lens -> индекс концентрации Джини KCI",
            ["l2_knowledge_map"]),
        suggestion!("abliterate", "🧭", "Вырезать отказы",
            "Пайплайн 4 шага: SVD-выделение направления -> project_out_direction_ -> Vespasian",
            ["l3_apply_yaml_plan", "l3_verify_and_commit"]),
        suggestion!("vespasian-guard", "🛡", "Проверить откат",
            "Пайплайн 3 шага: снимок весов -> стресс-атака scale_: -50 -> побитовый авто-откат Vespasian",
            ["l3_apply_yaml_plan", "l3_verify_and_commit"]),
    ]
}

/// A loaded target model: weights + optional runtime + optional tokenizer.
pub struct LoadedModel {
    pub arch: Option<Arc<dyn ModelArch>>,
    pub tensors: Vec<(String, Tensor)>,
    pub tokenizer: Option<tokenizers::Tokenizer>,
    pub bos_token_id: Option<i64>,
    pub entry: ModelCatalogEntry,
    /// Original on-disk dtype names (computation is f32).
    pub dtypes: HashMap<String, String>,
}

pub struct ModelCatalog {
    entries: Vec<ModelCatalogEntry>,
    search_dirs: Vec<PathBuf>,
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelCatalog {
    pub fn new() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            entries: default_catalog_entries(),
            search_dirs: vec![cwd.clone(), cwd.join("models"), cwd.join("checkpoints")],
        }
    }

    pub fn refresh_cache_status(&mut self) {
        let hf_cache = hub_cache_dir();
        // 1. Re-verify the on-disk cache state of every HF-repo entry (real check,
        //    not a sticky flag): a repo is cached only if a snapshot folder exists.
        for entry in self.entries.iter_mut() {
            if entry.is_builtin_synthetic {
                entry.local_cached = true;
                continue;
            }
            let repo = &entry.hf_repo_id;
            if repo.contains('/') && !repo.contains('\\') && !repo.ends_with(".safetensors") {
                let folder = hf_cache.join(format!("models--{}", repo.replace('/', "--")));
                let cached = first_subdir(&folder.join("snapshots")).is_some();
                entry.local_cached = cached;
                entry.local_path = if cached { Some(folder.to_string_lossy().into_owned()) } else { None };
            }
        }
        // 2. Discover ALL repos present in the HF hub cache (downloaded by WeightX
        //    or any other tool) and surface them as catalog entries.
        if let Ok(rd) = std::fs::read_dir(&hf_cache) {
            for f in rd.flatten() {
                let name = f.file_name().to_string_lossy().into_owned();
                let Some(rest) = name.strip_prefix("models--") else { continue };
                let repo_id = rest.replacen("--", "/", 1);
                if repo_id.is_empty() || self.entries.iter().any(|e| e.hf_repo_id.eq_ignore_ascii_case(&repo_id)) {
                    continue;
                }
                let Some(snap) = first_subdir(&f.path().join("snapshots")) else { continue };
                let mut family = "safetensors".to_string();
                let mut num_layers = 0usize;
                let mut d_model = 0usize;
                if let Ok(cfg_text) = std::fs::read_to_string(snap.join("config.json")) {
                    if let Ok(cfg) = serde_json::from_str::<Value>(&cfg_text) {
                        family = cfg["model_type"].as_str().unwrap_or("safetensors").to_string();
                        num_layers = cfg["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
                        d_model = cfg["hidden_size"].as_u64().unwrap_or(0) as usize;
                    }
                }
                let bytes: u64 = std::fs::read_dir(&snap)
                    .map(|rd| rd.flatten()
                        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
                        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                        .sum())
                    .unwrap_or(0);
                if bytes == 0 {
                    continue; // incomplete download (no weights yet)
                }
                let size_mb = (bytes / (1024 * 1024)).max(1) as usize;
                self.entries.push(ModelCatalogEntry {
                    model_id: repo_id.clone(),
                    hf_repo_id: repo_id.clone(),
                    display_name: format!("{repo_id} (кэш Hugging Face)"),
                    family,
                    params_human: format!("~{size_mb} MB"),
                    num_layers,
                    d_model,
                    vram_fp32_mb: size_mb,
                    license_spdx: "—".into(),
                    description: format!("Скачана с Hugging Face · кэш: {}", snap.display()),
                    recommended_tasks: vec!["recon_full".into(), "apply_yaml_plan".into()],
                    is_builtin_synthetic: false,
                    local_cached: true,
                    local_path: Some(f.path().to_string_lossy().into_owned()),
                });
            }
        }
        // 3. Local .safetensors checkpoints in the workspace search dirs.
        for dir in &self.search_dirs {
            let Ok(rd) = std::fs::read_dir(dir) else { continue };
            for f in rd.flatten() {
                let p = f.path();
                if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    let stem = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
                    let key = format!("local:{stem}");
                    if self.entries.iter().any(|e| e.model_id == key) {
                        continue;
                    }
                    let size_mb = std::fs::metadata(&p)
                        .map(|m| (m.len() / (1024 * 1024)).max(1) as usize)
                        .unwrap_or(1);
                    let abs = p.canonicalize().unwrap_or(p.clone());
                    self.entries.push(ModelCatalogEntry {
                        model_id: key,
                        hf_repo_id: abs.to_string_lossy().into_owned(),
                        display_name: format!("Local Checkpoint: {}", p.file_name().unwrap_or_default().to_string_lossy()),
                        family: "safetensors".into(),
                        params_human: format!("~{size_mb} MB"),
                        num_layers: 0,
                        d_model: 0,
                        vram_fp32_mb: size_mb,
                        license_spdx: "Custom/Local".into(),
                        description: format!("Локальный чекпоинт safetensors: {}", abs.display()),
                        recommended_tasks: vec!["recon_full".into(), "apply_yaml_plan".into()],
                        is_builtin_synthetic: false,
                        local_cached: true,
                        local_path: Some(abs.to_string_lossy().into_owned()),
                    });
                }
            }
        }
    }

    pub fn list_entries(&mut self) -> Vec<ModelCatalogEntry> {
        self.refresh_cache_status();
        self.entries.clone()
    }

    pub fn get(&mut self, model_id_or_repo: &str) -> Option<ModelCatalogEntry> {
        self.refresh_cache_status();
        if let Some(e) = self.entries.iter().find(|e| e.model_id == model_id_or_repo) {
            return Some(e.clone());
        }
        self.entries
            .iter()
            .find(|e| e.hf_repo_id.to_lowercase() == model_id_or_repo.to_lowercase())
            .cloned()
    }

    pub fn to_toon(&mut self) -> String {
        let rows: Vec<Value> = self
            .list_entries()
            .iter()
            .map(|e| {
                json!({
                    "model_id": e.model_id,
                    "hf_repo_id": e.hf_repo_id,
                    "family": e.family,
                    "params": e.params_human,
                    "layers": e.num_layers,
                    "d_model": e.d_model,
                    "cached": e.local_cached,
                    "license": e.license_spdx,
                })
            })
            .collect();
        toon::dumps_toon(&json!({"model_catalog": rows}))
    }

    /// Load a model by catalog ID, HF repo ID, or local `.safetensors` path.
    pub fn load_model(&mut self, model_id_or_repo: &str) -> Result<LoadedModel> {
        let entry = self.get(model_id_or_repo);

        if let Some(e) = &entry {
            if e.is_builtin_synthetic {
                let bundle = minillama::build_corrupted_model_and_splits(2026)?;
                return Ok(LoadedModel {
                    arch: Some(Arc::new(bundle.model)),
                    tensors: bundle.tensors,
                    tokenizer: None,
                    bos_token_id: None,
                    entry: e.clone(),
                    dtypes: HashMap::new(),
                });
            }
        }

        let repo_id = entry
            .as_ref()
            .map(|e| e.hf_repo_id.clone())
            .unwrap_or_else(|| model_id_or_repo.to_string());

        // Local file path?
        let p = Path::new(&repo_id);
        if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let (tensors, dtypes) = tensor::load_safetensors_with_dtypes(p)?;
            let arch = infer_arch(&tensors, None);
            let synthetic_entry = entry.unwrap_or_else(|| ModelCatalogEntry {
                model_id: format!("local:{}", p.file_stem().unwrap_or_default().to_string_lossy()),
                hf_repo_id: repo_id.clone(),
                display_name: p.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                family: "safetensors".into(),
                params_human: "local".into(),
                num_layers: 0,
                d_model: 0,
                vram_fp32_mb: 10,
                license_spdx: "Local".into(),
                description: repo_id.clone(),
                recommended_tasks: vec![],
                is_builtin_synthetic: false,
                local_cached: true,
                local_path: Some(repo_id.clone()),
            });
            return Ok(LoadedModel { arch, tensors, tokenizer: None, bos_token_id: None, entry: synthetic_entry, dtypes });
        }

        // HuggingFace Hub download.
        self.load_hf_repo(&repo_id, entry)
    }

    fn load_hf_repo(&mut self, repo_id: &str, entry: Option<ModelCatalogEntry>) -> Result<LoadedModel> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| WeightXError::Value(format!("HF Hub init failed: {e}")))?;
        let repo = api.model(repo_id.to_string());

        let cfg_path = repo
            .get("config.json")
            .map_err(|e| WeightXError::Value(format!("Cannot fetch config.json for {repo_id}: {e}")))?;
        let cfg_value: Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
        let cfg = ModelConfig::from_hf_json(&cfg_value);

        // Weights: single file or sharded index.
        let mut tensors: Vec<(String, Tensor)> = Vec::new();
        let mut dtypes: HashMap<String, String> = HashMap::new();
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            let index: Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let mut shards: Vec<String> = index["weight_map"]
                .as_object()
                .map(|m| m.values().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            shards.sort();
            shards.dedup();
            for shard in shards {
                let sp = repo
                    .get(&shard)
                    .map_err(|e| WeightXError::Value(format!("Cannot fetch shard {shard}: {e}")))?;
                let (ts, ds) = tensor::load_safetensors_with_dtypes(&sp)?;
                tensors.extend(ts);
                dtypes.extend(ds);
            }
        } else {
            let wp = repo
                .get("model.safetensors")
                .map_err(|e| WeightXError::Value(format!("Cannot fetch model.safetensors for {repo_id}: {e}")))?;
            let (ts, ds) = tensor::load_safetensors_with_dtypes(&wp)?;
            tensors = ts;
            dtypes = ds;
        }

        // Tokenizer (optional): tokenizer.json, else GPT-2 vocab.json + merges.txt.
        let mut tokenizer = None;
        let mut bos_token_id = None;
        if let Ok(tp) = repo.get("tokenizer.json") {
            tokenizer = tokenizers::Tokenizer::from_file(tp).ok();
        }
        if tokenizer.is_none() {
            if let (Ok(vp), Ok(mp)) = (repo.get("vocab.json"), repo.get("merges.txt")) {
                tokenizer = build_gpt2_tokenizer(&vp, &mp).ok();
            }
        }
        if let Ok(tcp) = repo.get("tokenizer_config.json") {
            if let Ok(v) = serde_json::from_str::<Value>(&std::fs::read_to_string(tcp).unwrap_or_default()) {
                bos_token_id = v.get("bos_token_id").and_then(|x| x.as_i64());
            }
        }

        let arch = infer_arch(&tensors, Some(&cfg_value));
        let total: usize = tensors.iter().map(|(_, t)| t.len()).sum();
        let mut entry = entry.unwrap_or_else(|| ModelCatalogEntry {
            model_id: repo_id.to_string(),
            hf_repo_id: repo_id.to_string(),
            display_name: repo_id.to_string(),
            family: cfg.model_type.clone(),
            params_human: if total >= 1_000_000 {
                format!("{:.1}M", total as f64 / 1e6)
            } else {
                format!("{:.1}K", total as f64 / 1e3)
            },
            num_layers: cfg.num_hidden_layers,
            d_model: cfg.hidden_size,
            vram_fp32_mb: (total * 4 / (1024 * 1024)).max(1),
            license_spdx: "HuggingFace".into(),
            description: format!("HuggingFace model {repo_id}"),
            recommended_tasks: vec![],
            is_builtin_synthetic: false,
            local_cached: true,
            local_path: None,
        });
        entry.local_cached = true;
        // Register the loaded repo so it appears in the catalog / «Скаченные».
        if let Some(existing) = self.entries.iter_mut().find(|e| e.model_id == entry.model_id) {
            *existing = entry.clone();
        } else {
            self.entries.push(entry.clone());
        }
        Ok(LoadedModel { arch, tensors, tokenizer, bos_token_id, entry, dtypes })
    }
}

/// Real host resources (RAM + free disk on the HF cache drive) so the catalog
/// can auto-select models that actually fit this computer.
pub fn system_resources() -> Value {
    let cache = hub_cache_dir();
    let cache_str = cache.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        #[repr(C)]
        struct MemoryStatusEx {
            dw_length: u32,
            dw_memory_load: u32,
            ull_total_phys: u64,
            ull_avail_phys: u64,
            ull_total_page_file: u64,
            ull_avail_page_file: u64,
            ull_total_virtual: u64,
            ull_avail_virtual: u64,
            ull_avail_extended_virtual: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(lp: *mut MemoryStatusEx) -> i32;
            fn GetDiskFreeSpaceExW(
                dir: *const u16,
                free_bytes_avail: *mut u64,
                total_bytes: *mut u64,
                total_free: *mut u64,
            ) -> i32;
        }
        let mut mem = MemoryStatusEx {
            dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };
        let (ram_total, ram_avail) = if unsafe { GlobalMemoryStatusEx(&mut mem) } != 0 {
            (mem.ull_total_phys, mem.ull_avail_phys)
        } else {
            (0, 0)
        };
        // Anchor the disk query at an existing ancestor of the cache dir.
        let mut anchor = cache.clone();
        while !anchor.exists() {
            match anchor.parent() {
                Some(p) => anchor = p.to_path_buf(),
                None => break,
            }
        }
        let wide: Vec<u16> = anchor.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let mut disk_free_avail: u64 = 0;
        let mut disk_total: u64 = 0;
        let mut disk_free: u64 = 0;
        let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut disk_free_avail, &mut disk_total, &mut disk_free) };
        let (disk_free, disk_total) = if ok != 0 { (disk_free_avail, disk_total) } else { (0, 0) };
        return json!({
            "ram_total": ram_total, "ram_avail": ram_avail,
            "disk_free": disk_free, "disk_total": disk_total,
            "cache_dir": cache_str,
        });
    }
    #[cfg(not(windows))]
    {
        json!({ "ram_total": 0, "ram_avail": 0, "disk_free": 0, "disk_total": 0, "cache_dir": cache_str })
    }
}

/// Search/browse the Hugging Face Hub models API (native, no webview CORS limits).
/// Real API filters (task/license), sort, cursor pagination and card metadata
/// (thumbnail, safetensors param counts). `max_params` > 0 silently drops models
/// that cannot fit the host (unknown-size models kept only for explicit searches).
/// Returns { models, next_cursor }.
pub fn hf_search(query: &str, limit: usize, cursor: &str, task: &str, license: &str, sort: &str, max_params: i64) -> Result<Value> {
    let fetch_limit = if max_params > 0 { 100 } else { limit.clamp(1, 100) };
    let mut url = format!(
        "https://huggingface.co/api/models?sort={}&limit={}&expand=cardData&expand=safetensors&expand=lastModified&expand=tags&expand=pipeline_tag&expand=downloads&expand=likes",
        urlencode(if sort.is_empty() { "downloads" } else { sort }),
        fetch_limit
    );
    if !query.trim().is_empty() {
        url.push_str(&format!("&search={}", urlencode(query.trim())));
    }
    if !task.trim().is_empty() {
        url.push_str(&format!("&filter={}", urlencode(task.trim())));
    }
    if !license.trim().is_empty() {
        url.push_str(&format!("&filter={}", urlencode(&format!("license:{}", license.trim()))));
    }
    if !cursor.trim().is_empty() {
        // The cursor comes from the API's Link header and is ALREADY percent-encoded;
        // re-encoding it would double-escape %XX sequences and HF replies 400.
        url.push_str(&format!("&cursor={}", cursor.trim()));
    }
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(20))
        .build();
    let resp = agent
        .get(&url)
        .set("User-Agent", "WeightX/0.4.0 (desktop catalog)")
        .call()
        .map_err(|e| WeightXError::Value(format!("HF search failed: {e}")))?;
    // Cursor pagination: the API returns the next page in the Link header.
    let next_cursor = resp
        .header("Link")
        .and_then(|link| {
            let next_part = link.split(',').find(|p| p.contains("rel=\"next\""))?;
            let cstart = next_part.find("cursor=")? + "cursor=".len();
            let rest = &next_part[cstart..];
            let cend = rest.find(|c: char| c == '&' || c == '>' || c == ';').unwrap_or(rest.len());
            Some(rest[..cend].to_string())
        })
        .unwrap_or_default();
    let body = resp
        .into_string()
        .map_err(|e| WeightXError::Value(format!("HF search read failed: {e}")))?;
    let data: Value = serde_json::from_str(&body)
        .map_err(|e| WeightXError::Value(format!("HF search parse failed: {e}")))?;
    let items = data.as_array().cloned().unwrap_or_default();
    let has_query = !query.trim().is_empty();
    let task_trim = task.trim();
    let cards: Vec<Value> = items
        .iter()
        .map(|d| hf_card_json(d))
        .filter(|c| {
            // The API `filter` matches tags (a model may carry several task tags);
            // enforce the actual pipeline_tag so the task filter is exact.
            if !task_trim.is_empty() && c["family"].as_str() != Some(task_trim) {
                return false;
            }
            if max_params <= 0 {
                return true;
            }
            match c["params"].as_i64() {
                Some(n) => n <= max_params,
                // Unknown size: keep only when the user explicitly searches by name.
                None => has_query,
            }
        })
        .collect();
    Ok(json!({ "query": query, "models": cards, "next_cursor": next_cursor }))
}

/// Normalize one HF API model item into a catalog card.
fn hf_card_json(d: &Value) -> Value {
    let id = d["id"].as_str().or_else(|| d["modelId"].as_str()).unwrap_or("");
    let tags: Vec<String> = d["tags"]
        .as_array()
        .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let license = tags
        .iter()
        .find(|t| t.starts_with("license:"))
        .map(|t| t.trim_start_matches("license:").to_string())
        .unwrap_or_else(|| "—".into());
    let desc = tags
        .iter()
        .filter(|t| !t.starts_with("license:") && !t.starts_with("region:"))
        .take(4)
        .cloned()
        .collect::<Vec<_>>()
        .join(" · ");
    // Model card thumbnail (when the author set one); relative paths resolve
    // against the repo's main branch.
    let thumbnail = d["cardData"]["thumbnail"].as_str().map(|t| {
        if t.starts_with("http") {
            t.to_string()
        } else {
            format!("https://huggingface.co/{}/resolve/main/{}", id, t.trim_start_matches('/'))
        }
    }).unwrap_or_default();
    json!({
        "id": id,
        "family": d["pipeline_tag"].as_str().unwrap_or("—"),
        "license": license,
        "downloads": d["downloads"].as_i64().unwrap_or(0),
        "likes": d["likes"].as_i64().unwrap_or(0),
        "params": d["safetensors"]["total"].as_i64(),
        "last_modified": d["lastModified"].as_str().unwrap_or(""),
        "gated": d["gated"].as_bool().unwrap_or(false) || d["gated"].is_string(),
        "thumbnail": thumbnail,
        "desc": if desc.is_empty() { "Модель Hugging Face".to_string() } else { desc },
    })
}

fn avatar_cache() -> &'static std::sync::Mutex<std::collections::HashMap<String, String>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, String>>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Real org/user avatar from the HF organizations API (cached per namespace).
pub fn hf_org_avatar(org: &str) -> Result<String> {
    if org.is_empty() {
        return Ok(String::new());
    }
    if let Some(v) = avatar_cache().lock().unwrap().get(org) {
        return Ok(v.clone());
    }
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(10))
        .build();
    let url = format!("https://huggingface.co/api/organizations/{}/overview", urlencode(org));
    let avatar = match agent.get(&url).set("User-Agent", "WeightX/0.4.0").call() {
        Ok(resp) => resp
            .into_string()
            .ok()
            .and_then(|b| serde_json::from_str::<Value>(&b).ok())
            .and_then(|v| v["avatarUrl"].as_str().map(String::from))
            .unwrap_or_default(),
        Err(_) => String::new(),
    };
    avatar_cache().lock().unwrap().insert(org.to_string(), avatar.clone());
    Ok(avatar)
}

/// Full model card: API metadata + the repo's real README.md + org avatar.
pub fn hf_model_card(repo: &str) -> Result<Value> {
    if repo.is_empty() || repo.contains('\\') {
        return Err(WeightXError::Value("некорректный repo id".into()));
    }
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(25))
        .build();
    let info_url = format!(
        "https://huggingface.co/api/models/{}?expand=cardData&expand=safetensors&expand=lastModified",
        repo.split('/').map(urlencode).collect::<Vec<_>>().join("/")
    );
    let info: Value = agent
        .get(&info_url)
        .set("User-Agent", "WeightX/0.4.0")
        .call()
        .map_err(|e| WeightXError::Value(format!("HF model info failed: {e}")))?
        .into_string()
        .ok()
        .and_then(|b| serde_json::from_str(&b).ok())
        .ok_or_else(|| WeightXError::Value("HF model info: пустой ответ".into()))?;
    let readme_url = format!(
        "https://huggingface.co/{}/raw/main/README.md",
        repo.split('/').map(urlencode).collect::<Vec<_>>().join("/")
    );
    let mut readme = agent
        .get(&readme_url)
        .set("User-Agent", "WeightX/0.4.0")
        .call()
        .ok()
        .and_then(|r| r.into_string().ok())
        .unwrap_or_default();
    if readme.len() > 150_000 {
        readme.truncate(150_000);
        readme.push_str("\n\n…(описание обрезано)");
    }
    let org = repo.split('/').next().unwrap_or("");
    let avatar = hf_org_avatar(org).unwrap_or_default();
    let mut card = hf_card_json(&info);
    card["tags"] = info["tags"].clone();
    card["card_data"] = info["cardData"].clone();
    card["pipeline_tag"] = info["pipeline_tag"].clone();
    Ok(json!({ "model": card, "readme": readme, "avatar": avatar }))
}

/// Minimal percent-encoding for URL query values.
fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Real HF hub cache location (honors HF_HUB_CACHE / HF_HOME like hf-hub does).
fn hub_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        if !home.is_empty() {
            return PathBuf::from(home).join("hub");
        }
    }
    dirs::home_dir()
        .map(|h| h.join(".cache").join("huggingface").join("hub"))
        .unwrap_or_else(|| PathBuf::from(".cache/huggingface/hub"))
}

/// First existing subdirectory (e.g. a snapshot folder), if any.
fn first_subdir(dir: &Path) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    for f in rd.flatten() {
        if f.path().is_dir() {
            return Some(f.path());
        }
    }
    None
}

/// Build a GPT-2 BPE tokenizer from `vocab.json` + `merges.txt` (ByteLevel).
fn build_gpt2_tokenizer(vocab: &Path, merges: &Path) -> Result<tokenizers::Tokenizer> {
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    let bpe = BPE::from_file(vocab.to_str().unwrap_or_default(), merges.to_str().unwrap_or_default())
        .build()
        .map_err(|e| WeightXError::Value(format!("BPE init failed: {e}")))?;
    let mut tok = tokenizers::Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::default()));
    tok.with_decoder(Some(ByteLevel::default()));
    tok.add_special_tokens(&[tokenizers::AddedToken::from("<|endoftext|>", true)]);
    Ok(tok)
}

/// Infer the runtime architecture from tensor names (+ optional HF config).
/// In hybrid mode (`WEIGHTX_BACKEND`), large models are wrapped into the
/// torch sidecar backend; small/demo models keep the native Rust core.
pub fn infer_arch(tensors: &[(String, Tensor)], cfg_value: Option<&Value>) -> Option<Arc<dyn ModelArch>> {
    let has = |suffix: &str| tensors.iter().any(|(n, _)| n.ends_with(suffix));
    let default_cfg = Value::Null;
    let cv = cfg_value.unwrap_or(&default_cfg);
    let mut cfg = ModelConfig::from_hf_json(cv);
    let total_params: usize = tensors.iter().map(|(_, t)| t.len()).sum();
    let native: Option<Arc<dyn ModelArch>> = if has("transformer.h.0.attn.c_attn.weight") {
        cfg.model_type = "gpt2".into();
        Some(Arc::new(Gpt2Model::new(cfg)))
    } else if has("model.layers.0.self_attn.q_proj.weight") || has("model.layers.0.mlp.down_proj.weight") {
        Some(Arc::new(LlamaModel::new(cfg)))
    } else {
        None
    };
    native.map(|a| crate::torch_backend::maybe_wrap_torch(a, total_params, cfg_value.cloned()))
}

/// Tokenize texts with BOS prefix (port of `improver._tokenize_texts`).
pub fn tokenize_texts(tokenizer: &tokenizers::Tokenizer, bos: Option<i64>, texts: &[&str]) -> Result<Vec<Vec<i64>>> {
    let mut seqs = Vec::new();
    for t in texts {
        let enc = tokenizer
            .encode(*t, false)
            .map_err(|e| WeightXError::Value(format!("Tokenization failed: {e}")))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
        if let Some(b) = bos {
            ids.insert(0, b);
        }
        seqs.push(ids);
    }
    Ok(seqs)
}