# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX Model Catalog & Codex-Style Prompt Suggestions Library.

Provides:
  1. `ModelCatalog`: Built-in catalog of open-weight small/medium LLMs (0.1M - 1.5B+),
     automatic local `.safetensors` & HuggingFace cache discovery, and unified model loading
     (`AutoModelForCausalLM` + `AutoTokenizer` or standalone `.safetensors` checkpoints).
  2. `CODEX_PROMPT_SUGGESTIONS`: Curated ready-to-run Weight-Space Programming prompt templates
     inspired by OpenAI Codex / Claude Code starter cards for both CLI and Desktop UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from weightx.toon_io import dumps_toon


@dataclass
class ModelCatalogEntry:
    model_id: str
    hf_repo_id: str
    display_name: str
    family: str
    params_human: str
    num_layers: int
    d_model: int
    vram_fp32_mb: int
    license_spdx: str
    description: str
    recommended_tasks: list[str] = field(default_factory=list)
    is_builtin_synthetic: bool = False
    local_cached: bool = False
    local_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptSuggestion:
    prompt_id: str
    category: str
    icon: str
    title: str
    short_hint: str
    prompt_text: str
    target_tools: list[str]
    action_tag: str = "chat"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# DSH-Style Ultra-Compact 2-Word Skill Pills (for WeightX Desktop & CLI)
# ---------------------------------------------------------------------------
CODEX_PROMPT_SUGGESTIONS: list[PromptSuggestion] = [
    PromptSuggestion(
        prompt_id="coding-boost",
        category="Скилл /skill coding-boost",
        icon="⚡",
        title="Улучшить в кодинге",
        short_hint="Пайплайн 5 шагов: L1/L2 -> фокусировка внимания -> Code MEMIT -> IWCKEdit -> Vespasian",
        prompt_text="/skill coding-boost",
        target_tools=[
            "l1_autoarch_inspect",
            "l2_unilayer_profile",
            "l3_apply_yaml_plan",
            "l3_verify_and_commit",
        ],
        action_tag="skill:coding-boost",
    ),
    PromptSuggestion(
        prompt_id="memory-slim",
        category="Скилл /skill memory-slim",
        icon="🪶",
        title="Уменьшить память",
        short_hint="Пайплайн 4 шага: спектральный аудит ранга -> svd_low_rank_ -> ties_trim_ -> Vespasian",
        prompt_text="/skill memory-slim",
        target_tools=[
            "l2_unilayer_profile",
            "l3_apply_yaml_plan",
            "l3_verify_and_commit",
        ],
        action_tag="skill:memory-slim",
    ),
    PromptSuggestion(
        prompt_id="outlier-repair",
        category="Скилл /skill outlier-repair",
        icon="🛠",
        title="Починить выбросы",
        short_hint="Пайплайн 4 шага: поиск kurtosis-выбросов -> rescale_outlier_rows_ -> IWCKEdit -> Vespasian",
        prompt_text="/skill outlier-repair",
        target_tools=[
            "l2_unilayer_profile",
            "l3_describe_tensor",
            "l3_apply_yaml_plan",
            "l3_verify_and_commit",
        ],
        action_tag="skill:outlier-repair",
    ),
    PromptSuggestion(
        prompt_id="spectral-denoise",
        category="Скилл /skill spectral-denoise",
        icon="📉",
        title="Снизить перплексию",
        short_hint="Пайплайн 5 шагов: аудит анизотропии -> Closed-Form Domain MEMIT -> контроль KL -> Vespasian",
        prompt_text="/skill spectral-denoise",
        target_tools=["l3_describe_tensor", "l3_apply_yaml_plan", "l3_verify_and_commit"],
        action_tag="skill:spectral-denoise",
    ),
    PromptSuggestion(
        prompt_id="memit-facts",
        category="Скилл /skill memit-facts",
        icon="🧠",
        title="Записать факты",
        short_hint="Пайплайн 4 шага: локализация KCI -> оценка ковариации C0^-1 -> MEMIT-MASS -> Vespasian",
        prompt_text="/skill memit-facts",
        target_tools=["l3_batch_edit_memit", "l3_verify_and_commit"],
        action_tag="skill:memit-facts",
    ),
    PromptSuggestion(
        prompt_id="knowledge-map",
        category="Скилл /skill knowledge-map",
        icon="🗺",
        title="Карта знаний",
        short_hint="Пайплайн 3 шага: нормализованный Logit Lens -> индекс концентрации Джини KCI",
        prompt_text="/skill knowledge-map",
        target_tools=["l2_knowledge_map"],
        action_tag="skill:knowledge-map",
    ),
    PromptSuggestion(
        prompt_id="abliterate",
        category="Скилл /skill abliterate",
        icon="🧭",
        title="Вырезать отказы",
        short_hint="Пайплайн 4 шага: SVD-выделение направления -> project_out_direction_ -> Vespasian",
        prompt_text="/skill abliterate",
        target_tools=["l3_apply_yaml_plan", "l3_verify_and_commit"],
        action_tag="skill:abliterate",
    ),
    PromptSuggestion(
        prompt_id="vespasian-guard",
        category="Скилл /skill vespasian-guard",
        icon="🛡",
        title="Проверить откат",
        short_hint="Пайплайн 3 шага: снимок весов -> стресс-атака scale_: -50 -> побитовый авто-откат Vespasian",
        prompt_text="/skill vespasian-guard",
        target_tools=["l3_apply_yaml_plan", "l3_verify_and_commit"],
        action_tag="skill:vespasian-guard",
    ),
]
WSP_SKILL_SUGGESTIONS: list[PromptSuggestion] = CODEX_PROMPT_SUGGESTIONS


# ---------------------------------------------------------------------------
# Built-In Curated Model Catalog
# ---------------------------------------------------------------------------
DEFAULT_CATALOG_ENTRIES: list[ModelCatalogEntry] = [
    ModelCatalogEntry(
        model_id="minillama-demo",
        hf_repo_id="builtin/MiniLlama-4L-64D",
        display_name="MiniLlama-4L (Встроенный стенд с инъекцией выбросов)",
        family="llama",
        params_human="215K",
        num_layers=4,
        d_model=64,
        vram_fp32_mb=4,
        license_spdx="GPL-3.0-or-later",
        description="Быстрая встроенная модель семейства Llama (RMSNorm + SwiGLU + GQA/MHA) для мгновенных локальных экспериментов и отладки YAML-планов.",
        recommended_tasks=["outlier_repair", "memit_batch", "vespasian_rollback"],
        is_builtin_synthetic=True,
        local_cached=True,
    ),
    ModelCatalogEntry(
        model_id="tiny-gpt2",
        hf_repo_id="sshleifer/tiny-gpt2",
        display_name="Tiny-GPT2 (HuggingFace Pretrained Causal LM)",
        family="gpt2",
        params_human="103K",
        num_layers=2,
        d_model=2,
        vram_fp32_mb=5,
        license_spdx="MIT",
        description="Ультракомпактная предобученная модель GPT-2 из HuggingFace Hub с полноценным BPE-токенизатором для мгновенного тестирования реального пайплайна.",
        recommended_tasks=["spectral_denoise", "outlier_repair", "logit_lens"],
    ),
    ModelCatalogEntry(
        model_id="smollm2-135m",
        hf_repo_id="HuggingFaceTB/SmolLM2-135M",
        display_name="SmolLM2-135M Base (HuggingFaceTB) — рекомендуемая цель улучшения",
        family="llama",
        params_human="135M",
        num_layers=30,
        d_model=576,
        vram_fp32_mb=540,
        license_spdx="Apache-2.0",
        description="Базовая (pretrained) SmolLM2-135M. Используется в бенчмарке WeightX: closed-form MEMIT доменная калибровка с проверкой на скрытом held-out.",
        recommended_tasks=["domain_calibration", "unilayer_13_metrics", "knowledge_map"],
    ),
    ModelCatalogEntry(
        model_id="smollm2-135m-instruct",
        hf_repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        display_name="SmolLM2-135M-Instruct (HuggingFaceTB)",
        family="llama",
        params_human="135M",
        num_layers=30,
        d_model=576,
        vram_fp32_mb=540,
        license_spdx="Apache-2.0",
        description="Современная компактная Llama-совместимая модель (30 слоев, GQA, SwiGLU). Идеальна для исследования глубоких бутылочных слоев и MEMIT-MASS на CPU/GPU.",
        recommended_tasks=["unilayer_13_metrics", "knowledge_map", "memit_batch", "spectral_denoise"],
    ),
    ModelCatalogEntry(
        model_id="smollm2-360m",
        hf_repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        display_name="SmolLM2-360M-Instruct (HuggingFaceTB)",
        family="llama",
        params_human="360M",
        num_layers=32,
        d_model=960,
        vram_fp32_mb=1440,
        license_spdx="Apache-2.0",
        description="Сбалансированная модель 360M параметров с выраженной специализацией средних и поздних слоев.",
        recommended_tasks=["knowledge_map", "abliteration", "memit_batch"],
    ),
    ModelCatalogEntry(
        model_id="gpt2-124m",
        hf_repo_id="openai-community/gpt2",
        display_name="GPT-2 Base (124M, классический стенд ROME/MEMIT)",
        family="gpt2",
        params_human="124M",
        num_layers=12,
        d_model=768,
        vram_fp32_mb=500,
        license_spdx="MIT",
        description="Классическая 12-слойная трансформерная модель, используемая в оригинальных статьях ROME, MEMIT и Logit Lens.",
        recommended_tasks=["logit_lens", "memit_batch", "outlier_repair"],
    ),
    ModelCatalogEntry(
        model_id="qwen2.5-0.5b",
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        display_name="Qwen2.5-0.5B-Instruct (24 слоя, сильная мультиязычность)",
        family="qwen2",
        params_human="494M",
        num_layers=24,
        d_model=896,
        vram_fp32_mb=1980,
        license_spdx="Apache-2.0",
        description="Высококачественная модель 0.5B параметров (поддерживает русский и английский языки), отличная цель для точечной хирургии весов.",
        recommended_tasks=["unilayer_13_metrics", "knowledge_map", "memit_batch", "abliteration"],
    ),
    ModelCatalogEntry(
        model_id="tinyllama-1.1b",
        hf_repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        display_name="TinyLlama-1.1B-Chat-v1.0 (22 слоя, архитектура Llama-2)",
        family="llama",
        params_human="1.1B",
        num_layers=22,
        d_model=2048,
        vram_fp32_mb=4400,
        license_spdx="Apache-2.0",
        description="Полноценная 1.1B модель архитектуры Llama-2 для масштабирования YAML-планов BrainSurgery-WX.",
        recommended_tasks=["unilayer_13_metrics", "memit_batch", "spectral_denoise"],
    ),
]


class ModelCatalog:
    """Registry and loader for built-in, HuggingFace, and local `.safetensors` models."""

    def __init__(self, extra_search_dirs: list[Path | str] | None = None) -> None:
        self._entries: dict[str, ModelCatalogEntry] = {
            e.model_id: ModelCatalogEntry(**e.to_dict()) for e in DEFAULT_CATALOG_ENTRIES
        }
        self._search_dirs: list[Path] = [Path.cwd(), Path.cwd() / "models", Path.cwd() / "checkpoints"]
        if extra_search_dirs:
            self._search_dirs.extend(Path(p) for p in extra_search_dirs)
        self.refresh_cache_status()

    def refresh_cache_status(self) -> None:
        """Check HuggingFace Hub cache and local workspace directories for downloaded models."""
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        for entry in self._entries.values():
            if entry.is_builtin_synthetic:
                entry.local_cached = True
                continue
            repo_folder = "models--" + entry.hf_repo_id.replace("/", "--")
            candidate = hf_cache / repo_folder
            if candidate.exists():
                entry.local_cached = True
                entry.local_path = str(candidate)

        # Also scan local directories for .safetensors files
        for directory in self._search_dirs:
            if not directory.exists():
                continue
            for sf in directory.glob("*.safetensors"):
                key = f"local:{sf.stem}"
                if key not in self._entries:
                    size_mb = max(1, int(sf.stat().st_size / (1024 * 1024)))
                    self._entries[key] = ModelCatalogEntry(
                        model_id=key,
                        hf_repo_id=str(sf.resolve()),
                        display_name=f"Local Checkpoint: {sf.name}",
                        family="safetensors",
                        params_human=f"~{size_mb} MB",
                        num_layers=0,
                        d_model=0,
                        vram_fp32_mb=size_mb,
                        license_spdx="Custom/Local",
                        description=f"Локальный чекпоинт safetensors: {sf.resolve()}",
                        recommended_tasks=["recon_full", "apply_yaml_plan"],
                        local_cached=True,
                        local_path=str(sf.resolve()),
                    )

    def list_entries(self) -> list[ModelCatalogEntry]:
        self.refresh_cache_status()
        return list(self._entries.values())

    def get(self, model_id_or_repo: str) -> ModelCatalogEntry | None:
        self.refresh_cache_status()
        if model_id_or_repo in self._entries:
            return self._entries[model_id_or_repo]
        for entry in self._entries.values():
            if entry.hf_repo_id.lower() == model_id_or_repo.lower():
                return entry
        return None

    def to_toon(self) -> str:
        rows = [
            {
                "model_id": e.model_id,
                "hf_repo_id": e.hf_repo_id,
                "family": e.family,
                "params": e.params_human,
                "layers": e.num_layers,
                "d_model": e.d_model,
                "cached": e.local_cached,
                "license": e.license_spdx,
            }
            for e in self.list_entries()
        ]
        return dumps_toon({"model_catalog": rows})

    def load_model_and_tokenizer(
        self,
        model_id_or_repo: str = "minillama-demo",
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        corrupt_demo: bool = True,
    ) -> tuple[nn.Module, Any | None, ModelCatalogEntry]:
        """
        Load a model from the catalog (or arbitrary HuggingFace repo ID / local `.safetensors` path).
        Returns `(model, tokenizer_or_none, catalog_entry)`.
        """
        entry = self.get(model_id_or_repo)
        if entry is not None and entry.is_builtin_synthetic:
            from examples.mvp_end_to_end_demo import MiniLlamaForCausalLM, build_corrupted_model_and_splits

            if corrupt_demo:
                model, _, _, _ = build_corrupted_model_and_splits()
            else:
                model = MiniLlamaForCausalLM()
                model.eval()
            return model.to(device=device, dtype=dtype), None, entry

        repo_id = entry.hf_repo_id if entry is not None else model_id_or_repo
        p = Path(repo_id)
        if p.exists() and p.is_file() and p.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(p))

            class _SafetensorsModule(nn.Module):
                def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
                    super().__init__()
                    self._tensors = {k: v.to(device=device) for k, v in tensors.items()}

                def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:  # type: ignore[override]
                    return self._tensors

            synthetic_entry = entry or ModelCatalogEntry(
                model_id=f"local:{p.stem}",
                hf_repo_id=str(p.resolve()),
                display_name=p.name,
                family="safetensors",
                params_human="local",
                num_layers=0,
                d_model=0,
                vram_fp32_mb=10,
                license_spdx="Local",
                description=str(p.resolve()),
                local_cached=True,
                local_path=str(p.resolve()),
            )
            return _SafetensorsModule(state), None, synthetic_entry

        # Load via HuggingFace transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype)
        model.to(device)
        model.eval()

        if entry is None:
            cfg = getattr(model, "config", None)
            n_params = sum(p.numel() for p in model.parameters())
            entry = ModelCatalogEntry(
                model_id=model_id_or_repo,
                hf_repo_id=repo_id,
                display_name=repo_id,
                family=getattr(cfg, "model_type", "unknown"),
                params_human=f"{n_params / 1e6:.1f}M" if n_params >= 1e6 else f"{n_params / 1e3:.1f}K",
                num_layers=int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))),
                d_model=int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0))),
                vram_fp32_mb=max(1, int(n_params * 4 / (1024 * 1024))),
                license_spdx="HuggingFace",
                description=f"HuggingFace model {repo_id}",
                local_cached=True,
            )
        else:
            entry.local_cached = True

        return model, tokenizer, entry
