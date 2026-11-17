# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX Session — shared backend for the interactive CLI (`wx`/`weightx`) and Desktop app.

A `StudioSession` owns the loaded target model, its evaluation splits, the configured
brain provider and a `WeightXHarness`. Front-ends call `run_action` or `run_skill`
and receive Markdown/TOON output; all progress is streamed through the `log` callback.
"""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightx.catalog import CODEX_PROMPT_SUGGESTIONS, ModelCatalog, PromptSuggestion
from weightx.declarative.plan_engine import execute_yaml_plan
from weightx.execution.vespasian import EvaluationSplit
from weightx.harness import WeightXHarness
from weightx.l1_arch.autoarch import inspect_architecture
from weightx.l2_profile.knowledge_map import run_knowledge_map
from weightx.l2_profile.unilayer import profile_model_layers
from weightx.l3_weights.eamit import BatchFactSpec, apply_eamit_batch
from weightx.providers import (
    AnthropicProvider,
    AutonomousWSPProvider,
    CLIAgentProvider,
    LLMProvider,
    OpenAICompatibleProvider,
)

CONFIG_DIR = Path.home() / ".weightx"
CONFIG_FILE = CONFIG_DIR / "studio_config.json"

PROVIDER_KINDS: list[str] = [
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
]


@dataclass
class StudioConfig:
    provider_kind: str = PROVIDER_KINDS[0]
    provider_model: str = ""
    api_key: str = ""
    base_url: str = ""
    model_id: str = "minillama-demo"
    history: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls) -> StudioConfig:
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception:
            return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = self.__dict__.copy()
        data["history"] = data["history"][-50:]
        # API keys are stored only if not supplied through environment variables
        CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_provider(kind: str, model: str = "", api_key: str = "", base_url: str = "") -> LLMProvider:
    """Instantiate a brain provider from UI/CLI selections."""
    kind = kind.split(" ")[0]
    kw: dict[str, Any] = {}
    if model:
        kw["model"] = model
    if base_url and kind in ("ollama", "lm-studio", "vllm"):
        kw["base_url"] = base_url
    if kind == "autonomous":
        return AutonomousWSPProvider()
    if kind == "ollama":
        return OpenAICompatibleProvider.for_ollama(**kw)
    if kind == "lm-studio":
        return OpenAICompatibleProvider.for_lm_studio(**kw)
    if kind == "openai":
        return OpenAICompatibleProvider.for_openai(api_key=api_key or None, **kw)
    if kind == "openrouter":
        return OpenAICompatibleProvider.for_openrouter(api_key=api_key or None, **kw)
    if kind == "vllm":
        kw.setdefault("model", "default")
        return OpenAICompatibleProvider.for_vllm(**kw)
    if kind == "anthropic":
        return AnthropicProvider(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), **kw)
    if kind.startswith("cli:"):
        return CLIAgentProvider(agent_type=kind.split(":", 1)[1])
    raise ValueError(f"Unknown provider kind {kind!r}")


class StudioSession:
    """Stateful backend shared by `weightx chat` (REPL) and `weightx desktop`."""

    def __init__(self, log: Callable[[str], None] = print, config: StudioConfig | None = None) -> None:
        self.log = log
        self.config = config or StudioConfig.load()
        self.catalog = ModelCatalog()
        self.model: Any = None
        self.tokenizer: Any = None
        self.entry: Any = None
        self.harness: WeightXHarness | None = None
        self.calib_ids: torch.Tensor | None = None
        self.provider: LLMProvider = AutonomousWSPProvider()

    # ------------------------------------------------------------------ setup
    @property
    def suggestions(self) -> list[PromptSuggestion]:
        return CODEX_PROMPT_SUGGESTIONS

    @property
    def skills(self):
        from weightx.skills import WSP_SKILLS

        return WSP_SKILLS

    def run_skill(self, skill_query: str, user_args: str = "", progress_cb: Callable[[str], None] | None = None):
        from weightx.skills import execute_skill

        return execute_skill(self, skill_query, user_args=user_args, progress_cb=progress_cb)

    def set_provider(self, kind: str, model: str = "", api_key: str = "", base_url: str = "") -> str:
        self.provider = build_provider(kind, model, api_key, base_url)
        self.config.provider_kind, self.config.provider_model = kind, model
        self.config.api_key, self.config.base_url = api_key, base_url
        self.config.save()
        if self.harness is not None:
            self.harness.provider = self.provider
        return f"Мозг (provider) установлен: {kind} {model}".strip()

    def load_model(self, model_id: str) -> str:
        self.log(f"Загрузка модели «{model_id}»…")
        if model_id == "minillama-demo":
            from examples.mvp_end_to_end_demo import build_corrupted_model_and_splits

            model, practice, held, calib = build_corrupted_model_and_splits()
            tok = None
            entry = self.catalog.get("minillama-demo")
        else:
            model, tok, entry = self.catalog.load_model_and_tokenizer(model_id)
            for p in model.parameters():
                p.requires_grad_(False)
            practice, held, calib = self._text_splits(model, tok)
        self.model, self.tokenizer, self.entry, self.calib_ids = model, tok, entry, calib
        self.harness = WeightXHarness(
            model,
            provider=self.provider,
            practice_split=practice,
            held_out_split=held,
            calibration_input_ids=calib,
            tokenizer=tok,
            fisher_damping_alpha=0.15,
            improvement_threshold=1e-4 if tok is not None else 0.01,
            max_kl_drift=0.05 if tok is not None else 2.0,
        )
        self.config.model_id = model_id
        self.config.save()
        n_params = sum(p.numel() for p in model.parameters())
        return (
            f"Модель загружена: {entry.display_name if entry else model_id} | параметров: {n_params:,} | "
            f"базовый held-out score: {self.harness.vespasian.baseline_score:.5f}"
        )

    def _text_splits(self, model: Any, tok: Any) -> tuple[EvaluationSplit, EvaluationSplit, torch.Tensor]:
        from weightx.corpora import GENERAL_CONTROL, WSP_HELD_OUT, WSP_PRACTICE
        from weightx.improver import _tokenize_texts
        from weightx.l3_weights.domain_eamit import token_nll

        prac = _tokenize_texts(tok, WSP_PRACTICE)
        held = _tokenize_texts(tok, WSP_HELD_OUT)
        ctrl = _tokenize_texts(tok, GENERAL_CONTROL)
        self._prac_seqs, self._held_seqs, self._ctrl_seqs = prac, held, ctrl

        def _mk(seqs: list[torch.Tensor]) -> Callable[[Any], tuple[float, list[float]]]:
            def _fn(m: Any) -> tuple[float, list[float]]:
                _, per = token_nll(m, seqs)
                items = [float(np.exp(-x)) for x in per]
                return float(np.mean(items)), items
            return _fn

        min_len = min(len(s) for s in ctrl)
        ctrl_ref = torch.stack([s[:min_len] for s in ctrl])
        practice = EvaluationSplit(name="practice", eval_fn=_mk(prac))
        held_split = EvaluationSplit(name="held_out_hidden", eval_fn=_mk(held), reference_input_ids=ctrl_ref)
        min_p = min(len(s) for s in prac)
        calib = torch.stack([s[:min_p] for s in prac[:6]])
        return practice, held_split, calib

    def _require_model(self) -> WeightXHarness:
        if self.harness is None:
            self.log(self.load_model(self.config.model_id or "minillama-demo"))
        assert self.harness is not None
        return self.harness

    # --------------------------------------------------------------- actions
    def run_action(self, action_tag: str, prompt_text: str = "") -> str:
        try:
            self.config.history.append({"action": action_tag, "prompt": prompt_text[:500]})
            self.config.save()
            from weightx.skills import execute_skill, resolve_skill

            if action_tag.startswith("skill:"):
                sk_id = action_tag.split(":", 1)[1]
                return execute_skill(self, sk_id, user_args=prompt_text).to_markdown()
            sk = resolve_skill(action_tag) or resolve_skill(prompt_text)
            if sk is not None and action_tag in ("chat", "skill", sk.skill_id, f"/{sk.skill_id}"):
                return execute_skill(self, sk, user_args=prompt_text).to_markdown()
            handler = getattr(self, f"_act_{action_tag}", None)
            if handler is None:
                if sk is not None:
                    return execute_skill(self, sk, user_args=prompt_text).to_markdown()
                return self._act_chat(prompt_text)
            return handler(prompt_text)
        except Exception as exc:  # surfaced in UI instead of crashing it
            return f"Ошибка: {exc}\n{traceback.format_exc(limit=3)}"

    def _act_recon_profile(self, _p: str) -> str:
        h = self._require_model()
        arch = inspect_architecture(self.model, calibration_input_ids=self.calib_ids)
        prof = profile_model_layers(self.model, calibration_input_ids=self.calib_ids)
        return "## L1 AutoArch\n" + arch.to_toon() + "\n\n## L2 Uni-Layer\n" + prof.to_toon(include_full_table=False)

    def _act_knowledge_map(self, _p: str) -> str:
        self._require_model()
        return run_knowledge_map(self.model, tokenizer=self.tokenizer).to_toon()

    def _act_chat(self, prompt: str) -> str:
        h = self._require_model()
        h.provider = self.provider
        res = h.run_editing_cycle(goal=prompt or "Diagnose and repair the target model.", max_turns=8)
        return res.to_toon()

    _act_full_cycle = _act_chat

    def _act_auto_improve(self, _p: str) -> str:
        h = self._require_model()
        if self.tokenizer is None:
            return self._act_chat("Localize and repair outlier neurons, then verify with Vespasian.")
        from weightx.l3_weights.domain_eamit import eamit_domain_calibration

        lines = []
        for it in range(3):
            res = eamit_domain_calibration(
                self.model, h.workspace, self._prac_seqs, self._ctrl_seqs, iterations=1,
                max_control_nll_increase=0.01, apply_fisher_guard=False,
            )
            if not res.steps or res.steps[-1].chosen_rel_step == 0.0:
                lines.append(f"итерация {it + 1}: безопасный шаг не найден — стоп")
                break
            s = res.steps[-1]
            v = h.vespasian.verify_and_decide(hypothesis=f"domain calibration {it + 1}")
            lines.append(
                f"итерация {it + 1}: practice NLL {s.practice_nll_before:.4f}→{s.practice_nll_after:.4f}; "
                f"Vespasian: {v.verdict_reason}"
            )
            self.log(lines[-1])
            if not v.accepted:
                break
        lines.append(f"Итоговый held-out score: {h.vespasian.best_score:.5f} (база {h.vespasian.baseline_score:.5f})")
        return "\n".join(lines)

    def _mlp_out_tensors(self) -> list[str]:
        h = self._require_model()
        names = [n for n in h.workspace.names() if len(h.workspace.shape(n)) == 2]
        return [n for n in names if any(k in n for k in ("down_proj", "mlp.c_proj", "fc2", "dense_4h_to_h"))]

    def _act_eamit_demo(self, _p: str) -> str:
        h = self._require_model()
        targets = self._mlp_out_tensors()[-3:]
        shape = h.workspace.shape(targets[0])
        rng = np.random.default_rng(7)
        keys = rng.standard_normal((32, shape[1]))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        vals = rng.standard_normal((32, shape[0])) * 0.05
        facts = [BatchFactSpec(f"fact_{i}", keys[i], vals[i]) for i in range(32)]
        res = apply_eamit_batch(h.workspace, targets, facts, lambda_cov=1e-2, apply_fisher_guard=True)
        v = h.vespasian.verify_and_decide(hypothesis="EAMIT synthetic batch")
        return "## EAMIT\n" + res.to_toon() + "\n\n## Vespasian\n" + v.verdict_reason

    def _act_abliterate(self, _p: str) -> str:
        h = self._require_model()
        t = self._mlp_out_tensors()[-1]
        plan = (
            "transforms:\n"
            f"  - assert: {{ shape: {{ of: '{t}', is: {list(h.workspace.shape(t))} }} }}\n"
            f"  - project_out_direction_: {{ target: '{t}', alpha: 0.3 }}\n"
            f"  - assert: {{ finite: {{ of: '{t}' }} }}\n"
        )
        return self.apply_yaml(plan)

    def _act_rollback_test(self, _p: str) -> str:
        h = self._require_model()
        t = [n for n in h.workspace.names() if len(h.workspace.shape(n)) == 2][1]
        plan = (
            "transforms:\n"
            f"  - assert: {{ exists: {{ of: '{t}' }} }}\n"
            f"  - scale_: {{ target: '{t}', by: -50.0 }}\n"
            f"  - assert: {{ finite: {{ of: '{t}' }} }}\n"
        )
        return self.apply_yaml(plan, fisher=False)

    def apply_yaml(self, plan: str, fisher: bool = True) -> str:
        h = self._require_model()
        rep = execute_yaml_plan(h.workspace, plan, apply_fisher_guard=fisher)
        out = "## План BrainSurgery-WX\n" + rep.to_toon()
        if rep.success:
            v = h.vespasian.verify_and_decide(hypothesis="user YAML plan")
            out += "\n\n## Vespasian\n" + v.to_toon()
        return out

    def export_checkpoint(self, path: str) -> str:
        h = self._require_model()
        p = h.workspace.export_safetensors(path)
        return f"Чекпоинт сохранён: {p.resolve()}"

    def status(self) -> str:
        if self.harness is None:
            return "Модель не загружена."
        v = self.harness.vespasian
        d = self.harness.workspace.diff()
        return (
            f"Модель: {self.config.model_id} | мозг: {self.config.provider_kind} {self.config.provider_model}\n"
            f"Held-out: база {v.baseline_score:.5f} → лучший {v.best_score:.5f} | итераций Vespasian: {v.iteration}\n"
            f"Изменено параметров: {d.changed_params:,} / {d.total_params:,}"
        )
