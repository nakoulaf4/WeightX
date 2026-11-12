# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX CLI (`wx` / `weightx`).

Case-insensitive CLI & WSP Skill runner:
  - `wx` / `weightx` (any case: `wx`, `WX`, `Wx`, `weightx`, `WeightX`, `WEIGHTX`)
    launches the interactive DSH-styled WeightX TUI.
  - `wx desktop` / `wx gui`           : Launch WeightX Desktop app.
  - `wx skills` / `wx prompts`        : List the 8 compact 2-word WSP skills and pipelines.
  - `wx skill <name>` / `wx /<skill>` : Execute a multi-stage WSP skill pipeline directly.
  - `wx catalog`                      : List all models in the WeightX Model Catalog (TOON).
  - `wx improve [model_id]`           : Run closed-form WeightX improvement pipeline on a small LLM.
  - `wx demo`                         : Run the full 5-stage WeightX MVP cycle on a causal Transformer LM.
  - `wx agents` / `wx providers`      : List detected CLI agents and supported LLM providers.
  - `wx tools [--toon]`               : Print JSON Schemas or TOON table of all isolated WeightX tools.
  - `wx recon <path>` / `inspect`     : Run L1 AutoArch on a `.safetensors` checkpoint.
  - `wx apply-plan <ckpt> <plan>`     : Apply a BrainSurgery-WX YAML plan.
  - `wx install-desktop`              : Install WeightX Desktop shortcut (`WeightX.lnk`) & CLI (`wx`/`weightx`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch.nn as nn
from safetensors.torch import load_file

from weightx.catalog import CODEX_PROMPT_SUGGESTIONS, ModelCatalog
from weightx.declarative.plan_engine import execute_yaml_plan
from weightx.l1_arch.autoarch import inspect_architecture
from weightx.l3_weights.workspace import WeightWorkspace
from weightx.providers.cli_agent import CLIAgentProvider
from weightx.skills import WSP_SKILLS, resolve_skill
from weightx.tools.registry import build_default_tool_registry
from weightx.toon_io import dumps_toon

KNOWN_SUBCOMMANDS = {
    "agents",
    "providers",
    "catalog",
    "prompts",
    "skills",
    "skill",
    "tools",
    "recon",
    "inspect",
    "apply-plan",
    "improve",
    "chat",
    "repl",
    "desktop",
    "gui",
    "install-desktop",
    "demo",
}


def cmd_agents() -> int:
    detected = CLIAgentProvider.detect_installed_agents()
    info = {
        "api_providers": ["openai", "openrouter", "vllm", "ollama", "lm-studio", "anthropic", "custom"],
        "cli_agents": [
            {"agent": k, "installed": bool(v), "binary_path": v or "not_found"}
            for k, v in detected.items()
        ],
    }
    print(dumps_toon(info))
    return 0


def cmd_catalog() -> int:
    cat = ModelCatalog()
    print(cat.to_toon())
    return 0


def cmd_prompts() -> int:
    rows = [
        {
            "idx": i + 1,
            "skill": f"/{sk.skill_id}",
            "pill": sk.pill_label,
            "pipeline": sk.pipeline_title,
            "stages": len(sk.stage_names),
        }
        for i, sk in enumerate(WSP_SKILLS)
    ]
    print(dumps_toon({"weightx_skills": rows, "codex_prompt_suggestions": len(CODEX_PROMPT_SUGGESTIONS)}))
    return 0


def cmd_run_skill(skill_query: str, model_id: str = "minillama-demo") -> int:
    from weightx.studio import StudioSession

    sk = resolve_skill(skill_query)
    if sk is None:
        print(f"Неизвестный навык: {skill_query!r}. Доступные навыки: {[s.skill_id for s in WSP_SKILLS]}", file=sys.stderr)
        return 1
    sess = StudioSession()
    sess.load_model(model_id)
    res = sess.run_skill(sk.skill_id)
    print(res.to_markdown())
    return 0


def cmd_tools(use_toon: bool = False) -> int:
    dummy = nn.Linear(4, 4, bias=False)
    ws = WeightWorkspace(dummy)
    reg = build_default_tool_registry(ws)
    if use_toon:
        print(reg.to_prompt_catalog())
    else:
        print(json.dumps(reg.to_openai_tools(), indent=2, ensure_ascii=False))
    return 0


def cmd_recon(path_str: str) -> int:
    p = Path(path_str)
    if not p.exists():
        print(f"Error: checkpoint path not found: {p}", file=sys.stderr)
        return 1
    rep = inspect_architecture(p)
    print(rep.to_toon(include_tensor_table=True))
    return 0


def cmd_apply_plan(model_path_str: str, plan_path_str: str, out_path_str: str | None) -> int:
    mp = Path(model_path_str)
    pp = Path(plan_path_str)
    if not mp.exists() or not pp.exists():
        print(f"Error: checkpoint ({mp}) or plan ({pp}) not found.", file=sys.stderr)
        return 1
    tensors = load_file(str(mp))
    ws = WeightWorkspace(tensors)
    plan_res = execute_yaml_plan(ws, pp.read_text(encoding="utf-8"), apply_fisher_guard=False)
    print(plan_res.to_toon())
    if out_path_str:
        ws.export_safetensors(out_path_str)
        print(f"Saved edited checkpoint to: {out_path_str}")
    return 0


def cmd_improve(model_id: str, save_dir: str) -> int:
    from weightx.improver import improve_pretrained_lm, improve_small_model

    if model_id.lower() == "minillama-demo":
        rep = improve_small_model("minillama-demo", save_dir=save_dir)
        print(rep.to_toon())
    else:
        rep_pre = improve_pretrained_lm(model_id, save_dir=save_dir)
        print(rep_pre.to_toon())
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # Case-insensitive normalization for subcommands & flags, plus direct `/skill` invocation
    if raw_args:
        first_low = raw_args[0].lower()
        # Allow `wx wx ...` or `wx weightx ...` in any case
        if first_low in ("wx", "weightx"):
            raw_args = raw_args[1:]
            first_low = raw_args[0].lower() if raw_args else ""

        if raw_args:
            if first_low in KNOWN_SUBCOMMANDS:
                raw_args[0] = first_low
            elif first_low in ("-h", "--help", "/?"):
                raw_args[0] = "--help"
            else:
                # Check if first argument is a direct skill invocation (`/coding-boost`, `"Улучшить в кодинге"`, etc.)
                sk = resolve_skill(" ".join(raw_args)) or resolve_skill(raw_args[0])
                if sk is not None:
                    return cmd_run_skill(sk.skill_id)

    parser = argparse.ArgumentParser(
        prog="wx",
        description="WeightX: Open-source (GPLv3) Weight-Space Programming Harness (CLI wx/weightx + Desktop).",
    )
    sub = parser.add_subparsers(dest="subcommand")

    sub.add_parser("agents", help="List supported API providers and detected CLI agents.")
    sub.add_parser("providers", help="Alias for `agents`: list supported API providers and CLI agents.")
    sub.add_parser("catalog", help="List all models in the WeightX Model Catalog.")
    sub.add_parser("prompts", help="List ready-to-run WSP skill pills and pipelines.")
    sub.add_parser("skills", help="List all 8 WSP skills and their multi-stage pipelines.")

    p_skill = sub.add_parser("skill", help="Execute a WSP skill pipeline by ID or 2-word pill name.")
    p_skill.add_argument("name", help="Skill ID (e.g. coding-boost, memory-slim) or 2-word label.")
    p_skill.add_argument("--model", default="minillama-demo", help="Target model ID.")

    p_tools = sub.add_parser("tools", help="List all isolated WeightX LLM tools.")
    p_tools.add_argument("--toon", action="store_true", help="Format tool catalog as TOON.")

    p_recon = sub.add_parser("recon", help="Run L1 AutoArch inspection on a .safetensors file.")
    p_recon.add_argument("path", help="Path to .safetensors checkpoint.")

    p_inspect = sub.add_parser("inspect", help="Alias for `recon`: run L1 AutoArch inspection.")
    p_inspect.add_argument("path", help="Path to .safetensors checkpoint.")

    p_apply = sub.add_parser("apply-plan", help="Apply a BrainSurgery-WX declarative YAML plan.")
    p_apply.add_argument("checkpoint", help="Path to input .safetensors checkpoint.")
    p_apply.add_argument("plan", help="Path to BrainSurgery-WX .yaml plan.")
    p_apply.add_argument("-o", "--output", default=None, help="Optional output .safetensors path.")

    p_improve = sub.add_parser("improve", help="Improve a small LLM using WeightX closed-form surgery.")
    p_improve.add_argument("model", nargs="?", default="smollm2-135m", help="Catalog model ID or HF repo.")
    p_improve.add_argument("--save-dir", default="checkpoints", help="Directory for improved checkpoint.")

    p_chat = sub.add_parser("chat", help="Launch interactive DSH-style WeightX CLI REPL.")
    p_chat.add_argument("--model", default="minillama-demo", help="Initial model ID to load.")

    p_repl = sub.add_parser("repl", help="Alias for `chat`: launch interactive WeightX CLI REPL.")
    p_repl.add_argument("--model", default="minillama-demo", help="Initial model ID to load.")

    p_desktop = sub.add_parser("desktop", help="Launch WeightX Desktop GUI.")
    p_desktop.add_argument("--model", default="minillama-demo", help="Initial model ID to load.")
    p_desktop.add_argument("--smoke-test-ms", type=int, default=None, help="Auto-close after N ms (for testing).")

    p_gui = sub.add_parser("gui", help="Alias for `desktop`: launch WeightX Desktop GUI.")
    p_gui.add_argument("--model", default="minillama-demo", help="Initial model ID to load.")
    p_gui.add_argument("--smoke-test-ms", type=int, default=None, help="Auto-close after N ms (for testing).")

    sub.add_parser("install-desktop", help="Install WeightX shortcut (WeightX.lnk) on Windows Desktop & CLI PATH.")
    sub.add_parser("demo", help="Run the full 5-stage WeightX MVP demonstration.")

    args = parser.parse_args(raw_args)
    if args.subcommand in ("agents", "providers"):
        return cmd_agents()
    if args.subcommand == "catalog":
        return cmd_catalog()
    if args.subcommand in ("prompts", "skills"):
        return cmd_prompts()
    if args.subcommand == "skill":
        return cmd_run_skill(args.name, model_id=args.model)
    if args.subcommand == "tools":
        return cmd_tools(use_toon=args.toon)
    if args.subcommand in ("recon", "inspect"):
        return cmd_recon(args.path)
    if args.subcommand == "apply-plan":
        return cmd_apply_plan(args.checkpoint, args.plan, args.output)
    if args.subcommand == "improve":
        return cmd_improve(args.model, args.save_dir)
    if args.subcommand in ("chat", "repl"):
        from weightx.tui import run_interactive_cli

        return run_interactive_cli(initial_model=args.model)
    if args.subcommand in ("desktop", "gui"):
        from weightx.desktop import launch_desktop_app

        return launch_desktop_app(initial_model=args.model, smoke_test_ms=args.smoke_test_ms)
    if args.subcommand == "install-desktop":
        from weightx.desktop_installer import install_desktop_shortcut

        info = install_desktop_shortcut()
        print(dumps_toon({"desktop_installation": info}))
        return 0
    if args.subcommand == "demo":
        from examples.mvp_end_to_end_demo import run_mvp_demo

        run_mvp_demo()
        return 0

    if sys.stdin is not None and sys.stdin.isatty():
        from weightx.tui import run_interactive_cli

        return run_interactive_cli()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
