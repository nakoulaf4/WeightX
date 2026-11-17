# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Tests for ModelCatalog, Codex Prompt Suggestions, Expanded BrainSurgery-WX ops, StudioSession, and Desktop installer."""

from pathlib import Path

from weightx.catalog import CODEX_PROMPT_SUGGESTIONS, ModelCatalog
from weightx.declarative.plan_engine import execute_yaml_plan
from weightx.desktop_installer import install_desktop_shortcut
from weightx.improver import improve_small_model
from weightx.studio import StudioSession


def test_model_catalog_and_codex_prompt_suggestions():
    cat = ModelCatalog()
    entries = cat.list_entries()
    assert len(entries) >= 8
    ids = {e.model_id for e in entries}
    assert "minillama-demo" in ids
    assert "smollm2-135m" in ids
    assert "qwen2.5-0.5b" in ids
    assert len(CODEX_PROMPT_SUGGESTIONS) == 8
    assert all(p.prompt_text and p.title and p.icon for p in CODEX_PROMPT_SUGGESTIONS)


def test_expanded_brainsurgery_yaml_transforms():
    session = StudioSession(log=lambda _m: None)
    session.load_model("minillama-demo")
    ws = session.harness.workspace

    yaml_plan = """
transforms:
  - assert:
      exists: { of: "model.layers.1.mlp.down_proj.weight" }
  - spectral_norm_cap_:
      target: "model.layers.1.mlp.down_proj.weight"
      max_median_ratio: 2.5
  - svd_low_rank_:
      target: "model.layers.1.mlp.down_proj.weight"
      energy_fraction: 0.99
  - project_out_direction_:
      target: "model.layers.1.mlp.down_proj.weight"
      alpha: 0.25
  - ties_trim_:
      target: "model.layers.1.mlp.down_proj.weight"
      keep_top_fraction: 0.5
  - slerp_:
      target: "model.layers.1.mlp.down_proj.weight"
      t: 0.2
  - assert:
      finite: { of: "model.layers.1.mlp.down_proj.weight" }
"""
    rep = execute_yaml_plan(ws, yaml_plan, apply_fisher_guard=True)
    assert rep.success is True
    assert rep.mutations_executed == 5
    assert rep.assertions_verified == 2


def test_studio_session_actions_and_small_model_improver(tmp_path: Path):
    logs: list[str] = []
    session = StudioSession(log=logs.append)
    session.load_model("minillama-demo")

    out_recon = session.run_action("recon_profile", "")
    assert "AutoArch" in out_recon and "Uni-Layer" in out_recon

    out_rollback = session.run_action("rollback_test", "")
    assert "AUTOMATIC ROLLBACK" in out_rollback

    imp_rep = improve_small_model("minillama-demo", save_dir=tmp_path)
    assert imp_rep.final_metrics.top1_accuracy >= imp_rep.initial_metrics.top1_accuracy
    assert imp_rep.final_metrics.perplexity <= imp_rep.initial_metrics.perplexity
    assert imp_rep.saved_checkpoint_path is not None
    assert Path(imp_rep.saved_checkpoint_path).exists()

    info = install_desktop_shortcut(desktop_dir=tmp_path)
    assert Path(info["shortcut_lnk"]).exists()
    assert Path(info["shortcut_lnk"]).name == "WeightX.lnk"
    assert not (tmp_path / "WeightX Studio.lnk").exists()
    assert Path(info["launcher_cmd"]).exists()
    assert Path(info["icon_path"]).exists()


def test_wsp_skills_and_case_insensitive_resolution():
    from weightx.catalog import CODEX_PROMPT_SUGGESTIONS
    from weightx.skills import WSP_SKILLS, resolve_skill

    expected_ids = {
        "coding-boost",
        "memory-slim",
        "outlier-repair",
        "spectral-denoise",
        "eamit-facts",
        "knowledge-map",
        "abliterate",
        "vespasian-guard",
    }
    assert len(WSP_SKILLS) == 8
    assert len(CODEX_PROMPT_SUGGESTIONS) == 8

    for pill in CODEX_PROMPT_SUGGESTIONS:
        assert 1 <= len(pill.title.split()) <= 3
        assert pill.action_tag.startswith("skill:")
        assert pill.action_tag.removeprefix("skill:") in expected_ids

    s_coding_upper = resolve_skill("/CODING-BOOST")
    s_coding_plain = resolve_skill("coding-boost")
    s_coding_label = resolve_skill("Улучшить в кодинге")
    assert s_coding_upper is not None
    assert s_coding_upper == s_coding_plain == s_coding_label

    s_mem_mixed = resolve_skill("Уменьшить Память")
    s_mem_id = resolve_skill("memory-slim")
    assert s_mem_mixed is not None and s_mem_mixed == s_mem_id

    s_guard = resolve_skill("vespasian-guard")
    s_guard_upper = resolve_skill("VESPASIAN-GUARD")
    assert s_guard is not None and s_guard == s_guard_upper


def test_studio_session_run_skill_pipelines():
    logs: list[str] = []
    session = StudioSession(log=logs.append)
    session.load_model("minillama-demo")

    all_skills = (
        "coding-boost",
        "memory-slim",
        "outlier-repair",
        "spectral-denoise",
        "eamit-facts",
        "knowledge-map",
        "abliterate",
        "vespasian-guard",
    )
    for skill_id in all_skills:
        res = session.run_skill(skill_id)
        assert isinstance(res.accepted, bool)
        assert len(res.steps) >= 3
        md = res.to_markdown()
        assert isinstance(md, str) and len(md.strip()) > 0
        if skill_id == "knowledge-map":
            assert res.accepted is True
            assert "global_gini_kci" in res.metrics
        if skill_id == "vespasian-guard":
            assert res.accepted is True
            assert res.metrics["rolled_back"] is True
            assert res.metrics["bit_exact_max_diff"] < 1e-6


def test_cli_case_insensitive_subcommands_and_skills():
    from weightx.cli import main as cli_main

    assert cli_main(["SKILLS"]) == 0
    assert cli_main(["Catalog"]) == 0
    assert cli_main(["PROMPTS"]) == 0
    assert cli_main(["skill", "VESPASIAN-GUARD"]) == 0
    assert cli_main(["/skill", "VESPASIAN-GUARD"]) == 0
    assert cli_main(["SKILL", "/coding-boost"]) == 0


def test_parse_skill_command_and_catalog_skill_naming():
    from weightx.catalog import WSP_SKILL_SUGGESTIONS
    from weightx.skills import parse_skill_command

    assert WSP_SKILL_SUGGESTIONS == CODEX_PROMPT_SUGGESTIONS
    assert all(p.prompt_text.startswith("/skill ") for p in WSP_SKILL_SUGGESTIONS)
    assert all(p.category.startswith("Скилл ") for p in WSP_SKILL_SUGGESTIONS)

    sk, rem, exp = parse_skill_command("/skill coding-boost")
    assert exp is True and sk is not None and sk.skill_id == "coding-boost" and rem == ""

    sk2, _, exp2 = parse_skill_command("/SKILL Улучшить в кодинге")
    assert exp2 is True and sk2 is not None and sk2.skill_id == "coding-boost"

    sk3, _, exp3 = parse_skill_command("/skill")
    assert exp3 is True and sk3 is None


def test_cli_slash_suggestions_and_completions():
    from weightx.tui import get_cli_suggestions

    root_sugs = get_cli_suggestions("/")
    root_texts = {s.text for s in root_sugs}
    assert any(t.startswith("/skill") for t in root_texts)
    assert any(t.startswith("/load") for t in root_texts)

    all_skills = get_cli_suggestions("/skill ")
    assert len(all_skills) == 8
    assert any("coding-boost" in s.text for s in all_skills)

    mem_sugs = get_cli_suggestions("/skill mem")
    mem_texts = " ".join(s.text for s in mem_sugs)
    assert "memory-slim" in mem_texts
    eam_sugs = get_cli_suggestions("/skill eam")
    eam_texts = " ".join(s.text for s in eam_sugs)
    assert "eamit-facts" in eam_texts

    load_sugs = get_cli_suggestions("/load smol", model_ids=["minillama-demo", "smollm2-135m"])
    assert any("smollm2-135m" in s.text for s in load_sugs)

    brain_sugs = get_cli_suggestions("/brain ")
    assert len(brain_sugs) >= 1


def test_desktop_animations_bg_modes_and_smoke():
    from weightx.desktop import BG_MODES, ease_in_out_cubic, ease_out_cubic, launch_desktop_app, lerp_hex

    mid = lerp_hex("#000000", "#ffffff", 0.5)
    assert mid.startswith("#") and len(mid) == 7
    assert ease_out_cubic(0.0) == 0.0 and ease_out_cubic(1.0) == 1.0
    assert 0.0 < ease_out_cubic(0.5) < 1.0
    assert ease_in_out_cubic(0.0) == 0.0 and ease_in_out_cubic(1.0) == 1.0
    assert 0.0 < ease_in_out_cubic(0.5) < 1.0
    assert len(BG_MODES) == 4
    assert launch_desktop_app(smoke_test_ms=700) == 0


