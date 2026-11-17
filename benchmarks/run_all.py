# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Orchestrator for the WeightX Model Improvement Benchmark Suite.

Runs each `(suite, model)` benchmark in an isolated subprocess so system RAM
and GPU VRAM stay clean between models, saving structured JSON artifacts under
`benchmarks/results/`.

Usage:
  python -m benchmarks.run_all --all
  python -m benchmarks.run_all --job domain:smollm2-135m
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

REAL_MODELS = [
    "smollm2-135m",
    "smollm2-135m-instruct",
    "gpt2-124m",
    "smollm2-360m",
    "qwen2.5-0.5b",
]

EAMIT_FACT_MODELS = [
    "smollm2-135m",
    "smollm2-135m-instruct",
    "smollm2-360m",
    "qwen2.5-0.5b",
]

ABLATION_MODELS = {"smollm2-135m", "qwen2.5-0.5b"}
HUMANEVAL_GEN_MODELS = {"qwen2.5-0.5b"}


def run_minillama_suite() -> dict[str, Any]:
    """Benchmark WeightX's built-in `minillama-demo` stand (outlier repair + skills + EAMIT)."""
    import numpy as np

    from examples.mvp_end_to_end_demo import build_corrupted_model_and_splits
    from weightx.improver import improve_small_model
    from weightx.l3_weights.eamit import BatchFactSpec, apply_eamit_batch
    from weightx.l3_weights.workspace import WeightWorkspace
    from weightx.studio import StudioSession

    t0 = time.perf_counter()
    rep = improve_small_model("minillama-demo", inject_realistic_corruption=True, save_dir=None, seed=2026)
    elapsed_imp = round(time.perf_counter() - t0, 3)

    # All 8 WSP skills smoke & metrics benchmark on minillama-demo
    skill_results: list[dict[str, Any]] = []
    for sk_id in (
        "outlier-repair",
        "spectral-denoise",
        "coding-boost",
        "memory-slim",
        "eamit-facts",
        "knowledge-map",
        "abliterate",
        "vespasian-guard",
    ):
        sess = StudioSession(log=lambda _m: None)
        sess.load_model("minillama-demo")
        ts = time.perf_counter()
        sr = sess.run_skill(sk_id)
        skill_results.append({
            "skill_id": sk_id,
            "accepted": bool(sr.accepted),
            "elapsed_s": round(time.perf_counter() - ts, 3),
            "metrics": sr.metrics,
            "summary": sr.summary,
        })

    # EAMIT associative scaling on minillama-demo
    model, _, _, _ = build_corrupted_model_and_splits(seed=2026)
    ws = WeightWorkspace(model)
    ws.checkpoint("clean")
    targets = [
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.2.mlp.down_proj.weight",
        "model.layers.3.mlp.down_proj.weight",
    ]
    d_out, d_in = ws.shape(targets[0])
    rng = np.random.default_rng(7)
    scaling: list[dict[str, Any]] = []
    for n_f in (10, 50, 100, 250):
        ws.restore("clean")
        keys = rng.standard_normal((n_f, d_in))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8
        vals = rng.standard_normal((n_f, d_out)) * 0.25
        facts = [BatchFactSpec(f"m_{i}", keys[i], vals[i]) for i in range(n_f)]
        ts = time.perf_counter()
        er = apply_eamit_batch(ws, targets, facts, lambda_cov=1e-2, apply_fisher_guard=True)
        scaling.append({
            "num_facts": n_f,
            "elapsed_ms": round((time.perf_counter() - ts) * 1000.0, 2),
            "pre_edit_mse": round(er.pre_edit_mse, 6),
            "post_edit_mse": round(er.post_edit_mse, 6),
            "error_reduction_pct": round(er.relative_error_reduction * 100.0, 2),
            "pre_alignment_kl": round(er.pre_alignment_kl, 5),
            "post_alignment_kl": round(er.post_alignment_kl, 5),
        })

    return {
        "model_id": "minillama-demo",
        "improver_elapsed_s": elapsed_imp,
        "improver_report": rep.to_dict(),
        "skills": skill_results,
        "eamit_scaling": scaling,
    }


def _collect_env_meta() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "cpu_count": os.cpu_count(),
    }


def _run_job(job_spec: str, device: str = "cuda") -> dict[str, Any]:
    suite, _, model_id = job_spec.partition(":")
    if suite == "minillama":
        return run_minillama_suite()
    if suite == "domain":
        from benchmarks.suite_domain import run_model_domain_suite

        return run_model_domain_suite(
            model_id,
            run_ablation=(model_id in ABLATION_MODELS),
            run_humaneval_gen=(model_id in HUMANEVAL_GEN_MODELS),
            device=device,
        )
    if suite == "facts":
        from benchmarks.suite_eamit_facts import run_model_eamit_facts_suite

        return run_model_eamit_facts_suite(model_id, device=device)
    if suite == "repair":
        from benchmarks.suite_repair_and_slim import run_model_repair_slim_kmap

        return run_model_repair_slim_kmap(model_id, device=device)
    raise ValueError(f"Unknown job spec: {job_spec!r}")


def _job_filename(job_spec: str) -> Path:
    safe = job_spec.replace(":", "__").replace("/", "_")
    return RESULTS_DIR / f"{safe}.json"


def run_all(force: bool = False, device: str = "cuda") -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[str] = ["minillama:demo"]
    for m in REAL_MODELS:
        jobs.append(f"domain:{m}")
    for m in EAMIT_FACT_MODELS:
        jobs.append(f"facts:{m}")
    for m in REAL_MODELS:
        jobs.append(f"repair:{m}")

    summary: dict[str, Any] = {
        "environment": _collect_env_meta(),
        "jobs": {},
    }

    total_t0 = time.perf_counter()
    for idx, job in enumerate(jobs, start=1):
        out_path = _job_filename(job)
        if out_path.exists() and not force:
            print(f"[{idx}/{len(jobs)}] [cached] {job} -> {out_path.name}", flush=True)
            summary["jobs"][job] = json.loads(out_path.read_text(encoding="utf-8"))
            continue

        print(f"[{idx}/{len(jobs)}] [running] {job} ...", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "benchmarks.run_all", "--job", job, "--device", device],
            cwd=str(ROOT),
        )
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"Benchmark job {job} failed with exit code {proc.returncode}")
        print(f"[{idx}/{len(jobs)}] [done] {job} in {time.perf_counter() - t0:.1f}s", flush=True)
        summary["jobs"][job] = json.loads(out_path.read_text(encoding="utf-8"))

    summary["total_wall_time_s"] = round(time.perf_counter() - total_t0, 2)
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    from benchmarks.report_generator import generate_report

    md_p, svg_p = generate_report()
    print(f"All benchmark suites finished in {summary['total_wall_time_s']}s -> {summary_path}", flush=True)
    print(f"Generated report -> {md_p}, chart -> {svg_p}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="WeightX Model Improvement Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all suites across all models")
    parser.add_argument("--job", type=str, default="", help="Run single job (e.g. domain:smollm2-135m)")
    parser.add_argument("--force", action="store_true", help="Re-run even if result JSON exists")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.job:
        res = _run_job(args.job, device=args.device)
        out_path = _job_filename(args.job)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {out_path}")
        return 0
    if args.all:
        run_all(force=args.force, device=args.device)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
