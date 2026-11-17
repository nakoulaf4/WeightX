# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Download and deterministically cache canonical evaluation splits from HuggingFace Hub.

All benchmark data is saved locally under `benchmarks/cache/` (gitignored) with a
SHA-256 integrity manifest so every run is offline-reproducible and auditable.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

CACHE_DIR = Path(__file__).resolve().parent / "cache"
DATASETS_API = "https://datasets-server.huggingface.co/rows"


def _fetch_rows(
    client: httpx.Client,
    dataset: str,
    config: str,
    split: str,
    max_rows: int | None = None,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Fetch paginated rows from HuggingFace datasets-server with retry backoff."""
    rows: list[dict[str, Any]] = []
    offset = 0
    total_reported: int | None = None

    while True:
        batch_len = page_size if max_rows is None else min(page_size, max_rows - len(rows))
        if batch_len <= 0:
            break

        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": batch_len,
        }
        payload: dict[str, Any] | None = None
        for attempt in range(8):
            resp = client.get(DATASETS_API, params=params, timeout=30.0)
            if resp.status_code == 200:
                payload = resp.json()
                break
            if resp.status_code in (429, 500, 502, 503, 504):
                wait_s = min(30.0, 3.0 * (attempt + 1))
                print(f"  [retry {attempt+1}] {dataset}/{split} offset={offset} status={resp.status_code}, sleep {wait_s}s")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
        if payload is None:
            raise RuntimeError(f"Failed fetching {dataset}/{config}/{split} at offset={offset}")

        if total_reported is None:
            total_reported = int(payload.get("num_rows_total", 0))
        batch = payload.get("rows", [])
        if not batch:
            break
        for item in batch:
            rows.append(item["row"])
        offset += len(batch)
        if len(batch) < batch_len:
            break
        if total_reported and offset >= total_reported:
            break
        if max_rows is not None and len(rows) >= max_rows:
            break
        time.sleep(0.8)

    return rows


def prepare_all(force: bool = False) -> dict[str, Any]:
    """Download and cache all benchmark splits; returns the integrity manifest."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("wikitext2_test", "Salesforce/wikitext", "wikitext-2-raw-v1", "test", None),
        ("lambada_test", "cimec/lambada", "plain_text", "test", 2000),
        ("arc_easy_test", "allenai/ai2_arc", "ARC-Easy", "test", 1000),
        ("hellaswag_val", "Rowan/hellaswag", "default", "validation", 600),
        ("sciq_train", "allenai/sciq", "default", "train", 100),
        ("sciq_test", "allenai/sciq", "default", "test", 500),
        ("mbpp_train", "google-research-datasets/mbpp", "full", "train", 100),
        ("mbpp_test", "google-research-datasets/mbpp", "full", "test", None),
        ("humaneval_test", "openai/openai_humaneval", "openai_humaneval", "test", None),
        ("counterfact_train", "NeelNanda/counterfact-tracing", "default", "train", 800),
    ]

    manifest: dict[str, Any] = {}
    with httpx.Client(http2=False) as client:
        for name, ds, cfg, split, max_rows in specs:
            out_path = CACHE_DIR / f"{name}.json"
            if out_path.exists() and not force:
                raw_bytes = out_path.read_bytes()
                data = json.loads(raw_bytes.decode("utf-8"))
                manifest[name] = {
                    "dataset": ds,
                    "config": cfg,
                    "split": split,
                    "rows": len(data),
                    "sha256": hashlib.sha256(raw_bytes).hexdigest()[:16],
                }
                print(f"[cache] {name}: {len(data)} rows (cached)")
                continue

            t0 = time.perf_counter()
            rows = _fetch_rows(client, ds, cfg, split, max_rows=max_rows)
            raw_text = json.dumps(rows, ensure_ascii=False, indent=2)
            out_path.write_text(raw_text, encoding="utf-8")
            sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
            manifest[name] = {
                "dataset": ds,
                "config": cfg,
                "split": split,
                "rows": len(rows),
                "sha256": sha,
            }
            print(f"[fetch] {name}: {len(rows)} rows in {time.perf_counter() - t0:.1f}s -> {out_path.name}")

    (CACHE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def load_cached(name: str) -> list[dict[str, Any]]:
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cached dataset split missing: {path}. Run prepare_datasets.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    prepare_all()
