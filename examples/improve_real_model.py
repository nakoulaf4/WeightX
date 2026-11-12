# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark: improve a real small pretrained LM (default SmolLM2-135M) with WeightX."""

import json
import sys
from pathlib import Path

from weightx.improver import improve_pretrained_lm

if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "smollm2-135m"
    rep = improve_pretrained_lm(model_id, save_dir="checkpoints")
    print("=" * 70)
    print(rep.to_toon())
    out = Path("checkpoints") / f"benchmark_{model_id.replace('/', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report -> {out}")
