# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Unit tests for TOON serialization and token-efficiency diagnostics."""

from weightx.toon_io import compare_toon_vs_json, dumps_toon, loads_toon


def test_toon_roundtrip_and_tabular_compression():
    payload = {
        "model": "MiniLlama-4L",
        "layers": [
            {
                "layer": f"layer_{i}",
                "effective_rank": round(32.5 - i * 1.2, 3),
                "stable_rank": round(14.1 + i * 0.5, 3),
                "block_influence": round(0.18 + i * 0.02, 4),
                "fisher_information": round(0.0012 * (i + 1), 6),
                "wanda_score": round(0.85 - i * 0.03, 4),
            }
            for i in range(12)
        ],
    }

    toon_text = dumps_toon(payload)
    # Tabular header must declare length [12] and field list {...}
    assert "layers[12]{" in toon_text

    decoded = loads_toon(toon_text)
    assert decoded["model"] == "MiniLlama-4L"
    assert len(decoded["layers"]) == 12
    assert decoded["layers"][0]["layer"] == "layer_0"

    savings = compare_toon_vs_json(payload)
    assert savings.toon_chars < savings.json_chars
    assert savings.token_reduction_ratio >= 0.30
