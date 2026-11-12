# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Unit tests for L2 13-Metric Uni-Layer Profiler and Clean-Room Knowledge Map (Logit Lens + KCI)."""

import numpy as np
import torch

from examples.mvp_end_to_end_demo import build_corrupted_model_and_splits
from weightx.l2_profile import (
    compute_gini_kci,
    profile_model_layers,
    run_knowledge_map,
)
from weightx.l2_profile.unilayer import METRIC_NAMES_13


def test_l2_unilayer_13_metrics_and_outlier_localization():
    model, _practice, _held_out, calib_ids = build_corrupted_model_and_splits()
    report = profile_model_layers(model, calibration_input_ids=calib_ids, model_name="TestLlama")

    assert report.num_layers == 4
    assert len(report.metrics_computed) == 13
    for m_name in METRIC_NAMES_13:
        assert m_name in report.per_layer_metrics[0]

    # Must detect the injected outlier neurons in layer_2 (rows 5 and 11 of down_proj)
    assert len(report.outlier_neurons) >= 1
    layer2_outliers = [
        o for o in report.outlier_neurons if o.tensor_name == "model.layers.2.mlp.down_proj.weight"
    ]
    assert len(layer2_outliers) == 1
    assert set(layer2_outliers[0].row_indices[:2]) == {5, 11}
    assert "outlier neurons" in report.summary


def test_l2_knowledge_map_logit_lens_and_kci():
    model, _practice, _held_out, _calib_ids = build_corrupted_model_and_splits()
    kmap = run_knowledge_map(model)

    assert kmap.n_layers == 4
    assert kmap.n_heads == 4
    assert 0.0 <= kmap.kci_overall <= 1.0
    assert set(kmap.kci_per_domain.keys()) == {"factual", "syntactic", "reasoning"}
    assert len(kmap.logit_lens_layers) == 4

    # Verify Gini KCI mathematical properties: uniform -> 0, one-hot -> approaches (N-1)/N
    uniform_gini = compute_gini_kci(np.ones(16))
    sparse_gini = compute_gini_kci([0.0] * 15 + [100.0])
    assert uniform_gini < 1e-6
    assert sparse_gini > 0.85
