# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Unit tests for L3 WeightWorkspace, closed-form ridge solver, and MEMIT-MASS batch editing."""

import numpy as np

from examples.mvp_end_to_end_demo import MiniLlamaConfig, MiniLlamaForCausalLM
from weightx.l3_weights import (
    BatchFactSpec,
    WeightWorkspace,
    apply_closed_form_ridge,
    apply_memit_mass_batch,
)
from weightx.safety import FisherGuard


def test_workspace_describe_slice_checkpoint_and_revert():
    model = MiniLlamaForCausalLM(MiniLlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2))
    ws = WeightWorkspace(model)

    tname = "model.layers.0.mlp.down_proj.weight"
    desc = ws.describe_tensor(tname, top_k=4)
    assert desc["shape"] == [32, 64]
    assert len(desc["largest_rows"]) == 4
    assert "top_singular_values" in desc

    sl = ws.read_slice(tname, "0:2, 0:4")
    assert sl["slice_shape"] == [2, 4]
    assert sl["slice_numel"] == 8

    orig = ws.get_readonly(tname).copy()
    ws.set_tensor(tname, orig * 2.0, apply_fisher_guard=False)
    diff = ws.diff()
    assert diff.changed_params > 0

    ws.revert_to_original(tname)
    assert ws.diff().changed_params == 0
    np.testing.assert_allclose(ws.get_readonly(tname), orig, atol=1e-6)


def test_memit_mass_batch_associative_editing_and_ridge():
    rng = np.random.default_rng(123)
    tensors = {
        "layers.0.mlp.down_proj.weight": rng.standard_normal((32, 64)).astype(np.float32) * 0.1,
        "layers.1.mlp.down_proj.weight": rng.standard_normal((32, 64)).astype(np.float32) * 0.1,
    }
    fg = FisherGuard(damping_alpha=0.05, freeze_top_fraction=0.0, enabled=True)
    ws = WeightWorkspace(tensors, fisher_guard=fg)

    # Batch of 20 associative facts
    keys = rng.standard_normal((20, 64))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    vals = rng.standard_normal((20, 32)) * 0.5
    facts = [
        BatchFactSpec(fact_id=f"f_{i}", key_vector=keys[i], target_value_vector=vals[i])
        for i in range(20)
    ]

    res = apply_memit_mass_batch(
        ws,
        ["layers.0.mlp.down_proj.weight", "layers.1.mlp.down_proj.weight"],
        facts,
        lambda_cov=1e-3,
        apply_fisher_guard=True,
    )
    assert res.num_facts == 20
    assert res.post_edit_mse < res.pre_edit_mse * 0.25
    assert res.relative_error_reduction > 0.75

    # Closed-form ridge on single matrix
    ridge_info = apply_closed_form_ridge(
        ws,
        "layers.0.mlp.down_proj.weight",
        keys[:8],
        vals[:8],
        lambda_reg=1e-3,
        apply_fisher_guard=False,
    )
    assert ridge_info["post_mse"] < ridge_info["pre_mse"]
