# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Unit tests for L3 WeightWorkspace, closed-form ridge solver, EAMIT embedding alignment & batch editing, and domain calibration."""

import numpy as np
import torch

from examples.mvp_end_to_end_demo import MiniLlamaConfig, MiniLlamaForCausalLM
from weightx.declarative.plan_engine import execute_yaml_plan
from weightx.l3_weights import (
    BatchFactSpec,
    EamitAlignConfig,
    WeightWorkspace,
    align_residuals_to_keys,
    apply_closed_form_ridge,
    apply_eamit_batch,
)
from weightx.l3_weights.domain_eamit import eamit_domain_calibration
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


def test_eamit_batch_associative_editing_and_ridge():
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

    res = apply_eamit_batch(
        ws,
        ["layers.0.mlp.down_proj.weight", "layers.1.mlp.down_proj.weight"],
        facts,
        lambda_cov=1e-3,
        apply_fisher_guard=True,
    )
    assert res.num_facts == 20
    assert res.post_edit_mse < res.pre_edit_mse * 0.25
    assert res.relative_error_reduction > 0.75
    assert res.post_alignment_kl < res.pre_alignment_kl
    assert res.post_alignment_top_m_mse < res.pre_alignment_top_m_mse

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


def test_eamit_clustered_alignment_and_yaml_op():
    rng = np.random.default_rng(7)
    centers = rng.standard_normal((3, 64))
    keys = []
    vals = []
    for i in range(18):
        k = centers[i // 6] + 0.12 * rng.standard_normal(64)
        k /= np.linalg.norm(k)
        v = rng.standard_normal(32) * 0.25
        keys.append(k)
        vals.append(v)
    k_mat = np.column_stack(keys)
    r_mat = np.column_stack(vals)
    r_aligned, stats = align_residuals_to_keys(k_mat, r_mat, EamitAlignConfig())
    assert r_aligned.shape == r_mat.shape
    assert stats.post_kl < stats.pre_kl * 0.6
    assert stats.post_top_m_mse < stats.pre_top_m_mse

    ws = WeightWorkspace({"w": rng.standard_normal((32, 64)).astype(np.float32) * 0.05})
    plan = {
        "transforms": [
            {"assert": {"exists": {"of": "w"}}},
            {
                "eamit_batch_": {
                    "target": "w",
                    "facts": [{"id": f"c_{i}", "key": keys[i].tolist(), "value": vals[i].tolist()} for i in range(18)],
                }
            },
        ]
    }
    rep = execute_yaml_plan(ws, plan, apply_fisher_guard=False)
    assert rep.success, rep.error
    assert ws.diff().changed_params > 0


def test_eamit_domain_calibration_on_minillama():
    model = MiniLlamaForCausalLM(MiniLlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2))
    ws = WeightWorkspace(model)
    prac = [torch.tensor([1, 5, 3, 9, 2, 4], dtype=torch.long), torch.tensor([3, 7, 11, 2, 8, 6], dtype=torch.long)]
    ctrl = [torch.tensor([2, 4, 6, 8, 10, 12], dtype=torch.long)]
    res = eamit_domain_calibration(model, ws, prac, ctrl, n_layers=2, iterations=1, apply_fisher_guard=False)
    assert len(res.steps) == 1
    assert res.steps[0].alignment_kl_after <= res.steps[0].alignment_kl_before + 1e-9
