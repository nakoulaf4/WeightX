# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Unit tests for BrainSurgery-WX Declarative YAML Plan Engine and Invariant Assertions."""

import numpy as np

from weightx.declarative import execute_yaml_plan
from weightx.l3_weights import WeightWorkspace


def test_yaml_plan_mandatory_assertions_and_filesystem_lockdown():
    ws = WeightWorkspace({"layer.0.weight": np.ones((8, 16), dtype=np.float32)})

    # 1. Plan without assertions must be rejected when require_assertions=True
    unguarded_plan = """
transforms:
  - scale_: { target: "layer.0.weight", by: 0.5 }
"""
    rep_no_assert = execute_yaml_plan(ws, unguarded_plan, require_assertions=True)
    assert not rep_no_assert.success
    assert "Invariant Policy Violation" in (rep_no_assert.error or "")

    # 2. Plan attempting filesystem save/load must be rejected
    fs_plan = """
transforms:
  - assert: { finite: { of: "layer.0.weight" } }
  - save: { path: "/tmp/pwn.safetensors" }
"""
    rep_fs = execute_yaml_plan(ws, fs_plan)
    assert not rep_fs.success
    assert "Security Policy Violation" in (rep_fs.error or "")


def test_yaml_plan_transforms_and_atomic_rollback_on_failed_assertion():
    w = np.ones((8, 16), dtype=np.float32)
    w[3, :] = 20.0  # Outlier row 3
    ws = WeightWorkspace({"layer.0.weight": w})

    valid_plan = """
transforms:
  - assert:
      all:
        - exists: { of: "layer.0.weight" }
        - count: { of: "layer.*", is: 1 }
        - shape: { of: "layer.0.weight", is: [8, 16] }
        - dimensions: { of: "layer.0.weight", is: 2 }
        - dtype: { of: "layer.0.weight", is: "float32" }
        - finite: { of: "layer.0.weight" }
  - rescale_outlier_rows_:
      target: "layer.0.weight"
      min_ratio: 2.5
  - assert:
      all:
        - finite: { of: "layer.0.weight" }
        - norm: { of: "layer.0.weight", ord: "max", le: 1.5 }
"""
    rep = execute_yaml_plan(ws, valid_plan, apply_fisher_guard=False)
    assert rep.success
    assert rep.assertions_verified == 2
    assert rep.mutations_executed == 1
    np.testing.assert_allclose(ws.get_readonly("layer.0.weight"), np.ones((8, 16)), atol=1e-5)

    # Now test atomic rollback when a post-mutation assertion fails
    bad_post_assert_plan = """
transforms:
  - assert: { exists: { of: "layer.0.weight" } }
  - scale_: { target: "layer.0.weight", by: 100.0 }
  - assert: { norm: { of: "layer.0.weight", ord: "max", le: 2.0 } }
"""
    rep_fail = execute_yaml_plan(ws, bad_post_assert_plan, apply_fisher_guard=False)
    assert not rep_fail.success
    # Weights must have been rolled back to 1.0 (not 100.0)
    np.testing.assert_allclose(ws.get_readonly("layer.0.weight"), np.ones((8, 16)), atol=1e-5)
