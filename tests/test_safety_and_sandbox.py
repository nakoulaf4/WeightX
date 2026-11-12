# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Unit tests for IWCKEdit Fisher Protection and the Process-Isolated Python Sandbox."""

import numpy as np
import pytest

from weightx.execution import IsolatedSandbox, SandboxSecurityError, validate_sandbox_ast
from weightx.l3_weights import WeightWorkspace
from weightx.safety import FisherGuard


def test_iwck_fisher_guard_protects_critical_weights_and_allows_outlier_repair():
    orig = np.ones((10, 10), dtype=np.float32)
    orig[0, :] = 15.0  # Row 0 is a corrupted outlier neuron

    # Row 9 has ultra-high Fisher importance for general capabilities
    fisher_mat = np.ones((10, 10), dtype=np.float32) * 0.1
    fisher_mat[9, :] = 50.0

    guard = FisherGuard(
        {"w": fisher_mat},
        damping_alpha=0.5,
        freeze_top_fraction=0.10,
        outlier_bypass_ratio=2.5,
        enabled=True,
    )

    # Propose repairing row 0 back to 1.0 AND perturbing critical row 9 to 5.0
    proposed = orig.copy()
    proposed[0, :] = 1.0
    proposed[9, :] = 5.0

    safe_w, rep = guard.protect_update("w", orig, proposed, allow_outlier_repair=True)
    # Row 0 (outlier) must be fully repaired to 1.0
    np.testing.assert_allclose(safe_w[0, :], 1.0, atol=1e-5)
    # Row 9 (top 10% Fisher critical) must be frozen at its original value 1.0!
    np.testing.assert_allclose(safe_w[9, :], 1.0, atol=1e-5)
    assert rep.frozen_critical_elements == 10
    assert rep.outlier_bypass_elements == 10


def test_sandbox_ast_gatekeeper_blocks_dangerous_code():
    for forbidden_snippet in [
        "import os\nos.system('echo pwn')",
        "import socket",
        "from subprocess import Popen",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "W.__class__.__mro__[1].__subclasses__()",
        "__import__('os')",
    ]:
        with pytest.raises(SandboxSecurityError):
            validate_sandbox_ast(forbidden_snippet)


def test_sandbox_subprocess_execution_and_weight_modification():
    ws = WeightWorkspace(
        {"layer.0.weight": np.arange(16, dtype=np.float32).reshape(4, 4)}
    )
    sbx = IsolatedSandbox(timeout_sec=10.0, apply_fisher_guard=False)

    code = """
import numpy as np
arr = W["layer.0.weight"]
u, s, vh = np.linalg.svd(arr, full_matrices=False)
print("Top SV:", round(float(s[0]), 3))
W["layer.0.weight"] = arr * 0.5
"""
    res = sbx.run(ws, code, tensor_pattern="layer.0.*")
    assert res.success
    assert "Top SV:" in res.stdout
    assert res.modified_tensors == ["layer.0.weight"]
    np.testing.assert_allclose(
        ws.get_readonly("layer.0.weight"),
        np.arange(16, dtype=np.float32).reshape(4, 4) * 0.5,
        atol=1e-6,
    )
