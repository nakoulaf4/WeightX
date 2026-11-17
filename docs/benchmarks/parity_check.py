# SPDX-License-Identifier: GPL-3.0-or-later
"""Parity checks: Rust vs Python safetensors exports + MiniLlama demo weights."""
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

RUST = Path(r"C:\Users\runbo\AppData\Local\Temp\wx_bench_export.safetensors")
PY = Path(r"C:\Users\runbo\AppData\Local\Temp\qwenwork\wx-bench\py_bench_export.safetensors")

a = load_file(str(RUST))
b = load_file(str(PY))
assert set(a) == set(b), f"key mismatch: {set(a) ^ set(b)}"
max_abs = 0.0
n_bitexact = 0
worst = None
for k in a:
    x, y = a[k], b[k]
    assert x.shape == y.shape and x.dtype == y.dtype, k
    if np.array_equal(x, y):
        n_bitexact += 1
    d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
    if d > max_abs:
        max_abs, worst = d, k
print(json.dumps({
    "tensors": len(a),
    "bit_exact": n_bitexact,
    "max_abs_diff": max_abs,
    "worst_tensor": worst,
}))