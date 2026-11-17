# SPDX-License-Identifier: GPL-3.0-or-later
"""TOON cross-compatibility: Rust decodes Python's TOON and vice versa."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\runbo\Downloads\WeightX")
sys.path.insert(0, str(ROOT / "src"))
WORK = Path(__file__).parent
BENCH = ROOT / "rust" / "target" / "release" / "wx-bench.exe"

from weightx.toon_io import dumps_toon, loads_toon

payload = json.loads((WORK / "payload.json").read_text())

# Python -> TOON file
py_text = dumps_toon(payload)
(WORK / "py.toon").write_text(py_text, encoding="utf-8")

# Rust -> TOON file
r = subprocess.run([str(BENCH), "toon-dump", str(WORK / "payload.json"), str(WORK / "rust.toon")],
                   capture_output=True, text=True, check=True)
rust_dump = json.loads(r.stdout)

# Rust decodes Python's TOON
r = subprocess.run([str(BENCH), "toon-load", str(WORK / "py.toon")],
                   capture_output=True, text=True, check=True)
rust_decodes_py = json.loads(r.stdout)["json"]

# Python decodes Rust's TOON
py_decodes_rust = loads_toon((WORK / "rust.toon").read_text(encoding="utf-8"))


def close(a, b, path=""):
    """Recursive equality with float tolerance; returns list of mismatches."""
    errs = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            errs.append(f"{path}: keys {set(a) ^ set(b)}")
        for k in set(a) & set(b):
            errs += close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            errs.append(f"{path}: len {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            errs += close(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > 1e-6 * max(1.0, abs(a), abs(b)):
            errs.append(f"{path}: {a} vs {b}")
    elif a != b:
        errs.append(f"{path}: {a!r} vs {b!r}")
    return errs


e1 = close(payload, rust_decodes_py)
e2 = close(payload, py_decodes_rust)
print(json.dumps({
    "py_toon_chars": len(py_text),
    "rust_toon_chars": rust_dump["chars"],
    "rust_decodes_python": "OK" if not e1 else e1[:10],
    "python_decodes_rust": "OK" if not e2 else e2[:10],
    "n_mismatch_rust_dec_py": len(e1),
    "n_mismatch_py_dec_rust": len(e2),
}))