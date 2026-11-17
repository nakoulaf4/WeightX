# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""Console-script entry point for the WeightX hybrid (Rust core + torch sidecar).

Since v0.4.0 the user-facing WeightX is the Rust binary (`wx`). This shim only
locates and execs it; the Python package itself remains as the numeric sidecar
library used by the hybrid backend (see weightx-core/src/torch_worker.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _candidate_bins() -> list[Path]:
    cands: list[Path] = []
    env_bin = os.environ.get("WEIGHTX_BIN")
    if env_bin:
        cands.append(Path(env_bin))
    # Editable install: <repo>/src/weightx/_launcher.py -> <repo>
    repo = Path(__file__).resolve().parent.parent.parent
    cands.append(repo / "rust" / "target" / "release" / "wx.exe")
    cands.append(repo / "rust" / "target" / "debug" / "wx.exe")
    cands.append(repo / "dist" / "cli" / "wx.exe")
    cands.append(Path.home() / ".local" / "bin" / "wx.exe")
    return cands


def find_wx_binary() -> Path | None:
    for c in _candidate_bins():
        if c.is_file():
            return c
    found = shutil.which("wx")
    if found:
        p = Path(found).resolve()
        # Avoid recursing into this shim's own console script.
        if p.suffix.lower() == ".exe" and p.stat().st_size > 1_000_000:
            return p
    return None


def main() -> None:
    bin_path = find_wx_binary()
    if bin_path is None:
        sys.stderr.write(
            "WeightX hybrid binary (wx) not found.\n"
            "Build it:  cargo build --release --bin wx   (in the rust/ workspace)\n"
            "Or set WEIGHTX_BIN to the full path of wx.exe.\n"
        )
        raise SystemExit(1)
    env = dict(os.environ)
    # Make the repo discoverable for the torch sidecar regardless of cwd.
    repo = Path(__file__).resolve().parent.parent.parent
    if (repo / "src" / "weightx").is_dir():
        env.setdefault("WEIGHTX_HOME", str(repo))
    raise SystemExit(subprocess.call([str(bin_path), *sys.argv[1:]], env=env))


if __name__ == "__main__":
    main()