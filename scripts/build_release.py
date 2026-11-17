# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
Build the WeightX v0.4.0 HYBRID Windows distribution.

Since v0.4.0 the user-facing product is the Rust hybrid (Rust core + PyTorch
numeric sidecar, Tauri desktop). This script:
  1. builds the Rust binaries (wx, wx-bench, weightx-desktop) if cargo is available;
  2. places them into dist/cli/wx.exe, dist/cli/weightx.exe, dist/gui/WeightX.exe
     (the targets used by Desktop shortcuts and ~/.local/bin shims);
  3. reinstalls the editable Python package (sidecar library + entry-point shims)
     and builds the wheel;
  4. refreshes Desktop shortcuts and global wx/weightx shims;
  5. packages the portable zip and release-named assets;
  6. moves stale v0.3.0 (Python-launcher) artifacts to the Recycle Bin.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.4.0"
RUST_RELEASE = ROOT / "rust" / "target" / "release"


def _recycle(path: Path) -> None:
    """Send a stale build artifact to the Recycle Bin (never permanent delete)."""
    ps = (
        "Add-Type -AssemblyName Microsoft.VisualBasic; "
        "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile("
        f"'{path}', 'OnlyErrorDialogs', 'SendToRecycleBin')"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=False)


def build_rust() -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        print("cargo not found; using prebuilt binaries in rust/target/release")
        return
    for args in (
        ["build", "--release", "-p", "wx-cli", "--bin", "wx", "--bin", "wx-bench"],
        ["build", "--release", "-p", "weightx-desktop"],
    ):
        subprocess.run([cargo, *args], cwd=ROOT / "rust", check=True)


def build_all() -> None:
    dist = ROOT / "dist"
    gui_dir = dist / "gui"
    cli_dir = dist / "cli"
    for d in (dist, gui_dir, cli_dir):
        d.mkdir(parents=True, exist_ok=True)

    build_rust()

    wx_src = RUST_RELEASE / "wx.exe"
    bench_src = RUST_RELEASE / "wx-bench.exe"
    gui_src = RUST_RELEASE / "weightx-desktop.exe"
    for src in (wx_src, gui_src):
        if not src.is_file():
            raise SystemExit(f"Missing Rust binary: {src} (run cargo build --release)")

    # Real native binaries replace the old Python-launcher stubs.
    shutil.copy2(wx_src, cli_dir / "wx.exe")
    shutil.copy2(wx_src, cli_dir / "weightx.exe")
    shutil.copy2(gui_src, gui_dir / "WeightX.exe")
    if bench_src.is_file():
        shutil.copy2(bench_src, cli_dir / "wx-bench.exe")

    # Release-named standalone assets for GitHub Releases.
    shutil.copy2(gui_dir / "WeightX.exe", dist / f"WeightX-v{VERSION}-win64.exe")
    shutil.copy2(cli_dir / "wx.exe", dist / f"wx-cli-v{VERSION}-win64.exe")

    # Editable reinstall (regenerates weightx/wx console shims -> weightx._launcher)
    # and wheel build.
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    py_exe = str(py) if py.is_file() else sys.executable
    subprocess.run([py_exe, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(ROOT)], check=False)
    subprocess.run([py_exe, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(dist), str(ROOT)], check=False)

    # Desktop shortcuts + global shims (targets dist/gui and dist/cli, copies to ~/.local/bin).
    from weightx.desktop_installer import install_desktop_shortcut
    info = install_desktop_shortcut()
    print("Desktop & CLI installed:", info)

    # Portable zip: native binaries + Python sidecar library + launchers + docs.
    zip_path = dist / f"WeightX-v{VERSION}-win64-portable.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(gui_dir / "WeightX.exe", arcname="WeightX/WeightX.exe")
        zf.write(cli_dir / "wx.exe", arcname="WeightX/bin/wx.exe")
        zf.write(cli_dir / "weightx.exe", arcname="WeightX/bin/weightx.exe")
        if bench_src.is_file():
            zf.write(cli_dir / "wx-bench.exe", arcname="WeightX/bin/wx-bench.exe")
        for f in ("README.md", "LICENSE", "pyproject.toml", "Launch_WeightX.cmd", "Launch_WeightX_CLI.cmd"):
            fp = ROOT / f
            if fp.exists():
                zf.write(fp, arcname=f"WeightX/{f}")
        for asset in (ROOT / "assets").glob("*"):
            if asset.is_file():
                zf.write(asset, arcname=f"WeightX/assets/{asset.name}")
        for docf in (ROOT / "docs").glob("*.md"):
            if docf.is_file():
                zf.write(docf, arcname=f"WeightX/docs/{docf.name}")
        for pyf in (ROOT / "src").rglob("*.py"):
            zf.write(pyf, arcname=f"WeightX/{pyf.relative_to(ROOT).as_posix()}")
    print("Packaged portable release:", zip_path)

    # Stale v0.3.0 Python-launcher artifacts -> Recycle Bin.
    for stale in (
        dist / f"WeightX-v0.3.0-win64.exe",
        dist / f"wx-cli-v0.3.0-win64.exe",
        dist / f"WeightX-v0.3.0-win64-portable.zip",
        dist / "weightx-0.3.0-py3-none-any.whl",
    ):
        if stale.exists():
            _recycle(stale)
            print("Recycled stale artifact:", stale.name)


if __name__ == "__main__":
    build_all()