# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
Build native Windows executables (`dist/gui/WeightX.exe`, `dist/cli/wx.exe`, `dist/cli/weightx.exe`),
clean legacy v0.2.0 / WeightX Studio artifacts, build the `.whl` package, install Desktop shortcuts
(`WeightX.lnk` & `WeightX CLI.lnk`), register case-insensitive `wx` / `weightx` CLI shims,
and package the v0.3.0 release assets.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from weightx.desktop_installer import generate_weightx_icon, install_desktop_shortcut


def build_all() -> None:
    root = Path(__file__).resolve().parent.parent
    dist = root / "dist"
    gui_dir = dist / "gui"
    cli_dir = dist / "cli"
    dist.mkdir(exist_ok=True)
    gui_dir.mkdir(parents=True, exist_ok=True)
    cli_dir.mkdir(parents=True, exist_ok=True)

    # Refresh logo & icon first
    ico = generate_weightx_icon(root / "assets" / "weightx.ico")

    # Clean up legacy v0.2.0 / WeightX Studio files to prevent stale binaries or case collisions
    for stale in [
        root / "WeightX.exe",
        root / "WeightX-Studio.exe",
        root / "Launch_WeightX_Studio.cmd",
        root / "assets" / "weightx_studio.ico",
        dist / "WeightX.exe",
        dist / "wx.exe",
        dist / "weightx.exe",
        dist / "WeightX-Studio.exe",
        dist / "WeightX-Studio-v0.2.0-win64.exe",
        dist / "weightx-cli-v0.2.0-win64.exe",
        dist / "WeightX-v0.2.0-win64-portable.zip",
        dist / "weightx-0.2.0-py3-none-any.whl",
    ]:
        if stale.exists():
            try:
                stale.unlink()
            except Exception:
                pass

    csc = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    pyw = root / ".venv" / "Scripts" / "pythonw.exe"
    py = root / ".venv" / "Scripts" / "python.exe"

    gui_cs = f"""using System;
using System.Diagnostics;
using System.IO;

class WeightXGuiLauncher {{
    [STAThread]
    static int Main(string[] args) {{
        string root = @"{root}";
        string pyw = @"{pyw}";
        if (!File.Exists(pyw)) pyw = "pythonw.exe";
        string argStr = "-m weightx.desktop";
        if (args.Length > 0) argStr += " " + string.Join(" ", args);
        var psi = new ProcessStartInfo(pyw, argStr);
        psi.WorkingDirectory = root;
        psi.UseShellExecute = false;
        Process.Start(psi);
        return 0;
    }}
}}
"""

    cli_cs = f"""using System;
using System.Diagnostics;
using System.IO;
using System.Text;

class WeightXCliLauncher {{
    static int Main(string[] args) {{
        Console.OutputEncoding = Encoding.UTF8;
        Console.InputEncoding = Encoding.UTF8;
        string root = @"{root}";
        string py = @"{py}";
        if (!File.Exists(py)) py = "python.exe";
        var sb = new StringBuilder("-m weightx.cli");
        foreach (var a in args) {{
            sb.Append(" \\"").Append(a.Replace("\\"", "\\\\\\"")).Append("\\"");
        }}
        var psi = new ProcessStartInfo(py, sb.ToString());
        psi.WorkingDirectory = root;
        psi.UseShellExecute = false;
        var p = Process.Start(psi);
        p.WaitForExit();
        return p.ExitCode;
    }}
}}
"""

    gui_src = dist / "gui_launcher.cs"
    cli_src = dist / "cli_launcher.cs"
    gui_src.write_text(gui_cs, encoding="utf-8")
    cli_src.write_text(cli_cs, encoding="utf-8")

    gui_exe = gui_dir / "WeightX.exe"
    wx_exe = cli_dir / "wx.exe"
    weightx_cli_exe = cli_dir / "weightx.exe"

    subprocess.run(
        [str(csc), "/nologo", "/target:winexe", f"/win32icon:{ico}", f"/out:{gui_exe}", str(gui_src)],
        check=True,
    )
    subprocess.run(
        [str(csc), "/nologo", "/target:exe", f"/win32icon:{ico}", f"/out:{wx_exe}", str(cli_src)],
        check=True,
    )
    subprocess.run(
        [str(csc), "/nologo", "/target:exe", f"/win32icon:{ico}", f"/out:{weightx_cli_exe}", str(cli_src)],
        check=True,
    )

    gui_src.unlink(missing_ok=True)
    cli_src.unlink(missing_ok=True)

    # Copy distinct release-named standalone binaries for GitHub Releases
    shutil.copy2(gui_exe, dist / "WeightX-v0.3.0-win64.exe")
    shutil.copy2(wx_exe, dist / "wx-cli-v0.3.0-win64.exe")

    # Reinstall editable package & build wheel v0.3.0 if pip/wheel available
    subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(root)], check=False)
    subprocess.run([str(py), "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(dist), str(root)], check=False)

    # Install Desktop shortcuts (`WeightX.lnk` + `WeightX CLI.lnk`) & global `wx` / `weightx`
    info = install_desktop_shortcut()
    print("Desktop & CLI installed:", info)

    # Package portable zip `dist/WeightX-v0.3.0-win64-portable.zip` (avoiding NTFS case collision inside zip)
    zip_path = dist / "WeightX-v0.3.0-win64-portable.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(gui_exe, arcname="WeightX/WeightX.exe")
        zf.write(wx_exe, arcname="WeightX/bin/wx.exe")
        zf.write(weightx_cli_exe, arcname="WeightX/bin/weightx.exe")
        for f in ["README.md", "LICENSE", "pyproject.toml", "Launch_WeightX.cmd", "Launch_WeightX_CLI.cmd"]:
            fp = root / f
            if fp.exists():
                zf.write(fp, arcname=f"WeightX/{f}")
        for asset in (root / "assets").glob("*"):
            if asset.is_file():
                zf.write(asset, arcname=f"WeightX/assets/{asset.name}")
        for docf in (root / "docs").glob("*.md"):
            if docf.is_file():
                zf.write(docf, arcname=f"WeightX/docs/{docf.name}")
        for pyf in (root / "src").rglob("*.py"):
            zf.write(pyf, arcname=f"WeightX/{pyf.relative_to(root).as_posix()}")
    print("Packaged portable release:", zip_path)


if __name__ == "__main__":
    build_all()
