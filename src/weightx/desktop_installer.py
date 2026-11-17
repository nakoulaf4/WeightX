# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
Brand Assets Generator, Case-Insensitive Global CLI Installer (`wx` / `weightx`),
and Windows Desktop Shortcut Installer (`wx install-desktop`).

Design System (1:1 DSH / DeepSeek Harness Minimalist Palette):
  - Superellipse squircle (`#1B1B1C`) with subtle hairline (`#353638`)
  - Symmetrical W-X tensor graph in crisp off-white (`#F9FAFB`) and muted slate (`#878D96`)
  - Central Weight-Space focal node in `#7AAAFF`
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
  <rect x="14" y="14" width="228" height="228" rx="58" fill="#1B1B1C" stroke="#353638" stroke-width="3"/>
  <!-- Subtle outer W tensor links -->
  <line x1="62" y1="72" x2="95" y2="184" stroke="#878D96" stroke-width="9" stroke-linecap="round"/>
  <line x1="194" y1="72" x2="161" y2="184" stroke="#878D96" stroke-width="9" stroke-linecap="round"/>
  <line x1="95" y1="184" x2="128" y2="128" stroke="#878D96" stroke-width="9" stroke-linecap="round"/>
  <line x1="161" y1="184" x2="128" y2="128" stroke="#878D96" stroke-width="9" stroke-linecap="round"/>
  <!-- Primary X weight-space diagonals -->
  <line x1="62" y1="72" x2="161" y2="184" stroke="#F9FAFB" stroke-width="15" stroke-linecap="round"/>
  <line x1="194" y1="72" x2="95" y2="184" stroke="#F9FAFB" stroke-width="15" stroke-linecap="round"/>
  <!-- Tensor nodes -->
  <circle cx="62" cy="72" r="11" fill="#F9FAFB"/>
  <circle cx="194" cy="72" r="11" fill="#F9FAFB"/>
  <circle cx="95" cy="184" r="11" fill="#F9FAFB"/>
  <circle cx="161" cy="184" r="11" fill="#F9FAFB"/>
  <!-- Central focal node -->
  <circle cx="128" cy="128" r="21" fill="#1B1B1C"/>
  <circle cx="128" cy="128" r="16" fill="#7AAAFF"/>
</svg>
"""

BANNER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 220" width="960" height="220">
  <rect x="2" y="2" width="956" height="216" rx="24" fill="#151517" stroke="#2C2D30" stroke-width="2"/>
  <g transform="translate(46, 32) scale(0.61)">
    <rect x="14" y="14" width="228" height="228" rx="58" fill="#1B1B1C" stroke="#353638" stroke-width="4"/>
    <line x1="62" y1="72" x2="95" y2="184" stroke="#878D96" stroke-width="9" stroke-linecap="round"/>
    <line x1="194" y1="72" x2="161" y2="184" stroke="#878D96" stroke-width="9" stroke-linecap="round"/>
    <line x1="62" y1="72" x2="161" y2="184" stroke="#F9FAFB" stroke-width="15" stroke-linecap="round"/>
    <line x1="194" y1="72" x2="95" y2="184" stroke="#F9FAFB" stroke-width="15" stroke-linecap="round"/>
    <circle cx="62" cy="72" r="11" fill="#F9FAFB"/>
    <circle cx="194" cy="72" r="11" fill="#F9FAFB"/>
    <circle cx="95" cy="184" r="11" fill="#F9FAFB"/>
    <circle cx="161" cy="184" r="11" fill="#F9FAFB"/>
    <circle cx="128" cy="128" r="21" fill="#1B1B1C"/>
    <circle cx="128" cy="128" r="16" fill="#7AAAFF"/>
  </g>
  <text x="248" y="98" fill="#F9FAFB" font-family="Segoe UI, Inter, system-ui, sans-serif" font-size="46" font-weight="700" letter-spacing="1">WeightX</text>
  <rect x="454" y="68" width="58" height="26" rx="13" fill="#283142"/>
  <text x="483" y="86" text-anchor="middle" fill="#7AAAFF" font-family="Consolas, monospace" font-size="13" font-weight="700">WSP</text>
  <text x="250" y="138" fill="#CFD3D6" font-family="Segoe UI, Inter, system-ui, sans-serif" font-size="18">Open-Source Weight-Space Programming Harness for LLMs (GPLv3)</text>
  <text x="250" y="170" fill="#81858C" font-family="Consolas, monospace" font-size="13">L1 AutoArch  ·  L2 Uni-Layer &amp; Knowledge Map  ·  L3 BrainSurgery-WX &amp; EAMIT  ·  Vespasian Rollback</text>
</svg>
"""


def generate_weightx_icon(icon_path: Path) -> Path:
    """
    Generate the minimalist DSH-style WeightX icon (`weightx.ico` + `weightx_logo.png`)
    using 4x supersampling (`2048x2048` -> Lanczos downscale) and write vector SVGs.
    """
    assets_dir = icon_path.parent
    assets_dir.mkdir(parents=True, exist_ok=True)

    (assets_dir / "logo.svg").write_text(LOGO_SVG.strip(), encoding="utf-8")
    (assets_dir / "banner.svg").write_text(BANNER_SVG.strip(), encoding="utf-8")

    S = 2048
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(S * 0.06)
    r = int(S * 0.24)
    d.rounded_rectangle(
        [pad, pad, S - pad, S - pad],
        radius=r,
        fill=(27, 27, 28, 255),
        outline=(53, 54, 56, 255),
        width=14,
    )

    white = (249, 250, 251, 255)
    blue = (122, 170, 255, 255)
    muted = (135, 141, 150, 255)
    w = int(S * 0.058)
    w_thin = int(S * 0.036)

    xL, xML, xC, xMR, xR = int(S * 0.24), int(S * 0.37), S // 2, int(S * 0.63), int(S * 0.76)
    yT, yM, yB = int(S * 0.28), int(S * 0.50), int(S * 0.72)

    d.line([(xL, yT), (xML, yB)], fill=muted, width=w_thin)
    d.line([(xR, yT), (xMR, yB)], fill=muted, width=w_thin)
    d.line([(xML, yB), (xC, yM)], fill=muted, width=w_thin)
    d.line([(xMR, yB), (xC, yM)], fill=muted, width=w_thin)

    d.line([(xL, yT), (xMR, yB)], fill=white, width=w)
    d.line([(xR, yT), (xML, yB)], fill=white, width=w)

    nr = int(S * 0.044)
    for nx, ny in [(xL, yT), (xR, yT), (xML, yB), (xMR, yB)]:
        d.ellipse([nx - nr, ny - nr, nx + nr, ny + nr], fill=white)

    cr = int(S * 0.062)
    ring = int(S * 0.082)
    d.ellipse([xC - ring, yM - ring, xC + ring, yM + ring], fill=(27, 27, 28, 255))
    d.ellipse([xC - cr, yM - cr, xC + cr, yM + cr], fill=blue)

    out_512 = img.resize((512, 512), resample=Image.Resampling.LANCZOS)
    png_path = assets_dir / "weightx_logo.png"
    out_512.save(str(png_path), format="PNG")

    ico_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    out_512.save(str(icon_path), format="ICO", sizes=ico_sizes)
    # Also keep weightx.ico in sync if another path was requested
    canonical_ico = assets_dir / "weightx.ico"
    if canonical_ico.resolve() != icon_path.resolve():
        out_512.save(str(canonical_ico), format="ICO", sizes=ico_sizes)
    return icon_path


def get_user_desktop_dir() -> Path:
    """Resolve the active user's Windows Desktop directory."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
            text=True,
            timeout=10,
        ).strip()
        if out and Path(out).exists():
            return Path(out)
    except Exception:
        pass
    fallback = Path.home() / "Desktop"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def install_global_cli_shims(project_root: Path, python_exe: Path) -> list[str]:
    """
    Install case-insensitive `wx` and `weightx` shims into `~/.local/bin`, add `~/.local/bin`
    to User PATH, and register PowerShell profile functions so `wx`, `WX`, `Wx`, `weightx`,
    `WeightX`, `WEIGHTX` work from any terminal.
    """
    installed: list[str] = []
    user_bin = Path.home() / ".local" / "bin"
    cli_dist = project_root / "dist" / "cli"
    try:
        user_bin.mkdir(parents=True, exist_ok=True)

        # Copy native wx.exe and weightx.exe if present in dist/cli/ (hybrid v0.4.0+).
        native_wx = cli_dist / "wx.exe"
        for exe_name in ("wx.exe", "weightx.exe"):
            src_exe = cli_dist / exe_name
            if src_exe.exists():
                dst_exe = user_bin / exe_name
                try:
                    dst_exe.write_bytes(src_exe.read_bytes())
                    installed.append(str(dst_exe))
                except Exception:
                    pass

        # .cmd wrappers: native binary when available, legacy Python CLI otherwise.
        if native_wx.exists():
            shim_body = (
                "@echo off\r\n"
                "chcp 65001 >nul\r\n"
                f'"{user_bin / "wx.exe"}" %*\r\n'
            )
            profile_wx = str(user_bin / "wx.exe")
            profile_call = f"& '{profile_wx}' @args"
        else:
            shim_body = (
                "@echo off\r\n"
                "chcp 65001 >nul\r\n"
                f'"{python_exe}" -m weightx.cli %*\r\n'
            )
            profile_call = f"& '{python_exe}' -m weightx.cli @args"
        for name in ("wx.cmd", "weightx.cmd"):
            target = user_bin / name
            target.write_text(shim_body, encoding="utf-8")
            installed.append(str(target))

        # Ensure user_bin is in HKCU\Environment\Path (never put GUI folder in PATH)
        ps_path_cmd = (
            f"$ub = '{user_bin}'; "
            "$cur = [Environment]::GetEnvironmentVariable('Path', 'User'); "
            "if (-not $cur) { $cur = '' }; "
            "$parts = $cur -split ';' | Where-Object { $_ -ne '' -and $_ -notlike '*WeightX\\dist' }; "
            "if ($parts -notcontains $ub) { $parts += $ub }; "
            "[Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User'); "
            "if ($PROFILE) { "
            "  $pdir = Split-Path -Parent $PROFILE; "
            "  if (-not (Test-Path $pdir)) { New-Item -ItemType Directory -Path $pdir -Force | Out-Null }; "
            f"  $fn = \"`n# WeightX CLI (case-insensitive wx / weightx)`nfunction global:wx {{ {profile_call} }}`nfunction global:weightx {{ {profile_call} }}`n\"; "
            "  if (Test-Path $PROFILE) { "
            "    $c = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue; "
            "    if ($c) { $c = $c -replace '(?m)^# WeightX CLI[^\n]*\nfunction global:wx[^\n]*\nfunction global:weightx[^\n]*\n?', ''; Set-Content -Path $PROFILE -Value $c -Encoding UTF8 -NoNewline } "
            "  }; "
            "  Add-Content -Path $PROFILE -Value $fn -Encoding UTF8 "
            "}"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_path_cmd],
            check=False,
            timeout=12,
        )
    except Exception:
        pass
    return installed


def install_desktop_shortcut(desktop_dir: Path | str | None = None) -> dict[str, str]:
    """
    Install/update:
      1. Minimalist DSH-style WeightX icon (`assets/weightx.ico` + `assets/weightx_logo.png`),
      2. Desktop shortcut `WeightX.lnk` (removes old `WeightX Studio.lnk`),
      3. Desktop shortcut `WeightX CLI.lnk`,
      4. Global case-insensitive CLI `wx` and `weightx`.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    icon_path = generate_weightx_icon(project_root / "assets" / "weightx.ico")

    target_desktop = Path(desktop_dir) if desktop_dir else get_user_desktop_dir()
    target_desktop.mkdir(parents=True, exist_ok=True)

    python_exe = Path(sys.executable).resolve()
    pythonw_exe = python_exe.with_name("pythonw.exe")
    gui_launcher_exe = pythonw_exe if pythonw_exe.exists() else python_exe

    # Remove legacy WeightX Studio shortcuts/files from Desktop
    if desktop_dir is None:
        for legacy_name in ("WeightX Studio.lnk", "WeightX Studio.cmd", "WeightX-Studio.lnk"):
            legacy_p = target_desktop / legacy_name
            if legacy_p.exists():
                try:
                    legacy_p.unlink()
                except Exception:
                    pass

    # Project-root launchers: native hybrid binaries when present (v0.4.0+),
    # legacy Python desktop/CLI otherwise.
    native_gui = project_root / "dist" / "gui" / "WeightX.exe"
    native_cli = project_root / "dist" / "cli" / "wx.exe"
    if native_gui.exists():
        cmd_gui = (
            "@echo off\r\n"
            "rem WeightX hybrid — Tauri desktop (Rust core + PyTorch sidecar)\r\n"
            "chcp 65001 >nul\r\n"
            f'cd /d "{project_root}"\r\n'
            f'start "" "{native_gui}" %*\r\n'
        )
    else:
        cmd_gui = (
            "@echo off\r\n"
            "chcp 65001 >nul\r\n"
            f'cd /d "{project_root}"\r\n'
            f'start "" "{gui_launcher_exe}" -m weightx.desktop\r\n'
        )
    local_gui_cmd = project_root / "Launch_WeightX.cmd"
    local_gui_cmd.write_text(cmd_gui, encoding="utf-8")

    if native_cli.exists():
        cmd_cli = (
            "@echo off\r\n"
            "rem WeightX hybrid — interactive CLI (Rust core + PyTorch sidecar)\r\n"
            "chcp 65001 >nul\r\n"
            f'cd /d "{project_root}"\r\n'
            f'"{native_cli}" chat %*\r\n'
        )
    else:
        cmd_cli = (
            "@echo off\r\n"
            "chcp 65001 >nul\r\n"
            f'cd /d "{project_root}"\r\n'
            f'"{python_exe}" -m weightx.cli chat\r\n'
        )
    local_cli_cmd = project_root / "Launch_WeightX_CLI.cmd"
    local_cli_cmd.write_text(cmd_cli, encoding="utf-8")

    if desktop_dir is not None:
        (target_desktop / "WeightX.cmd").write_text(cmd_gui, encoding="utf-8")

    # Native Windows .lnk shortcuts on Desktop named strictly `WeightX.lnk` and `WeightX CLI.lnk`
    lnk_weightx = target_desktop / "WeightX.lnk"
    lnk_cli = target_desktop / "WeightX CLI.lnk"

    native_exe_candidates = [
        project_root / "dist" / "gui" / "WeightX.exe",
    ]
    native_exe = next((p for p in native_exe_candidates if p.exists()), None)
    gui_target = native_exe if native_exe is not None else gui_launcher_exe
    gui_args = "" if native_exe is not None else "-m weightx.desktop"

    native_wx_cli = project_root / "dist" / "cli" / "wx.exe"
    cli_target = native_wx_cli if native_wx_cli.exists() else python_exe
    cli_args = "" if native_wx_cli.exists() else "-m weightx.cli chat"

    ps_script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s1 = $ws.CreateShortcut('{lnk_weightx}'); "
        f"$s1.TargetPath = '{gui_target}'; "
        f"$s1.Arguments = '{gui_args}'; "
        f"$s1.WorkingDirectory = '{project_root}'; "
        f"$s1.IconLocation = '{icon_path},0'; "
        "$s1.Description = 'WeightX — Weight-Space Programming Harness'; "
        "$s1.Save(); "
        f"$s2 = $ws.CreateShortcut('{lnk_cli}'); "
        f"$s2.TargetPath = '{cli_target}'; "
        f"$s2.Arguments = '{cli_args}'; "
        f"$s2.WorkingDirectory = '{project_root}'; "
        f"$s2.IconLocation = '{icon_path},0'; "
        "$s2.Description = 'WeightX CLI (wx / weightx)'; "
        "$s2.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
        timeout=15,
    )

    shims = install_global_cli_shims(project_root, python_exe)

    return {
        "desktop_dir": str(target_desktop),
        "shortcut_lnk": str(lnk_weightx),
        "cli_shortcut_lnk": str(lnk_cli),
        "launcher_cmd": str(target_desktop / "WeightX.cmd") if desktop_dir is not None else str(local_gui_cmd),
        "icon_path": str(icon_path),
        "logo_png": str(project_root / "assets" / "weightx_logo.png"),
        "global_cli_shims": ", ".join(shims),
    }
