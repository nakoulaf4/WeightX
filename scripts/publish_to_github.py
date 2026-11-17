# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
Publish WeightX to GitHub (`nakoulaf4/WeightX`) and upload Release v0.4.0 assets.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
DEFAULT_GIT_EXE = Path(
    r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd\git.exe"
)

TARGET_OWNER = "nakoulaf4"
REPO_NAME = "WeightX"
FULL_REPO = f"{TARGET_OWNER}/{REPO_NAME}"
RELEASE_TAG = "v0.4.0"
GITHUB_API = "https://api.github.com"


def resolve_git_exe() -> str:
    """Locate git executable without assuming a single hardcoded install."""
    which_git = shutil.which("git")
    if which_git:
        return which_git
    if DEFAULT_GIT_EXE.exists():
        return str(DEFAULT_GIT_EXE)
    raise FileNotFoundError("Could not locate git.exe in PATH or Visual Studio installation.")


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _query_gcm_for_target(git_exe: str) -> str | None:
    """Query git credential helper specifically for TARGET_OWNER without accepting other accounts."""
    try:
        proc = subprocess.run(
            [git_exe, "credential", "fill"],
            input=f"protocol=https\nhost=github.com\nusername={TARGET_OWNER}\n\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None

    parsed: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k.strip()] = v.strip()

    user = parsed.get("username", "")
    password = parsed.get("password", "")
    if user.lower() == TARGET_OWNER.lower() and password:
        return password
    return None


def resolve_target_token(git_exe: str) -> str:
    """Resolve and validate a GitHub token belonging strictly to `nakoulaf4`."""
    candidates: list[tuple[str, str]] = []
    for env_key in ("WEIGHTX_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            candidates.append((f"env:{env_key}", val))

    gcm_token = _query_gcm_for_target(git_exe)
    if gcm_token:
        candidates.append(("git-credential-manager", gcm_token))

    if not candidates:
        raise RuntimeError(
            f"Missing GitHub credentials for account '{TARGET_OWNER}'.\n"
            f"Refusing to fall back to any other account (e.g. End0ez).\n"
            f"Provide a GitHub Personal Access Token for '{TARGET_OWNER}' (with 'repo' scope) via:\n"
            f"  $env:GITHUB_TOKEN = 'ghp_...'\n"
            f"or store '{TARGET_OWNER}' credentials in Git Credential Manager, then re-run this script."
        )

    with httpx.Client(timeout=20.0) as client:
        for source, token in candidates:
            resp = client.get(f"{GITHUB_API}/user", headers=_headers(token))
            if resp.status_code != 200:
                continue
            login = str(resp.json().get("login", ""))
            if login.lower() == TARGET_OWNER.lower():
                print(f"[publish] Authenticated as GitHub user '{login}' via {source}.")
                return token
            raise RuntimeError(
                f"Credential from {source} belongs to GitHub user '{login}', not required '{TARGET_OWNER}'.\n"
                f"Aborted to avoid publishing to the wrong account."
            )

    raise RuntimeError(
        f"Available GitHub token(s) failed authentication against {GITHUB_API}/user.\n"
        f"Please provide a valid token for '{TARGET_OWNER}' in $env:GITHUB_TOKEN."
    )


def run_git(git_exe: str, *args: str, redact: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git in repo root with token redaction on output/errors."""
    proc = subprocess.run(
        [git_exe, *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        combined = (proc.stdout + "\n" + proc.stderr).strip()
        if redact:
            combined = combined.replace(redact, "***")
        safe_args = [a.replace(redact, "***") if redact else a for a in args]
        raise RuntimeError(f"git {' '.join(safe_args)} failed ({proc.returncode}):\n{combined}")
    return proc


def ensure_github_repo(client: httpx.Client) -> dict[str, Any]:
    """Fetch `nakoulaf4/WeightX` or create it if it does not exist yet."""
    r = client.get(f"{GITHUB_API}/repos/{FULL_REPO}")
    if r.status_code == 200:
        print(f"[publish] Using existing repository: https://github.com/{FULL_REPO}")
        return r.json()
    if r.status_code != 404:
        raise RuntimeError(f"Failed to query {FULL_REPO}: {r.status_code} {r.text}")

    payload = {
        "name": REPO_NAME,
        "description": (
            "WeightX: Open-source Weight-Space Programming (WSP) harness with DSH minimalist UI, "
            "case-insensitive wx/weightx CLI, 8 built-in WSP skills, and Vespasian rollback."
        ),
        "private": False,
        "has_issues": True,
        "has_wiki": False,
    }
    created = client.post(f"{GITHUB_API}/user/repos", json=payload)
    if created.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create repository {FULL_REPO}: {created.status_code} {created.text}")
    print(f"[publish] Created repository: https://github.com/{FULL_REPO}")
    return created.json()


def sync_git_repo_and_push(git_exe: str, token: str) -> None:
    """Commit local changes on `main`, set clean origin, and push `main` + `v0.3.0`."""
    if not (ROOT / ".git").exists():
        run_git(git_exe, "init", "-b", "main")
    run_git(git_exe, "branch", "-M", "main")

    name_check = run_git(git_exe, "config", "--get", "user.name", check=False)
    if not name_check.stdout.strip():
        run_git(git_exe, "config", "user.name", TARGET_OWNER)
    email_check = run_git(git_exe, "config", "--get", "user.email", check=False)
    if not email_check.stdout.strip():
        run_git(git_exe, "config", "user.email", f"{TARGET_OWNER}@users.noreply.github.com")

    clean_remote_url = f"https://github.com/{FULL_REPO}.git"
    has_origin = run_git(git_exe, "remote", "get-url", "origin", check=False)
    if has_origin.returncode == 0:
        run_git(git_exe, "remote", "set-url", "origin", clean_remote_url)
    else:
        run_git(git_exe, "remote", "add", "origin", clean_remote_url)

    # Stage tracked updates and explicit WeightX project directories (never slurp unrelated local folders)
    run_git(git_exe, "add", "-u")
    project_paths = [
        ".gitignore",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "Launch_WeightX.cmd",
        "Launch_WeightX_CLI.cmd",
        "assets",
        "benchmarks",
        "checkpoints",
        "docs",
        "examples",
        "rust",
        "scripts",
        "src",
        "tests",
    ]
    existing_paths = [p for p in project_paths if (ROOT / p).exists()]
    if existing_paths:
        run_git(git_exe, "add", "--", *existing_paths)
    diff_cached = run_git(git_exe, "diff", "--cached", "--quiet", check=False)
    if diff_cached.returncode != 0:
        run_git(git_exe, "commit", "-m", f"WeightX {RELEASE_TAG}: hybrid Rust+PyTorch core, EAMIT & model benchmarks")

    run_git(git_exe, "tag", "-f", RELEASE_TAG)

    auth_push_url = f"https://x-access-token:{token}@github.com/{FULL_REPO}.git"
    run_git(
        git_exe,
        "-c",
        "credential.helper=",
        "push",
        auth_push_url,
        "main:main",
        redact=token,
    )
    run_git(
        git_exe,
        "-c",
        "credential.helper=",
        "push",
        auth_push_url,
        f"refs/tags/{RELEASE_TAG}:refs/tags/{RELEASE_TAG}",
        "--force",
        redact=token,
    )
    run_git(git_exe, "config", "branch.main.remote", "origin")
    run_git(git_exe, "config", "branch.main.merge", "refs/heads/main")
    print(f"[publish] Pushed main and tag {RELEASE_TAG} to {clean_remote_url}")


def upsert_release_and_upload_assets(client: httpx.Client) -> str:
    """Create/update GitHub release `v0.4.0` and upload release binaries."""
    release_body = (
        f"## WeightX {RELEASE_TAG}\n\n"
        "- **Hybrid Rust + PyTorch Core**: Native Rust CLI (`wx`/`weightx`), Tauri Desktop, TOON, YAML plans, and persistent PyTorch/CUDA numeric sidecar.\n"
        "- **EAMIT Batch & Domain Calibration**: Embedding-aligned closed-form editing (KL + top-M MSE + $C_0$ covariance preservation) with IWCKEdit Fisher Guard and Vespasian rollback.\n"
        "- **Comprehensive Model Improvement Benchmarks**: Cross-architecture evaluations (SmolLM2-135M/360M, GPT-2 124M, Qwen2.5-0.5B, MiniLlama) across WikiText-2, LAMBADA, ARC-Easy, HellaSwag, SciQ, MBPP, HumanEval, and CounterFact (`docs/MODEL_BENCHMARKS.md`).\n"
        "- **Windows Standalone Releases**: `WeightX-v0.4.0-win64.exe` (Desktop GUI), `wx-cli-v0.4.0-win64.exe` (CLI), portable ZIP, Python wheel, and improved `smollm2-135m` checkpoint.\n"
    )

    r_rel = client.get(f"{GITHUB_API}/repos/{FULL_REPO}/releases/tags/{RELEASE_TAG}")
    if r_rel.status_code == 200:
        rel_id = r_rel.json()["id"]
        upd = client.patch(
            f"{GITHUB_API}/repos/{FULL_REPO}/releases/{rel_id}",
            json={"tag_name": RELEASE_TAG, "name": f"WeightX {RELEASE_TAG}", "body": release_body, "draft": False},
        )
        if upd.status_code != 200:
            raise RuntimeError(f"Failed updating release {RELEASE_TAG}: {upd.status_code} {upd.text}")
        rel = upd.json()
    elif r_rel.status_code == 404:
        crt = client.post(
            f"{GITHUB_API}/repos/{FULL_REPO}/releases",
            json={
                "tag_name": RELEASE_TAG,
                "target_commitish": "main",
                "name": f"WeightX {RELEASE_TAG}",
                "body": release_body,
                "draft": False,
                "prerelease": False,
            },
        )
        if crt.status_code not in (200, 201):
            raise RuntimeError(f"Failed creating release {RELEASE_TAG}: {crt.status_code} {crt.text}")
        rel = crt.json()
    else:
        raise RuntimeError(f"Failed querying release {RELEASE_TAG}: {r_rel.status_code} {r_rel.text}")

    upload_url_tmpl: str = rel["upload_url"].split("{", 1)[0]
    existing_assets = {a["name"]: a["id"] for a in rel.get("assets", [])}

    assets_to_upload = [
        (DIST / f"WeightX-{RELEASE_TAG}-win64-portable.zip", "application/zip"),
        (DIST / f"WeightX-{RELEASE_TAG}-win64.exe", "application/vnd.microsoft.portable-executable"),
        (DIST / f"wx-cli-{RELEASE_TAG}-win64.exe", "application/vnd.microsoft.portable-executable"),
        (DIST / "weightx-0.4.0-py3-none-any.whl", "application/octet-stream"),
        (ROOT / "checkpoints" / "smollm2-135m_weightx_improved.safetensors", "application/octet-stream"),
    ]

    for path, content_type in assets_to_upload:
        if not path.exists():
            raise FileNotFoundError(f"Expected release asset is missing: {path}. Run scripts/build_release.py first.")
        if path.name in existing_assets:
            del_r = client.delete(f"{GITHUB_API}/repos/{FULL_REPO}/releases/assets/{existing_assets[path.name]}")
            if del_r.status_code not in (200, 204):
                raise RuntimeError(f"Failed deleting existing asset {path.name}: {del_r.status_code}")

        with path.open("rb") as fh:
            data = fh.read()
        up_r = client.post(
            upload_url_tmpl,
            params={"name": path.name},
            content=data,
            headers={"Content-Type": content_type},
            timeout=120.0,
        )
        if up_r.status_code not in (200, 201):
            raise RuntimeError(f"Failed uploading {path.name}: {up_r.status_code} {up_r.text}")
        print(f"[publish] Uploaded release asset: {path.name} ({len(data):,} bytes)")

    return str(rel["html_url"])


def main() -> int:
    git_exe = resolve_git_exe()
    token = resolve_target_token(git_exe)

    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_release.py")], cwd=str(ROOT), check=True)
    from weightx.desktop_installer import install_desktop_shortcut

    install_desktop_shortcut()

    with httpx.Client(headers=_headers(token), timeout=30.0) as client:
        ensure_github_repo(client)
        sync_git_repo_and_push(git_exe, token)
        release_url = upsert_release_and_upload_assets(client)

    print(f"[publish] Successfully published https://github.com/{FULL_REPO}")
    print(f"[publish] Release URL: {release_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
