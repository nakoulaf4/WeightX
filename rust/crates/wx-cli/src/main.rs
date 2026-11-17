// SPDX-License-Identifier: GPL-3.0-or-later
//! WeightX CLI (`wx` / `weightx`) — Rust port of `weightx.cli`.
//!
//! Case-insensitive CLI & WSP skill runner with an interactive REPL.

mod repl;

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::json;
use weightx_core::catalog::ModelCatalog;
use weightx_core::error::Result;
use weightx_core::l1;
use weightx_core::plan;
use weightx_core::skills::{parse_skill_command, resolve_skill, WSP_SKILLS};
use weightx_core::studio::StudioSession;
use weightx_core::tensor;
use weightx_core::toon;
use weightx_core::tools::ToolCatalog;
use weightx_core::workspace::WeightWorkspace;
use weightx_agent::CliAgentProvider;

const KNOWN_SUBCOMMANDS: &[&str] = &[
    "agents", "providers", "catalog", "prompts", "skills", "skill", "tools", "recon",
    "inspect", "apply-plan", "improve", "chat", "repl", "desktop", "gui",
    "install-desktop", "demo",
];

fn make_session() -> StudioSession {
    let mut s = StudioSession::new();
    s.factory = weightx_agent::agent_factory();
    let spec = s.provider_spec();
    if let Ok(p) = weightx_agent::build_provider(&spec) {
        s.provider = p;
    }
    s
}

fn cmd_agents() -> i32 {
    let detected = CliAgentProvider::detect_installed_agents();
    let mut agents: Vec<serde_json::Value> = detected
        .iter()
        .map(|(k, v)| {
            json!({
                "agent": k,
                "installed": v.is_some(),
                "binary_path": v.clone().unwrap_or_else(|| "not_found".into()),
            })
        })
        .collect();
    agents.sort_by(|a, b| a["agent"].as_str().cmp(&b["agent"].as_str()));
    let info = json!({
        "api_providers": ["openai", "openrouter", "vllm", "ollama", "lm-studio", "anthropic", "custom"],
        "cli_agents": agents,
    });
    println!("{}", toon::dumps_toon(&info));
    0
}

fn cmd_catalog() -> i32 {
    let mut cat = ModelCatalog::new();
    println!("{}", cat.to_toon());
    0
}

fn cmd_skills() -> i32 {
    let rows: Vec<serde_json::Value> = WSP_SKILLS
        .iter()
        .enumerate()
        .map(|(i, sk)| {
            json!({
                "idx": i + 1,
                "skill": format!("/skill {}", sk.skill_id),
                "pill": sk.pill_label,
                "pipeline": sk.pipeline_title,
                "stages": sk.stage_names.len(),
            })
        })
        .collect();
    println!("{}", toon::dumps_toon(&json!({"weightx_skills": rows, "total_skills": WSP_SKILLS.len()})));
    0
}

fn cmd_run_skill(skill_query: &str, model_id: &str) -> i32 {
    let Some(sk) = resolve_skill(skill_query) else {
        eprintln!(
            "Неизвестный скилл: {skill_query:?}. Используйте `/skill <название>`: {:?}",
            WSP_SKILLS.iter().map(|s| format!("/skill {}", s.skill_id)).collect::<Vec<_>>()
        );
        return 1;
    };
    let mut sess = make_session();
    if let Err(e) = sess.load_model(model_id) {
        eprintln!("Ошибка загрузки модели: {e}");
        return 1;
    }
    match sess.run_skill(sk.skill_id, "") {
        Ok(res) => {
            println!("{}", res.to_markdown());
            0
        }
        Err(e) => {
            eprintln!("Ошибка: {e}");
            1
        }
    }
}

fn cmd_tools(use_toon: bool) -> i32 {
    let cat = ToolCatalog::new(use_toon);
    if use_toon {
        println!("{}", cat.to_prompt_catalog());
    } else {
        println!("{}", serde_json::to_string_pretty(&cat.to_openai_tools()).unwrap_or_default());
    }
    0
}

fn cmd_recon(path_str: &str) -> i32 {
    let p = Path::new(path_str);
    if !p.exists() {
        eprintln!("Error: checkpoint path not found: {}", p.display());
        return 1;
    }
    match l1::inspect_safetensors(p) {
        Ok(rep) => {
            println!("{}", rep.to_toon(true));
            0
        }
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn cmd_apply_plan(model_path: &str, plan_path: &str, out_path: Option<&str>) -> i32 {
    let mp = Path::new(model_path);
    let pp = Path::new(plan_path);
    if !mp.exists() || !pp.exists() {
        eprintln!("Error: checkpoint ({}) or plan ({}) not found.", mp.display(), pp.display());
        return 1;
    }
    let tensors = match tensor::load_safetensors(mp) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    let yaml = match std::fs::read_to_string(pp) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    let mut ws = WeightWorkspace::from_tensors(tensors, None);
    let rep = plan::execute_yaml_plan(&mut ws, &yaml, false, true, false);
    println!("{}", rep.to_toon());
    if let Some(out) = out_path {
        match ws.export_safetensors(Path::new(out), false) {
            Ok(p) => println!("Saved edited checkpoint to: {}", p.display()),
            Err(e) => {
                eprintln!("Error: {e}");
                return 1;
            }
        }
    }
    0
}

fn cmd_improve(model_id: &str, save_dir: &str) -> i32 {
    let dir = PathBuf::from(save_dir);
    let result: Result<serde_json::Value> = if model_id.eq_ignore_ascii_case("minillama-demo") {
        weightx_core::improver::improve_small_model("minillama-demo", true, Some(&dir), 2026)
            .map(|r| r.to_value())
    } else {
        weightx_core::improver::improve_pretrained_lm(model_id, 3, 4, 0.02, Some(&dir), &|m| println!("{m}"))
            .map(|r| r.to_value())
    };
    match result {
        Ok(v) => {
            println!("{}", toon::dumps_toon(&v));
            // Benchmark JSON artifact (parity with examples/improve_real_model.py).
            let safe = model_id.replace('/', "_").replace(':', "_");
            let report_path = dir.join(format!("benchmark_{safe}.json"));
            if let Ok(s) = serde_json::to_string_pretty(&v) {
                if std::fs::write(&report_path, s).is_ok() {
                    println!("Benchmark report: {}", report_path.display());
                }
            }
            0
        }
        Err(e) => {
            eprintln!("Ошибка: {e}");
            1
        }
    }
}

fn desktop_binary() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let name = if cfg!(windows) { "weightx-desktop.exe" } else { "weightx-desktop" };
    let gui_name = if cfg!(windows) { "WeightX.exe" } else { "WeightX" };
    // Search: next to wx, the dist/gui layout, and the cargo release dir.
    // NB: no `dir/WeightX.exe` candidate — on case-insensitive NTFS it would
    // collide with the `weightx.exe` CLI copy and re-launch wx itself.
    let candidates = [
        dir.join(name),
        dir.join("../gui").join(gui_name),
        dir.join("../gui").join(name),
        dir.join("../../rust/target/release").join(name),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

fn cmd_desktop(model: &str, smoke_test_ms: Option<u64>) -> i32 {
    let Some(bin) = desktop_binary() else {
        eprintln!(
            "WeightX Desktop (Tauri) binary not found next to wx. Build it with: cargo build -p weightx-desktop"
        );
        return 1;
    };
    let mut cmd = Command::new(bin);
    cmd.arg("--model").arg(model);
    if let Some(ms) = smoke_test_ms {
        cmd.arg("--smoke-test-ms").arg(ms.to_string());
    }
    match cmd.status() {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("Failed to launch desktop: {e}");
            1
        }
    }
}

fn cmd_install_desktop() -> i32 {
    if !cfg!(windows) {
        eprintln!("install-desktop currently supports Windows shortcuts only.");
        return 1;
    }
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("wx.exe"));
    let dir = exe.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let desktop_bin = dir.join("weightx-desktop.exe");
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let user_bin = home.join(".local").join("bin");
    let wx_dir = home.join(".weightx");
    let _ = std::fs::create_dir_all(&user_bin);
    let _ = std::fs::create_dir_all(&wx_dir);

    // 1. Global CLI shims: copy wx.exe / weightx.exe + .cmd wrappers into ~/.local/bin.
    let mut installed: Vec<String> = Vec::new();
    let wx_dst = user_bin.join("wx.exe");
    let weightx_dst = user_bin.join("weightx.exe");
    if std::fs::copy(&exe, &wx_dst).is_ok() {
        installed.push(wx_dst.display().to_string());
    }
    if std::fs::copy(&exe, &weightx_dst).is_ok() {
        installed.push(weightx_dst.display().to_string());
    }
    for name in ["wx.cmd", "weightx.cmd"] {
        let target_exe = if name.starts_with("wx") { &wx_dst } else { &weightx_dst };
        let body = format!("@echo off\r\nchcp 65001 >nul\r\n\"{}\" %*\r\n", target_exe.display());
        let p = user_bin.join(name);
        if std::fs::write(&p, body).is_ok() {
            installed.push(p.display().to_string());
        }
    }

    // 2. Icon for shortcuts (embedded crisp WeightX W-X icon -> ~/.weightx/weightx.ico).
    let icon_path = wx_dir.join("weightx.ico");
    let _ = std::fs::write(&icon_path, include_bytes!("../../../../assets/weightx.ico"));

    // 3. PATH registration + PowerShell profile functions + shortcuts (single PS call).
    let gui_target = if desktop_bin.is_file() { desktop_bin.clone() } else { exe.clone() };
    let gui_args = if desktop_bin.is_file() { "" } else { "desktop" };
    let script = format!(
        "$ErrorActionPreference = 'SilentlyContinue'; \
         $ub = '{user_bin}'; \
         $cur = [Environment]::GetEnvironmentVariable('Path', 'User'); if (-not $cur) {{ $cur = '' }}; \
         $parts = $cur -split ';' | Where-Object {{ $_ -ne '' -and $_ -notlike '*WeightX\\dist' }}; \
         if ($parts -notcontains $ub) {{ $parts += $ub }}; \
         [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User'); \
         if ($PROFILE) {{ \
           $pdir = Split-Path -Parent $PROFILE; \
           if (-not (Test-Path $pdir)) {{ New-Item -ItemType Directory -Path $pdir -Force | Out-Null }}; \
           $fn = \"`n# WeightX CLI (case-insensitive wx / weightx)`nfunction global:wx {{ & '{wx_exe}' @args }}`nfunction global:weightx {{ & '{wx_exe}' @args }}`n\"; \
           if (-not (Test-Path $PROFILE) -or -not ((Get-Content $PROFILE -Raw) -match 'WeightX CLI')) {{ Add-Content -Path $PROFILE -Value $fn -Encoding UTF8 }} \
         }}; \
         $desktop = [Environment]::GetFolderPath('Desktop'); \
         if (-not $desktop -or -not (Test-Path $desktop)) {{ $desktop = Join-Path $env:USERPROFILE 'Desktop'; New-Item -ItemType Directory -Path $desktop -Force | Out-Null }}; \
         Add-Type -AssemblyName Microsoft.VisualBasic; \
         foreach ($legacy in @('WeightX Studio.lnk','WeightX Studio.cmd','WeightX-Studio.lnk')) {{ \
           $lp = Join-Path $desktop $legacy; \
           if (Test-Path $lp) {{ [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($lp, 'OnlyErrorDialogs', 'SendToRecycleBin') }} \
         }}; \
         $ws = New-Object -ComObject WScript.Shell; \
         $lnk = $ws.CreateShortcut((Join-Path $desktop 'WeightX.lnk')); \
         $lnk.TargetPath = '{gui_target}'; $lnk.Arguments = '{gui_args}'; $lnk.WorkingDirectory = '{wd}'; \
         $lnk.Description = 'WeightX Weight-Space Programming Harness'; \
         if (Test-Path '{icon}') {{ $lnk.IconLocation = '{icon}' }}; \
         $lnk.Save(); \
         $lnk2 = $ws.CreateShortcut((Join-Path $desktop 'WeightX CLI.lnk')); \
         $lnk2.TargetPath = '{cli_target}'; $lnk2.Arguments = 'chat'; $lnk2.WorkingDirectory = '{wd}'; \
         $lnk2.Description = 'WeightX interactive CLI'; \
         if (Test-Path '{icon}') {{ $lnk2.IconLocation = '{icon}' }}; \
         $lnk2.Save(); \
         Write-Output $desktop",
        user_bin = user_bin.display(),
        wx_exe = wx_dst.display(),
        gui_target = gui_target.display(),
        gui_args = gui_args,
        wd = dir.display(),
        icon = icon_path.display(),
        cli_target = wx_dst.display(),
    );
    let desktop_dir = match Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", &script])
        .output()
    {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Ok(out) => {
            eprintln!("Installer script failed: {}", String::from_utf8_lossy(&out.stderr));
            return 1;
        }
        Err(e) => {
            eprintln!("PowerShell not available: {e}");
            return 1;
        }
    };

    println!("{}", toon::dumps_toon(&json!({
        "desktop_installation": {
            "desktop_dir": desktop_dir,
            "shortcut_lnk": format!("{}\\WeightX.lnk", desktop_dir),
            "cli_shortcut_lnk": format!("{}\\WeightX CLI.lnk", desktop_dir),
            "icon_path": icon_path.display().to_string(),
            "global_cli_shims": installed,
            "gui_target": gui_target.display().to_string(),
        }
    })));
    0
}

fn print_help() {
    println!(
        "WeightX: Open-source (GPLv3) Weight-Space Programming Harness (CLI wx/weightx + Desktop).

Usage: wx <COMMAND>

Commands:
  agents, providers        List supported API providers and detected CLI agents
  catalog                  List all models in the WeightX Model Catalog
  skills, prompts          List all 8 WSP skills (/skill <name>) and their pipelines
  skill <name> [--model id]  Execute a WSP skill pipeline by ID or 2-word pill name
  tools [--toon]           List all isolated WeightX LLM tools
  recon, inspect <path>    Run L1 AutoArch inspection on a .safetensors file
  apply-plan <ckpt> <plan> [-o out]  Apply a BrainSurgery-WX declarative YAML plan
  improve [model] [--save-dir dir]   Improve a small LLM using closed-form surgery
  chat, repl [--model id]  Launch the interactive WeightX CLI REPL
  desktop, gui [--model id] [--smoke-test-ms N]  Launch WeightX Desktop (Tauri)
  install-desktop          Install WeightX.lnk shortcut on the Windows Desktop
  demo                     Run the full 5-stage WeightX MVP demonstration
  help                     Print this help

Direct skill invocation: wx /skill coding-boost [--model tiny-gpt2]

Global options:
  --backend rust|torch|auto  Numeric backend (default auto: PyTorch sidecar for
                             models >= 5M params, Rust core for small models).
                             Env: WEIGHTX_BACKEND, WEIGHTX_PYTHON, WEIGHTX_TORCH_DEVICE"
    );
}

fn collect_positionals(rest: &[String]) -> Vec<&str> {
    let value_flags = ["--model", "-m", "--save-dir", "-o", "--output", "--smoke-test-ms"];
    let mut positional: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < rest.len() {
        if value_flags.contains(&rest[i].as_str()) {
            i += 2;
            continue;
        }
        if !rest[i].starts_with('-') {
            positional.push(rest[i].as_str());
        }
        i += 1;
    }
    positional
}

#[cfg(test)]
mod tests {
    use super::collect_positionals;

    #[test]
    fn flag_values_are_not_positionals() {
        let args: Vec<String> = ["--save-dir", "out", "--toon"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(collect_positionals(&args).is_empty());

        let args: Vec<String> = ["-o", "out.safetensors", "ckpt.safetensors", "plan.yaml"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(collect_positionals(&args), vec!["ckpt.safetensors", "plan.yaml"]);

        let args: Vec<String> = ["3", "--model", "minillama-demo"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(collect_positionals(&args), vec!["3"]);
    }
}

fn main() {
    let mut raw_args: Vec<String> = std::env::args().skip(1).collect();

    // Global hybrid-backend flag: --backend rust|torch|auto (WEIGHTX_BACKEND).
    let mut i = 0;
    while i < raw_args.len() {
        if raw_args[i].eq_ignore_ascii_case("--backend") {
            if i + 1 < raw_args.len() {
                std::env::set_var("WEIGHTX_BACKEND", raw_args[i + 1].to_lowercase());
                raw_args.drain(i..i + 2);
                continue;
            }
            raw_args.remove(i);
            continue;
        }
        i += 1;
    }

    // Case-insensitive normalization + `wx wx ...` unwrapping + direct /skill.
    if !raw_args.is_empty() {
        let mut first_low = raw_args[0].to_lowercase();
        if first_low == "wx" || first_low == "weightx" {
            raw_args.remove(0);
            first_low = raw_args.first().map(|s| s.to_lowercase()).unwrap_or_default();
        }
        if !raw_args.is_empty() {
            if KNOWN_SUBCOMMANDS.contains(&first_low.as_str()) {
                raw_args[0] = first_low;
            } else if matches!(first_low.as_str(), "-h" | "--help" | "/?") {
                raw_args[0] = "--help".to_string();
            } else if first_low == "/skills" {
                std::process::exit(cmd_skills());
            } else {
                // Extract optional --model/-m and try direct skill invocation.
                let mut model_override = "minillama-demo".to_string();
                let mut clean: Vec<String> = Vec::new();
                let mut i = 0;
                while i < raw_args.len() {
                    if matches!(raw_args[i].to_lowercase().as_str(), "--model" | "-m") && i + 1 < raw_args.len() {
                        model_override = raw_args[i + 1].clone();
                        i += 2;
                        continue;
                    }
                    clean.push(raw_args[i].clone());
                    i += 1;
                }
                let joined = clean.join(" ");
                let (sk, _rem, explicit) = parse_skill_command(&joined);
                if explicit && sk.is_none() {
                    std::process::exit(cmd_skills());
                }
                if let Some(sk) = sk {
                    std::process::exit(cmd_run_skill(sk.skill_id, &model_override));
                }
            }
        }
    }

    let sub = raw_args.first().map(|s| s.as_str()).unwrap_or("");
    let rest = if raw_args.is_empty() { &[][..] } else { &raw_args[1..] };
    let flag = |name: &str| -> Option<String> {
        rest.iter()
            .position(|a| a == name)
            .and_then(|i| rest.get(i + 1))
            .cloned()
    };
    // Positionals must exclude values consumed by value-taking flags.
    let positional = collect_positionals(rest);

    let code = match sub {
        "agents" | "providers" => cmd_agents(),
        "catalog" => cmd_catalog(),
        "skills" | "prompts" => cmd_skills(),
        "skill" => {
            let name = positional.first().copied().unwrap_or("");
            let model = flag("--model").unwrap_or_else(|| "minillama-demo".into());
            cmd_run_skill(name, &model)
        }
        "tools" => cmd_tools(rest.iter().any(|a| a == "--toon")),
        "recon" | "inspect" => cmd_recon(positional.first().copied().unwrap_or("")),
        "apply-plan" => {
            let ckpt = positional.first().copied().unwrap_or("");
            let plan = positional.get(1).copied().unwrap_or("");
            let out = flag("-o").or_else(|| flag("--output"));
            cmd_apply_plan(ckpt, plan, out.as_deref())
        }
        "improve" => {
            let model = positional.first().copied().unwrap_or("smollm2-135m");
            let dir = flag("--save-dir").unwrap_or_else(|| "checkpoints".into());
            cmd_improve(model, &dir)
        }
        "chat" | "repl" => {
            let model = flag("--model").unwrap_or_else(|| "minillama-demo".into());
            repl::run_interactive_cli(&model)
        }
        "desktop" | "gui" => {
            let model = flag("--model").unwrap_or_else(|| "minillama-demo".into());
            let ms = flag("--smoke-test-ms").and_then(|s| s.parse::<u64>().ok());
            cmd_desktop(&model, ms)
        }
        "install-desktop" => cmd_install_desktop(),
        "demo" => match weightx_core::demo::run_mvp_demo() {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("Ошибка: {e}");
                1
            }
        },
        "--help" | "help" | "-h" => {
            print_help();
            0
        }
        "" => {
            // No args: launch REPL on a TTY, otherwise print help.
            use std::io::IsTerminal;
            if std::io::stdin().is_terminal() {
                repl::run_interactive_cli("minillama-demo")
            } else {
                print_help();
                0
            }
        }
        other => {
            eprintln!("Unknown command: {other}\n");
            print_help();
            1
        }
    };
    std::process::exit(code);
}