// SPDX-License-Identifier: GPL-3.0-or-later
//! Interactive WeightX CLI REPL — minimalist terminal workspace.

use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use weightx_core::skills::{parse_skill_command, WSP_SKILLS};
use weightx_core::studio::{StudioSession, PROVIDER_KINDS};
use weightx_core::toon;

use crate::make_session;

static COLOR_ON: AtomicBool = AtomicBool::new(false);

fn init_terminal_style() {
    if std::env::var_os("NO_COLOR").is_some() || !std::io::stdout().is_terminal() {
        COLOR_ON.store(false, Ordering::Relaxed);
        return;
    }
    #[cfg(windows)]
    unsafe {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetStdHandle(nStdHandle: u32) -> isize;
            fn GetConsoleMode(hConsoleHandle: isize, lpMode: *mut u32) -> i32;
            fn SetConsoleMode(hConsoleHandle: isize, dwMode: u32) -> i32;
        }
        const STD_OUTPUT_HANDLE: u32 = (-11i32) as u32;
        const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        if handle != 0 && handle != -1 {
            let mut mode = 0u32;
            if GetConsoleMode(handle, &mut mode) != 0 {
                let _ = SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            }
        }
    }
    COLOR_ON.store(true, Ordering::Relaxed);
}

fn c(code: &str, text: &str) -> String {
    if COLOR_ON.load(Ordering::Relaxed) {
        format!("{code}{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

const ACCENT: &str = "\x1b[38;2;122;170;255m";
const ACCENT_BOLD: &str = "\x1b[1;38;2;122;170;255m";
const FG_BOLD: &str = "\x1b[1;38;2;244;246;251m";
const MUTED: &str = "\x1b[38;2;129;133;140m";
const OK: &str = "\x1b[38;2;78;209;126m";
const WARN: &str = "\x1b[38;2;247;173;49m";
const ERR: &str = "\x1b[38;2;242;90;90m";

/// Graceful Ctrl+C exit (parity with the Python TUI's KeyboardInterrupt handling).
fn install_ctrlc_handler() {
    #[cfg(windows)]
    unsafe {
        #[link(name = "kernel32")]
        extern "system" {
            fn SetConsoleCtrlHandler(
                handler: Option<unsafe extern "system" fn(u32) -> i32>,
                add: i32,
            ) -> i32;
        }
        unsafe extern "system" fn handler(_ctrl_type: u32) -> i32 {
            println!("\n  Сеанс WeightX завершён.");
            std::process::exit(0);
        }
        SetConsoleCtrlHandler(Some(handler), 1);
    }
}

fn render_header(session: &StudioSession) {
    let (held, diff, dirty) = match &session.harness {
        Some(h) => {
            let d = h.workspace.diff();
            (
                format!("{:.4} (base {:.4})", h.vespasian.best_score, h.vespasian.baseline_score),
                format!("{}/{}", d.changed_params, d.total_params),
                d.changed_params > 0,
            )
        }
        None => ("ready".to_string(), "0".to_string(), false),
    };
    let brain = session.config.provider_kind.split(' ').next().unwrap_or("autonomous");
    let top = c(MUTED, "╭──────────────────────────────────────────────────────────────────────╮");
    let bot = c(MUTED, "╰──────────────────────────────────────────────────────────────────────╯");
    let mark = c(ACCENT_BOLD, "◈ WeightX");
    let wsp = c(ACCENT, "[WSP]");
    let sub = c(MUTED, "Weight-Space Programming Harness · GPLv3");
    let bar = c(MUTED, "│");
    let diff_str = if dirty { c(WARN, &diff) } else { c(OK, &diff) };
    println!("{top}");
    println!("  {mark}  {wsp}  {sub}");
    println!(
        "  {} {}  {}  {} {}  {}  {} {}  {}  {} {}",
        c(MUTED, "модель:"),
        c(FG_BOLD, &session.config.model_id),
        bar,
        c(MUTED, "мозг:"),
        c(ACCENT, brain),
        bar,
        c(MUTED, "held-out:"),
        c(FG_BOLD, &held),
        bar,
        c(MUTED, "Δ:"),
        diff_str
    );
    println!("{bot}");
}

fn render_skill_pills() {
    println!("\n{}", c(FG_BOLD, "Скиллы WSP (введите номер 1–8, /skill <имя> или задачу):"));
    for pair in WSP_SKILLS.chunks(2) {
        let mut row = String::from("  ");
        for (idx_in_pair, sk) in pair.iter().enumerate() {
            let num = WSP_SKILLS.iter().position(|s| s.skill_id == sk.skill_id).unwrap_or(0) + 1;
            let styled = format!(
                "{} {:<20} {}",
                c(ACCENT_BOLD, &format!("[{num}]")),
                c(FG_BOLD, sk.pill_label),
                c(MUTED, &format!("/{:<18}", sk.skill_id))
            );
            row.push_str(&styled);
            if idx_in_pair == 0 {
                row.push_str("  ");
            }
        }
        println!("{row}");
    }
    println!(
        "\n  {}  {}\n",
        c(MUTED, "команды:"),
        c(ACCENT, "/skills  /catalog  /load <id>  /brain <kind>  /recon  /kmap  /diff  /export <path>  /status  /clear  /quit")
    );
}

fn render_skills_table() {
    println!("\n{}", c(ACCENT_BOLD, "WeightX WSP Skills (8 встроенных скиллов)"));
    for (i, sk) in WSP_SKILLS.iter().enumerate() {
        println!(
            "  {}  {}  {}  {}",
            c(ACCENT_BOLD, &format!("[{}]", i + 1)),
            c(FG_BOLD, &format!("{:<22}", sk.pill_label)),
            c(ACCENT, &format!("/skill {:<17}", sk.skill_id)),
            c(MUTED, sk.short_hint)
        );
    }
    println!();
}

fn render_catalog(session: &mut StudioSession) {
    println!("\n{}", c(ACCENT_BOLD, "Каталог моделей WeightX"));
    for e in session.catalog.list_entries() {
        let state = if e.local_cached {
            c(OK, "● локально")
        } else {
            c(WARN, "○ HF Hub")
        };
        println!(
            "  {} {} {} {:>8}  {:>3} слоя  {}",
            c(FG_BOLD, &format!("{:<22}", e.model_id)),
            c(MUTED, &format!("{:<36}", e.hf_repo_id)),
            c(ACCENT, &format!("{:<10}", e.family)),
            e.params_human,
            e.num_layers,
            state
        );
    }
    println!();
}

fn format_output_line(line: &str) -> String {
    let t = line.trim_start();
    if t.starts_with("## ") {
        return c(ACCENT_BOLD, line);
    }
    if t.contains("ROLLED BACK") || t.starts_with("✗") || t.starts_with("Ошибка") {
        return c(ERR, line);
    }
    if t.contains("COMMITTED") || t.starts_with("✓") {
        return c(OK, line);
    }
    line.to_string()
}

fn print_execution_block(tag: &str, title: &str, output: &str, elapsed_ms: f64) {
    println!(
        "  {} {} {} {}",
        c(MUTED, "┌─"),
        c(ACCENT_BOLD, tag),
        c(FG_BOLD, &format!("· {title}")),
        c(MUTED, &format!("({elapsed_ms:.0} ms)"))
    );
    let pipe = c(MUTED, "│");
    for line in output.trim().lines() {
        println!("  {pipe}  {}", format_output_line(line));
    }
    println!("  {} {}\n", c(MUTED, "└─"), c(OK, "готово"));
}

pub fn run_interactive_cli(initial_model: &str) -> i32 {
    init_terminal_style();
    install_ctrlc_handler();
    let mut session = make_session();
    session.set_log(std::sync::Arc::new(|msg: &str| {
        println!("  {} {}", c(MUTED, "├─"), c(MUTED, msg));
    }));
    match session.load_model(initial_model) {
        Ok(msg) => println!("  {} {}", c(OK, "●"), c(FG_BOLD, &msg)),
        Err(e) => {
            eprintln!("{} {}", c(ERR, "Ошибка загрузки модели:"), e);
            return 1;
        }
    }
    render_header(&session);
    render_skill_pills();

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("  {} ", c(ACCENT_BOLD, "wx ❯"));
        let _ = std::io::stdout().flush();
        let raw = match lines.next() {
            Some(Ok(l)) => l.trim().to_string(),
            _ => {
                println!("\n  Сеанс WeightX завершён.");
                break;
            }
        };
        if raw.is_empty() {
            continue;
        }
        let mut raw = raw;
        let tokens: Vec<&str> = raw.splitn(2, char::is_whitespace).collect();
        if tokens[0].eq_ignore_ascii_case("wx") || tokens[0].eq_ignore_ascii_case("weightx") {
            if tokens.len() == 1 {
                render_header(&session);
                render_skill_pills();
                continue;
            }
            raw = tokens[1].trim().to_string();
        }
        let cmd_low = raw.to_lowercase();

        if matches!(cmd_low.as_str(), "/quit" | "/exit" | "exit" | "quit" | ":q") {
            println!("  Сеанс WeightX завершён.");
            break;
        }
        if matches!(cmd_low.as_str(), "/clear" | "clear" | "cls") {
            print!("\x1b[2J\x1b[H");
            render_header(&session);
            render_skill_pills();
            continue;
        }
        if matches!(cmd_low.as_str(), "/help" | "help" | "?" | "/prompts" | "/skills" | "skills") {
            render_skills_table();
            continue;
        }
        if matches!(cmd_low.as_str(), "/catalog" | "/models" | "catalog" | "models") {
            render_catalog(&mut session);
            continue;
        }
        if matches!(cmd_low.as_str(), "/status" | "status") {
            render_header(&session);
            continue;
        }
        if matches!(cmd_low.as_str(), "/diff" | "diff") {
            if let Some(h) = &session.harness {
                let d = h.workspace.diff();
                print_execution_block("diff", "workspace parameter diff", &toon::dumps_toon(&d.to_value()), 1.0);
            }
            continue;
        }
        if matches!(cmd_low.as_str(), "/kmap" | "kmap" | "/knowledge-map") {
            let t0 = Instant::now();
            match session.run_skill("knowledge-map", "") {
                Ok(res) => print_execution_block(
                    "/skill knowledge-map",
                    "Карта знаний",
                    &res.to_markdown(),
                    t0.elapsed().as_secs_f64() * 1000.0,
                ),
                Err(e) => eprintln!("Ошибка: {e}"),
            }
            continue;
        }
        if ["/load ", "/model ", "load ", "model "].iter().any(|p| cmd_low.starts_with(p)) {
            let target = raw.splitn(2, char::is_whitespace).nth(1).unwrap_or("").trim();
            let t0 = Instant::now();
            match session.load_model(target) {
                Ok(msg) => {
                    print_execution_block("load", target, &msg, t0.elapsed().as_secs_f64() * 1000.0);
                    render_header(&session);
                }
                Err(e) => eprintln!("Ошибка: {e}"),
            }
            continue;
        }
        if cmd_low.starts_with("/provider") || cmd_low.starts_with("/brain") || cmd_low.starts_with("brain") {
            let parts: Vec<&str> = raw.split_whitespace().collect();
            if parts.len() == 1 {
                println!("  провайдеры: {}\n", PROVIDER_KINDS.join(", "));
            } else {
                let kind = parts[1];
                let model_name = parts.get(2).copied().unwrap_or("");
                match session.set_provider(kind, model_name, "", "") {
                    Ok(msg) => print_execution_block("brain", kind, &msg, 1.0),
                    Err(e) => eprintln!("Ошибка: {e}"),
                }
            }
            continue;
        }
        if matches!(cmd_low.as_str(), "/recon" | "recon") {
            let t0 = Instant::now();
            let out = session.run_action("recon_profile", "");
            print_execution_block("recon_profile", "L1 AutoArch + L2 Uni-Layer", &out, t0.elapsed().as_secs_f64() * 1000.0);
            continue;
        }
        if cmd_low == "/export" || cmd_low == "export" || cmd_low.starts_with("/export ") || cmd_low.starts_with("export ") {
            let path = raw
                .splitn(2, char::is_whitespace)
                .nth(1)
                .unwrap_or("checkpoints/exported_model.safetensors")
                .trim();
            let t0 = Instant::now();
            match session.export_checkpoint(path) {
                Ok(out) => print_execution_block("export", path, &out, t0.elapsed().as_secs_f64() * 1000.0),
                Err(e) => eprintln!("Ошибка: {e}"),
            }
            continue;
        }

        let (sk_match, rem_args, explicit) = parse_skill_command(&raw);
        if explicit && sk_match.is_none() {
            if rem_args.is_empty() {
                render_skills_table();
            } else {
                println!("  Скилл не найден: {rem_args:?}. Введите `/` или `/skills` для списка.\n");
                render_skills_table();
            }
            continue;
        }
        if let Some(sk) = sk_match {
            let t0 = Instant::now();
            let args = if rem_args.is_empty() { raw.clone() } else { rem_args };
            match session.run_skill(sk.skill_id, &args) {
                Ok(res) => {
                    print_execution_block(
                        &format!("/skill {}", sk.skill_id),
                        sk.pill_label,
                        &res.to_markdown(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                    );
                    render_header(&session);
                }
                Err(e) => eprintln!("Ошибка: {e}"),
            }
            continue;
        }
        if raw.starts_with('/') {
            println!("  Неизвестная команда: {raw}");
            println!("  Введите `/` или `/skills` для просмотра всех скиллов.\n");
            continue;
        }

        // Free-form natural language WSP editing cycle.
        let t0 = Instant::now();
        let out = session.run_action("chat", &raw);
        let title: String = raw.chars().take(60).collect();
        print_execution_block("wsp_cycle", &title, &out, t0.elapsed().as_secs_f64() * 1000.0);
        render_header(&session);
    }
    0
}