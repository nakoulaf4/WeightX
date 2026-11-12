# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX Interactive CLI / TUI (`wx` / `weightx` / `wx chat`).

Minimalist 1:1 DSH (DeepSeek Harness) terminal aesthetic:
  - DSH neutral-bluish monochrome palette with `#7aaaff` focal accents and `#4ed17e` status badges
  - Ultra-compact 2-word WSP Skill Pills (`01`..`08` or `/<skill-id>`)
  - Step-by-step live WSP skill pipeline traces (`✓ [1/5] ...`)
  - Case-insensitive command & skill resolution (`wx`, `WX`, `WeightX`, `/CODING-BOOST`, etc.)
"""

from __future__ import annotations

import time
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from weightx.skills import WSP_SKILLS, resolve_skill
from weightx.studio import PROVIDER_KINDS, StudioSession

DSW_PRIMARY = "#F9FAFB"
DSW_SECONDARY = "#CFD3D6"
DSW_TERTIARY = "#ADB2B8"
DSW_CAPTION = "#81858C"
DSW_BORDER = "#353638"
DSW_ACCENT = "#7AAAFF"
DSW_SUCCESS = "#4ED17E"
DSW_WARN = "#F7AD31"
DSW_ERROR = "#F25A5A"


def render_dsh_header(console: Console, session: StudioSession) -> None:
    """Render a minimalist DSH-style Hero header & workspace status line."""
    v = session.harness.vespasian if session.harness else None
    d = session.harness.workspace.diff() if session.harness else None
    held_str = f"{v.best_score:.4f} (base {v.baseline_score:.4f})" if v else "ready"
    diff_str = f"{d.changed_params:,}/{d.total_params:,}" if d else "0"
    brain_str = session.config.provider_kind.split(" ")[0]

    header_text = (
        f"[bold {DSW_PRIMARY}]  ◈  WeightX[/bold {DSW_PRIMARY}]  "
        f"[bold {DSW_ACCENT}]WSP[/bold {DSW_ACCENT}]  "
        f"[{DSW_CAPTION}]Weight-Space Programming Harness · GPLv3[/{DSW_CAPTION}]\n"
        f"  [{DSW_CAPTION}]📁[/{DSW_CAPTION}] [bold {DSW_SECONDARY}]{session.config.model_id}[/bold {DSW_SECONDARY}]  "
        f"[{DSW_CAPTION}]·  мозг:[/{DSW_CAPTION}] [{DSW_TERTIARY}]{brain_str}[/{DSW_TERTIARY}]  "
        f"[{DSW_CAPTION}]·  held-out:[/{DSW_CAPTION}] [{DSW_SUCCESS}]{held_str}[/{DSW_SUCCESS}]  "
        f"[{DSW_CAPTION}]·  Δ:[/{DSW_CAPTION}] [{DSW_TERTIARY}]{diff_str}[/{DSW_TERTIARY}]"
    )
    console.print(
        Panel(
            header_text,
            box=box.ROUNDED,
            border_style=DSW_BORDER,
            padding=(0, 1),
        )
    )


def render_dsh_skill_pills(console: Console) -> None:
    """Render DSH-style ultra-compact 2-word Skill Pills in a clean 4-column strip."""
    grid = Table.grid(expand=True, padding=(0, 1))
    for _ in range(4):
        grid.add_column(ratio=1)

    pills: list[str] = []
    for idx, sk in enumerate(WSP_SKILLS, start=1):
        pills.append(
            f"[{DSW_CAPTION}]{idx}[/{DSW_CAPTION}] "
            f"[bold {DSW_SECONDARY}]( {sk.pill_label} )[/bold {DSW_SECONDARY}] "
            f"[{DSW_CAPTION}]/{sk.skill_id}[/{DSW_CAPTION}]"
        )

    for i in range(0, len(pills), 4):
        row = pills[i : i + 4]
        while len(row) < 4:
            row.append("")
        grid.add_row(*row)

    console.print(
        Panel(
            grid,
            title=f"[bold {DSW_SECONDARY}]Навыки WSP[/bold {DSW_SECONDARY}] [{DSW_CAPTION}](введите 1..8, /навык или двухсловное название)[/{DSW_CAPTION}]",
            title_align="left",
            box=box.ROUNDED,
            border_style=DSW_BORDER,
            padding=(0, 1),
        )
    )
    console.print(
        f"  [{DSW_CAPTION}]команды:[/{DSW_CAPTION}] [{DSW_TERTIARY}]/skills  /catalog  /load <id>  /brain <kind>  "
        f"/recon  /kmap  /diff  /export <path>  /status  /clear  /quit[/{DSW_TERTIARY}]\n"
    )


def render_skills_table(console: Console) -> None:
    table = Table(
        title=f"[bold {DSW_PRIMARY}]WeightX WSP Skills (Навыки)[/bold {DSW_PRIMARY}]",
        title_justify="left",
        box=box.ROUNDED,
        header_style=f"bold {DSW_ACCENT}",
        border_style=DSW_BORDER,
    )
    table.add_column("#", style=DSW_CAPTION, justify="right")
    table.add_column("Плашка (2 слова)", style=f"bold {DSW_PRIMARY}")
    table.add_column("Команда", style=f"bold {DSW_ACCENT}")
    table.add_column("Скрытый WSP-пайплайн", style=DSW_SECONDARY)

    for idx, sk in enumerate(WSP_SKILLS, start=1):
        table.add_row(str(idx), sk.pill_label, f"/{sk.skill_id}", sk.short_hint)
    console.print(table)


def render_catalog_table(console: Console, session: StudioSession) -> None:
    table = Table(
        title=f"[bold {DSW_PRIMARY}]Каталог моделей WeightX[/bold {DSW_PRIMARY}]",
        title_justify="left",
        box=box.ROUNDED,
        header_style=f"bold {DSW_ACCENT}",
        border_style=DSW_BORDER,
    )
    table.add_column("ID", style=f"bold {DSW_PRIMARY}")
    table.add_column("Источник", style=DSW_SECONDARY)
    table.add_column("Семейство", style=DSW_TERTIARY)
    table.add_column("Параметры", justify="right", style=DSW_PRIMARY)
    table.add_column("Слои", justify="right", style=DSW_TERTIARY)
    table.add_column("Статус", justify="center")

    for e in session.catalog.list_entries():
        state = f"[{DSW_SUCCESS}]● локально[/{DSW_SUCCESS}]" if e.local_cached else f"[{DSW_CAPTION}]○ HF Hub[/{DSW_CAPTION}]"
        table.add_row(
            e.model_id,
            e.hf_repo_id,
            e.family,
            e.params_human,
            str(e.num_layers),
            state,
        )
    console.print(table)


def _print_execution_block(console: Console, tag: str, title: str, output: str, elapsed_ms: float) -> None:
    console.print(
        f"  [{DSW_ACCENT}]┌─[/{DSW_ACCENT}] [bold {DSW_PRIMARY}]{tag}[/bold {DSW_PRIMARY}] "
        f"[{DSW_CAPTION}]· {title} ({elapsed_ms:.0f} ms)[/{DSW_CAPTION}]"
    )
    for line in output.strip().splitlines():
        if any(w in line for w in ("ACCEPTED", "COMMITTED", "ПРИНЯТО")):
            console.print(f"  [{DSW_ACCENT}]│[/{DSW_ACCENT}]  [{DSW_SUCCESS}]{line}[/{DSW_SUCCESS}]")
        elif any(w in line for w in ("REJECTED", "ROLLBACK", "ОТКАТ")):
            console.print(f"  [{DSW_ACCENT}]│[/{DSW_ACCENT}]  [{DSW_WARN}]{line}[/{DSW_WARN}]")
        elif line.startswith("###") or line.startswith("##"):
            console.print(f"  [{DSW_ACCENT}]├─[/{DSW_ACCENT}] [bold {DSW_ACCENT}]{line.lstrip('# ').strip()}[/bold {DSW_ACCENT}]")
        else:
            console.print(f"  [{DSW_ACCENT}]│[/{DSW_ACCENT}]  [{DSW_SECONDARY}]{line}[/{DSW_SECONDARY}]")
    console.print(f"  [{DSW_ACCENT}]└─[/{DSW_ACCENT}] [{DSW_CAPTION}]готово[/{DSW_CAPTION}]\n")


def run_interactive_cli(initial_model: str = "minillama-demo") -> int:
    """Launch the minimalist DSH-styled interactive WeightX CLI (`wx` / `weightx`)."""
    console = Console()
    session = StudioSession(log=lambda msg: console.print(f"  [{DSW_CAPTION}]├─ {msg}[/{DSW_CAPTION}]"))
    session.load_model(initial_model)

    render_dsh_header(console, session)
    render_dsh_skill_pills(console)

    while True:
        try:
            raw = console.input(f"  [bold {DSW_PRIMARY}]wx[/bold {DSW_PRIMARY}] [{DSW_ACCENT}]❯[/{DSW_ACCENT}] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n  [{DSW_CAPTION}]Сеанс WeightX завершён.[/{DSW_CAPTION}]")
            break

        if not raw:
            continue

        # Allow typing `wx ...` or `weightx ...` in any case inside the REPL too!
        tokens = raw.split(None, 1)
        if tokens[0].lower() in ("wx", "weightx"):
            if len(tokens) == 1:
                render_dsh_header(console, session)
                render_dsh_skill_pills(console)
                continue
            raw = tokens[1].strip()

        cmd_low = raw.lower()

        if cmd_low in ("/quit", "/exit", "exit", "quit", ":q"):
            console.print(f"  [{DSW_CAPTION}]Сеанс WeightX завершён.[/{DSW_CAPTION}]")
            break

        if cmd_low in ("/clear", "clear", "cls"):
            console.clear()
            render_dsh_header(console, session)
            render_dsh_skill_pills(console)
            continue

        if cmd_low in ("/help", "help", "?", "/prompts", "/skills", "skills"):
            render_skills_table(console)
            continue

        if cmd_low in ("/catalog", "/models", "catalog", "models"):
            render_catalog_table(console, session)
            continue

        if cmd_low in ("/status", "status"):
            render_dsh_header(console, session)
            continue

        if cmd_low in ("/diff", "diff"):
            if session.harness:
                d = session.harness.workspace.diff()
                _print_execution_block(console, "diff", "workspace parameter diff", d.to_toon(), 1.0)
            continue

        if cmd_low.startswith(("/load ", "/model ", "load ", "model ")):
            target_id = raw.split(None, 1)[1].strip()
            t0 = time.perf_counter()
            msg = session.load_model(target_id)
            _print_execution_block(console, "load", target_id, msg, (time.perf_counter() - t0) * 1000)
            render_dsh_header(console, session)
            continue

        if cmd_low.startswith(("/provider", "/brain", "brain")):
            parts = raw.split()
            if len(parts) == 1:
                console.print(f"  [{DSW_CAPTION}]провайдеры:[/{DSW_CAPTION}] [{DSW_SECONDARY}]{', '.join(PROVIDER_KINDS)}[/{DSW_SECONDARY}]\n")
            else:
                kind = parts[1]
                model_name = parts[2] if len(parts) > 2 else ""
                msg = session.set_provider(kind, model=model_name)
                _print_execution_block(console, "brain", kind, msg, 1.0)
            continue

        if cmd_low in ("/recon", "recon"):
            t0 = time.perf_counter()
            out = session.run_action("recon_profile", "")
            _print_execution_block(console, "recon_profile", "L1 AutoArch + L2 Uni-Layer", out, (time.perf_counter() - t0) * 1000)
            continue

        if cmd_low in ("/export", "export") or cmd_low.startswith(("/export ", "export ")):
            parts = raw.split(None, 1)
            path = parts[1] if len(parts) > 1 else "checkpoints/exported_model.safetensors"
            t0 = time.perf_counter()
            out = session.export_checkpoint(path)
            _print_execution_block(console, "export", path, out, (time.perf_counter() - t0) * 1000)
            continue

        # Numbered skill pill (1..8 or 01..08)
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(WSP_SKILLS):
                sk = WSP_SKILLS[idx]
                t0 = time.perf_counter()
                res = session.run_skill(sk.skill_id)
                _print_execution_block(
                    console,
                    f"/{sk.skill_id}",
                    sk.pill_label,
                    res.to_markdown(),
                    (time.perf_counter() - t0) * 1000,
                )
                render_dsh_header(console, session)
                continue

        # Skill invocation by `/skill-id` or 2-word pill label (case-insensitive)
        sk_match = resolve_skill(raw)
        if sk_match is not None:
            t0 = time.perf_counter()
            res = session.run_skill(sk_match.skill_id, user_args=raw)
            _print_execution_block(
                console,
                f"/{sk_match.skill_id}",
                sk_match.pill_label,
                res.to_markdown(),
                (time.perf_counter() - t0) * 1000,
            )
            render_dsh_header(console, session)
            continue

        # Free-form natural language WSP editing cycle
        t0 = time.perf_counter()
        out = session.run_action("chat", raw)
        _print_execution_block(console, "wsp_cycle", raw[:60], out, (time.perf_counter() - t0) * 1000)
        render_dsh_header(console, session)

    return 0
