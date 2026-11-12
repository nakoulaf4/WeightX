# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX Interactive CLI / TUI (`wx` / `weightx` / `wx chat`).

Minimalist 1:1 DSH (DeepSeek Harness) terminal aesthetic with live slash autocomplete:
  - DSH neutral-bluish monochrome palette with `#7aaaff` focal accents and `#4ed17e` status badges
  - Live interactive `/` autocomplete dropdown (`prompt_toolkit`) as soon as you type `/`, `/sk`, `/skill ...`, `/load ...`, `/brain ...`
  - Ultra-compact 2-word WSP Skill Pills invoked via `/skill <название>` (or `1..8`, `/<id>`, pill name)
  - Case-insensitive command & skill resolution (`wx`, `WX`, `WeightX`, `/SKILL CODING-BOOST`)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style as PTStyle
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from weightx.skills import WSP_SKILLS, parse_skill_command, resolve_skill
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

REPL_SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/skills", "Список всех 8 скиллов WSP и их пайплайнов"),
    ("/catalog", "Каталог моделей WeightX (локальные + HuggingFace)"),
    ("/load ", "Загрузить модель из каталога (/load <model_id>)"),
    ("/brain ", "Выбрать LLM-провайдер (auto, openai, claude, codex, ollama…)"),
    ("/recon", "L1 AutoArch + L2 Uni-Layer диагностика текущей модели"),
    ("/kmap", "Построить Logit-Lens карту знаний (Gini KCI)"),
    ("/diff", "Показать изменённые тензоры в рабочем пространстве (TOON)"),
    ("/export ", "Экспортировать веса в .safetensors"),
    ("/status", "Состояние активной модели и метрики Vespasian"),
    ("/clear", "Очистить экран и показать стартовые скиллы"),
    ("/quit", "Выйти из WeightX CLI"),
)


@dataclass(frozen=True)
class CLISuggestion:
    """Single autocomplete suggestion for the interactive WeightX CLI."""

    text: str
    display: str
    meta: str
    replace_len: int


def get_cli_suggestions(
    line_before_cursor: str,
    model_ids: Iterable[str] = (),
    provider_kinds: Iterable[str] = PROVIDER_KINDS,
) -> list[CLISuggestion]:
    """Return filtered `/` autocomplete hints for `/skill <name>`, `/load <model>`, `/brain <kind>`, and commands."""
    raw = line_before_cursor.lstrip()
    low = raw.lower()
    if not low:
        return []

    # 1. Contextual argument completion after `/load <model_prefix>`
    if low.startswith(("/load ", "/model ", "load ", "model ")):
        prefix = raw.split(None, 1)[1] if len(raw.split(None, 1)) > 1 else ""
        p_low = prefix.lower()
        out: list[CLISuggestion] = []
        for mid in model_ids:
            if not p_low or p_low in mid.lower():
                out.append(
                    CLISuggestion(
                        text=mid,
                        display=mid,
                        meta="Модель из каталога WeightX",
                        replace_len=len(prefix),
                    )
                )
        return out

    # 2. Contextual completion after `/brain <provider_prefix>` or `/provider <prefix>`
    if low.startswith(("/brain ", "/provider ", "brain ")):
        prefix = raw.split(None, 1)[1] if len(raw.split(None, 1)) > 1 else ""
        p_low = prefix.lower()
        out = []
        for kind in provider_kinds:
            short = kind.split()[0].lower()
            if not p_low or p_low in kind.lower() or p_low in short:
                out.append(
                    CLISuggestion(
                        text=short,
                        display=f"{short}",
                        meta=kind,
                        replace_len=len(prefix),
                    )
                )
        return out

    # Only trigger auto-suggestions cleanly when user types `/` or `skill`
    if not (low.startswith("/") or low.startswith("skill")):
        return []

    replace_len = len(raw)
    suggestions: list[CLISuggestion] = []

    # 3. Always surface `/skill <skill_id>` entries matching `/`, `/sk`, `/skill ...`, `/<alias>`, or pill name
    if low.startswith(("/skill", "skill")):
        arg = low[6:].strip() if low.startswith("/skill") else low[5:].strip()
        arg = arg.lstrip(":").strip()
        for idx, sk in enumerate(WSP_SKILLS, start=1):
            haystack = " ".join(
                (sk.skill_id, sk.pill_label, sk.short_hint, *sk.aliases)
            ).lower()
            if not arg or arg in haystack or str(idx) == arg:
                suggestions.append(
                    CLISuggestion(
                        text=f"/skill {sk.skill_id}",
                        display=f"/skill {sk.skill_id}",
                        meta=f"{sk.pill_label} · {sk.short_hint}",
                        replace_len=replace_len,
                    )
                )
        return suggestions

    # 4. General `/` prefix: combine `/skill <id>` skills first, then built-in slash commands
    q = low.lstrip("/")
    for sk in WSP_SKILLS:
        cmd_full = f"/skill {sk.skill_id}"
        haystack = f"skill {sk.skill_id} {sk.pill_label} {' '.join(sk.aliases)}".lower()
        if not q or q in haystack:
            suggestions.append(
                CLISuggestion(
                    text=cmd_full,
                    display=cmd_full,
                    meta=f"{sk.pill_label} · {sk.short_hint}",
                    replace_len=replace_len,
                )
            )

    for cmd, desc in REPL_SLASH_COMMANDS:
        if not q or q in cmd.lower() or q in desc.lower():
            suggestions.append(
                CLISuggestion(
                    text=cmd,
                    display=cmd.strip(),
                    meta=desc,
                    replace_len=replace_len,
                )
            )
    return suggestions


class WeightXSlashCompleter(Completer):
    """Prompt-toolkit completer powered by `get_cli_suggestions`."""

    def __init__(self, session: StudioSession) -> None:
        self.session = session

    def get_completions(self, document, complete_event):  # type: ignore[override]
        model_ids = [e.model_id for e in self.session.catalog.list_entries()]
        items = get_cli_suggestions(
            document.text_before_cursor,
            model_ids=model_ids,
            provider_kinds=PROVIDER_KINDS,
        )
        for item in items:
            yield Completion(
                text=item.text,
                start_position=-item.replace_len,
                display=item.display,
                display_meta=item.meta[:68],
            )


def _dsh_prompt_style() -> PTStyle:
    return PTStyle.from_dict(
        {
            "completion-menu": "bg:#1b1b1c #cfd3d6",
            "completion-menu.completion": "bg:#1b1b1c #cfd3d6",
            "completion-menu.completion.current": "bg:#2c2c2e #7aaaff bold",
            "completion-menu.meta.completion": "bg:#1b1b1c #81858c",
            "completion-menu.meta.completion.current": "bg:#2c2c2e #f9fafb",
            "scrollbar.background": "bg:#1b1b1c",
            "scrollbar.button": "bg:#353638",
            "bottom-toolbar": "bg:#151517 #81858c",
        }
    )


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
    """Render DSH-style ultra-compact 2-word Skill Pills in a clean grid."""
    grid = Table.grid(expand=True, padding=(0, 1))
    for _ in range(2):
        grid.add_column(ratio=1)

    pills: list[str] = []
    for idx, sk in enumerate(WSP_SKILLS, start=1):
        pills.append(
            f"[{DSW_CAPTION}]{idx}[/{DSW_CAPTION}] "
            f"[bold {DSW_SECONDARY}]( {sk.pill_label} )[/bold {DSW_SECONDARY}]  "
            f"[{DSW_ACCENT}]/skill {sk.skill_id}[/{DSW_ACCENT}]"
        )

    for i in range(0, len(pills), 2):
        row = pills[i : i + 2]
        while len(row) < 2:
            row.append("")
        grid.add_row(*row)

    console.print(
        Panel(
            grid,
            title=(
                f"[bold {DSW_SECONDARY}]Скиллы WSP (Skills)[/bold {DSW_SECONDARY}] "
                f"[{DSW_CAPTION}](начните ввод с / для автодополнения или /skill <название>)[/{DSW_CAPTION}]"
            ),
            title_align="left",
            box=box.ROUNDED,
            border_style=DSW_BORDER,
            padding=(0, 1),
        )
    )
    console.print(
        f"  [{DSW_CAPTION}]команды:[/{DSW_CAPTION}] [{DSW_TERTIARY}]/skill <имя>  /skills  /catalog  /load <id>  "
        f"/brain <kind>  /recon  /kmap  /diff  /export <path>  /status  /clear  /quit[/{DSW_TERTIARY}]\n"
    )


def render_skills_table(console: Console) -> None:
    table = Table(
        title=f"[bold {DSW_PRIMARY}]WeightX WSP Skills (Скиллы)[/bold {DSW_PRIMARY}]",
        title_justify="left",
        box=box.ROUNDED,
        header_style=f"bold {DSW_ACCENT}",
        border_style=DSW_BORDER,
    )
    table.add_column("#", style=DSW_CAPTION, justify="right")
    table.add_column("Плашка (2 слова)", style=f"bold {DSW_PRIMARY}")
    table.add_column("Команда вызова", style=f"bold {DSW_ACCENT}")
    table.add_column("Скрытый WSP-пайплайн", style=DSW_SECONDARY)

    for idx, sk in enumerate(WSP_SKILLS, start=1):
        table.add_row(str(idx), sk.pill_label, f"/skill {sk.skill_id}", sk.short_hint)
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
        state = (
            f"[{DSW_SUCCESS}]● локально[/{DSW_SUCCESS}]"
            if e.local_cached
            else f"[{DSW_CAPTION}]○ HF Hub[/{DSW_CAPTION}]"
        )
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
            console.print(
                f"  [{DSW_ACCENT}]├─[/{DSW_ACCENT}] [bold {DSW_ACCENT}]{line.lstrip('# ').strip()}[/bold {DSW_ACCENT}]"
            )
        else:
            console.print(f"  [{DSW_ACCENT}]│[/{DSW_ACCENT}]  [{DSW_SECONDARY}]{line}[/{DSW_SECONDARY}]")
    console.print(f"  [{DSW_ACCENT}]└─[/{DSW_ACCENT}] [{DSW_CAPTION}]готово[/{DSW_CAPTION}]\n")


def _create_prompt_session(session: StudioSession) -> PromptSession[str] | None:
    """Create an interactive `prompt_toolkit` session with live `/` dropdowns when running in a real TTY."""
    if not (sys.stdin and sys.stdout and sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        pt_session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            completer=WeightXSlashCompleter(session),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            style=_dsh_prompt_style(),
            reserve_space_for_menu=7,
        )
        return pt_session
    except Exception:
        return None


def _bottom_toolbar(pt_session: PromptSession[str], session: StudioSession) -> HTML:
    buf_text = pt_session.default_buffer.text.strip()
    model_ids = [e.model_id for e in session.catalog.list_entries()]
    hits = get_cli_suggestions(buf_text, model_ids=model_ids)
    if hits:
        preview = hits[0]
        return HTML(
            f"<style fg='#7aaaff'>  ✦ Подсказка:</style> "
            f"<style fg='#f9fafb'>{preview.display}</style> "
            f"<style fg='#81858c'>— {preview.meta[:64]}  (↑↓ выбор · Tab заполнить · Enter запуск)</style>"
        )
    return HTML(
        "<style fg='#81858c'>  Подсказка: введите <style fg='#7aaaff'>/</style> для меню команд и скиллов · "
        "<style fg='#cfd3d6'>/skill &lt;название&gt;</style> · ↑↓ история</style>"
    )


def run_interactive_cli(initial_model: str = "minillama-demo") -> int:
    """Launch the minimalist DSH-styled interactive WeightX CLI (`wx` / `weightx`)."""
    console = Console()
    session = StudioSession(log=lambda msg: console.print(f"  [{DSW_CAPTION}]├─ {msg}[/{DSW_CAPTION}]"))
    session.load_model(initial_model)

    render_dsh_header(console, session)
    render_dsh_skill_pills(console)

    pt_session = _create_prompt_session(session)

    while True:
        try:
            if pt_session is not None:
                raw = pt_session.prompt(
                    HTML("<style fg='#f9fafb'><b>  wx</b></style> <style fg='#7aaaff'>❯ </style>"),
                    bottom_toolbar=lambda: _bottom_toolbar(pt_session, session),
                ).strip()
            else:
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

        if cmd_low in ("/kmap", "kmap", "/knowledge-map"):
            t0 = time.perf_counter()
            res = session.run_skill("knowledge-map")
            _print_execution_block(console, "/skill knowledge-map", "Карта знаний", res.to_markdown(), (time.perf_counter() - t0) * 1000)
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
                console.print(
                    f"  [{DSW_CAPTION}]провайдеры:[/{DSW_CAPTION}] [{DSW_SECONDARY}]{', '.join(PROVIDER_KINDS)}[/{DSW_SECONDARY}]\n"
                )
            else:
                kind = parts[1]
                model_name = parts[2] if len(parts) > 2 else ""
                msg = session.set_provider(kind, model=model_name)
                _print_execution_block(console, "brain", kind, msg, 1.0)
            continue

        if cmd_low in ("/recon", "recon"):
            t0 = time.perf_counter()
            out = session.run_action("recon_profile", "")
            _print_execution_block(
                console, "recon_profile", "L1 AutoArch + L2 Uni-Layer", out, (time.perf_counter() - t0) * 1000
            )
            continue

        if cmd_low in ("/export", "export") or cmd_low.startswith(("/export ", "export ")):
            parts = raw.split(None, 1)
            path = parts[1] if len(parts) > 1 else "checkpoints/exported_model.safetensors"
            t0 = time.perf_counter()
            out = session.export_checkpoint(path)
            _print_execution_block(console, "export", path, out, (time.perf_counter() - t0) * 1000)
            continue

        # `/skill` / `/skill <name>` / shorthand `/<id>` / `1..8` / pill label
        sk_match, rem_args, explicit_skill_cmd = parse_skill_command(raw)
        if explicit_skill_cmd and sk_match is None:
            if not rem_args:
                render_skills_table(console)
            else:
                console.print(
                    f"  [{DSW_WARN}]Скилл не найден:[/{DSW_WARN}] {rem_args!r}. "
                    f"[{DSW_CAPTION}]Введите `/` или `/skills` для списка.[/{DSW_CAPTION}]\n"
                )
                render_skills_table(console)
            continue

        if sk_match is not None:
            t0 = time.perf_counter()
            res = session.run_skill(sk_match.skill_id, user_args=rem_args or raw)
            _print_execution_block(
                console,
                f"/skill {sk_match.skill_id}",
                sk_match.pill_label,
                res.to_markdown(),
                (time.perf_counter() - t0) * 1000,
            )
            render_dsh_header(console, session)
            continue

        # Also support unknown `/` command gracefully with live nearest hints instead of treating as LLM goal
        if raw.startswith("/"):
            near = get_cli_suggestions(raw, model_ids=[e.model_id for e in session.catalog.list_entries()])
            console.print(f"  [{DSW_WARN}]Неизвестная команда:[/{DSW_WARN}] {raw}")
            if near:
                console.print(
                    f"  [{DSW_CAPTION}]Возможно, вы имели в виду:[/{DSW_CAPTION}] "
                    + ", ".join(f"[{DSW_ACCENT}]{s.display}[/{DSW_ACCENT}]" for s in near[:4])
                )
            else:
                console.print(f"  [{DSW_CAPTION}]Введите `/` или `/skills` для просмотра всех скиллов.[/{DSW_CAPTION}]")
            console.print()
            continue

        # Free-form natural language WSP editing cycle
        t0 = time.perf_counter()
        out = session.run_action("chat", raw)
        _print_execution_block(console, "wsp_cycle", raw[:60], out, (time.perf_counter() - t0) * 1000)
        render_dsh_header(console, session)

    return 0
