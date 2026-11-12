# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX Desktop (`src/weightx/desktop.py`).

Minimalist 1:1 DSH (DeepSeek Harness) desktop interface for WeightX:
- Exact `--dsw-static-neutral-bluish-*` dark palette (`#151517` base, `#1b1b1c` sidebar,
  `#2c2c2e` rounded composer card, `#353638` selector pills, `#f9fafb` primary action).
- 3-column collapsible `AppFrame`:
  1. Left Sidebar (`256px` expanded / `56px` collapsed icon rail via `Ctrl+B` or `◧`).
  2. Uncluttered Center Surface: centered Hero (`HeroShell` + `ComposerCard` + ultra-compact
     2-word Skill Pills) that transitions into a clean Conversation & Skill Trajectory feed.
  3. On-demand Right Inspector Drawer (closed `0px` by default; toggled via `Ctrl+I` or `◨`).
"""

from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageGrab

from weightx.skills import WSP_SKILLS, SkillExecutionResult, WeightXSkill, resolve_skill
from weightx.studio import PROVIDER_KINDS, StudioSession

# ---------------------------------------------------------------------------
# Exact DSH (`@deepseek-ai/dsh-client-ui-theme`) Dark Palette Tokens
# ---------------------------------------------------------------------------
DSW_BG_BASE = "#151517"          # --dsw-alias-bg-base (--dsw-static-neutral-bluish-950)
DSW_SIDEBAR_FILL = "#1b1b1c"     # --dsw-specific-sidebar-fill (--dsw-static-neutral-bluish-900)
DSW_LAYER_1 = "#232324"          # --dsw-alias-bg-layer-1 (--dsw-static-neutral-bluish-875)
DSW_INPUT_MAJOR = "#2c2c2e"      # --dsw-specific-input-major (--dsw-static-neutral-bluish-850)
DSW_SELECTOR = "#353638"         # --dsw-specific-selector (--dsw-static-neutral-bluish-800)
DSW_NAV_ACTIVE = "#43454a"       # --dsw-specific-sidebar-nav-item-active (--dsw-static-neutral-bluish-750)
DSW_BORDER_L2 = "#2c2d30"        # --dsw-alias-border-l2 hairline
DSW_BORDER_L3 = "#3a3b40"        # --dsw-alias-border-l3

DSW_TEXT_PRIMARY = "#f9fafb"     # --dsw-alias-label-primary (--dsw-static-neutral-bluish-50)
DSW_TEXT_SECONDARY = "#cfd3d6"   # --dsw-alias-label-secondary (--dsw-static-neutral-bluish-300)
DSW_TEXT_TERTIARY = "#adb2b8"    # --dsw-alias-label-tertiary (--dsw-static-neutral-bluish-400)
DSW_TEXT_CAPTION = "#81858c"     # --dsw-alias-label-caption (--dsw-static-neutral-bluish-600)
DSW_TEXT_INVERT = "#0f1115"      # --dsw-alias-label-primary-foreground

DSW_ACCENT = "#7aaaff"           # --dsw-alias-state-business-primary (--dsw-static-deepseek-400)
DSW_ACCENT_STRONG = "#5686fe"    # --dsw-static-deepseek-450
DSW_ACCENT_TERTIARY = "#283142"  # --dsw-static-deepseek-900

DSW_SUCCESS = "#4ed17e"          # --dsw-static-green-400
DSW_SUCCESS_BG = "#1f3124"
DSW_WARN = "#f7ad31"             # --dsw-static-amber-400
DSW_WARN_BG = "#27241f"
DSW_ERROR = "#f25a5a"            # --dsw-static-red-400

FONT_UI = "Segoe UI"
FONT_MONO = "Consolas"

PLACEHOLDER_HERO = "Опишите задачу для весов модели: /навыки, @тензоры или диагностика"


@dataclass
class ConversationMessage:
    role: str  # "user" | "assistant" | "skill"
    title: str
    body: str
    skill_result: SkillExecutionResult | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M"))


@dataclass
class SessionRecord:
    session_id: str
    title: str
    subtitle: str
    messages: list[ConversationMessage] = field(default_factory=list)


def _enable_windows_dark_caption(window: ctk.CTk) -> None:
    """Apply Windows 11 dark titlebar attribute matching DSH's #151517 frame."""
    if sys.platform != "win32":
        return
    try:
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        if not hwnd:
            hwnd = window.winfo_id()
        val = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(val), 4)
        # DWMWA_CAPTION_COLOR = 35 (0x00171515 in COLORREF BGR for #151517)
        caption_bgr = ctypes.c_int(0x00171515)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(caption_bgr), 4)
    except Exception:
        pass


def _load_brand_logo(size: int) -> ctk.CTkImage:
    """Load `assets/weightx_logo.png` or synthesize the minimalist DSH-style squircle mark."""
    candidates = [
        Path(__file__).resolve().parents[2] / "assets" / "weightx_logo.png",
        Path.cwd() / "assets" / "weightx_logo.png",
    ]
    for p in candidates:
        if p.exists():
            try:
                pil_img = Image.open(p).convert("RGBA")
                return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(size, size))
            except Exception:
                pass
    # Fallback procedural render
    S = 256
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([16, 16, S - 16, S - 16], radius=60, fill=(27, 27, 28, 255), outline=(53, 54, 56, 255), width=4)
    d.line([(62, 72), (160, 184)], fill=(249, 250, 251, 255), width=16)
    d.line([(194, 72), (96, 184)], fill=(249, 250, 251, 255), width=16)
    d.ellipse([112, 112, 144, 144], fill=(122, 170, 255, 255))
    return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))


class WeightXApp(ctk.CTk):
    """Minimalist 1:1 DSH-style desktop application for WeightX."""

    def __init__(
        self,
        screenshot_path: str | None = None,
        screenshot_chat_path: str | None = None,
    ) -> None:
        super().__init__(fg_color=DSW_BG_BASE)
        ctk.set_appearance_mode("dark")

        self.title("WeightX")
        self.geometry("1220x780")
        self.minsize(880, 560)
        self.configure(fg_color=DSW_BG_BASE)

        # Window icon
        for ico_candidate in [
            Path(__file__).resolve().parents[2] / "assets" / "weightx.ico",
            Path.cwd() / "assets" / "weightx.ico",
        ]:
            if ico_candidate.exists():
                try:
                    self.iconbitmap(str(ico_candidate))
                except Exception:
                    pass
                break

        self._screenshot_path = screenshot_path
        self._screenshot_chat_path = screenshot_chat_path

        # Backend session & UI state
        self._log_queue: queue.Queue[str] = queue.Queue()
        self.studio = StudioSession(log=self._enqueue_log)
        self._sidebar_collapsed = False
        self._inspector_open = False
        self._slash_menu_open = False
        self._placeholder_active = True
        self._busy = False

        # Sessions list (1st session starts as blank Hero; 2nd is pre-recorded benchmark reference)
        self._sessions: list[SessionRecord] = [
            SessionRecord(
                session_id="active",
                title="Новый сеанс",
                subtitle="Готов к работе",
                messages=[],
            ),
            SessionRecord(
                session_id="smollm2-bench",
                title="SmolLM2-135M · −73% PPL",
                subtitle="Domain MEMIT + Vespasian",
                messages=[
                    ConversationMessage(
                        role="user",
                        title="/spectral-denoise",
                        body="Снизить перплексию",
                    ),
                    ConversationMessage(
                        role="assistant",
                        title="Навык /spectral-denoise — Снизить перплексию",
                        body=(
                            "✓ [1/5] L1+L2 Bottleneck Recon — семейство=llama, слоёв=30, d_model=576\n"
                            "✓ [2/5] Outlier Conditioning — проверены выходные проекции down_proj (слои 27..29)\n"
                            "✓ [3/5] Closed-Form Domain MEMIT — итеративная калибровка C0^-1 (шаги 0.35, 0.35)\n"
                            "✓ [4/5] KL & Control Guard — контрольный PPL 13.80 → 13.29 (−3.66%), KL = 0.0019 <= 0.05\n"
                            "✓ [5/5] Vespasian Commit — Held-Out PPL 317.93 → 85.83 (−73.0%), Top-1 Acc 25.24% → 35.24% (+10.0 п.п.)\n\n"
                            "Чекпоинт сохранён: checkpoints/smollm2-135m_weightx_improved.safetensors"
                        ),
                    ),
                ],
            ),
        ]
        self._active_session_idx = 0

        # Build 3-column AppFrame
        self._build_app_frame()

        # Keyboard shortcuts matching DSH (`Ctrl+B` sidebar, `Ctrl+I` inspector, `Ctrl+N` new session)
        self.bind("<Control-b>", lambda _e: self.toggle_sidebar())
        self.bind("<Control-i>", lambda _e: self.toggle_inspector())
        self.bind("<Control-n>", lambda _e: self.new_session())

        self.after(80, lambda: _enable_windows_dark_caption(self))
        self.after(120, self._boot_model_async)
        self.after(100, self._drain_logs)

        if self._screenshot_path:
            self.after(1400, self._capture_screenshots_and_exit)

    # ------------------------------------------------------------------
    # Logging & Background Model Boot
    # ------------------------------------------------------------------
    def _enqueue_log(self, msg: str) -> None:
        self._log_queue.put(msg)

    def _drain_logs(self) -> None:
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._set_status_caption(msg)
        except queue.Empty:
            pass
        self.after(90, self._drain_logs)

    def _boot_model_async(self) -> None:
        def _job() -> None:
            model_id = self.studio.config.model_id or "minillama-demo"
            self.studio.load_model(model_id)
            self.after(0, self._sync_model_badges)

        threading.Thread(target=_job, daemon=True).start()

    def _sync_model_badges(self) -> None:
        try:
            while True:
                self._log_queue.get_nowait()
        except queue.Empty:
            pass
        mid = self.studio.config.model_id or "minillama-demo"
        entry = self.studio.catalog.get(mid)
        short_name = mid
        if entry is not None:
            short_name = f"{entry.model_id} ({entry.params_human})"
        self._hero_workspace_btn.configure(text=f"📁  {short_name}  ▾")
        self._composer_model_btn.configure(text=f"{mid} ▾")
        if self.studio.harness is not None:
            v = self.studio.harness.vespasian
            self._hero_score_badge.configure(text=f"held-out {v.best_score:.4f}")
        self._set_status_caption("")
        self._refresh_tensor_list()

    # ------------------------------------------------------------------
    # 3-Column DSH AppFrame Construction
    # ------------------------------------------------------------------
    def _build_app_frame(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # Left Sidebar
        self.grid_columnconfigure(1, weight=1)  # Center Surface (Hero / Chat)
        self.grid_columnconfigure(2, weight=0)  # Right Inspector Drawer (hidden by default)

        self._build_sidebar()
        self._build_center_surface()
        self._build_right_inspector()

    # ------------------------------------------------------------------
    # Column 1: DSH Left Sidebar (`#1b1b1c`, 252px expanded / 56px rail)
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> None:
        self._sidebar = ctk.CTkFrame(
            self,
            width=252,
            corner_radius=0,
            fg_color=DSW_SIDEBAR_FILL,
            border_width=0,
        )
        self._sidebar.grid(row=0, column=0, sticky="nsew")
        self._sidebar.grid_propagate(False)
        self._sidebar.grid_rowconfigure(2, weight=1)
        self._sidebar.grid_columnconfigure(0, weight=1)

        # 1. Brand Row (`sidebar.brand.mark` + `sidebar.brand.name` + collapse toggle)
        self._brand_row = ctk.CTkFrame(self._sidebar, fg_color="transparent", height=48)
        self._brand_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        self._brand_row.grid_columnconfigure(1, weight=1)

        self._logo_small = _load_brand_logo(24)
        self._brand_mark_btn = ctk.CTkButton(
            self._brand_row,
            text="",
            image=self._logo_small,
            width=28,
            height=28,
            corner_radius=8,
            fg_color="transparent",
            hover_color=DSW_LAYER_1,
            command=self.new_session,
        )
        self._brand_mark_btn.grid(row=0, column=0, padx=(2, 6))

        self._brand_name_lbl = ctk.CTkLabel(
            self._brand_row,
            text="WeightX",
            font=ctk.CTkFont(family=FONT_UI, size=15, weight="bold"),
            text_color=DSW_TEXT_PRIMARY,
            anchor="w",
        )
        self._brand_name_lbl.grid(row=0, column=1, sticky="w")
        self._brand_name_lbl.bind("<Button-1>", lambda _e: self.new_session())

        self._collapse_btn = ctk.CTkButton(
            self._brand_row,
            text="◧",
            width=28,
            height=28,
            corner_radius=8,
            fg_color="transparent",
            hover_color=DSW_LAYER_1,
            text_color=DSW_TEXT_TERTIARY,
            font=ctk.CTkFont(family=FONT_UI, size=14),
            command=self.toggle_sidebar,
        )
        self._collapse_btn.grid(row=0, column=2, sticky="e")

        # 2. New Session Button (`+ Новый сеанс`)
        self._new_session_btn = ctk.CTkButton(
            self._sidebar,
            text="+  Новый сеанс",
            height=34,
            corner_radius=10,
            fg_color=DSW_LAYER_1,
            hover_color=DSW_INPUT_MAJOR,
            border_width=1,
            border_color=DSW_BORDER_L2,
            text_color=DSW_TEXT_PRIMARY,
            font=ctk.CTkFont(family=FONT_UI, size=13),
            anchor="w",
            command=self.new_session,
        )
        self._new_session_btn.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 10))

        # 3. Scrollable Sessions & Skills List
        self._sidebar_scroll = ctk.CTkScrollableFrame(
            self._sidebar,
            fg_color="transparent",
            scrollbar_button_color=DSW_SIDEBAR_FILL,
            scrollbar_button_hover_color=DSW_SELECTOR,
        )
        self._sidebar_scroll.grid(row=2, column=0, sticky="nsew", padx=6, pady=0)
        self._sidebar_scroll.grid_columnconfigure(0, weight=1)
        self._render_sidebar_items()

        # 4. Bottom-Pinned Settings & Inspector Controls
        self._sidebar_footer = ctk.CTkFrame(self._sidebar, fg_color="transparent")
        self._sidebar_footer.grid(row=3, column=0, sticky="ew", padx=10, pady=(6, 12))
        self._sidebar_footer.grid_columnconfigure(0, weight=1)
        self._sidebar_footer.grid_columnconfigure(1, weight=0)

        self._settings_btn = ctk.CTkButton(
            self._sidebar_footer,
            text="⚙  Настройки",
            height=32,
            corner_radius=8,
            fg_color="transparent",
            hover_color=DSW_LAYER_1,
            text_color=DSW_TEXT_SECONDARY,
            font=ctk.CTkFont(family=FONT_UI, size=12),
            anchor="w",
            command=self._open_settings_modal,
        )
        self._settings_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._inspector_toggle_btn = ctk.CTkButton(
            self._sidebar_footer,
            text="◨",
            width=32,
            height=32,
            corner_radius=8,
            fg_color="transparent",
            hover_color=DSW_LAYER_1,
            text_color=DSW_TEXT_TERTIARY,
            font=ctk.CTkFont(family=FONT_UI, size=14),
            command=self.toggle_inspector,
        )
        self._inspector_toggle_btn.grid(row=0, column=1, sticky="e")

    def _render_sidebar_items(self) -> None:
        for child in self._sidebar_scroll.winfo_children():
            child.destroy()
        if self._sidebar_collapsed:
            return

        sec_lbl = ctk.CTkLabel(
            self._sidebar_scroll,
            text="Сеансы",
            font=ctk.CTkFont(family=FONT_UI, size=11),
            text_color=DSW_TEXT_CAPTION,
            anchor="w",
        )
        sec_lbl.grid(row=0, column=0, sticky="ew", padx=8, pady=(2, 4))

        for idx, sess in enumerate(self._sessions):
            is_active = idx == self._active_session_idx
            btn = ctk.CTkButton(
                self._sidebar_scroll,
                text=sess.title,
                height=32,
                corner_radius=8,
                fg_color=DSW_NAV_ACTIVE if is_active else "transparent",
                hover_color=DSW_NAV_ACTIVE if is_active else DSW_LAYER_1,
                text_color=DSW_TEXT_PRIMARY if is_active else DSW_TEXT_SECONDARY,
                font=ctk.CTkFont(family=FONT_UI, size=12, weight="bold" if is_active else "normal"),
                anchor="w",
                command=lambda i=idx: self._select_session(i),
            )
            btn.grid(row=idx + 1, column=0, sticky="ew", padx=2, pady=1)

    def toggle_sidebar(self) -> None:
        """Collapse sidebar to DSH 56px rail or expand to 252px."""
        self._sidebar_collapsed = not self._sidebar_collapsed
        if self._sidebar_collapsed:
            self._sidebar.configure(width=56)
            self._brand_name_lbl.grid_remove()
            self._brand_mark_btn.grid_remove()
            self._new_session_btn.configure(text="+", width=36, anchor="center")
            self._settings_btn.configure(text="⚙", width=36, anchor="center")
            self._inspector_toggle_btn.grid_remove()
            self._render_sidebar_items()
        else:
            self._sidebar.configure(width=252)
            self._brand_mark_btn.grid()
            self._brand_name_lbl.grid()
            self._new_session_btn.configure(text="+  Новый сеанс", anchor="w")
            self._settings_btn.configure(text="⚙  Настройки", anchor="w")
            self._inspector_toggle_btn.grid()
            self._render_sidebar_items()

    # ------------------------------------------------------------------
    # Column 2: DSH Uncluttered Center Surface (`#151517`)
    # ------------------------------------------------------------------
    def _build_center_surface(self) -> None:
        self._center = ctk.CTkFrame(self, fg_color=DSW_BG_BASE, corner_radius=0)
        self._center.grid(row=0, column=1, sticky="nsew")
        self._center.grid_rowconfigure(1, weight=1)
        self._center.grid_columnconfigure(0, weight=1)

        # Subtle top bar ONLY visible in Conversation mode (hidden in Hero mode!)
        self._chat_topbar = ctk.CTkFrame(self._center, fg_color=DSW_BG_BASE, height=42, corner_radius=0)
        self._chat_topbar.grid_columnconfigure(0, weight=1)
        self._chat_title_lbl = ctk.CTkLabel(
            self._chat_topbar,
            text="WeightX",
            font=ctk.CTkFont(family=FONT_UI, size=13, weight="bold"),
            text_color=DSW_TEXT_SECONDARY,
            anchor="w",
        )
        self._chat_title_lbl.grid(row=0, column=0, sticky="w", padx=20, pady=8)

        self._chat_top_inspector_btn = ctk.CTkButton(
            self._chat_topbar,
            text="Инспектор ◨",
            height=26,
             width=104,
            corner_radius=13,
            fg_color=DSW_LAYER_1,
            hover_color=DSW_SELECTOR,
            text_color=DSW_TEXT_SECONDARY,
            font=ctk.CTkFont(family=FONT_UI, size=11),
            command=self.toggle_inspector,
        )
        self._chat_top_inspector_btn.grid(row=0, column=1, sticky="e", padx=16, pady=8)

        # Main Stage Container (switches between Centered Hero and Chat Scroll + Bottom Composer)
        self._stage = ctk.CTkFrame(self._center, fg_color=DSW_BG_BASE, corner_radius=0)
        self._stage.grid(row=1, column=0, sticky="nsew")
        self._stage.grid_columnconfigure(0, weight=1)

        # Build Hero Header, Conversation Feed, and Shared Composer+Pills Dock
        self._build_hero_view()
        self._build_conversation_view()
        self._build_shared_composer_and_pills()
        self._refresh_center_mode()

    def _build_hero_view(self) -> None:
        """Build DSH 1:1 `HeroShell` header (`[Logo 34px] WeightX (WSP)` + Workspace row)."""
        self._hero_stack = ctk.CTkFrame(self._stage, fg_color="transparent", width=680)
        self._hero_stack.grid_columnconfigure(0, weight=1)

        # 1. DSH Hero Headline (`[Logo 34px]  WeightX  (WSP)`)
        headline_row = ctk.CTkFrame(self._hero_stack, fg_color="transparent")
        headline_row.grid(row=0, column=0, pady=(0, 14))

        self._logo_hero = _load_brand_logo(36)
        logo_lbl = ctk.CTkLabel(headline_row, text="", image=self._logo_hero)
        logo_lbl.pack(side="left", padx=(0, 12))

        title_lbl = ctk.CTkLabel(
            headline_row,
            text="WeightX",
            font=ctk.CTkFont(family=FONT_UI, size=26, weight="bold"),
            text_color=DSW_TEXT_PRIMARY,
        )
        title_lbl.pack(side="left", padx=(0, 8))

        preview_badge = ctk.CTkLabel(
            headline_row,
            text="WSP",
            font=ctk.CTkFont(family=FONT_MONO, size=11, weight="bold"),
            text_color=DSW_ACCENT,
            fg_color=DSW_ACCENT_TERTIARY,
            corner_radius=999,
            padx=8,
            pady=1,
        )
        preview_badge.pack(side="left", pady=(2, 0))

        # 2. DSH `heroWorkspaceRow` (subtle folder chip right above the composer card)
        ws_row = ctk.CTkFrame(self._hero_stack, fg_color="transparent", width=680)
        ws_row.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 6))

        self._hero_workspace_btn = ctk.CTkButton(
            ws_row,
            text="📁  minillama-demo (215K)  ▾",
            height=28,
            corner_radius=8,
            fg_color="transparent",
            hover_color=DSW_LAYER_1,
            text_color=DSW_TEXT_PRIMARY,
            font=ctk.CTkFont(family=FONT_UI, size=13),
            anchor="w",
            command=self._cycle_target_model,
        )
        self._hero_workspace_btn.pack(side="left")

        self._hero_score_badge = ctk.CTkLabel(
            ws_row,
            text="held-out готов",
            font=ctk.CTkFont(family=FONT_MONO, size=11),
            text_color=DSW_TEXT_CAPTION,
        )
        self._hero_score_badge.pack(side="right", padx=6)

    def _build_shared_composer_and_pills(self) -> None:
        """Create the DSH Composer Card (`#2c2c2e`) and the ultra-compact 2-word Skill Pills."""
        self._composer_dock = ctk.CTkFrame(self._stage, fg_color="transparent", width=680)
        self._composer_dock.grid_columnconfigure(0, weight=1)

        # Composer Card (`--dsw-specific-input-major: #2c2c2e`)
        self._composer_card = ctk.CTkFrame(
            self._composer_dock,
            fg_color=DSW_INPUT_MAJOR,
            corner_radius=20,
            border_width=1,
            border_color=DSW_BORDER_L3,
            width=680,
        )
        self._composer_card.grid(row=0, column=0, sticky="ew")
        self._composer_card.grid_columnconfigure(0, weight=1)

        # Slash `/` Skills Popup Menu (hidden unless `/` clicked or typed)
        self._slash_popup = ctk.CTkFrame(
            self._composer_card,
            fg_color=DSW_SIDEBAR_FILL,
            corner_radius=12,
            border_width=1,
            border_color=DSW_BORDER_L3,
        )
        self._slash_popup.grid_columnconfigure(0, weight=1)
        for idx, sk in enumerate(WSP_SKILLS):
            row_btn = ctk.CTkButton(
                self._slash_popup,
                text=f"/{sk.skill_id:<17}  {sk.pill_label:<20}  —  {sk.short_hint}",
                height=28,
                corner_radius=6,
                fg_color="transparent",
                hover_color=DSW_INPUT_MAJOR,
                text_color=DSW_TEXT_SECONDARY,
                font=ctk.CTkFont(family=FONT_MONO, size=11),
                anchor="w",
                command=lambda skill=sk: self._pick_slash_skill(skill),
            )
            row_btn.grid(row=idx, column=0, sticky="ew", padx=6, pady=1)

        # Multi-line borderless input inside Composer Card
        self._input_box = ctk.CTkTextbox(
            self._composer_card,
            height=58,
            width=650,
            fg_color="transparent",
            border_width=0,
            text_color=DSW_TEXT_CAPTION,
            font=ctk.CTkFont(family=FONT_UI, size=14),
            wrap="word",
            activate_scrollbars=False,
        )
        self._input_box.grid(row=1, column=0, sticky="ew", padx=12, pady=(10, 2))
        self._input_box.insert("1.0", PLACEHOLDER_HERO)
        self._input_box.bind("<FocusIn>", self._on_input_focus_in)
        self._input_box.bind("<FocusOut>", self._on_input_focus_out)
        self._input_box.bind("<Return>", self._on_input_enter)
        self._input_box.bind("<KeyRelease>", self._on_input_key_release)

        # Bottom Control Row inside Composer Card (`GRgSLq_row`)
        ctrl_row = ctk.CTkFrame(self._composer_card, fg_color="transparent", height=34)
        ctrl_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 10))
        ctrl_row.grid_columnconfigure(3, weight=1)

        # Leading `/` Skill Trigger button (`28x28`, `#353638`, `border-radius: 999px`)
        self._slash_btn = ctk.CTkButton(
            ctrl_row,
            text="/",
            width=28,
            height=28,
            corner_radius=14,
            fg_color=DSW_SELECTOR,
            hover_color=DSW_NAV_ACTIVE,
            text_color=DSW_TEXT_PRIMARY,
            font=ctk.CTkFont(family=FONT_MONO, size=13, weight="bold"),
            command=self._toggle_slash_popup,
        )
        self._slash_btn.grid(row=0, column=0, padx=(2, 6))

        # Model Pill (`SmolLM2-135M ▾`)
        self._composer_model_btn = ctk.CTkButton(
            ctrl_row,
            text="minillama-demo ▾",
            height=28,
            width=136,
            corner_radius=14,
            fg_color=DSW_SELECTOR,
            hover_color=DSW_NAV_ACTIVE,
            text_color=DSW_TEXT_SECONDARY,
            font=ctk.CTkFont(family=FONT_UI, size=12),
            command=self._cycle_target_model,
        )
        self._composer_model_btn.grid(row=0, column=1, padx=(0, 6))

        # Provider / Brain Mode Pill (`Авто (L1→L3) ▾`)
        self._composer_provider_btn = ctk.CTkButton(
            ctrl_row,
            text="Авто (L1→L3) ▾",
            height=28,
            width=124,
            corner_radius=14,
            fg_color="transparent",
            hover_color=DSW_SELECTOR,
            text_color=DSW_TEXT_TERTIARY,
            font=ctk.CTkFont(family=FONT_UI, size=12),
            command=self._open_settings_modal,
        )
        self._composer_provider_btn.grid(row=0, column=2, padx=(0, 6))

        # Trailing Circular Send Button (`↑` in `#f9fafb` with dark `#0f1115` arrow)
        self._send_btn = ctk.CTkButton(
            ctrl_row,
            text="↑",
            width=30,
            height=30,
            corner_radius=15,
            fg_color=DSW_TEXT_PRIMARY,
            hover_color=DSW_TEXT_SECONDARY,
            text_color=DSW_TEXT_INVERT,
            font=ctk.CTkFont(family=FONT_UI, size=15, weight="bold"),
            command=self.submit_composer,
        )
        self._send_btn.grid(row=0, column=4, padx=(6, 2))

        # Ultra-Compact 2-Word Skill Pills Container (two centered fit-width wrap rows)
        self._pills_strip = ctk.CTkFrame(self._composer_dock, fg_color="transparent")
        self._pills_strip.grid(row=1, column=0, pady=(12, 0))

        row1 = ctk.CTkFrame(self._pills_strip, fg_color="transparent")
        row1.pack(pady=2)
        row2 = ctk.CTkFrame(self._pills_strip, fg_color="transparent")
        row2.pack(pady=2)

        for idx, sk in enumerate(WSP_SKILLS):
            parent_row = row1 if idx < 4 else row2
            pill = ctk.CTkButton(
                parent_row,
                text=f"  {sk.pill_label}  ",
                height=26,
                width=24,
                corner_radius=13,
                fg_color=DSW_LAYER_1,
                hover_color=DSW_SELECTOR,
                border_width=1,
                border_color=DSW_BORDER_L2,
                text_color=DSW_TEXT_SECONDARY,
                font=ctk.CTkFont(family=FONT_UI, size=12),
                command=lambda skill=sk: self.trigger_skill(skill),
            )
            pill.pack(side="left", padx=4)
            pill.bind(
                "<Enter>",
                lambda _e, s=sk: self._set_status_caption(f"/{s.skill_id} — {s.short_hint}"),
            )
            pill.bind(
                "<Leave>",
                lambda _e: self._set_status_caption(""),
            )

        # Subtle 1-line status/pipeline hint below the pills (`#81858c`, `11px`)
        self._status_caption = ctk.CTkLabel(
            self._composer_dock,
            text="",
            font=ctk.CTkFont(family=FONT_UI, size=11),
            text_color=DSW_TEXT_CAPTION,
            height=18,
        )
        self._status_caption.grid(row=2, column=0, pady=(4, 0))

    def _build_conversation_view(self) -> None:
        """Build DSH Conversation View (scrollable turn list)."""
        self._conv_scroll = ctk.CTkScrollableFrame(
            self._stage,
            fg_color=DSW_BG_BASE,
            scrollbar_button_color=DSW_LAYER_1,
            scrollbar_button_hover_color=DSW_SELECTOR,
        )
        self._conv_scroll.grid_columnconfigure(0, weight=1)

    def _refresh_center_mode(self) -> None:
        """Switch seamlessly between Hero Empty State and Conversation State using DSH grid weights."""
        sess = self._sessions[self._active_session_idx]
        is_hero = len(sess.messages) == 0

        if is_hero:
            self._chat_topbar.grid_remove()
            self._conv_scroll.grid_remove()
            self._stage.grid_rowconfigure(0, weight=2)
            self._stage.grid_rowconfigure(1, weight=0)
            self._stage.grid_rowconfigure(2, weight=0)
            self._stage.grid_rowconfigure(3, weight=3)
            self._hero_stack.grid(row=1, column=0, padx=32, pady=(0, 4))
            self._composer_dock.grid(row=2, column=0, padx=32, pady=(0, 0))
        else:
            self._hero_stack.grid_remove()
            self._chat_topbar.grid(row=0, column=0, sticky="ew")
            self._chat_title_lbl.configure(text=sess.title)
            self._stage.grid_rowconfigure(0, weight=0)
            self._stage.grid_rowconfigure(1, weight=1)
            self._stage.grid_rowconfigure(2, weight=0)
            self._stage.grid_rowconfigure(3, weight=0)
            self._conv_scroll.grid(row=1, column=0, sticky="nsew", padx=28, pady=(4, 6))
            self._composer_dock.grid(row=2, column=0, padx=32, pady=(4, 12))
            self._render_conversation_messages()

    def _render_conversation_messages(self) -> None:
        for child in self._conv_scroll.winfo_children():
            child.destroy()

        sess = self._sessions[self._active_session_idx]
        for idx, msg in enumerate(sess.messages):
            row = ctk.CTkFrame(self._conv_scroll, fg_color="transparent")
            row.grid(row=idx, column=0, sticky="ew", pady=6)
            row.grid_columnconfigure(0, weight=1)

            if msg.role == "user":
                # DSH right-aligned rounded user bubble (`--dsw-specific-bubble: #2c2c2e`)
                bubble = ctk.CTkFrame(
                    row,
                    fg_color=DSW_INPUT_MAJOR,
                    corner_radius=16,
                    border_width=1,
                    border_color=DSW_BORDER_L2,
                )
                bubble.pack(anchor="e", padx=8)
                lbl = ctk.CTkLabel(
                    bubble,
                    text=msg.body,
                    font=ctk.CTkFont(family=FONT_UI, size=13),
                    text_color=DSW_TEXT_PRIMARY,
                    justify="left",
                    wraplength=560,
                )
                lbl.pack(padx=14, pady=8)
            else:
                # DSH Skill / Assistant Trajectory Card (`#1b1b1c`, subtle border)
                card = ctk.CTkFrame(
                    row,
                    fg_color=DSW_SIDEBAR_FILL,
                    corner_radius=14,
                    border_width=1,
                    border_color=DSW_BORDER_L2,
                )
                card.pack(fill="x", padx=8)

                hdr = ctk.CTkFrame(card, fg_color="transparent")
                hdr.pack(fill="x", padx=14, pady=(10, 4))

                dot_color = DSW_SUCCESS if (msg.skill_result is None or msg.skill_result.accepted) else DSW_WARN
                dot = ctk.CTkLabel(
                    hdr,
                    text="●",
                    font=ctk.CTkFont(family=FONT_UI, size=11),
                    text_color=dot_color,
                )
                dot.pack(side="left", padx=(0, 6))

                t_lbl = ctk.CTkLabel(
                    hdr,
                    text=msg.title,
                    font=ctk.CTkFont(family=FONT_UI, size=13, weight="bold"),
                    text_color=DSW_TEXT_PRIMARY,
                )
                t_lbl.pack(side="left")

                if msg.skill_result is not None:
                    badge_txt = "ПРИНЯТО" if msg.skill_result.accepted else "ОТКАТ VESPASIAN"
                    badge_fg = DSW_SUCCESS_BG if msg.skill_result.accepted else DSW_WARN_BG
                    badge_tc = DSW_SUCCESS if msg.skill_result.accepted else DSW_WARN
                    b_lbl = ctk.CTkLabel(
                        hdr,
                        text=badge_txt,
                        font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
                        text_color=badge_tc,
                        fg_color=badge_fg,
                        corner_radius=999,
                        padx=8,
                        pady=1,
                    )
                    b_lbl.pack(side="right")

                body_box = ctk.CTkTextbox(
                    card,
                    height=min(230, max(78, 22 * (msg.body.count("\n") + 2))),
                    fg_color=DSW_BG_BASE,
                    corner_radius=10,
                    border_width=1,
                    border_color=DSW_BORDER_L2,
                    text_color=DSW_TEXT_SECONDARY,
                    font=ctk.CTkFont(family=FONT_MONO, size=12),
                    wrap="word",
                )
                body_box.pack(fill="x", padx=12, pady=(2, 12))
                body_box.insert("1.0", msg.body)
                body_box.configure(state="disabled")

        self.after(30, lambda: self._conv_scroll._parent_canvas.yview_moveto(1.0))

    # ------------------------------------------------------------------
    # Column 3: On-Demand Right Inspector Drawer (Hidden `0px` by default!)
    # ------------------------------------------------------------------
    def _build_right_inspector(self) -> None:
        self._inspector = ctk.CTkFrame(
            self,
            width=360,
            corner_radius=0,
            fg_color=DSW_SIDEBAR_FILL,
            border_width=1,
            border_color=DSW_BORDER_L2,
        )
        self._inspector.grid_propagate(False)
        self._inspector.grid_rowconfigure(1, weight=1)
        self._inspector.grid_columnconfigure(0, weight=1)

        # Header
        ihdr = ctk.CTkFrame(self._inspector, fg_color="transparent", height=42)
        ihdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        ihdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            ihdr,
            text="Инспектор весов (L3 / YAML)",
            font=ctk.CTkFont(family=FONT_UI, size=13, weight="bold"),
            text_color=DSW_TEXT_PRIMARY,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            ihdr,
            text="✕",
            width=26,
            height=26,
            corner_radius=6,
            fg_color="transparent",
            hover_color=DSW_LAYER_1,
            text_color=DSW_TEXT_TERTIARY,
            command=self.toggle_inspector,
        ).grid(row=0, column=1, sticky="e")

        # Minimalist Tabview inside Right Drawer
        self._insp_tabs = ctk.CTkTabview(
            self._inspector,
            fg_color=DSW_BG_BASE,
            segmented_button_fg_color=DSW_LAYER_1,
            segmented_button_selected_color=DSW_SELECTOR,
            segmented_button_selected_hover_color=DSW_NAV_ACTIVE,
            segmented_button_unselected_color=DSW_LAYER_1,
            segmented_button_unselected_hover_color=DSW_INPUT_MAJOR,
            text_color=DSW_TEXT_PRIMARY,
            corner_radius=12,
        )
        self._insp_tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        tab_yaml = self._insp_tabs.add("YAML План")
        tab_tensors = self._insp_tabs.add("Тензоры")
        tab_export = self._insp_tabs.add("Экспорт")

        # YAML Tab
        tab_yaml.grid_rowconfigure(0, weight=1)
        tab_yaml.grid_columnconfigure(0, weight=1)
        self._yaml_editor = ctk.CTkTextbox(
            tab_yaml,
            fg_color=DSW_SIDEBAR_FILL,
            text_color=DSW_TEXT_PRIMARY,
            font=ctk.CTkFont(family=FONT_MONO, size=11),
            corner_radius=8,
            border_width=1,
            border_color=DSW_BORDER_L2,
        )
        self._yaml_editor.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._yaml_editor.insert(
            "1.0",
            "transforms:\n"
            "  - assert: { exists: { of: 'layers.1.mlp.down_proj.weight' } }\n"
            "  - rescale_outlier_rows_: { target: 'layers.1.mlp.down_proj.weight', threshold_factor: 5.5, target_factor: 2.0 }\n"
            "  - assert: { finite: { of: 'layers.1.mlp.down_proj.weight' } }\n",
        )
        ctk.CTkButton(
            tab_yaml,
            text="Применить YAML + Vespasian",
            height=32,
            corner_radius=8,
            fg_color=DSW_TEXT_PRIMARY,
            hover_color=DSW_TEXT_SECONDARY,
            text_color=DSW_TEXT_INVERT,
            font=ctk.CTkFont(family=FONT_UI, size=12, weight="bold"),
            command=self._run_custom_yaml,
        ).grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))

        # Tensors Tab
        tab_tensors.grid_rowconfigure(1, weight=1)
        tab_tensors.grid_columnconfigure(0, weight=1)
        self._tensor_menu = ctk.CTkOptionMenu(
            tab_tensors,
            values=["layers.1.mlp.down_proj.weight"],
            fg_color=DSW_SELECTOR,
            button_color=DSW_NAV_ACTIVE,
            text_color=DSW_TEXT_PRIMARY,
            command=self._inspect_selected_tensor,
        )
        self._tensor_menu.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self._tensor_info_box = ctk.CTkTextbox(
            tab_tensors,
            fg_color=DSW_SIDEBAR_FILL,
            text_color=DSW_TEXT_SECONDARY,
            font=ctk.CTkFont(family=FONT_MONO, size=11),
            corner_radius=8,
        )
        self._tensor_info_box.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        # Export Tab
        tab_export.grid_columnconfigure(0, weight=1)
        self._export_entry = ctk.CTkEntry(
            tab_export,
            fg_color=DSW_SIDEBAR_FILL,
            text_color=DSW_TEXT_PRIMARY,
            font=ctk.CTkFont(family=FONT_MONO, size=11),
        )
        self._export_entry.grid(row=0, column=0, sticky="ew", padx=6, pady=(12, 8))
        self._export_entry.insert(0, "checkpoints/weightx_edited.safetensors")
        ctk.CTkButton(
            tab_export,
            text="Сохранить .safetensors",
            height=32,
            corner_radius=8,
            fg_color=DSW_TEXT_PRIMARY,
            hover_color=DSW_TEXT_SECONDARY,
            text_color=DSW_TEXT_INVERT,
            command=self._export_safetensors,
        ).grid(row=1, column=0, sticky="ew", padx=6)

    def toggle_inspector(self) -> None:
        self._inspector_open = not self._inspector_open
        if self._inspector_open:
            self._inspector.grid(row=0, column=2, sticky="nsew")
            self._refresh_tensor_list()
        else:
            self._inspector.grid_remove()

    def _refresh_tensor_list(self) -> None:
        if self.studio.harness is None:
            return
        names = self.studio.harness.workspace.names()
        if names:
            self._tensor_menu.configure(values=names[:60])
            self._tensor_menu.set(names[min(5, len(names) - 1)])
            self._inspect_selected_tensor(self._tensor_menu.get())

    def _inspect_selected_tensor(self, name: str) -> None:
        if self.studio.harness is None:
            return
        try:
            desc = self.studio.harness.workspace.describe_tensor(name)
            self._tensor_info_box.delete("1.0", "end")
            self._tensor_info_box.insert("1.0", desc.to_toon())
        except Exception as exc:
            self._tensor_info_box.delete("1.0", "end")
            self._tensor_info_box.insert("1.0", str(exc))

    def _run_custom_yaml(self) -> None:
        plan = self._yaml_editor.get("1.0", "end").strip()
        if not plan:
            return
        self._run_async_turn(
            user_label="Применить YAML-план BrainSurgery-WX",
            worker=lambda: self.studio.apply_yaml(plan),
        )

    def _export_safetensors(self) -> None:
        path = self._export_entry.get().strip() or "checkpoints/weightx_edited.safetensors"
        msg = self.studio.export_checkpoint(path)
        self._set_status_caption(msg)

    # ------------------------------------------------------------------
    # Skills & Composer Interaction
    # ------------------------------------------------------------------
    def _set_status_caption(self, text: str) -> None:
        self._status_caption.configure(text=text[:115])

    def _on_input_focus_in(self, _event: Any = None) -> None:
        if self._placeholder_active:
            self._input_box.delete("1.0", "end")
            self._input_box.configure(text_color=DSW_TEXT_PRIMARY)
            self._placeholder_active = False

    def _on_input_focus_out(self, _event: Any = None) -> None:
        raw = self._input_box.get("1.0", "end").strip()
        if not raw:
            self._placeholder_active = True
            self._input_box.delete("1.0", "end")
            self._input_box.configure(text_color=DSW_TEXT_CAPTION)
            self._input_box.insert("1.0", PLACEHOLDER_HERO)

    def _on_input_key_release(self, _event: Any = None) -> None:
        if self._placeholder_active:
            return
        raw = self._input_box.get("1.0", "end").strip()
        if raw == "/" and not self._slash_menu_open:
            self._toggle_slash_popup(force=True)
        elif not raw.startswith("/") and self._slash_menu_open:
            self._toggle_slash_popup(force=False)

    def _on_input_enter(self, event: Any) -> str | None:
        if event.state & 0x0001:  # Shift+Enter inserts newline
            return None
        self.submit_composer()
        return "break"

    def _toggle_slash_popup(self, force: bool | None = None) -> None:
        self._slash_menu_open = (not self._slash_menu_open) if force is None else force
        if self._slash_menu_open:
            self._slash_popup.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 2))
        else:
            self._slash_popup.grid_remove()

    def _pick_slash_skill(self, skill: WeightXSkill) -> None:
        self._toggle_slash_popup(force=False)
        self.trigger_skill(skill)

    def trigger_skill(self, skill: WeightXSkill) -> None:
        """Invoke a full multi-stage WSP skill pipeline from a 2-word pill or `/skill` command."""
        if self._busy:
            return
        self._toggle_slash_popup(force=False)

        sess = self._sessions[self._active_session_idx]
        if sess.title == "Новый сеанс":
            sess.title = f"{skill.pill_label}"
            self._render_sidebar_items()

        sess.messages.append(
            ConversationMessage(
                role="user",
                title=f"/{skill.skill_id}",
                body=f"/{skill.skill_id}  ·  {skill.pill_label}",
            )
        )
        # Placeholder live trajectory card
        live_msg = ConversationMessage(
            role="skill",
            title=f"Skill · /{skill.skill_id} ({skill.pill_label})",
            body=f"Запуск пайплайна «{skill.pipeline_title}»…",
        )
        sess.messages.append(live_msg)
        self._refresh_center_mode()

        self._busy = True
        self._send_btn.configure(text="■", fg_color=DSW_NAV_ACTIVE)

        def _worker() -> None:
            progress_lines: list[str] = []

            def _on_step(line: str) -> None:
                progress_lines.append("✓ " + line)
                live_msg.body = "\n".join(progress_lines)
                self.after(0, self._render_conversation_messages)

            try:
                res = self.studio.run_skill(skill.skill_id, progress_cb=_on_step)
                live_msg.skill_result = res
                body_parts = [f"✓ [{s.index}/{s.total}] {s.title} — {s.detail}" for s in res.steps]
                if res.metrics:
                    from weightx.toon_io import dumps_toon as toon_dumps

                    body_parts.append("\n[Метрики TOON]\n" + toon_dumps(res.metrics))
                if res.yaml_plan.strip():
                    body_parts.append("\n[BrainSurgery-WX YAML]\n" + res.yaml_plan.strip())
                body_parts.append(f"\nИтог: {res.summary}")
                live_msg.body = "\n".join(body_parts)
                if res.yaml_plan.strip():
                    self.after(0, lambda p=res.yaml_plan: self._set_yaml_editor(p))
            except Exception as exc:
                live_msg.body = f"Ошибка выполнения навыка /{skill.skill_id}: {exc}"
            finally:
                self._busy = False
                self.after(0, self._finish_turn)

        threading.Thread(target=_worker, daemon=True).start()

    def _set_yaml_editor(self, plan_text: str) -> None:
        self._yaml_editor.delete("1.0", "end")
        self._yaml_editor.insert("1.0", plan_text)

    def submit_composer(self) -> None:
        if self._busy:
            return
        raw = "" if self._placeholder_active else self._input_box.get("1.0", "end").strip()
        if not raw:
            # Default quick skill if empty
            self.trigger_skill(WSP_SKILLS[0])
            return

        # Clear input
        self._input_box.delete("1.0", "end")
        self._placeholder_active = True
        self._input_box.configure(text_color=DSW_TEXT_CAPTION)
        self._input_box.insert("1.0", PLACEHOLDER_HERO)
        self._toggle_slash_popup(force=False)

        # Check if user typed a skill `/...` or 2-word pill name
        sk = resolve_skill(raw)
        if sk is not None:
            self.trigger_skill(sk)
            return

        self._run_async_turn(
            user_label=raw,
            worker=lambda: self.studio.run_action("chat", raw),
        )

    def _run_async_turn(self, user_label: str, worker: Any) -> None:
        sess = self._sessions[self._active_session_idx]
        if sess.title == "Новый сеанс":
            sess.title = user_label[:26]
            self._render_sidebar_items()
        sess.messages.append(ConversationMessage(role="user", title="Запрос", body=user_label))
        reply_msg = ConversationMessage(
            role="assistant",
            title="WeightX WSP Trajectory",
            body="Выполнение цикла разведки и редактирования весов…",
        )
        sess.messages.append(reply_msg)
        self._refresh_center_mode()

        self._busy = True
        self._send_btn.configure(text="■", fg_color=DSW_NAV_ACTIVE)

        def _bg() -> None:
            try:
                reply_msg.body = worker()
            except Exception as exc:
                reply_msg.body = f"Ошибка: {exc}"
            finally:
                self._busy = False
                self.after(0, self._finish_turn)

        threading.Thread(target=_bg, daemon=True).start()

    def _finish_turn(self) -> None:
        self._send_btn.configure(text="↑", fg_color=DSW_TEXT_PRIMARY)
        self._sync_model_badges()
        self._render_conversation_messages()

    # ------------------------------------------------------------------
    # Session & Model Switching + Settings Modal
    # ------------------------------------------------------------------
    def new_session(self) -> None:
        """Start or switch to a clean blank Hero session (`+ Новый сеанс`)."""
        if self._sessions and len(self._sessions[0].messages) == 0:
            self._active_session_idx = 0
        else:
            self._sessions.insert(
                0,
                SessionRecord(session_id=f"s_{int(time.time())}", title="Новый сеанс", subtitle="Пустой сеанс"),
            )
            self._active_session_idx = 0
        self._render_sidebar_items()
        self._refresh_center_mode()

    def _select_session(self, idx: int) -> None:
        self._active_session_idx = idx
        self._render_sidebar_items()
        self._refresh_center_mode()

    def _cycle_target_model(self) -> None:
        entries = self.studio.catalog.list_entries()
        ids = [e.model_id for e in entries]
        cur = self.studio.config.model_id or "minillama-demo"
        next_id = ids[(ids.index(cur) + 1) % len(ids)] if cur in ids else ids[0]
        self._set_status_caption(f"Переключение целевой модели на {next_id}…")

        def _load() -> None:
            self.studio.load_model(next_id)
            self.after(0, self._sync_model_badges)

        threading.Thread(target=_load, daemon=True).start()

    def _open_settings_modal(self) -> None:
        modal = ctk.CTkToplevel(self)
        modal.title("WeightX — Настройки и модели")
        modal.geometry("520x420")
        modal.configure(fg_color=DSW_SIDEBAR_FILL)
        modal.transient(self)
        modal.grab_set()

        ctk.CTkLabel(
            modal,
            text="Настройки подключения и целевой модели",
            font=ctk.CTkFont(family=FONT_UI, size=15, weight="bold"),
            text_color=DSW_TEXT_PRIMARY,
        ).pack(anchor="w", padx=20, pady=(18, 10))

        # Target Model Selector
        ctk.CTkLabel(modal, text="Целевая модель (Target LLM):", text_color=DSW_TEXT_SECONDARY).pack(
            anchor="w", padx=20, pady=(4, 2)
        )
        model_ids = [e.model_id for e in self.studio.catalog.list_entries()]
        model_opt = ctk.CTkOptionMenu(
            modal,
            values=model_ids,
            fg_color=DSW_INPUT_MAJOR,
            button_color=DSW_SELECTOR,
            text_color=DSW_TEXT_PRIMARY,
        )
        model_opt.set(self.studio.config.model_id or "minillama-demo")
        model_opt.pack(fill="x", padx=20, pady=(0, 10))

        # Brain Provider Selector
        ctk.CTkLabel(modal, text="Мозг-провайдер (Provider Layer):", text_color=DSW_TEXT_SECONDARY).pack(
            anchor="w", padx=20, pady=(4, 2)
        )
        prov_opt = ctk.CTkOptionMenu(
            modal,
            values=PROVIDER_KINDS,
            fg_color=DSW_INPUT_MAJOR,
            button_color=DSW_SELECTOR,
            text_color=DSW_TEXT_PRIMARY,
        )
        prov_opt.set(self.studio.config.provider_kind or PROVIDER_KINDS[0])
        prov_opt.pack(fill="x", padx=20, pady=(0, 10))

        # API Key / Base URL
        ctk.CTkLabel(modal, text="API Key (опционально):", text_color=DSW_TEXT_SECONDARY).pack(
            anchor="w", padx=20, pady=(4, 2)
        )
        key_entry = ctk.CTkEntry(modal, fg_color=DSW_INPUT_MAJOR, text_color=DSW_TEXT_PRIMARY, show="*")
        key_entry.pack(fill="x", padx=20, pady=(0, 16))
        if self.studio.config.api_key:
            key_entry.insert(0, self.studio.config.api_key)

        def _save() -> None:
            chosen_model = model_opt.get()
            chosen_prov = prov_opt.get()
            self.studio.set_provider(chosen_prov, api_key=key_entry.get().strip())
            short_p = chosen_prov.split(" ")[0]
            self._composer_provider_btn.configure(text=f"{short_p} ▾")
            if chosen_model != self.studio.config.model_id:
                threading.Thread(target=lambda: (self.studio.load_model(chosen_model), self.after(0, self._sync_model_badges)), daemon=True).start()
            modal.destroy()

        ctk.CTkButton(
            modal,
            text="Сохранить",
            height=34,
            corner_radius=10,
            fg_color=DSW_TEXT_PRIMARY,
            hover_color=DSW_TEXT_SECONDARY,
            text_color=DSW_TEXT_INVERT,
            font=ctk.CTkFont(family=FONT_UI, size=13, weight="bold"),
            command=_save,
        ).pack(fill="x", padx=20, pady=8)

    # ------------------------------------------------------------------
    # Automated Visual Verification Screenshots
    # ------------------------------------------------------------------
    def _grab_window_image(self) -> Image.Image:
        self.update_idletasks()
        self.update()
        try:
            x = self.winfo_rootx()
            y = self.winfo_rooty()
            w = self.winfo_width()
            h = self.winfo_height()
            return ImageGrab.grab(bbox=(x, y, x + w, y + h))
        except Exception:
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id()) or self.winfo_id()
            rect = (ctypes.c_long * 4)()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = max(800, rect[2] - rect[0])
            h = max(560, rect[3] - rect[1])
            hwnd_dc = ctypes.windll.user32.GetWindowDC(hwnd)
            mfc_dc = ctypes.windll.gdi32.CreateCompatibleDC(hwnd_dc)
            bmp = ctypes.windll.gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
            ctypes.windll.gdi32.SelectObject(mfc_dc, bmp)
            ctypes.windll.user32.PrintWindow(hwnd, mfc_dc, 2)

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", ctypes.c_uint32),
                    ("biWidth", ctypes.c_int32),
                    ("biHeight", ctypes.c_int32),
                    ("biPlanes", ctypes.c_uint16),
                    ("biBitCount", ctypes.c_uint16),
                    ("biCompression", ctypes.c_uint32),
                    ("biSizeImage", ctypes.c_uint32),
                    ("biXPelsPerMeter", ctypes.c_int32),
                    ("biYPelsPerMeter", ctypes.c_int32),
                    ("biClrUsed", ctypes.c_uint32),
                    ("biClrImportant", ctypes.c_uint32),
                ]

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = -h
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0
            buf = ctypes.create_string_buffer(w * h * 4)
            ctypes.windll.gdi32.GetDIBits(mfc_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
            img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")
            ctypes.windll.gdi32.DeleteObject(bmp)
            ctypes.windll.gdi32.DeleteDC(mfc_dc)
            ctypes.windll.user32.ReleaseDC(hwnd, hwnd_dc)
            return img

    def _capture_screenshots_and_exit(self) -> None:
        try:
            if self._screenshot_path:
                img = self._grab_window_image()
                Path(self._screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                img.save(self._screenshot_path)
            if self._screenshot_chat_path:
                self._select_session(1)
                self.update_idletasks()
                self.update()
                time.sleep(0.3)
                img2 = self._grab_window_image()
                Path(self._screenshot_chat_path).parent.mkdir(parents=True, exist_ok=True)
                img2.save(self._screenshot_chat_path)
        finally:
            self.destroy()


def launch_desktop(
    screenshot_path: str | None = None,
    screenshot_chat_path: str | None = None,
) -> None:
    app = WeightXApp(screenshot_path=screenshot_path, screenshot_chat_path=screenshot_chat_path)
    app.mainloop()


def launch_desktop_app(
    initial_model: str = "minillama-demo",
    smoke_test_ms: int | None = None,
) -> int:
    app = WeightXApp()
    if initial_model:
        app.studio.config.model_id = initial_model
    if smoke_test_ms is not None:
        app.after(smoke_test_ms, app.destroy)
    app.mainloop()
    return 0


if __name__ == "__main__":
    shot_hero: str | None = None
    shot_chat: str | None = None
    args = sys.argv[1:]
    if "--screenshot" in args:
        i = args.index("--screenshot")
        if i + 1 < len(args):
            shot_hero = args[i + 1]
    if "--screenshot-chat" in args:
        i = args.index("--screenshot-chat")
        if i + 1 < len(args):
            shot_chat = args[i + 1]
    launch_desktop(screenshot_path=shot_hero, screenshot_chat_path=shot_chat)
