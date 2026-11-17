# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 WeightX Contributors
"""
WeightX Desktop (`src/weightx/desktop.py`).

Minimalist 1:1 DSH (DeepSeek Harness) desktop interface enhanced with:
  - Smooth color, border, hover, press, and slide animations across ALL elements
    (collapsible sidebar `252px <-> 56px`, right inspector `0 <-> 360px`, skill pills,
    floating model/provider popovers, session rows, and conversation action buttons).
  - Interactive real-time Weight-Space background canvas behind the minimalist center
    stage with 3 switchable interactive modes (`Нейро-сеть`, `Спектр SVD`, `Тензор-поток`),
    global cursor gravity synapses, click shockwaves, and Vespasian commit/rollback ripples.
  - Ultra-compact 2-word WSP Skill Pills + live filtered `/skill <название>` autocomplete
    popup supporting arrow navigation, `Tab` completion, and `Enter` execution.
"""

from __future__ import annotations

import ctypes
import math
import queue
import random
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageGrab

from weightx.skills import WSP_SKILLS, SkillExecutionResult, WeightXSkill, parse_skill_command
from weightx.studio import PROVIDER_KINDS, StudioSession
from weightx.toon_io import dumps_toon

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
DSW_ACCENT_SOFT = "#2b384e"
DSW_ACCENT_STRONG = "#5686fe"    # --dsw-static-deepseek-450
DSW_ACCENT_TERTIARY = "#283142"  # --dsw-static-deepseek-900

DSW_SUCCESS = "#4ed17e"          # --dsw-static-green-400
DSW_SUCCESS_BG = "#1f3124"
DSW_WARN = "#f7ad31"             # --dsw-static-amber-400
DSW_WARN_BG = "#27241f"
DSW_ERROR = "#f25a5a"            # --dsw-static-red-400

FONT_UI = "Segoe UI"
FONT_MONO = "Consolas"

PLACEHOLDER_HERO = "Опишите задачу или введите /skill <название> (например /skill coding-boost)…"

BG_MODES: tuple[tuple[str, str], ...] = (
    ("neural", "Нейро-сеть"),
    ("wave", "Спектр SVD"),
    ("rain", "Тензор-поток"),
    ("off", "Чистый DSH"),
)


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


# ---------------------------------------------------------------------------
# Color & Motion Animation Primitives
# ---------------------------------------------------------------------------
def _parse_hex(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) != 6:
        return (21, 21, 23)
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _format_hex(r: float, g: float, b: float) -> str:
    ri = max(0, min(255, int(round(r))))
    gi = max(0, min(255, int(round(g))))
    bi = max(0, min(255, int(round(b))))
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def lerp_hex(c1: str, c2: str, t: float) -> str:
    """Linearly interpolate two `#RRGGBB` colors with clamped factor `t`."""
    t = max(0.0, min(1.0, float(t)))
    r1, g1, b1 = _parse_hex(c1)
    r2, g2, b2 = _parse_hex(c2)
    return _format_hex(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def ease_out_cubic(t: float) -> float:
    u = 1.0 - max(0.0, min(1.0, t))
    return 1.0 - (u * u * u)


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def animate_colors(
    app: ctk.CTk,
    widget: Any,
    targets: dict[str, str],
    *,
    duration_ms: int = 150,
    steps: int = 9,
) -> None:
    """Smoothly interpolate widget color properties (`fg_color`, `border_color`, `text_color`)."""
    if getattr(app, "_destroyed", False) or not widget.winfo_exists():
        return

    prev_job = getattr(widget, "_wx_color_anim_job", None)
    if prev_job is not None:
        try:
            app.after_cancel(prev_job)
        except Exception:
            pass
        widget._wx_color_anim_job = None

    starts: dict[str, str] = {}
    fallback_surface = getattr(widget, "_wx_surface_hex", DSW_BG_BASE)
    for key, target_hex in targets.items():
        try:
            cur_val = widget.cget(key)
        except Exception:
            continue
        if isinstance(cur_val, (list, tuple)):
            cur_val = cur_val[-1]
        if not isinstance(cur_val, str) or cur_val == "transparent" or not cur_val.startswith("#"):
            cur_val = fallback_surface
        starts[key] = cur_val

    if not starts:
        return

    interval = max(12, duration_ms // max(1, steps))

    def _step(idx: int) -> None:
        if getattr(app, "_destroyed", False) or not widget.winfo_exists():
            return
        t = ease_out_cubic(idx / float(steps))
        update_kwargs = {k: lerp_hex(starts[k], targets[k], t) for k in starts}
        try:
            widget.configure(**update_kwargs)
        except Exception:
            return
        if idx < steps:
            widget._wx_color_anim_job = app.after(interval, lambda: _step(idx + 1))
        else:
            widget._wx_color_anim_job = None
            restore_tp = getattr(widget, "_wx_transparent_on_rest", None)
            if restore_tp and targets == restore_tp:
                try:
                    widget.configure(fg_color="transparent")
                except Exception:
                    pass

    _step(1)


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
    S = 256
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([16, 16, S - 16, S - 16], radius=60, fill=(27, 27, 28, 255), outline=(53, 54, 56, 255), width=4)
    d.line([(62, 72), (160, 184)], fill=(249, 250, 251, 255), width=16)
    d.line([(194, 72), (96, 184)], fill=(249, 250, 251, 255), width=16)
    d.ellipse([112, 112, 144, 144], fill=(122, 170, 255, 255))
    return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))


# ---------------------------------------------------------------------------
# Interactive Weight-Space Neural Canvas Background
# ---------------------------------------------------------------------------
@dataclass
class _NeuralNode:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    phase: float
    label: str


@dataclass
class _BgRipple:
    x: float
    y: float
    radius: float
    max_radius: float
    color: str
    speed: float


@dataclass
class _RainGlyph:
    x: float
    y: float
    vy: float
    text: str
    accent: bool


class _InteractiveNeuralBackground:
    """Real-time interactive canvas behind DSH's minimalist center surface."""

    _GLYPHS = ("ΔW", "C0⁻¹", "SVD", "Q·Kᵀ", "KCI", "σ₁", "L2", "λI", "WSP", "Vesp")

    def __init__(self, app: WeightXApp, parent: ctk.CTkFrame) -> None:
        self.app = app
        self.parent = parent
        self.canvas = tk.Canvas(parent, bg=DSW_BG_BASE, highlightthickness=0, bd=0)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.ensure_z_order()

        self.mode_idx = 0  # neural | wave | rain | off
        self.cursor_x: float = -1000.0
        self.cursor_y: float = -1000.0
        self.cursor_inside = False
        self.t = 0.0
        self.energy_boost = 0.0

        rng = random.Random(42)
        self.nodes: list[_NeuralNode] = []
        for i in range(34):
            ang = rng.uniform(0.0, math.tau)
            spd = rng.uniform(0.28, 0.72)
            self.nodes.append(
                _NeuralNode(
                    x=rng.uniform(40, 920),
                    y=rng.uniform(40, 720),
                    vx=math.cos(ang) * spd,
                    vy=math.sin(ang) * spd,
                    radius=rng.uniform(1.8, 3.4),
                    phase=rng.uniform(0.0, math.tau),
                    label=self._GLYPHS[i % len(self._GLYPHS)] if i % 2 == 0 else "",
                )
            )
        self.ripples: list[_BgRipple] = []
        self.rain: list[_RainGlyph] = [
            _RainGlyph(
                x=rng.uniform(20, 940),
                y=rng.uniform(0, 740),
                vy=rng.uniform(0.9, 2.2),
                text=rng.choice(("ΔW", "0.98", "SVD", "σ₀", "KCI", "λI", "0.02", "C₀⁻¹", "WSP")),
                accent=(i % 4 == 0),
            )
            for i in range(44)
        ]

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda _e: self.ensure_z_order(), add="+")
        self._tick()

    def ensure_z_order(self) -> None:
        """Keep the interactive canvas above parent CTkFrame._canvas yet below all UI widgets."""
        try:
            tk.Misc.lower(self.canvas)
            base_canvas = getattr(self.parent, "_canvas", None)
            if base_canvas is not None:
                tk.Misc.lower(base_canvas)
        except Exception:
            pass

    @property
    def mode_key(self) -> str:
        return BG_MODES[self.mode_idx][0]

    @property
    def mode_label(self) -> str:
        return BG_MODES[self.mode_idx][1]

    def cycle_mode(self) -> str:
        self.mode_idx = (self.mode_idx + 1) % len(BG_MODES)
        w = max(self.canvas.winfo_width(), 640)
        h = max(self.canvas.winfo_height(), 480)
        self.spawn_ripple(w * 0.5, h * 0.45, color=DSW_ACCENT, max_radius=320.0)
        return self.mode_label

    def update_global_pointer(self, root_x: int, root_y: int) -> None:
        if getattr(self.app, "_destroyed", False):
            return
        try:
            cx = root_x - self.canvas.winfo_rootx()
            cy = root_y - self.canvas.winfo_rooty()
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            self.cursor_inside = (0 <= cx <= w and 0 <= cy <= h)
            if self.cursor_inside:
                self.cursor_x = float(cx)
                self.cursor_y = float(cy)
        except Exception:
            pass

    def spawn_ripple(
        self,
        x: float | None = None,
        y: float | None = None,
        *,
        color: str = DSW_ACCENT,
        max_radius: float = 240.0,
    ) -> None:
        w = max(self.canvas.winfo_width(), 640)
        h = max(self.canvas.winfo_height(), 480)
        rx = w * 0.5 if x is None else x
        ry = h * 0.48 if y is None else y
        self.ripples.append(_BgRipple(rx, ry, 8.0, max_radius, color, speed=7.5))
        self.ripples.append(_BgRipple(rx, ry, 2.0, max_radius * 0.65, lerp_hex(color, DSW_BG_BASE, 0.35), speed=5.5))
        self.energy_boost = min(1.0, self.energy_boost + 0.55)

    def spawn_ripple_from_widget(self, widget: Any, color: str = DSW_ACCENT) -> None:
        try:
            wx = widget.winfo_rootx() + widget.winfo_width() * 0.5 - self.canvas.winfo_rootx()
            wy = widget.winfo_rooty() + widget.winfo_height() * 0.5 - self.canvas.winfo_rooty()
            self.spawn_ripple(wx, wy, color=color, max_radius=210.0)
        except Exception:
            self.spawn_ripple(color=color)

    def _on_canvas_click(self, event: Any) -> None:
        self.spawn_ripple(float(event.x), float(event.y), color=DSW_ACCENT, max_radius=260.0)
        # Subtle scatter impulse to nearby nodes
        for n in self.nodes:
            dx = n.x - event.x
            dy = n.y - event.y
            d2 = dx * dx + dy * dy
            if 4.0 < d2 < 26000.0:
                d = math.sqrt(d2)
                push = 2.4 * (1.0 - d / 165.0)
                n.vx += (dx / d) * push
                n.vy += (dy / d) * push

    def _draw_ripples(self) -> None:
        alive: list[_BgRipple] = []
        for rp in self.ripples:
            rp.radius += rp.speed
            if rp.radius < rp.max_radius:
                frac = rp.radius / rp.max_radius
                col = lerp_hex(rp.color, DSW_BG_BASE, min(0.95, 0.30 + frac * 0.70))
                self.canvas.create_oval(
                    rp.x - rp.radius,
                    rp.y - rp.radius,
                    rp.x + rp.radius,
                    rp.y + rp.radius,
                    outline=col,
                    width=1.5 if frac < 0.5 else 1.0,
                    tags="dyn",
                )
                alive.append(rp)
        self.ripples = alive

    def _draw_neural_mode(self, w: int, h: int, busy: bool) -> None:
        # 1. Subtle interactive coordinate dot lattice near cursor
        step = 56
        cx, cy = self.cursor_x, self.cursor_y
        if self.cursor_inside:
            gx0 = max(step, int((cx - 190) // step) * step)
            gx1 = min(w - step // 2, int(cx + 190))
            gy0 = max(step, int((cy - 190) // step) * step)
            gy1 = min(h - step // 2, int(cy + 190))
            for gx in range(gx0, gx1 + 1, step):
                for gy in range(gy0, gy1 + 1, step):
                    dist = math.hypot(gx - cx, gy - cy)
                    if dist < 185:
                        k = (1.0 - dist / 185.0) ** 1.6
                        dot_c = lerp_hex("#1c1d21", DSW_ACCENT, 0.18 + 0.55 * k)
                        r = 1.1 + 1.1 * k
                        self.canvas.create_oval(gx - r, gy - r, gx + r, gy + r, fill=dot_c, outline="", tags="dyn")

        # 2. Update physics of neural nodes
        speed_mul = (1.55 if busy else 1.0) + self.energy_boost * 0.9
        margin = 18
        for n in self.nodes:
            if self.cursor_inside:
                dx = cx - n.x
                dy = cy - n.y
                d = math.hypot(dx, dy)
                if 28.0 < d < 220.0:
                    pull = 0.045 * (1.0 - d / 220.0)
                    n.vx += (dx / d) * pull
                    n.vy += (dy / d) * pull
            # Gentle damping back to calm orbital drift
            n.vx *= 0.985
            n.vy *= 0.985
            if abs(n.vx) + abs(n.vy) < 0.28:
                n.vx += math.cos(self.t + n.phase) * 0.03
                n.vy += math.sin(self.t * 0.8 + n.phase) * 0.03

            n.x += n.vx * speed_mul
            n.y += n.vy * speed_mul
            if n.x < margin:
                n.x = margin
                n.vx = abs(n.vx) + 0.15
            elif n.x > w - margin:
                n.x = w - margin
                n.vx = -(abs(n.vx) + 0.15)
            if n.y < margin:
                n.y = margin
                n.vy = abs(n.vy) + 0.15
            elif n.y > h - margin:
                n.y = h - margin
                n.vy = -(abs(n.vy) + 0.15)

        # 3. Inter-node synaptic edges
        max_link = 152.0
        for i in range(len(self.nodes)):
            ni = self.nodes[i]
            for j in range(i + 1, len(self.nodes)):
                nj = self.nodes[j]
                dx = ni.x - nj.x
                dy = ni.y - nj.y
                if abs(dx) > max_link or abs(dy) > max_link:
                    continue
                d = math.hypot(dx, dy)
                if d < max_link:
                    prox = 1.0 - d / max_link
                    near_cursor = 0.0
                    if self.cursor_inside:
                        mid_d = math.hypot((ni.x + nj.x) * 0.5 - cx, (ni.y + nj.y) * 0.5 - cy)
                        if mid_d < 175.0:
                            near_cursor = 1.0 - mid_d / 175.0
                    base_edge = lerp_hex("#181c26", "#374157", 0.30 + 0.70 * prox)
                    edge_col = lerp_hex(base_edge, DSW_ACCENT, min(0.90, near_cursor * 0.72 + (0.28 if busy else 0.0)))
                    self.canvas.create_line(ni.x, ni.y, nj.x, nj.y, fill=edge_col, width=1.1, tags="dyn")

        # 4. Cursor magnetic synapses & glow halo
        if self.cursor_inside:
            halo_col = lerp_hex("#1b2130", DSW_ACCENT, 0.42)
            self.canvas.create_oval(cx - 28, cy - 28, cx + 28, cy + 28, outline=halo_col, width=1.2, tags="dyn")
            nearest = sorted(self.nodes, key=lambda n: (n.x - cx) ** 2 + (n.y - cy) ** 2)[:6]
            for n in nearest:
                d = math.hypot(n.x - cx, n.y - cy)
                if d < 260.0:
                    k = 1.0 - d / 260.0
                    ray_col = lerp_hex("#2a3244", DSW_ACCENT, 0.35 + 0.65 * k)
                    self.canvas.create_line(cx, cy, n.x, n.y, fill=ray_col, width=1.25, tags="dyn")

        # 5. Nodes + interactive glyphs near pointer
        for n in self.nodes:
            pulse = 0.5 + 0.5 * math.sin(self.t * 2.2 + n.phase)
            dist_c = math.hypot(n.x - cx, n.y - cy) if self.cursor_inside else 999.0
            glow_k = max(0.0, 1.0 - dist_c / 210.0) if self.cursor_inside else 0.0
            node_col = lerp_hex("#3a4358", DSW_ACCENT, min(1.0, 0.28 * pulse + 0.78 * glow_k + (0.28 if busy else 0.0)))
            r = n.radius + 1.6 * glow_k + 0.55 * pulse
            self.canvas.create_oval(n.x - r, n.y - r, n.x + r, n.y + r, fill=node_col, outline="", tags="dyn")
            if n.label and (glow_k > 0.20 or self.energy_boost > 0.12):
                vis = max(glow_k, self.energy_boost * 0.65)
                lbl_col = lerp_hex("#3c4456", DSW_TEXT_SECONDARY, vis)
                self.canvas.create_text(
                    n.x + 10,
                    n.y - 8,
                    text=n.label,
                    fill=lbl_col,
                    font=(FONT_MONO, 9),
                    anchor="w",
                    tags="dyn",
                )

    def _draw_wave_mode(self, w: int, h: int, busy: bool) -> None:
        mx_bias = (self.cursor_x / max(w, 1) - 0.5) * 2.0 if self.cursor_inside else 0.0
        my_amp = (1.0 - abs(self.cursor_y / max(h, 1) - 0.5)) if self.cursor_inside else 0.55
        ribbons = (
            (h * 0.24, 0.012, 1.0, "#232a36", 14.0),
            (h * 0.50, 0.016, 1.4, "#2d3b54", 22.0),
            (h * 0.78, 0.010, 0.85, "#243042", 16.0),
        )
        for idx, (y0, freq, spd, base_c, amp) in enumerate(ribbons):
            pts: list[float] = []
            col = lerp_hex(base_c, DSW_ACCENT, (0.45 if busy else 0.20) + 0.35 * my_amp)
            phase_shift = self.t * (spd + (0.9 if busy else 0.0)) + idx * 1.3 + mx_bias * 2.5
            for x in range(0, w + 24, 24):
                y = y0 + math.sin(x * freq + phase_shift) * amp * (0.65 + 0.65 * my_amp)
                y += math.cos(x * freq * 0.5 - phase_shift * 0.7) * (amp * 0.4)
                pts.extend((float(x), y))
            if len(pts) >= 4:
                self.canvas.create_line(*pts, fill=col, width=1.4, smooth=True, tags="dyn")

    def _draw_rain_mode(self, w: int, h: int, busy: bool) -> None:
        spd_mul = 1.6 if busy else 1.0
        for g in self.rain:
            g.y += g.vy * spd_mul
            if g.y > h + 20:
                g.y = -18.0
                g.x = random.uniform(20.0, max(float(w - 20), 100.0))
            if self.cursor_inside:
                dx = g.x - self.cursor_x
                dy = g.y - self.cursor_y
                d = math.hypot(dx, dy)
                if 1.0 < d < 120.0:
                    g.x += (dx / d) * 2.2
            dist_c = math.hypot(g.x - self.cursor_x, g.y - self.cursor_y) if self.cursor_inside else 999.0
            bright = max(0.0, 1.0 - dist_c / 160.0)
            col = lerp_hex("#22252b", DSW_ACCENT if g.accent else DSW_TEXT_SECONDARY, bright * 0.85 + (0.18 if g.accent else 0.05))
            self.canvas.create_text(g.x, g.y, text=g.text, fill=col, font=(FONT_MONO, 10), tags="dyn")

    def render_frame(self) -> None:
        if getattr(self.app, "_destroyed", False):
            return
        try:
            if not self.canvas.winfo_exists():
                return
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w > 80 and h > 80 and self.canvas.winfo_viewable():
                self.canvas.delete("dyn")
                self.t += 0.032
                self.energy_boost = max(0.0, self.energy_boost - 0.018)
                busy = getattr(self.app, "_busy", False)
                mode = self.mode_key
                if mode == "neural":
                    self._draw_neural_mode(w, h, busy)
                elif mode == "wave":
                    self._draw_wave_mode(w, h, busy)
                elif mode == "rain":
                    self._draw_rain_mode(w, h, busy)
                self._draw_ripples()
        except Exception:
            pass

    def _tick(self) -> None:
        self.render_frame()
        if not getattr(self.app, "_destroyed", False):
            job = getattr(self.app, "safe_after", None)
            if callable(job):
                job(34, self._tick)
            else:
                try:
                    self.app.after(34, self._tick)
                except (RuntimeError, tk.TclError):
                    pass


# ---------------------------------------------------------------------------
# Main WeightX Desktop Application
# ---------------------------------------------------------------------------
class WeightXApp(ctk.CTk):
    """Minimalist 1:1 DSH-style desktop application with fluid animations & interactive neural canvas."""

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

        self._destroyed = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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
        self._sidebar_skills_open = False
        self._sidebar_anim_job: str | None = None
        self._inspector_open = False
        self._inspector_anim_job: str | None = None
        self._slash_menu_open = False
        self._slash_selected_idx = 0
        self._slash_matches: list[WeightXSkill] = list(WSP_SKILLS)
        self._slash_row_buttons: list[ctk.CTkButton] = []
        self._popover_widget: ctk.CTkFrame | None = None
        self._placeholder_active = True
        self._busy = False
        self._spinner_phase = 0

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
                subtitle="Domain EAMIT + Vespasian",
                messages=[
                    ConversationMessage(
                        role="user",
                        title="/skill spectral-denoise",
                        body="/skill spectral-denoise  ·  Снизить перплексию",
                    ),
                    ConversationMessage(
                        role="assistant",
                        title="Скилл · /skill spectral-denoise (Снизить перплексию)",
                        body=(
                            "✓ [1/5] L1+L2 Bottleneck Recon — семейство=llama, слоёв=30, d_model=576\n"
                            "✓ [2/5] Outlier Conditioning — проверены выходные проекции down_proj (слои 27..29)\n"
                            "✓ [3/5] Closed-Form Domain EAMIT — итеративная калибровка C0^-1 с выравниванием эмбеддингов\n"
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

        # Global pointer tracking for interactive neural canvas
        self.bind_all("<Motion>", self._on_global_motion, add="+")

        # Keyboard shortcuts matching DSH (`Ctrl+B` sidebar, `Ctrl+I` inspector, `Ctrl+N` new session, `Escape` popups)
        self.bind("<Control-b>", lambda _e: self.toggle_sidebar())
        self.bind("<Control-i>", lambda _e: self.toggle_inspector())
        self.bind("<Control-n>", lambda _e: self.new_session())
        self.bind("<Escape>", lambda _e: self._close_all_popups())

        self.safe_after(80, lambda: _enable_windows_dark_caption(self))
        self.safe_after(120, self._boot_model_async)
        self.safe_after(100, self._drain_logs)

        if self._screenshot_path:
            self.safe_after(1450, self._capture_screenshots_and_exit)

    def safe_after(self, delay_ms: int, callback: Callable[[], Any]) -> str | None:
        """Schedule a Tkinter callback guarding against closed-loop or teardown races."""
        if self._destroyed:
            return None
        jobs = getattr(self, "_scheduled_jobs", None)
        if jobs is None:
            jobs = set()
            self._scheduled_jobs = jobs

        def _wrapped() -> None:
            if self._destroyed:
                return
            try:
                callback()
            except (RuntimeError, tk.TclError):
                pass

        try:
            job_id = self.after(delay_ms, _wrapped)
            jobs.add(job_id)
            return job_id
        except (RuntimeError, tk.TclError):
            return None

    def _on_close(self) -> None:
        self._destroyed = True
        for job_id in list(getattr(self, "_scheduled_jobs", ())):
            try:
                self.after_cancel(job_id)
            except Exception:
                pass
        if hasattr(self, "_scheduled_jobs"):
            self._scheduled_jobs.clear()
        self.destroy()

    def destroy(self) -> None:
        self._destroyed = True
        super().destroy()

    def _on_global_motion(self, event: Any) -> None:
        if hasattr(self, "_bg_field") and event is not None:
            self._bg_field.update_global_pointer(event.x_root, event.y_root)

    # ------------------------------------------------------------------
    # Universal Animated Button/Hover Helper
    # ------------------------------------------------------------------
    def _attach_motion(
        self,
        widget: Any,
        *,
        base_fg: str,
        hover_fg: str,
        press_fg: str | None = None,
        base_border: str | None = None,
        hover_border: str | None = None,
        base_text: str | None = None,
        hover_text: str | None = None,
        surface_hex: str = DSW_BG_BASE,
        transparent_rest: bool = False,
        active_fn: Callable[[], bool] | None = None,
        on_hover: Callable[[], None] | None = None,
        on_leave: Callable[[], None] | None = None,
        ripple_color: str = DSW_ACCENT,
        ripple_on_click: bool = True,
    ) -> None:
        """Bind smooth color interpolation, hand cursor, press feedback, and background ripple to any widget."""
        widget._wx_surface_hex = surface_hex
        if transparent_rest:
            rest_cfg: dict[str, str] = {"fg_color": surface_hex}
            widget._wx_transparent_on_rest = dict(rest_cfg)
        try:
            widget.configure(cursor="hand2")
        except Exception:
            pass

        def _build_state(fg: str, border: str | None, txt: str | None) -> dict[str, str]:
            d: dict[str, str] = {"fg_color": fg}
            if border is not None:
                d["border_color"] = border
            if txt is not None:
                d["text_color"] = txt
            return d

        def _on_enter(_e: Any = None) -> None:
            if active_fn and active_fn():
                return
            if transparent_rest:
                widget._wx_transparent_on_rest = None
            animate_colors(self, widget, _build_state(hover_fg, hover_border, hover_text), duration_ms=130, steps=8)
            if on_hover is not None:
                on_hover()

        def _on_leave(_e: Any = None) -> None:
            if active_fn and active_fn():
                return
            targets = _build_state(base_fg, base_border, base_text)
            if transparent_rest:
                widget._wx_transparent_on_rest = dict(targets)
            animate_colors(self, widget, targets, duration_ms=160, steps=9)
            if on_leave is not None:
                on_leave()

        def _on_press(_e: Any = None) -> None:
            if press_fg is not None:
                try:
                    widget.configure(fg_color=press_fg)
                except Exception:
                    pass
            if ripple_on_click and hasattr(self, "_bg_field"):
                self._bg_field.spawn_ripple_from_widget(widget, color=ripple_color)

        def _on_release(_e: Any = None) -> None:
            target = _build_state(hover_fg, hover_border, hover_text)
            if active_fn and active_fn():
                return
            animate_colors(self, widget, target, duration_ms=110, steps=6)

        widget.bind("<Enter>", _on_enter, add="+")
        widget.bind("<Leave>", _on_leave, add="+")
        widget.bind("<ButtonPress-1>", _on_press, add="+")
        widget.bind("<ButtonRelease-1>", _on_release, add="+")

    # ------------------------------------------------------------------
    # Logging & Background Model Boot
    # ------------------------------------------------------------------
    def _enqueue_log(self, msg: str) -> None:
        self._log_queue.put(msg)

    def _drain_logs(self) -> None:
        if self._destroyed:
            return
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._set_status_caption(msg)
        except queue.Empty:
            pass
        if not self._destroyed:
            self.safe_after(90, self._drain_logs)

    def _boot_model_async(self) -> None:
        def _job() -> None:
            model_id = self.studio.config.model_id or "minillama-demo"
            self.studio.load_model(model_id)
            if not self._destroyed:
                self.safe_after(0, self._sync_model_badges)

        threading.Thread(target=_job, daemon=True).start()

    def _sync_model_badges(self) -> None:
        if self._destroyed:
            return
        try:
            while True:
                self._log_queue.get_nowait()
        except queue.Empty:
            pass
        mid = self.studio.config.model_id or "minillama-demo"
        entry = self.studio.catalog.get(mid)
        short_name = f"{entry.model_id} ({entry.params_human})" if entry is not None else mid
        self._hero_workspace_btn.configure(text=f"📁  {short_name}  ▾")
        self._composer_model_btn.configure(text=f"{mid} ▾")
        if self.studio.harness is not None:
            v = self.studio.harness.vespasian
            self._hero_score_badge.configure(text=f"held-out {v.best_score:.4f}")
            # Keep default YAML sample valid for current workspace tensors
            two_d = [n for n in self.studio.harness.workspace.names() if len(self.studio.harness.workspace.shape(n)) == 2]
            if two_d and "layers.1.mlp.down_proj.weight" in self._yaml_editor.get("1.0", "end"):
                sample_t = next((n for n in two_d if "down_proj" in n or "c_proj" in n), two_d[0])
                self._set_yaml_editor(
                    "transforms:\n"
                    f"  - assert: {{ exists: {{ of: '{sample_t}' }} }}\n"
                    f"  - rescale_outlier_rows_: {{ target: '{sample_t}', min_ratio: 2.5 }}\n"
                    f"  - assert: {{ finite: {{ of: '{sample_t}' }} }}\n"
                )
        self._set_status_caption("")
        self._refresh_tensor_list()

    # ------------------------------------------------------------------
    # 3-Column DSH AppFrame Construction
    # ------------------------------------------------------------------
    def _build_app_frame(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # Left Sidebar
        self.grid_columnconfigure(1, weight=1)  # Center Surface
        self.grid_columnconfigure(2, weight=0)  # Right Inspector Drawer

        self._build_sidebar()
        self._build_center_surface()
        self._build_right_inspector()

    # ------------------------------------------------------------------
    # Column 1: Animated DSH Left Sidebar (`#1b1b1c`, 252px <-> 56px)
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

        # 1. Brand Row (`WeightX` + animated collapse toggle)
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
        self._attach_motion(
            self._brand_mark_btn,
            base_fg=DSW_SIDEBAR_FILL,
            hover_fg=DSW_SELECTOR,
            press_fg=DSW_NAV_ACTIVE,
            surface_hex=DSW_SIDEBAR_FILL,
            transparent_rest=True,
            on_hover=lambda: self._set_status_caption("WeightX · Главный экран (новый сеанс)"),
            on_leave=lambda: self._set_status_caption(""),
        )

        self._brand_name_lbl = ctk.CTkLabel(
            self._brand_row,
            text="WeightX",
            font=ctk.CTkFont(family=FONT_UI, size=15, weight="bold"),
            text_color=DSW_TEXT_PRIMARY,
            anchor="w",
            cursor="hand2",
        )
        self._brand_name_lbl.grid(row=0, column=1, sticky="w")
        self._brand_name_lbl.bind("<Button-1>", lambda _e: self.new_session())
        self._brand_name_lbl.bind("<Enter>", lambda _e: self._brand_name_lbl.configure(text_color=DSW_ACCENT))
        self._brand_name_lbl.bind("<Leave>", lambda _e: self._brand_name_lbl.configure(text_color=DSW_TEXT_PRIMARY))

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
        self._attach_motion(
            self._collapse_btn,
            base_fg=DSW_SIDEBAR_FILL,
            hover_fg=DSW_SELECTOR,
            press_fg=DSW_NAV_ACTIVE,
            base_text=DSW_TEXT_TERTIARY,
            hover_text=DSW_TEXT_PRIMARY,
            surface_hex=DSW_SIDEBAR_FILL,
            transparent_rest=True,
            on_hover=lambda: self._set_status_caption("Свернуть / развернуть боковую панель (Ctrl+B)"),
            on_leave=lambda: self._set_status_caption(""),
        )

        # 2. Animated `+ Новый сеанс` Button
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
        self._attach_motion(
            self._new_session_btn,
            base_fg=DSW_LAYER_1,
            hover_fg=DSW_INPUT_MAJOR,
            press_fg=DSW_NAV_ACTIVE,
            base_border=DSW_BORDER_L2,
            hover_border=DSW_ACCENT,
            base_text=DSW_TEXT_PRIMARY,
            hover_text=DSW_TEXT_PRIMARY,
            surface_hex=DSW_SIDEBAR_FILL,
            on_hover=lambda: self._set_status_caption("Создать новый чистый сеанс (Ctrl+N)"),
            on_leave=lambda: self._set_status_caption(""),
        )

        # 3. Scrollable Interactive Sessions & Collapsible Skills List
        self._sidebar_scroll = ctk.CTkScrollableFrame(
            self._sidebar,
            fg_color="transparent",
            scrollbar_button_color=DSW_SIDEBAR_FILL,
            scrollbar_button_hover_color=DSW_SELECTOR,
        )
        self._sidebar_scroll.grid(row=2, column=0, sticky="nsew", padx=6, pady=0)
        self._sidebar_scroll.grid_columnconfigure(0, weight=1)
        self._render_sidebar_items()

        # 4. Bottom-Pinned Animated Footer Controls
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
        self._attach_motion(
            self._settings_btn,
            base_fg=DSW_SIDEBAR_FILL,
            hover_fg=DSW_INPUT_MAJOR,
            press_fg=DSW_NAV_ACTIVE,
            base_text=DSW_TEXT_SECONDARY,
            hover_text=DSW_TEXT_PRIMARY,
            surface_hex=DSW_SIDEBAR_FILL,
            transparent_rest=True,
            on_hover=lambda: self._set_status_caption("Выбор модели, LLM-провайдера и API-ключа"),
            on_leave=lambda: self._set_status_caption(""),
        )

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
        self._attach_motion(
            self._inspector_toggle_btn,
            base_fg=DSW_SIDEBAR_FILL,
            hover_fg=DSW_ACCENT_TERTIARY,
            press_fg=DSW_NAV_ACTIVE,
            base_text=DSW_TEXT_TERTIARY,
            hover_text=DSW_ACCENT,
            surface_hex=DSW_SIDEBAR_FILL,
            transparent_rest=True,
            active_fn=lambda: self._inspector_open,
            on_hover=lambda: self._set_status_caption("Открыть / скрыть инспектор тензоров и YAML (Ctrl+I)"),
            on_leave=lambda: self._set_status_caption(""),
        )

    def _toggle_sidebar_skills(self) -> None:
        self._sidebar_skills_open = not self._sidebar_skills_open
        self._render_sidebar_items()

    def _delete_session(self, idx: int) -> None:
        if len(self._sessions) <= 1 or not (0 <= idx < len(self._sessions)):
            return
        self._sessions.pop(idx)
        self._active_session_idx = max(0, min(self._active_session_idx, len(self._sessions) - 1))
        self._render_sidebar_items()
        self._refresh_center_mode()

    def _render_sidebar_items(self) -> None:
        for child in self._sidebar_scroll.winfo_children():
            child.destroy()
        if self._sidebar_collapsed:
            return

        row_idx = 0
        sec_lbl = ctk.CTkLabel(
            self._sidebar_scroll,
            text="Сеансы",
            font=ctk.CTkFont(family=FONT_UI, size=11),
            text_color=DSW_TEXT_CAPTION,
            anchor="w",
        )
        sec_lbl.grid(row=row_idx, column=0, sticky="ew", padx=8, pady=(2, 4))
        row_idx += 1

        for idx, sess in enumerate(self._sessions):
            is_active = idx == self._active_session_idx
            item_wrap = ctk.CTkFrame(
                self._sidebar_scroll,
                fg_color=DSW_NAV_ACTIVE if is_active else "transparent",
                corner_radius=8,
                border_width=1 if is_active else 0,
                border_color=DSW_BORDER_L3,
            )
            item_wrap.grid(row=row_idx, column=0, sticky="ew", padx=2, pady=1)
            item_wrap.grid_columnconfigure(1, weight=1)
            row_idx += 1

            # Active subtle accent indicator dot
            dot = ctk.CTkLabel(
                item_wrap,
                text="▍" if is_active else "·",
                font=ctk.CTkFont(family=FONT_UI, size=11, weight="bold"),
                text_color=DSW_ACCENT if is_active else DSW_TEXT_CAPTION,
                width=12,
            )
            dot.grid(row=0, column=0, padx=(6, 0))

            btn = ctk.CTkButton(
                item_wrap,
                text=sess.title,
                height=30,
                corner_radius=6,
                fg_color="transparent",
                hover_color=DSW_NAV_ACTIVE if is_active else DSW_INPUT_MAJOR,
                text_color=DSW_TEXT_PRIMARY if is_active else DSW_TEXT_SECONDARY,
                font=ctk.CTkFont(family=FONT_UI, size=12, weight="bold" if is_active else "normal"),
                anchor="w",
                command=lambda i=idx: self._select_session(i),
            )
            btn.grid(row=0, column=1, sticky="ew", padx=(2, 2))
            self._attach_motion(
                btn,
                base_fg=DSW_NAV_ACTIVE if is_active else DSW_SIDEBAR_FILL,
                hover_fg=DSW_NAV_ACTIVE if is_active else DSW_INPUT_MAJOR,
                press_fg=DSW_SELECTOR,
                base_text=DSW_TEXT_PRIMARY if is_active else DSW_TEXT_SECONDARY,
                hover_text=DSW_TEXT_PRIMARY,
                surface_hex=DSW_SIDEBAR_FILL,
                transparent_rest=True,
                active_fn=lambda a=is_active: a,
                on_hover=lambda s=sess: self._set_status_caption(f"Сеанс «{s.title}» — {s.subtitle}"),
                on_leave=lambda: self._set_status_caption(""),
            )

            if len(self._sessions) > 1:
                del_btn = ctk.CTkButton(
                    item_wrap,
                    text="×",
                    width=22,
                    height=22,
                    corner_radius=6,
                    fg_color="transparent",
                    hover_color=DSW_SELECTOR,
                    text_color=DSW_TEXT_CAPTION,
                    font=ctk.CTkFont(family=FONT_UI, size=12),
                    command=lambda i=idx: self._delete_session(i),
                )
                del_btn.grid(row=0, column=2, padx=(0, 4))
                self._attach_motion(
                    del_btn,
                    base_fg=DSW_NAV_ACTIVE if is_active else DSW_SIDEBAR_FILL,
                    hover_fg=DSW_SELECTOR,
                    press_fg=DSW_ERROR,
                    base_text=DSW_TEXT_CAPTION,
                    hover_text=DSW_ERROR,
                    surface_hex=DSW_SIDEBAR_FILL,
                    transparent_rest=True,
                )

        # Collapsible interactive Quick Skills group in sidebar
        acc_btn = ctk.CTkButton(
            self._sidebar_scroll,
            text=f"{'▾' if self._sidebar_skills_open else '▸'}  Скиллы ({len(WSP_SKILLS)})",
            height=28,
            corner_radius=8,
            fg_color="transparent",
            hover_color=DSW_LAYER_1,
            text_color=DSW_TEXT_CAPTION,
            font=ctk.CTkFont(family=FONT_UI, size=11, weight="bold"),
            anchor="w",
            command=self._toggle_sidebar_skills,
        )
        acc_btn.grid(row=row_idx, column=0, sticky="ew", padx=2, pady=(12, 2))
        self._attach_motion(
            acc_btn,
            base_fg=DSW_SIDEBAR_FILL,
            hover_fg=DSW_LAYER_1,
            press_fg=DSW_SELECTOR,
            base_text=DSW_TEXT_CAPTION,
            hover_text=DSW_ACCENT,
            surface_hex=DSW_SIDEBAR_FILL,
            transparent_rest=True,
            on_hover=lambda: self._set_status_caption("Быстрый вызов WSP скиллов из боковой панели"),
            on_leave=lambda: self._set_status_caption(""),
        )
        row_idx += 1

        if self._sidebar_skills_open:
            for sk in WSP_SKILLS:
                sk_btn = ctk.CTkButton(
                    self._sidebar_scroll,
                    text=f"/skill {sk.skill_id}",
                    height=26,
                    corner_radius=7,
                    fg_color="transparent",
                    hover_color=DSW_ACCENT_TERTIARY,
                    text_color=DSW_TEXT_TERTIARY,
                    font=ctk.CTkFont(family=FONT_MONO, size=11),
                    anchor="w",
                    command=lambda s=sk: self.trigger_skill(s),
                )
                sk_btn.grid(row=row_idx, column=0, sticky="ew", padx=6, pady=1)
                self._attach_motion(
                    sk_btn,
                    base_fg=DSW_SIDEBAR_FILL,
                    hover_fg=DSW_ACCENT_TERTIARY,
                    press_fg=DSW_ACCENT_SOFT,
                    base_text=DSW_TEXT_TERTIARY,
                    hover_text=DSW_ACCENT,
                    surface_hex=DSW_SIDEBAR_FILL,
                    transparent_rest=True,
                    on_hover=lambda s=sk: self._set_status_caption(f"/skill {s.skill_id} — {s.pill_label}: {s.short_hint}"),
                    on_leave=lambda: self._set_status_caption(""),
                )
                row_idx += 1

    def toggle_sidebar(self) -> None:
        """Smoothly animate sidebar width between DSH 252px expanded and 56px icon rail."""
        if self._sidebar_anim_job is not None:
            try:
                self.after_cancel(self._sidebar_anim_job)
            except Exception:
                pass
            self._sidebar_anim_job = None

        self._sidebar_collapsed = not self._sidebar_collapsed
        start_w = self._sidebar.cget("width")
        end_w = 56 if self._sidebar_collapsed else 252

        if self._sidebar_collapsed:
            self._brand_name_lbl.grid_remove()
            self._brand_mark_btn.grid_remove()
            self._new_session_btn.configure(text="+", width=36, anchor="center")
            self._settings_btn.configure(text="⚙", width=36, anchor="center")
            self._inspector_toggle_btn.grid_remove()
            self._render_sidebar_items()
        else:
            self._brand_mark_btn.grid()
            self._brand_name_lbl.grid()
            self._new_session_btn.configure(text="+  Новый сеанс", anchor="w")
            self._settings_btn.configure(text="⚙  Настройки", anchor="w")
            self._inspector_toggle_btn.grid()

        steps = 9
        def _step(idx: int) -> None:
            if self._destroyed:
                return
            t = ease_in_out_cubic(idx / float(steps))
            w = int(round(start_w + (end_w - start_w) * t))
            self._sidebar.configure(width=w)
            if idx < steps:
                self._sidebar_anim_job = self.after(15, lambda: _step(idx + 1))
            else:
                self._sidebar_anim_job = None
                if not self._sidebar_collapsed:
                    self._render_sidebar_items()

        _step(1)

    # ------------------------------------------------------------------
    # Column 2: DSH Uncluttered Center Surface + Interactive Neural Canvas
    # ------------------------------------------------------------------
    def _build_center_surface(self) -> None:
        self._center = ctk.CTkFrame(self, fg_color=DSW_BG_BASE, corner_radius=0)
        self._center.grid(row=0, column=1, sticky="nsew")
        self._center.grid_rowconfigure(1, weight=1)
        self._center.grid_columnconfigure(0, weight=1)

        # Subtle top bar (only visible in Conversation mode; hidden on Hero!)
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

        self._chat_top_bg_btn = ctk.CTkButton(
            self._chat_topbar,
            text="✦ Нейро-сеть",
            height=26,
            width=112,
            corner_radius=13,
            fg_color=DSW_LAYER_1,
            hover_color=DSW_SELECTOR,
            text_color=DSW_TEXT_TERTIARY,
            font=ctk.CTkFont(family=FONT_UI, size=11),
            command=self._cycle_bg_mode,
        )
        self._chat_top_bg_btn.grid(row=0, column=1, sticky="e", padx=(0, 8), pady=8)
        self._attach_motion(
            self._chat_top_bg_btn,
            base_fg=DSW_LAYER_1,
            hover_fg=DSW_ACCENT_TERTIARY,
            press_fg=DSW_SELECTOR,
            base_text=DSW_TEXT_TERTIARY,
            hover_text=DSW_ACCENT,
            surface_hex=DSW_BG_BASE,
        )

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
        self._chat_top_inspector_btn.grid(row=0, column=2, sticky="e", padx=(0, 16), pady=8)
        self._attach_motion(
            self._chat_top_inspector_btn,
            base_fg=DSW_LAYER_1,
            hover_fg=DSW_SELECTOR,
            press_fg=DSW_NAV_ACTIVE,
            base_text=DSW_TEXT_SECONDARY,
            hover_text=DSW_TEXT_PRIMARY,
            surface_hex=DSW_BG_BASE,
        )

        # Main Stage Container with Interactive Canvas underneath
        self._stage = ctk.CTkFrame(self._center, fg_color=DSW_BG_BASE, corner_radius=0)
        self._stage.grid(row=1, column=0, sticky="nsew")
        self._stage.grid_columnconfigure(0, weight=1)

        self._bg_field = _InteractiveNeuralBackground(self, self._stage)

        # Floating Hero background-mode selector chip in bottom-right of Hero
        self._hero_bg_pill = ctk.CTkButton(
            self._stage,
            text="✦ Фон: Нейро-сеть",
            height=24,
            width=128,
            corner_radius=12,
            fg_color=DSW_SIDEBAR_FILL,
            hover_color=DSW_INPUT_MAJOR,
            border_width=1,
            border_color=DSW_BORDER_L2,
            text_color=DSW_TEXT_CAPTION,
            font=ctk.CTkFont(family=FONT_UI, size=11),
            command=self._cycle_bg_mode,
        )
        self._hero_bg_pill.place(relx=1.0, rely=1.0, x=-18, y=-14, anchor="se")
        self._attach_motion(
            self._hero_bg_pill,
            base_fg=DSW_SIDEBAR_FILL,
            hover_fg=DSW_ACCENT_TERTIARY,
            press_fg=DSW_SELECTOR,
            base_border=DSW_BORDER_L2,
            hover_border=DSW_ACCENT,
            base_text=DSW_TEXT_CAPTION,
            hover_text=DSW_ACCENT,
            surface_hex=DSW_BG_BASE,
            on_hover=lambda: self._set_status_caption("Переключить интерактивный режим фона (Нейро-сеть / Спектр SVD / Тензор-поток)"),
            on_leave=lambda: self._set_status_caption(""),
        )

        # Build Hero Header, Conversation Feed, and Shared Composer+Skills Dock
        self._build_hero_view()
        self._build_conversation_view()
        self._build_shared_composer_and_pills()
        self._refresh_center_mode()

    def _cycle_bg_mode(self) -> str:
        label = self._bg_field.cycle_mode()
        self._hero_bg_pill.configure(text=f"✦ Фон: {label}")
        self._chat_top_bg_btn.configure(text=f"✦ {label}")
        self._set_status_caption(f"Интерактивный фон: {label} (кликните по фону для волны)")
        return label

    def _build_hero_view(self) -> None:
        """Build DSH 1:1 `HeroShell` header (`[Logo 36px] WeightX (WSP)` + interactive workspace row)."""
        self._hero_stack = ctk.CTkFrame(self._stage, fg_color=DSW_BG_BASE, width=680, corner_radius=16)
        self._hero_stack.grid_columnconfigure(0, weight=1)

        headline_row = ctk.CTkFrame(self._hero_stack, fg_color="transparent")
        headline_row.grid(row=0, column=0, pady=(4, 14), padx=12)

        self._logo_hero = _load_brand_logo(36)
        logo_lbl = ctk.CTkLabel(headline_row, text="", image=self._logo_hero, cursor="hand2")
        logo_lbl.pack(side="left", padx=(0, 12))
        logo_lbl.bind("<Button-1>", lambda _e: self._bg_field.spawn_ripple_from_widget(logo_lbl, color=DSW_ACCENT))

        title_lbl = ctk.CTkLabel(
            headline_row,
            text="WeightX",
            font=ctk.CTkFont(family=FONT_UI, size=26, weight="bold"),
            text_color=DSW_TEXT_PRIMARY,
            cursor="hand2",
        )
        title_lbl.pack(side="left", padx=(0, 8))
        title_lbl.bind("<Enter>", lambda _e: title_lbl.configure(text_color=DSW_ACCENT))
        title_lbl.bind("<Leave>", lambda _e: title_lbl.configure(text_color=DSW_TEXT_PRIMARY))
        title_lbl.bind("<Button-1>", lambda _e: self._bg_field.spawn_ripple_from_widget(title_lbl, color=DSW_ACCENT))

        self._wsp_badge_btn = ctk.CTkButton(
            headline_row,
            text="WSP",
            height=22,
            width=46,
            font=ctk.CTkFont(family=FONT_MONO, size=11, weight="bold"),
            text_color=DSW_ACCENT,
            fg_color=DSW_ACCENT_TERTIARY,
            hover_color=DSW_ACCENT_SOFT,
            corner_radius=999,
            command=lambda: self._bg_field.spawn_ripple_from_widget(self._wsp_badge_btn, color=DSW_ACCENT),
        )
        self._wsp_badge_btn.pack(side="left", pady=(2, 0))
        self._attach_motion(
            self._wsp_badge_btn,
            base_fg=DSW_ACCENT_TERTIARY,
            hover_fg=DSW_ACCENT_SOFT,
            press_fg=DSW_ACCENT,
            base_text=DSW_ACCENT,
            hover_text=DSW_TEXT_PRIMARY,
            surface_hex=DSW_BG_BASE,
            on_hover=lambda: self._set_status_caption("Weight-Space Programming · Клик — энергетический импульс по графу"),
            on_leave=lambda: self._set_status_caption(""),
        )

        ws_row = ctk.CTkFrame(self._hero_stack, fg_color="transparent", width=680)
        ws_row.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

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
            command=self._toggle_model_popover,
        )
        self._hero_workspace_btn.pack(side="left")
        self._attach_motion(
            self._hero_workspace_btn,
            base_fg=DSW_BG_BASE,
            hover_fg=DSW_LAYER_1,
            press_fg=DSW_SELECTOR,
            base_text=DSW_TEXT_PRIMARY,
            hover_text=DSW_ACCENT,
            surface_hex=DSW_BG_BASE,
            transparent_rest=True,
            on_hover=lambda: self._set_status_caption("Выбрать целевую модель из каталога WeightX"),
            on_leave=lambda: self._set_status_caption(""),
        )

        self._hero_score_badge = ctk.CTkButton(
            ws_row,
            text="held-out готов",
            height=24,
            width=110,
            corner_radius=999,
            fg_color=DSW_LAYER_1,
            hover_color=DSW_SUCCESS_BG,
            font=ctk.CTkFont(family=FONT_MONO, size=11),
            text_color=DSW_TEXT_CAPTION,
            command=lambda: self.trigger_skill(WSP_SKILLS[7]),
        )
        self._hero_score_badge.pack(side="right", padx=6)
        self._attach_motion(
            self._hero_score_badge,
            base_fg=DSW_LAYER_1,
            hover_fg=DSW_SUCCESS_BG,
            press_fg=DSW_SELECTOR,
            base_text=DSW_TEXT_CAPTION,
            hover_text=DSW_SUCCESS,
            surface_hex=DSW_BG_BASE,
            ripple_color=DSW_SUCCESS,
            on_hover=lambda: self._set_status_caption("Проверить инварианты Vespasian (/skill vespasian-guard)"),
            on_leave=lambda: self._set_status_caption(""),
        )

    def _build_shared_composer_and_pills(self) -> None:
        """Create the DSH Composer Card (`#2c2c2e`), live `/` autocomplete, and compact 2-word Skill Pills."""
        self._composer_dock = ctk.CTkFrame(self._stage, fg_color=DSW_BG_BASE, width=680, corner_radius=20)
        self._composer_dock.grid_columnconfigure(0, weight=1)

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

        # Live filtered Slash `/` Skills Popup Menu (hidden unless `/` clicked or typed)
        self._slash_popup = ctk.CTkFrame(
            self._composer_card,
            fg_color=DSW_SIDEBAR_FILL,
            corner_radius=12,
            border_width=1,
            border_color=DSW_BORDER_L3,
        )
        self._slash_popup.grid_columnconfigure(0, weight=1)

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
        self._input_box.bind("<Tab>", self._on_input_tab)
        self._input_box.bind("<Up>", lambda e: self._move_slash_selection(-1, e))
        self._input_box.bind("<Down>", lambda e: self._move_slash_selection(1, e))
        self._input_box.bind("<KeyRelease>", self._on_input_key_release)

        # Bottom Control Row inside Composer Card
        ctrl_row = ctk.CTkFrame(self._composer_card, fg_color="transparent", height=34)
        ctrl_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 10))
        ctrl_row.grid_columnconfigure(3, weight=1)

        # Leading `/` Skill Trigger button
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
        self._attach_motion(
            self._slash_btn,
            base_fg=DSW_SELECTOR,
            hover_fg=DSW_ACCENT_TERTIARY,
            press_fg=DSW_ACCENT_SOFT,
            base_text=DSW_TEXT_PRIMARY,
            hover_text=DSW_ACCENT,
            surface_hex=DSW_INPUT_MAJOR,
            on_hover=lambda: self._set_status_caption("Открыть меню скиллов (/skill <название>)"),
            on_leave=lambda: self._set_status_caption(""),
        )

        # Model Pill (opens interactive Model Selector Popover!)
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
            command=self._toggle_model_popover,
        )
        self._composer_model_btn.grid(row=0, column=1, padx=(0, 6))
        self._attach_motion(
            self._composer_model_btn,
            base_fg=DSW_SELECTOR,
            hover_fg=DSW_NAV_ACTIVE,
            press_fg=DSW_ACCENT_TERTIARY,
            base_text=DSW_TEXT_SECONDARY,
            hover_text=DSW_TEXT_PRIMARY,
            surface_hex=DSW_INPUT_MAJOR,
            on_hover=lambda: self._set_status_caption("Выбрать модель из каталога"),
            on_leave=lambda: self._set_status_caption(""),
        )

        # Provider / Brain Mode Pill (opens interactive Provider Selector Popover!)
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
            command=self._toggle_provider_popover,
        )
        self._composer_provider_btn.grid(row=0, column=2, padx=(0, 6))
        self._attach_motion(
            self._composer_provider_btn,
            base_fg=DSW_INPUT_MAJOR,
            hover_fg=DSW_SELECTOR,
            press_fg=DSW_NAV_ACTIVE,
            base_text=DSW_TEXT_TERTIARY,
            hover_text=DSW_TEXT_PRIMARY,
            surface_hex=DSW_INPUT_MAJOR,
            transparent_rest=True,
            on_hover=lambda: self._set_status_caption("Выбрать мозг-провайдер (Provider Layer)"),
            on_leave=lambda: self._set_status_caption(""),
        )

        # Trailing Circular Send Button (`↑` / animated spinner when busy)
        self._send_btn = ctk.CTkButton(
            ctrl_row,
            text="↑",
            width=30,
            height=30,
            corner_radius=15,
            fg_color=DSW_TEXT_PRIMARY,
            hover_color=DSW_ACCENT,
            text_color=DSW_TEXT_INVERT,
            font=ctk.CTkFont(family=FONT_UI, size=15, weight="bold"),
            command=self.submit_composer,
        )
        self._send_btn.grid(row=0, column=4, padx=(6, 2))
        self._attach_motion(
            self._send_btn,
            base_fg=DSW_TEXT_PRIMARY,
            hover_fg=DSW_ACCENT,
            press_fg=DSW_ACCENT_STRONG,
            base_text=DSW_TEXT_INVERT,
            hover_text=DSW_TEXT_INVERT,
            surface_hex=DSW_INPUT_MAJOR,
            active_fn=lambda: self._busy,
            on_hover=lambda: self._set_status_caption("Запустить скилл или цикл редактирования (Enter)"),
            on_leave=lambda: self._set_status_caption(""),
        )

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
                hover_color=DSW_ACCENT_TERTIARY,
                border_width=1,
                border_color=DSW_BORDER_L2,
                text_color=DSW_TEXT_SECONDARY,
                font=ctk.CTkFont(family=FONT_UI, size=12),
                command=lambda skill=sk: self.trigger_skill(skill),
            )
            pill.pack(side="left", padx=4)
            self._attach_motion(
                pill,
                base_fg=DSW_LAYER_1,
                hover_fg=DSW_ACCENT_TERTIARY,
                press_fg=DSW_ACCENT_SOFT,
                base_border=DSW_BORDER_L2,
                hover_border=DSW_ACCENT,
                base_text=DSW_TEXT_SECONDARY,
                hover_text=DSW_TEXT_PRIMARY,
                surface_hex=DSW_BG_BASE,
                on_hover=lambda s=sk: self._set_status_caption(
                    f"Скилл: /skill {s.skill_id} — {s.short_hint}  (ПКМ — вставить)"
                ),
                on_leave=lambda: self._set_status_caption(""),
            )
            pill.bind("<Button-3>", lambda _e, s=sk: self._insert_skill_into_composer(s))

        # Subtle 1-line status/pipeline hint below the pills (`#81858c`, `11px`)
        self._status_caption = ctk.CTkLabel(
            self._composer_dock,
            text="",
            font=ctk.CTkFont(family=FONT_UI, size=11),
            text_color=DSW_TEXT_CAPTION,
            height=18,
        )
        self._status_caption.grid(row=2, column=0, pady=(4, 2))

    # ------------------------------------------------------------------
    # Interactive Floating Model & Provider Popovers
    # ------------------------------------------------------------------
    def _close_all_popups(self) -> None:
        self._toggle_slash_popup(force=False)
        if self._popover_widget is not None:
            try:
                self._popover_widget.destroy()
            except Exception:
                pass
            self._popover_widget = None

    def _toggle_model_popover(self) -> None:
        if self._popover_widget is not None and getattr(self._popover_widget, "_wx_kind", "") == "model":
            self._close_all_popups()
            return
        self._close_all_popups()

        pop = ctk.CTkFrame(
            self._stage,
            fg_color=DSW_SIDEBAR_FILL,
            corner_radius=14,
            border_width=1,
            border_color=DSW_ACCENT,
            width=360,
        )
        pop._wx_kind = "model"
        self._popover_widget = pop
        ctk.CTkLabel(
            pop,
            text="Целевая модель (WeightX Каталог)",
            font=ctk.CTkFont(family=FONT_UI, size=12, weight="bold"),
            text_color=DSW_TEXT_PRIMARY,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        cur_mid = self.studio.config.model_id or "minillama-demo"
        for entry in self.studio.catalog.list_entries()[:9]:
            is_cur = entry.model_id == cur_mid
            dot = "●" if entry.local_cached else "○"
            btn = ctk.CTkButton(
                pop,
                text=f"{dot}  {entry.model_id:<22}  {entry.params_human} · {entry.num_layers}L",
                height=28,
                corner_radius=8,
                fg_color=DSW_ACCENT_TERTIARY if is_cur else "transparent",
                hover_color=DSW_INPUT_MAJOR,
                text_color=DSW_ACCENT if is_cur else DSW_TEXT_SECONDARY,
                font=ctk.CTkFont(family=FONT_MONO, size=11),
                anchor="w",
                command=lambda mid=entry.model_id: self._select_model_from_popover(mid),
            )
            btn.pack(fill="x", padx=8, pady=1)
            self._attach_motion(
                btn,
                base_fg=DSW_ACCENT_TERTIARY if is_cur else DSW_SIDEBAR_FILL,
                hover_fg=DSW_INPUT_MAJOR,
                press_fg=DSW_SELECTOR,
                base_text=DSW_ACCENT if is_cur else DSW_TEXT_SECONDARY,
                hover_text=DSW_TEXT_PRIMARY,
                surface_hex=DSW_SIDEBAR_FILL,
                transparent_rest=not is_cur,
                active_fn=lambda c=is_cur: c,
            )

        pop.place(relx=0.5, rely=0.52, anchor="center")
        pop.lift()

    def _select_model_from_popover(self, model_id: str) -> None:
        self._close_all_popups()
        self._set_status_caption(f"Загрузка целевой модели «{model_id}»…")

        def _load() -> None:
            self.studio.load_model(model_id)
            if not self._destroyed:
                self.safe_after(0, self._sync_model_badges)

        threading.Thread(target=_load, daemon=True).start()

    def _toggle_provider_popover(self) -> None:
        if self._popover_widget is not None and getattr(self._popover_widget, "_wx_kind", "") == "provider":
            self._close_all_popups()
            return
        self._close_all_popups()

        pop = ctk.CTkFrame(
            self._stage,
            fg_color=DSW_SIDEBAR_FILL,
            corner_radius=14,
            border_width=1,
            border_color=DSW_ACCENT,
            width=360,
        )
        pop._wx_kind = "provider"
        self._popover_widget = pop
        ctk.CTkLabel(
            pop,
            text="Мозг-провайдер (Provider Layer)",
            font=ctk.CTkFont(family=FONT_UI, size=12, weight="bold"),
            text_color=DSW_TEXT_PRIMARY,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        cur_p = self.studio.config.provider_kind or PROVIDER_KINDS[0]
        for kind in PROVIDER_KINDS:
            is_cur = kind == cur_p
            btn = ctk.CTkButton(
                pop,
                text=f"{'● ' if is_cur else '  '}{kind}",
                height=28,
                corner_radius=8,
                fg_color=DSW_ACCENT_TERTIARY if is_cur else "transparent",
                hover_color=DSW_INPUT_MAJOR,
                text_color=DSW_ACCENT if is_cur else DSW_TEXT_SECONDARY,
                font=ctk.CTkFont(family=FONT_UI, size=12),
                anchor="w",
                command=lambda k=kind: self._select_provider_from_popover(k),
            )
            btn.pack(fill="x", padx=8, pady=1)
            self._attach_motion(
                btn,
                base_fg=DSW_ACCENT_TERTIARY if is_cur else DSW_SIDEBAR_FILL,
                hover_fg=DSW_INPUT_MAJOR,
                press_fg=DSW_SELECTOR,
                base_text=DSW_ACCENT if is_cur else DSW_TEXT_SECONDARY,
                hover_text=DSW_TEXT_PRIMARY,
                surface_hex=DSW_SIDEBAR_FILL,
                transparent_rest=not is_cur,
                active_fn=lambda c=is_cur: c,
            )

        adv = ctk.CTkButton(
            pop,
            text="⚙  Указать API-ключ / URL…",
            height=28,
            corner_radius=8,
            fg_color=DSW_LAYER_1,
            hover_color=DSW_SELECTOR,
            text_color=DSW_TEXT_PRIMARY,
            font=ctk.CTkFont(family=FONT_UI, size=11),
            command=lambda: (self._close_all_popups(), self._open_settings_modal()),
        )
        adv.pack(fill="x", padx=8, pady=(6, 10))
        self._attach_motion(
            adv,
            base_fg=DSW_LAYER_1,
            hover_fg=DSW_SELECTOR,
            press_fg=DSW_NAV_ACTIVE,
            surface_hex=DSW_SIDEBAR_FILL,
        )

        pop.place(relx=0.5, rely=0.52, anchor="center")
        pop.lift()

    def _select_provider_from_popover(self, kind: str) -> None:
        self._close_all_popups()
        self.studio.set_provider(kind)
        short_p = kind.split(" ")[0]
        self._composer_provider_btn.configure(text=f"{short_p} ▾")
        self._set_status_caption(f"Мозг-провайдер: {kind}")

    # ------------------------------------------------------------------
    # Conversation View & Interactive Cards
    # ------------------------------------------------------------------
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
        """Switch seamlessly between Hero Empty State and Conversation State."""
        sess = self._sessions[self._active_session_idx]
        is_hero = len(sess.messages) == 0

        if is_hero:
            self._chat_topbar.grid_remove()
            self._conv_scroll.grid_remove()
            self._stage.grid_rowconfigure(0, weight=2)
            self._stage.grid_rowconfigure(1, weight=0)
            self._stage.grid_rowconfigure(2, weight=0)
            self._stage.grid_rowconfigure(3, weight=3)
            self._hero_stack.grid(row=1, column=0, padx=32, pady=(0, 6))
            self._composer_dock.grid(row=2, column=0, padx=32, pady=(0, 0))
            self._hero_bg_pill.place(relx=1.0, rely=1.0, x=-18, y=-14, anchor="se")
            self._hero_bg_pill.lift()
        else:
            self._hero_bg_pill.place_forget()
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

        self._bg_field.ensure_z_order()

    def _copy_to_clipboard(self, text: str) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._set_status_caption("Скопировано в буфер обмена ✓")
        except Exception:
            pass

    def _render_conversation_messages(self) -> None:
        if self._destroyed:
            return
        for child in self._conv_scroll.winfo_children():
            child.destroy()

        sess = self._sessions[self._active_session_idx]
        for idx, msg in enumerate(sess.messages):
            row = ctk.CTkFrame(self._conv_scroll, fg_color="transparent")
            row.grid(row=idx, column=0, sticky="ew", pady=6)
            row.grid_columnconfigure(0, weight=1)

            if msg.role == "user":
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
                bubble.bind("<Enter>", lambda _e, b=bubble: animate_colors(self, b, {"border_color": DSW_ACCENT}, duration_ms=120, steps=6))
                bubble.bind("<Leave>", lambda _e, b=bubble: animate_colors(self, b, {"border_color": DSW_BORDER_L2}, duration_ms=150, steps=6))
            else:
                card = ctk.CTkFrame(
                    row,
                    fg_color=DSW_SIDEBAR_FILL,
                    corner_radius=14,
                    border_width=1,
                    border_color=DSW_BORDER_L2,
                )
                card.pack(fill="x", padx=8)
                card.bind("<Enter>", lambda _e, c=card: animate_colors(self, c, {"border_color": DSW_BORDER_L3}, duration_ms=120, steps=6))
                card.bind("<Leave>", lambda _e, c=card: animate_colors(self, c, {"border_color": DSW_BORDER_L2}, duration_ms=150, steps=6))

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

                # Interactive card action buttons (Copy, YAML -> Inspector, Badge)
                copy_btn = ctk.CTkButton(
                    hdr,
                    text="⧉",
                    width=26,
                    height=22,
                    corner_radius=6,
                    fg_color="transparent",
                    hover_color=DSW_INPUT_MAJOR,
                    text_color=DSW_TEXT_CAPTION,
                    font=ctk.CTkFont(family=FONT_UI, size=11),
                    command=lambda txt=msg.body: self._copy_to_clipboard(txt),
                )
                copy_btn.pack(side="right", padx=(4, 0))
                self._attach_motion(
                    copy_btn,
                    base_fg=DSW_SIDEBAR_FILL,
                    hover_fg=DSW_INPUT_MAJOR,
                    press_fg=DSW_SELECTOR,
                    base_text=DSW_TEXT_CAPTION,
                    hover_text=DSW_TEXT_PRIMARY,
                    surface_hex=DSW_SIDEBAR_FILL,
                    transparent_rest=True,
                )

                if msg.skill_result is not None:
                    if msg.skill_result.yaml_plan.strip():
                        yaml_btn = ctk.CTkButton(
                            hdr,
                            text="YAML ◨",
                            width=64,
                            height=22,
                            corner_radius=6,
                            fg_color=DSW_LAYER_1,
                            hover_color=DSW_ACCENT_TERTIARY,
                            text_color=DSW_TEXT_TERTIARY,
                            font=ctk.CTkFont(family=FONT_MONO, size=10),
                            command=lambda p=msg.skill_result.yaml_plan: self._send_yaml_to_inspector(p),
                        )
                        yaml_btn.pack(side="right", padx=(4, 0))
                        self._attach_motion(
                            yaml_btn,
                            base_fg=DSW_LAYER_1,
                            hover_fg=DSW_ACCENT_TERTIARY,
                            press_fg=DSW_SELECTOR,
                            base_text=DSW_TEXT_TERTIARY,
                            hover_text=DSW_ACCENT,
                            surface_hex=DSW_SIDEBAR_FILL,
                        )

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

        self.after(30, lambda: self._conv_scroll._parent_canvas.yview_moveto(1.0) if not self._destroyed else None)

    def _send_yaml_to_inspector(self, plan_text: str) -> None:
        self._set_yaml_editor(plan_text)
        if not self._inspector_open:
            self.toggle_inspector()
        self._insp_tabs.set("YAML План")
        self._set_status_caption("YAML-план загружен в Инспектор")

    # ------------------------------------------------------------------
    # Column 3: Animated Right Inspector Drawer (0 <-> 360px)
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

        close_insp_btn = ctk.CTkButton(
            ihdr,
            text="✕",
            width=26,
            height=26,
            corner_radius=6,
            fg_color="transparent",
            hover_color=DSW_LAYER_1,
            text_color=DSW_TEXT_TERTIARY,
            command=self.toggle_inspector,
        )
        close_insp_btn.grid(row=0, column=1, sticky="e")
        self._attach_motion(
            close_insp_btn,
            base_fg=DSW_SIDEBAR_FILL,
            hover_fg=DSW_LAYER_1,
            press_fg=DSW_ERROR,
            base_text=DSW_TEXT_TERTIARY,
            hover_text=DSW_TEXT_PRIMARY,
            surface_hex=DSW_SIDEBAR_FILL,
            transparent_rest=True,
        )

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
            "  - rescale_outlier_rows_: { target: 'layers.1.mlp.down_proj.weight', min_ratio: 2.5 }\n"
            "  - assert: { finite: { of: 'layers.1.mlp.down_proj.weight' } }\n",
        )
        apply_yaml_btn = ctk.CTkButton(
            tab_yaml,
            text="Применить YAML + Vespasian",
            height=32,
            corner_radius=8,
            fg_color=DSW_TEXT_PRIMARY,
            hover_color=DSW_ACCENT,
            text_color=DSW_TEXT_INVERT,
            font=ctk.CTkFont(family=FONT_UI, size=12, weight="bold"),
            command=self._run_custom_yaml,
        )
        apply_yaml_btn.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        self._attach_motion(
            apply_yaml_btn,
            base_fg=DSW_TEXT_PRIMARY,
            hover_fg=DSW_ACCENT,
            press_fg=DSW_ACCENT_STRONG,
            surface_hex=DSW_BG_BASE,
        )

        # Tensors Tab
        tab_tensors.grid_rowconfigure(1, weight=1)
        tab_tensors.grid_columnconfigure(0, weight=1)
        self._tensor_menu = ctk.CTkOptionMenu(
            tab_tensors,
            values=["model.layers.0.mlp.down_proj.weight"],
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
        export_btn = ctk.CTkButton(
            tab_export,
            text="Сохранить .safetensors",
            height=32,
            corner_radius=8,
            fg_color=DSW_TEXT_PRIMARY,
            hover_color=DSW_ACCENT,
            text_color=DSW_TEXT_INVERT,
            font=ctk.CTkFont(family=FONT_UI, size=12, weight="bold"),
            command=self._export_safetensors,
        )
        export_btn.grid(row=1, column=0, sticky="ew", padx=6)
        self._attach_motion(
            export_btn,
            base_fg=DSW_TEXT_PRIMARY,
            hover_fg=DSW_ACCENT,
            press_fg=DSW_ACCENT_STRONG,
            surface_hex=DSW_BG_BASE,
        )

    def toggle_inspector(self) -> None:
        """Smoothly slide-in or slide-out the Right Inspector Drawer (`0 <-> 360px`)."""
        if self._inspector_anim_job is not None:
            try:
                self.after_cancel(self._inspector_anim_job)
            except Exception:
                pass
            self._inspector_anim_job = None

        self._inspector_open = not self._inspector_open
        steps = 9

        if self._inspector_open:
            self._inspector_toggle_btn.configure(fg_color=DSW_ACCENT_TERTIARY, text_color=DSW_ACCENT)
            self._inspector.configure(width=16)
            self._inspector.grid(row=0, column=2, sticky="nsew")
            self._refresh_tensor_list()

            def _open_step(idx: int) -> None:
                if self._destroyed:
                    return
                t = ease_out_cubic(idx / float(steps))
                self._inspector.configure(width=int(round(16 + (360 - 16) * t)))
                if idx < steps:
                    self._inspector_anim_job = self.after(15, lambda: _open_step(idx + 1))
                else:
                    self._inspector_anim_job = None

            _open_step(1)
        else:
            self._inspector_toggle_btn.configure(fg_color="transparent", text_color=DSW_TEXT_TERTIARY)
            start_w = max(16, int(self._inspector.cget("width")))

            def _close_step(idx: int) -> None:
                if self._destroyed:
                    return
                t = ease_out_cubic(idx / float(steps))
                w = int(round(start_w * (1.0 - t)))
                if idx < steps and w > 18:
                    self._inspector.configure(width=w)
                    self._inspector_anim_job = self.after(14, lambda: _close_step(idx + 1))
                else:
                    self._inspector.grid_remove()
                    self._inspector.configure(width=360)
                    self._inspector_anim_job = None

            _close_step(1)

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
            self._tensor_info_box.insert("1.0", dumps_toon(desc))
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
    # Skills, Live `/` Filtering & Composer Interaction
    # ------------------------------------------------------------------
    def _set_status_caption(self, text: str) -> None:
        if self._destroyed:
            return
        self._status_caption.configure(text=text[:115])

    def _insert_skill_into_composer(self, skill: WeightXSkill) -> None:
        if self._placeholder_active:
            self._input_box.delete("1.0", "end")
            self._input_box.configure(text_color=DSW_TEXT_PRIMARY)
            self._placeholder_active = False
        else:
            self._input_box.delete("1.0", "end")
        self._input_box.insert("1.0", f"/skill {skill.skill_id}")
        self._input_box.focus_set()
        self._render_slash_popup(f"/skill {skill.skill_id}")
        self._set_status_caption(f"Вставлено `/skill {skill.skill_id}` — нажмите Enter или ↑ для запуска")

    def _on_input_focus_in(self, _event: Any = None) -> None:
        animate_colors(self, self._composer_card, {"border_color": DSW_ACCENT}, duration_ms=140, steps=7)
        if self._placeholder_active:
            self._input_box.delete("1.0", "end")
            self._input_box.configure(text_color=DSW_TEXT_PRIMARY)
            self._placeholder_active = False

    def _on_input_focus_out(self, _event: Any = None) -> None:
        animate_colors(self, self._composer_card, {"border_color": DSW_BORDER_L3}, duration_ms=160, steps=8)
        raw = self._input_box.get("1.0", "end").strip()
        if not raw:
            self._placeholder_active = True
            self._input_box.delete("1.0", "end")
            self._input_box.configure(text_color=DSW_TEXT_CAPTION)
            self._input_box.insert("1.0", PLACEHOLDER_HERO)

    def _filter_slash_skills(self, raw_query: str) -> list[WeightXSkill]:
        q = raw_query.strip().lower()
        for prefix in ("/skill:", "/skills:", "/skill ", "/skills ", "skill ", "skills "):
            if q.startswith(prefix):
                q = q[len(prefix) :].strip()
                break
        if q in ("/", "/skill", "/skills"):
            return list(WSP_SKILLS)
        q_clean = q.lstrip("/")
        if not q_clean:
            return list(WSP_SKILLS)
        hits = []
        for sk in WSP_SKILLS:
            hay = " ".join((sk.skill_id, sk.pill_label, sk.short_hint, *sk.aliases)).lower()
            if q_clean in hay:
                hits.append(sk)
        return hits

    def _render_slash_popup(self, query_text: str) -> None:
        for child in self._slash_popup.winfo_children():
            child.destroy()
        self._slash_row_buttons.clear()

        matches = self._filter_slash_skills(query_text)
        self._slash_matches = matches
        if not matches:
            self._slash_menu_open = False
            self._slash_popup.grid_remove()
            return

        self._slash_selected_idx = max(0, min(self._slash_selected_idx, len(matches) - 1))
        self._slash_menu_open = True
        self._slash_popup.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 2))

        for idx, sk in enumerate(matches):
            selected = idx == self._slash_selected_idx
            btn = ctk.CTkButton(
                self._slash_popup,
                text=f"/skill {sk.skill_id:<17}  {sk.pill_label:<20}  —  {sk.short_hint}",
                height=28,
                corner_radius=6,
                fg_color=DSW_ACCENT_TERTIARY if selected else "transparent",
                hover_color=DSW_INPUT_MAJOR,
                text_color=DSW_TEXT_PRIMARY if selected else DSW_TEXT_SECONDARY,
                font=ctk.CTkFont(family=FONT_MONO, size=11),
                anchor="w",
                command=lambda skill=sk: self._pick_slash_skill(skill),
            )
            btn.grid(row=idx, column=0, sticky="ew", padx=6, pady=1)
            self._attach_motion(
                btn,
                base_fg=DSW_ACCENT_TERTIARY if selected else DSW_SIDEBAR_FILL,
                hover_fg=DSW_INPUT_MAJOR,
                press_fg=DSW_SELECTOR,
                base_text=DSW_TEXT_PRIMARY if selected else DSW_TEXT_SECONDARY,
                hover_text=DSW_ACCENT,
                surface_hex=DSW_SIDEBAR_FILL,
                transparent_rest=not selected,
                active_fn=lambda sel=selected: sel,
            )
            self._slash_row_buttons.append(btn)

    def _on_input_key_release(self, event: Any = None) -> None:
        if self._placeholder_active:
            return
        if event is not None and getattr(event, "keysym", "") in ("Up", "Down", "Escape", "Return", "Tab"):
            return
        raw = self._input_box.get("1.0", "end").strip()
        if raw.startswith("/") or raw.lower().startswith("skill"):
            self._slash_selected_idx = 0
            self._render_slash_popup(raw)
        elif self._slash_menu_open:
            self._toggle_slash_popup(force=False)

    def _move_slash_selection(self, delta: int, _event: Any = None) -> str | None:
        if not self._slash_menu_open or not self._slash_matches:
            return None
        self._slash_selected_idx = (self._slash_selected_idx + delta) % len(self._slash_matches)
        raw = "" if self._placeholder_active else self._input_box.get("1.0", "end").strip()
        self._render_slash_popup(raw or "/")
        return "break"

    def _on_input_tab(self, _event: Any = None) -> str | None:
        if self._slash_menu_open and self._slash_matches:
            chosen = self._slash_matches[self._slash_selected_idx]
            self._insert_skill_into_composer(chosen)
            return "break"
        return None

    def _on_input_enter(self, event: Any) -> str | None:
        if event.state & 0x0001:  # Shift+Enter inserts newline
            return None
        if self._slash_menu_open and self._slash_matches:
            raw = "" if self._placeholder_active else self._input_box.get("1.0", "end").strip()
            sk, _rem, _exp = parse_skill_command(raw)
            if sk is None:
                chosen = self._slash_matches[self._slash_selected_idx]
                self._toggle_slash_popup(force=False)
                self._clear_composer_to_placeholder()
                self.trigger_skill(chosen)
                return "break"
        self.submit_composer()
        return "break"

    def _toggle_slash_popup(self, force: bool | None = None) -> None:
        want = (not self._slash_menu_open) if force is None else force
        if want:
            self._slash_selected_idx = 0
            raw = "" if self._placeholder_active else self._input_box.get("1.0", "end").strip()
            self._render_slash_popup(raw if raw.startswith("/") else "/")
        else:
            self._slash_menu_open = False
            self._slash_popup.grid_remove()

    def _pick_slash_skill(self, skill: WeightXSkill) -> None:
        self._toggle_slash_popup(force=False)
        self._clear_composer_to_placeholder()
        self.trigger_skill(skill)

    def _clear_composer_to_placeholder(self) -> None:
        self._input_box.delete("1.0", "end")
        self._placeholder_active = True
        self._input_box.configure(text_color=DSW_TEXT_CAPTION)
        self._input_box.insert("1.0", PLACEHOLDER_HERO)

    def _start_busy_spinner(self) -> None:
        self._busy = True
        self._spinner_phase = 0
        self._animate_spinner_tick()

    def _animate_spinner_tick(self) -> None:
        if self._destroyed or not self._busy:
            return
        frames = ("◐", "◓", "◑", "◒")
        ch = frames[self._spinner_phase % len(frames)]
        self._spinner_phase += 1
        try:
            self._send_btn.configure(text=ch, fg_color=DSW_ACCENT_TERTIARY, text_color=DSW_ACCENT)
        except Exception:
            return
        self.after(110, self._animate_spinner_tick)

    def trigger_skill(self, skill: WeightXSkill) -> None:
        """Invoke a full multi-stage WSP skill pipeline from a 2-word pill or `/skill <name>`."""
        if self._busy:
            return
        self._close_all_popups()
        self._bg_field.spawn_ripple(color=DSW_ACCENT, max_radius=300.0)

        sess = self._sessions[self._active_session_idx]
        if sess.title == "Новый сеанс":
            sess.title = f"{skill.pill_label}"
            self._render_sidebar_items()

        sess.messages.append(
            ConversationMessage(
                role="user",
                title=f"/skill {skill.skill_id}",
                body=f"/skill {skill.skill_id}  ·  {skill.pill_label}",
            )
        )
        live_msg = ConversationMessage(
            role="skill",
            title=f"Скилл · /skill {skill.skill_id} ({skill.pill_label})",
            body=f"Запуск пайплайна «{skill.pipeline_title}»…",
        )
        sess.messages.append(live_msg)
        self._refresh_center_mode()
        self._start_busy_spinner()

        def _worker() -> None:
            progress_lines: list[str] = []

            def _on_step(line: str) -> None:
                progress_lines.append("✓ " + line)
                live_msg.body = "\n".join(progress_lines)
                if not self._destroyed:
                    self.safe_after(0, self._render_conversation_messages)

            outcome_color = DSW_ACCENT
            try:
                res = self.studio.run_skill(skill.skill_id, progress_cb=_on_step)
                live_msg.skill_result = res
                outcome_color = DSW_SUCCESS if res.accepted else DSW_WARN
                body_parts = [f"✓ [{s.index}/{s.total}] {s.title} — {s.detail}" for s in res.steps]
                if res.metrics:
                    body_parts.append("\n[Метрики TOON]\n" + dumps_toon(res.metrics))
                if res.yaml_plan.strip():
                    body_parts.append("\n[BrainSurgery-WX YAML]\n" + res.yaml_plan.strip())
                body_parts.append(f"\nИтог: {res.summary}")
                live_msg.body = "\n".join(body_parts)
                if res.yaml_plan.strip() and not self._destroyed:
                    self.safe_after(0, lambda p=res.yaml_plan: self._set_yaml_editor(p))
            except Exception as exc:
                live_msg.body = f"Ошибка выполнения скилла /skill {skill.skill_id}: {exc}"
                outcome_color = DSW_ERROR
            finally:
                self._busy = False
                if not self._destroyed:
                    self.safe_after(0, lambda col=outcome_color: self._finish_turn(col))

        threading.Thread(target=_worker, daemon=True).start()

    def _set_yaml_editor(self, plan_text: str) -> None:
        self._yaml_editor.delete("1.0", "end")
        self._yaml_editor.insert("1.0", plan_text)

    def submit_composer(self) -> None:
        if self._busy:
            return
        raw = "" if self._placeholder_active else self._input_box.get("1.0", "end").strip()
        if not raw:
            self.trigger_skill(WSP_SKILLS[0])
            return

        self._clear_composer_to_placeholder()
        self._close_all_popups()

        sk, _rem, is_explicit = parse_skill_command(raw)
        if is_explicit and sk is None:
            self._toggle_slash_popup(force=True)
            self._set_status_caption("Укажите название скилла, например: /skill coding-boost")
            return
        if sk is not None:
            self.trigger_skill(sk)
            return

        self._run_async_turn(
            user_label=raw,
            worker=lambda: self.studio.run_action("chat", raw),
        )

    def _run_async_turn(self, user_label: str, worker: Any) -> None:
        self._bg_field.spawn_ripple(color=DSW_ACCENT, max_radius=260.0)
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
        self._start_busy_spinner()

        def _bg() -> None:
            try:
                reply_msg.body = worker()
                col = DSW_SUCCESS
            except Exception as exc:
                reply_msg.body = f"Ошибка: {exc}"
                col = DSW_ERROR
            finally:
                self._busy = False
                if not self._destroyed:
                    self.safe_after(0, lambda c=col: self._finish_turn(c))

        threading.Thread(target=_bg, daemon=True).start()

    def _finish_turn(self, outcome_color: str = DSW_SUCCESS) -> None:
        if self._destroyed:
            return
        self._send_btn.configure(text="↑", fg_color=DSW_TEXT_PRIMARY, text_color=DSW_TEXT_INVERT)
        self._bg_field.spawn_ripple(color=outcome_color, max_radius=340.0)
        self._sync_model_badges()
        self._render_conversation_messages()

    # ------------------------------------------------------------------
    # Session Switching + Settings Modal
    # ------------------------------------------------------------------
    def new_session(self) -> None:
        """Start or switch to a clean blank Hero session (`+ Новый сеанс`)."""
        self._close_all_popups()
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
        self._bg_field.spawn_ripple(color=DSW_ACCENT, max_radius=240.0)

    def _select_session(self, idx: int) -> None:
        self._close_all_popups()
        self._active_session_idx = idx
        self._render_sidebar_items()
        self._refresh_center_mode()

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
                threading.Thread(
                    target=lambda: (
                        self.studio.load_model(chosen_model),
                        self.safe_after(0, self._sync_model_badges) if not self._destroyed else None,
                    ),
                    daemon=True,
                ).start()
            modal.destroy()

        save_btn = ctk.CTkButton(
            modal,
            text="Сохранить",
            height=34,
            corner_radius=10,
            fg_color=DSW_TEXT_PRIMARY,
            hover_color=DSW_ACCENT,
            text_color=DSW_TEXT_INVERT,
            font=ctk.CTkFont(family=FONT_UI, size=13, weight="bold"),
            command=_save,
        )
        save_btn.pack(fill="x", padx=20, pady=8)

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
            # Stimulate a soft cursor ripple so Hero screenshot highlights the interactive neural mesh
            self._bg_field.ensure_z_order()
            self._bg_field.cursor_inside = True
            self._bg_field.cursor_x = 108.0
            self._bg_field.cursor_y = 245.0
            self._bg_field.spawn_ripple(108.0, 245.0, color=DSW_ACCENT, max_radius=170.0)
            for _ in range(7):
                self._bg_field.render_frame()
                self.update()
                time.sleep(0.025)

            if self._screenshot_path:
                img = self._grab_window_image()
                Path(self._screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                img.save(self._screenshot_path)
            if self._screenshot_chat_path:
                self._select_session(1)
                self.update_idletasks()
                self.update()
                time.sleep(0.28)
                img2 = self._grab_window_image()
                Path(self._screenshot_chat_path).parent.mkdir(parents=True, exist_ok=True)
                img2.save(self._screenshot_chat_path)
        finally:
            self._on_close()


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
        app.after(smoke_test_ms, app._on_close)
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
