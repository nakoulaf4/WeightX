"use strict";
/* ============================================================ HELPERS */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const icon = (name, cls = "icon") => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const fmt = (n, d = 2) => Number(n).toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtInt = (n) => Number(n).toLocaleString("ru-RU");
const nowTime = () => new Date().toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const store = { get(k, d) { try { const v = localStorage.getItem("wx." + k); return v === null ? d : JSON.parse(v); } catch { return d; } }, set(k, v) { try { localStorage.setItem("wx." + k, JSON.stringify(v)); } catch {} } };
function hexToRgb(h) { h = h.trim().replace("#", ""); if (h.length === 3) h = h.split("").map(c => c + c).join(""); const n = parseInt(h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function gini(values) { const v = values.filter(x => x >= 0).sort((a, b) => a - b); const n = v.length; if (!n) return 0; const sum = v.reduce((a, b) => a + b, 0); if (sum === 0) return 0; let cum = 0; v.forEach((x, i) => cum += (2 * (i + 1) - n - 1) * x); return Math.max(0, Math.min(1, cum / (n * sum))); }

/* ---- Tauri bridge: real native engine (Rust core + PyTorch sidecar) ---- */
const hasTauri = () => !!(window.__TAURI__ && window.__TAURI__.core);
async function invoke(cmd, args) {
  if (!hasTauri()) throw new Error("Движок WeightX недоступен: запустите приложение через WeightX Desktop (wx desktop).");
  return window.__TAURI__.core.invoke(cmd, args || {});
}
function listen(event, cb) { if (hasTauri()) return window.__TAURI__.event.listen(event, cb); return Promise.resolve(() => {}); }

/* ============================================================ DATA */
/* Presentation metadata for the real backend skills (icons/colors only;
   labels, hints, stages and plans come from the native engine). */
const SKILL_META = {
  "coding-boost": { icon: "zap", color: "var(--data-1)" },
  "memory-slim": { icon: "database", color: "var(--data-2)" },
  "outlier-repair": { icon: "wrench", color: "var(--data-4)" },
  "spectral-denoise": { icon: "waveform", color: "var(--data-7)" },
  "eamit-facts": { icon: "book", color: "var(--data-6)" },
  "knowledge-map": { icon: "map", color: "var(--data-3)" },
  "abliterate": { icon: "scissors", color: "var(--data-5)" },
  "vespasian-guard": { icon: "shield", color: "var(--data-8)" }
};
/* Real provider kinds from weightx-core PROVIDER_KINDS (first token = kind id). */
const PROVIDER_META = {
  "autonomous": { label: "Встроенный runtime", icon: "hard-drive", local: true },
  "ollama": { label: "Ollama", icon: "server" },
  "lm-studio": { label: "LM Studio", icon: "server" },
  "openai": { label: "OpenAI", icon: "cloud" },
  "openrouter": { label: "OpenRouter", icon: "cloud" },
  "vllm": { label: "vLLM", icon: "server" },
  "anthropic": { label: "Anthropic", icon: "sparkles" },
  "cli:claude-code": { label: "CLI · Claude Code", icon: "terminal" },
  "cli:codex": { label: "CLI · Codex", icon: "terminal" },
  "cli:opencode": { label: "CLI · OpenCode", icon: "terminal" },
  "cli:antigravity": { label: "CLI · Antigravity", icon: "terminal" },
  "cli:dsh": { label: "CLI · DSH", icon: "terminal" }
};
/* Model lists are fetched live from each provider's real API via the native
   provider_models command (ollama /api/tags, OpenAI-compatible /v1/models,
   Anthropic /v1/models). No hardcoded model catalogs. */
/* Offline fallback cards for the HF tab (real open-weight repos); live search
   replaces these via the native hf_search command when the network is up. */
const HF_FALLBACK = [
  { id: "HuggingFaceTB/SmolLM2-135M", family: "text-generation", params: "135M", layers: 30, vram: 540, license: "Apache-2.0", desc: "Компактная базовая модель — рекомендуемая цель WeightX." },
  { id: "HuggingFaceTB/SmolLM2-360M-Instruct", family: "text-generation", params: "360M", layers: 32, vram: 1440, license: "Apache-2.0", desc: "Сбалансированная малая модель с сильным reasoning." },
  { id: "HuggingFaceTB/SmolLM2-1.7B-Instruct", family: "text-generation", params: "1.7B", layers: 24, vram: 6800, license: "Apache-2.0", desc: "Лучшая в классе малая модель для кода и фактов." },
  { id: "Qwen/Qwen2.5-0.5B-Instruct", family: "text-generation", params: "0.5B", layers: 24, vram: 2000, license: "Apache-2.0", desc: "Мультиязычная малая модель, удобна для аблитерации." },
  { id: "Qwen/Qwen2.5-1.5B-Instruct", family: "text-generation", params: "1.5B", layers: 28, vram: 6000, license: "Apache-2.0", desc: "Сильная мультиязычная модель для редактирования фактов." },
  { id: "Qwen/Qwen2.5-Coder-1.5B-Instruct", family: "text-generation", params: "1.5B", layers: 28, vram: 6000, license: "Apache-2.0", desc: "Специализирована на коде — цель для coding-boost." },
  { id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", family: "text-generation", params: "1.1B", layers: 22, vram: 4400, license: "Apache-2.0", desc: "Популярная компактная chat-модель архитектуры Llama-2." },
  { id: "openai-community/gpt2", family: "text-generation", params: "124M", layers: 12, vram: 500, license: "MIT", desc: "Классический GPT-2 для экспериментов с вниманием (ROME/EAMIT)." },
  { id: "meta-llama/Llama-3.2-1B", family: "text-generation", params: "1B", layers: 16, vram: 4000, license: "Llama 3.2", desc: "Компактная Llama для быстрых экспериментов." },
  { id: "microsoft/Phi-3.5-mini-instruct", family: "text-generation", params: "3.8B", layers: 32, vram: 15200, license: "MIT", desc: "Мощная малая модель Microsoft с длинным контекстом." },
  { id: "google/gemma-2-2b", family: "text-generation", params: "2B", layers: 26, vram: 8000, license: "Gemma", desc: "Лёгкая открытая модель Google." },
  { id: "mistralai/Mistral-7B-v0.3", family: "text-generation", params: "7B", layers: 32, vram: 28000, license: "Apache-2.0", desc: "Крупная цель для спектральной денoйзинг-хирургии." }
];
/* Real L2 Uni-Layer metrics (13) as produced by weightx-core::l2. */
const LAYER_METRICS = [
  { key: "weight_l2_norm", label: "Норма", d: 2 }, { key: "weight_kurtosis", label: "Эксцесс", d: 1 },
  { key: "activation_entropy", label: "Энтропия акт.", d: 3 }, { key: "block_influence", label: "Влияние блока", d: 3 },
  { key: "effective_rank", label: "Эфф. ранг", d: 1 }, { key: "stable_rank", label: "Стаб. ранг", d: 2 },
  { key: "weight_sparsity", label: "Разреженность", d: 3 }, { key: "weight_outlier_ratio", label: "Выбросы весов", d: 4 },
  { key: "neuron_outlier_ratio", label: "Выбросы нейронов", d: 3 }, { key: "condition_number", label: "Обусловленность", d: 1 },
  { key: "fisher_information", label: "Fisher trace", d: 6 }, { key: "gradient_norm", label: "Норма градиента", d: 4 },
  { key: "wanda_score", label: "Wanda", d: 4 }
];
/* Styles — each patches the seed token set */
const STYLES = [
  { id: "graphite", name: "Графит", desc: "Глубокий тёмный · синий акцент", mode: "dark", bg: "#0b0d11", fg: "#f4f6fb", primary: "#eef1f7", accent: "#7aaaff", surface: "#13161d", radius: 8, glow: "25%" },
  { id: "amoled", name: "AMOLED", desc: "Чистый чёрный · синий акцент", mode: "dark", bg: "#000000", fg: "#ffffff", primary: "#ededed", accent: "#7aaaff", surface: "#0e0f13", radius: 6, glow: "25%" },
  { id: "polar", name: "Полярный", desc: "Светлый · синий акцент", mode: "light", bg: "#f6f7fa", fg: "#111318", primary: "#181a20", accent: "#2d63d8", surface: "#ffffff", radius: 8, glow: "16%" },
  { id: "carmine", name: "Кармин", desc: "Тёмный · алый акцент", mode: "dark", bg: "#0a0506", fg: "#fff5f5", primary: "#f5e9e9", accent: "#ff5a6e", surface: "#170d0f", radius: 4, glow: "35%" },
  { id: "emerald", name: "Изумруд", desc: "Тёмный · зелёный акцент", mode: "dark", bg: "#030a07", fg: "#f2fff7", primary: "#e9f5ee", accent: "#34d399", surface: "#0b1712", radius: 6, glow: "30%" },
  { id: "amber", name: "Янтарь", desc: "Тёмный · тёплый акцент", mode: "dark", bg: "#0a0703", fg: "#fff9f0", primary: "#f5ede0", accent: "#f7ad31", surface: "#17110a", radius: 10, glow: "30%" },
  { id: "amethyst", name: "Аметист", desc: "Тёмный · фиолетовый акцент", mode: "dark", bg: "#08050d", fg: "#faf5ff", primary: "#efe9f5", accent: "#a78bfa", surface: "#130d1c", radius: 12, glow: "35%" },
  { id: "coffee", name: "Кофе", desc: "Эспрессо · карамельная пенка · мягкий уют", mode: "dark", bg: "#1c110b", fg: "#f6ead9", primary: "#f2dcc0", accent: "#db934f", surface: "#281810", radius: 14, glow: "38%" },
  { id: "bubblegum", name: "Жевачка", desc: "Светлый · розовый акцент · круглые формы", mode: "light", bg: "#fff0f6", fg: "#3d1230", primary: "#4a1638", accent: "#e8447f", surface: "#ffe1ee", radius: 18, glow: "30%" },
  { id: "neon", name: "Неон", desc: "Тёмный · кислотный акцент · острые формы", mode: "dark", bg: "#05010d", fg: "#eafff5", primary: "#e6fff2", accent: "#39ff8e", surface: "#0c0618", radius: 2, glow: "60%" },
  { id: "ocean", name: "Океан", desc: "Тёмный · бирюзовый акцент", mode: "dark", bg: "#02080d", fg: "#eaf7ff", primary: "#e2f2fb", accent: "#2dd4bf", surface: "#08141c", radius: 10, glow: "35%" },
  { id: "system", name: "Системный", desc: "Как в ОС", mode: "auto", bg: "#0b0d11", fg: "#f4f6fb", primary: "#eef1f7", accent: "#7aaaff", surface: "#13161d", radius: 8, glow: "25%" }
];
const STYLE_BY_ID = Object.fromEntries(STYLES.map(s => [s.id, s]));
/* Agent Activity Feed presentation variants */
const FEED_VARIANTS = [
  { id: "tools", label: "Tool calls", icon: "wrench", hint: "Раскрывающиеся вызовы инструментов, как у Claude/Devin" },
  { id: "timeline", label: "Таймлайн", icon: "history", hint: "Вертикальная хронология с узлами и временем" },
  { id: "console", label: "Консоль", icon: "terminal", hint: "Потоковый моно-лог агента с уровнями и авто-скроллом" },
  { id: "steps", label: "Шаги", icon: "layers", hint: "Классический степпер пайплайна + карточки инструментов" }
];
const STAGE_NAMES = ["План", "Диагностика", "Sandbox", "Верификация", "Экспорт"];
const BG_MODES = [
  { id: "neural", name: "Нейро-сеть", desc: "Синаптический граф с притяжением курсора" },
  { id: "wave", name: "Спектр SVD", desc: "Гармонические волны, реагирующие на курсор" },
  { id: "matrix", name: "Тензор-поток", desc: "Потоки σ/Δw и глифов с откликом на клик" },
  { id: "flow", name: "Вихрь весов", desc: "Силовые линии векторного поля и следы частиц" },
  { id: "lattice", name: "Весовая сетка", desc: "Матрица узлов с топографической волной" },
  { id: "coffee", name: "Кофе из нейронов", desc: "Силуэт чашки кофе, зёрен и пара из сети нейронов" },
  { id: "off", name: "Чистый фон", desc: "Без анимированного холста" }
];

/* ============================================================ STATE */
const state = {
  view: "session", model: null, modelInfo: null, style: store.get("style", "graphite"),
  provider: store.get("provider", { kind: "autonomous", model: "", apiKey: "", baseUrl: "" }),
  providerConfigs: store.get("providerConfigs", {}),
  providerModelsCache: {},
  guards: store.get("guards", { fisher: true, vespasian: true, sandbox: true }),
  pipeline: store.get("pipeline", null), // null = read live values from the engine
  busy: false, tensors: [], tensorsLoading: false, diag: null, diagLoading: false, selectedTensor: null, selectedLayer: null,
  tensorFilter: { q: "", role: "all", onlyModified: false }, messages: [], history: { iterations: [], baseline: 0, best: 0 }, lastYaml: "",
  sortDiag: { key: "block_influence", dir: -1 }, modelSource: "hf", modelSearch: "", feedVariant: store.get("feedVariantV2", "timeline"),
  bgMode: (() => { const m = store.get("bgMode", "neural"); return m === "orbits" ? "flow" : m; })(), motion: store.get("motion", "full"), gradients: store.get("gradients", true),
  catalog: [], skills: [], providerKinds: [],
  exportPath: store.get("exportPath", "checkpoints/exported_model.safetensors"),
  hfLive: false, hfLoading: false, hfResults: [], hfReqId: 0, hfCursor: "", hfLoadingMore: false,
  hfTask: "", hfLicense: "", hfSort: "downloads", avatarCache: {},
  sysResources: null, maxParams: 0, providerChecks: {}, settingsTab: "chat"
};
if (!state.providerConfigs[state.provider.kind]) state.providerConfigs[state.provider.kind] = { model: state.provider.model, apiKey: state.provider.apiKey, baseUrl: state.provider.baseUrl };
const skillById = (id) => state.skills.find(s => s.skill_id === id);
function providerCfg(kind) { return kind === state.provider.kind ? state.provider : (state.providerConfigs[kind] || { model: "", apiKey: "", baseUrl: "" }); }
/* The engine is the source of truth: exactly one active brain at a time. */
function activeBrain() {
  const info = state.modelInfo;
  if (info && info.provider_kind) return { kind: String(info.provider_kind).split(" ")[0], model: info.provider_model || "" };
  return { kind: "autonomous", model: "" };
}
function hasConfig(kind) {
  if (kind === activeBrain().kind) return true;
  // A provider explicitly saved by the user stays visible in the Brain picker
  // even without a model/key (the engine may still be able to use defaults).
  return !!state.providerConfigs[kind];
}
function configuredProviders() { return state.providerKinds.filter(p => p.kind === "autonomous" || hasConfig(p.kind)); }
function defaultBrainModel(kind) { const c = providerCfg(kind); return c.model || ((state.providerModelsCache[kind] || {}).models || [])[0] || ""; }
function brainLabel(short = false) { const b = activeBrain(); if (b.kind === "autonomous") return short ? "Встроенный" : "Встроенный runtime · без LLM"; const prov = state.providerKinds.find(p => p.kind === b.kind); const label = prov ? prov.label : b.kind; return short ? (b.model || label) : label + (b.model ? " · " + b.model : " · модель не указана"); }
function fullProviderKind(kind) { const p = state.providerKinds.find(x => x.kind === kind); return p ? p.full : kind; }
/* Tensor role inferred from the real tensor name (L1 AutoArch conventions). */
function tensorRole(name) {
  const n = name.toLowerCase();
  if (n.includes("q_proj") || n.includes("query")) return "q";
  if (n.includes("k_proj") || n.includes("key")) return "k";
  if (n.includes("v_proj") || n.includes("value")) return "v";
  if (n.includes("o_proj") || n.includes("out_proj") || n.includes("dense")) return "o";
  if (n.includes("mlp") || n.includes("gate_proj") || n.includes("up_proj") || n.includes("down_proj") || n.includes("c_proj") || n.includes("fc2")) return "mlp";
  if (n.includes("norm") || n.includes("ln_")) return "norm";
  if (n.includes("embed") || n.includes("wte") || n.includes("wpe") || n.includes("lm_head")) return "embed";
  return "other";
}
function localEntries() { return state.catalog.filter(e => e.local_cached); }
function isDownloaded(id) { const e = state.catalog.find(x => x.model_id === id || x.hf_repo_id === id); return !!(e && e.local_cached); }

/* ============================================================ STYLES */
let heatStopsCache = null;
function resolveStyleMode(st) { return st.mode === "auto" ? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark") : st.mode; }
function applyStyle() {
  const st = STYLE_BY_ID[state.style] || STYLE_BY_ID.graphite;
  const mode = resolveStyleMode(st);
  const root = document.documentElement;
  root.setAttribute("data-style", st.id); root.setAttribute("data-mode", mode);
  root.style.colorScheme = mode;
  const seeds = (mode === "light" && st.id === "system") ? { bg: "#f6f7fa", fg: "#111318", primary: "#181a20", surface: "#ffffff" } : { bg: st.bg, fg: st.fg, primary: st.primary, surface: st.surface };
  root.style.setProperty("--seed-bg", seeds.bg);
  root.style.setProperty("--seed-fg", seeds.fg);
  root.style.setProperty("--seed-primary", seeds.primary);
  root.style.setProperty("--seed-surface", seeds.surface);
  root.style.setProperty("--seed-accent", mode === "light" && st.id === "system" ? "#2d63d8" : st.accent);
  root.style.setProperty("--seed-radius", (st.radius ?? 8) + "px");
  root.style.setProperty("--seed-glow", st.glow || "25%");
  heatStopsCache = null;
  $("#styleBtnLabel").textContent = st.name;
  store.set("style", state.style);
  if (typeof renderStylesGrids === "function") renderStylesGrids();
  if (typeof bgRecolor === "function") bgRecolor();
}

/* ============================================================ ANIMATED BACKGROUND */
const bg = { nodes: [], streams: [], flow: [], coffeeNodes: [], coffeeLinks: [], ripples: [], mouse: { x: -9999, y: -9999 }, t: 0, raf: 0, accent: [122, 170, 255], accent2: [77, 208, 225] };
function bgRecolor() {
  const cs = getComputedStyle(document.documentElement);
  const a1 = cs.getPropertyValue("--seed-accent").trim() || "#7aaaff";
  const a2 = cs.getPropertyValue("--data-2").trim() || "#4dd0e1";
  bg.accent = a1.startsWith("#") ? hexToRgb(a1) : [122, 170, 255];
  bg.accent2 = a2.startsWith("#") ? hexToRgb(a2) : bg.accent;
  if (state.motion === "calm" && state.bgMode !== "off") bg.paintStatic?.();
}
function mulberry(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function bgRipple(x, y, maxR = 210) {
  if (state.bgMode === "off" || state.motion === "off") return;
  bg.ripples.push({ x, y, r: 8, max: maxR });
  if (bg.ripples.length > 8) bg.ripples.shift();
}
function bgInit() {
  const cv = $("#bgNet"); if (!cv) return; const ctx = cv.getContext("2d");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const rnd = mulberry(7);
  function seedCoffee() {
    const nodes = [], links = [];
    const s = Math.max(0.85, Math.min(1.5, Math.min(innerWidth, innerHeight) / 680));
    const cx = innerWidth * 0.5, cy = innerHeight * 0.48;
    const add = (x, y, kind, r = 2.0, phase = 0) => { nodes.push({ x, y, hx: x, hy: y, vx: 0, vy: 0, kind, r, phase }); return nodes.length - 1; };
    const chain = (idxs, close = false) => {
      for (let i = 0; i < idxs.length - 1; i++) links.push([idxs[i], idxs[i + 1]]);
      if (close && idxs.length > 2) links.push([idxs[idxs.length - 1], idxs[0]]);
    };
    // Saucer arc
    const saucer = [];
    for (let i = 0; i <= 14; i++) {
      const a = Math.PI * (0.12 + 0.76 * (i / 14));
      saucer.push(add(cx - Math.cos(a) * 164 * s, cy + 106 * s + Math.sin(a) * 26 * s, "saucer", 1.9, i * 0.35));
    }
    chain(saucer);
    // Rim ellipse
    const rim = [];
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      rim.push(add(cx + Math.cos(a) * 126 * s, cy - 28 * s + Math.sin(a) * 28 * s, "rim", 2.1, a));
    }
    chain(rim, true);
    // Crema swirl spiral inside cup top
    const crema = [];
    for (let i = 0; i < 16; i++) {
      const f = (i + 1) / 16, a = f * Math.PI * 3.2;
      crema.push(add(cx + Math.cos(a) * f * 84 * s, cy - 28 * s + Math.sin(a) * f * 17 * s, "crema", 1.8, a));
    }
    chain(crema);
    // Cup U-body contour + inner neural meridian ribs
    const bodyL = [], bodyR = [], midY = [], ribL = [], ribR = [];
    for (let i = 0; i <= 10; i++) {
      const f = i / 10, ang = f * Math.PI * 0.5;
      const dx = Math.cos(ang) * 124 * s, y = cy - 26 * s + Math.sin(ang) * 134 * s;
      bodyL.push(add(cx - dx, y, "cup", 2.1, f * 2));
      bodyR.push(add(cx + dx, y, "cup", 2.1, f * 2 + 1));
      if (i % 2 === 1 && i < 9) {
        ribL.push(add(cx - dx * 0.52, y - 6 * s, "cup", 1.6, f * 3));
        ribR.push(add(cx + dx * 0.52, y - 6 * s, "cup", 1.6, f * 3 + 0.8));
      }
      if (i === 5 || i === 8) midY.push(add(cx, y, "cup", 1.9, i));
    }
    chain(bodyL); chain(bodyR); links.push([bodyL[bodyL.length - 1], bodyR[bodyR.length - 1]]);
    chain(ribL); chain(ribR);
    links.push([rim[12], bodyL[0]], [rim[0], bodyR[0]]);
    for (let i = 0; i < Math.min(ribL.length, bodyL.length); i++) links.push([bodyL[i * 2 + 1], ribL[i]], [bodyR[i * 2 + 1], ribR[i]]);
    // Cup handle on the right
    const handleOuter = [], handleInner = [];
    for (let i = 0; i <= 10; i++) {
      const f = i / 10, a = -Math.PI * 0.48 + f * Math.PI * 0.96;
      handleOuter.push(add(cx + 114 * s + Math.cos(a) * 58 * s, cy + 28 * s + Math.sin(a) * 50 * s, "handle", 2.0, f * 4));
      if (i > 1 && i < 9) handleInner.push(add(cx + 114 * s + Math.cos(a) * 34 * s, cy + 28 * s + Math.sin(a) * 28 * s, "handle", 1.6, f * 4 + 0.5));
    }
    chain(handleOuter); chain(handleInner);
    links.push([bodyR[1], handleOuter[0]], [bodyR[7], handleOuter[handleOuter.length - 1]], [handleOuter[2], handleInner[0]], [handleOuter[8], handleInner[handleInner.length - 1]]);
    // Three rising steam wisps made of neural chains
    [-46, 0, 48].forEach((ox, wi) => {
      const wisp = [];
      for (let k = 0; k < 7; k++) {
        const f = k / 6, px = cx + ox * s + Math.sin(f * Math.PI * 2 + wi * 1.4) * 16 * s, py = cy - 60 * s - f * 145 * s;
        wisp.push(add(px, py, "steam", 1.7 + (1 - f) * 0.6, wi * 2 + f * 3));
      }
      chain(wisp);
    });
    // Neural coffee beans floating left and right
    const beans = [
      { bx: cx - 265 * s, by: cy + 32 * s, rot: -0.45, sc: 1.0 },
      { bx: cx + 275 * s, by: cy - 38 * s, rot: 0.52, sc: 0.92 },
      { bx: cx - 220 * s, by: cy - 132 * s, rot: 0.32, sc: 0.78 }
    ];
    for (const b of beans) {
      if (b.bx < 24 || b.bx > innerWidth - 24) continue;
      const ring = [], crease = [];
      for (let i = 0; i < 10; i++) {
        const a = (i / 10) * Math.PI * 2;
        const lx = Math.cos(a) * 24 * s * b.sc, ly = Math.sin(a) * 16 * s * b.sc;
        ring.push(add(b.bx + lx * Math.cos(b.rot) - ly * Math.sin(b.rot), b.by + lx * Math.sin(b.rot) + ly * Math.cos(b.rot), "bean", 1.8, a));
      }
      chain(ring, true);
      for (let i = -2; i <= 2; i++) {
        const lx = i * 7 * s * b.sc, ly = Math.sin(i * 0.8) * 3.5 * s * b.sc;
        crease.push(add(b.bx + lx * Math.cos(b.rot) - ly * Math.sin(b.rot), b.by + lx * Math.sin(b.rot) + ly * Math.cos(b.rot), "bean", 1.5, i));
      }
      chain(crease);
    }
    // Sparse ambient outer neurons
    for (let i = 0; i < 18; i++) {
      const ang = (i / 18) * Math.PI * 2, radX = (250 + rnd() * 210) * s, radY = (170 + rnd() * 140) * s;
      const ax = Math.max(18, Math.min(innerWidth - 18, cx + Math.cos(ang) * radX));
      const ay = Math.max(18, Math.min(innerHeight - 18, cy + Math.sin(ang) * radY));
      add(ax, ay, "amb", 1.3 + rnd() * 1.1, ang);
    }
    bg.coffeeNodes = nodes; bg.coffeeLinks = links;
  }
  function seedObjects() {
    const N = Math.min(72, Math.max(28, Math.floor(innerWidth / 22)));
    bg.nodes = Array.from({ length: N }, () => {
      const x = rnd() * innerWidth, y = rnd() * innerHeight;
      return { x, y, hx: x, hy: y, vx: (rnd() - 0.5) * 0.28, vy: (rnd() - 0.5) * 0.28, r: 1 + rnd() * 1.8 };
    });
    const glyphs = ["Δw", "σ₁", "∇L", "Wₓ", "L₂", "◈", "·", "λ"];
    const cols = Math.max(18, Math.floor(innerWidth / 48));
    bg.streams = Array.from({ length: cols }, (_, i) => ({
      x: ((i + 0.5) / cols) * innerWidth + (rnd() - 0.5) * 14,
      y: rnd() * innerHeight,
      speed: 0.55 + rnd() * 1.1,
      len: 4 + Math.floor(rnd() * 5),
      glyph: glyphs[i % glyphs.length],
      phase: rnd() * Math.PI * 2
    }));
    const F = Math.min(95, Math.max(48, Math.floor((innerWidth * innerHeight) / 14000)));
    bg.flow = Array.from({ length: F }, (_, i) => {
      const x = rnd() * innerWidth, y = rnd() * innerHeight;
      return { x, y, trail: [{ x, y }], speed: 1.1 + rnd() * 1.35, alt: i % 3 === 0, w: 1.0 + rnd() * 1.3 };
    });
    seedCoffee();
  }
  function drawRipples(ar, ag, ab) {
    const next = [];
    for (const rp of bg.ripples) {
      rp.r += 4.2;
      if (rp.r < rp.max) {
        const a = (1 - rp.r / rp.max) * 0.28;
        ctx.strokeStyle = `rgba(${ar},${ag},${ab},${a.toFixed(3)})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2); ctx.stroke();
        next.push(rp);
      }
    }
    bg.ripples = next;
  }
  function drawScene(dt) {
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    if (state.bgMode === "off") return;
    const [ar, ag, ab] = bg.accent;
    const [br, bgc, bb] = bg.accent2;
    const mx = bg.mouse.x, my = bg.mouse.y;
    const activePtr = mx > -500 && my > -500;
    const busyMul = state.busy ? 1.65 : 1.0;
    bg.t += dt * busyMul;
    const mode = state.bgMode;

    if (mode === "wave") {
      const bands = 6;
      for (let b = 0; b < bands; b++) {
        const baseY = innerHeight * (0.20 + b * 0.12);
        const col = b % 2 ? `${br},${bgc},${bb}` : `${ar},${ag},${ab}`;
        ctx.beginPath();
        for (let x = 0; x <= innerWidth + 18; x += 18) {
          const pull = activePtr ? Math.exp(-Math.pow((x - mx) / 170, 2)) * ((my - baseY) * 0.24) : 0;
          const amp = (14 + b * 4) * (state.busy ? 1.25 : 1);
          const y = baseY + Math.sin(x * 0.008 + bg.t * (0.9 + b * 0.15) + b) * amp + Math.cos(x * 0.017 - bg.t * 0.6) * 6 + pull;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = `rgba(${col},${0.13 + (b === 2 ? 0.08 : 0)})`;
        ctx.lineWidth = b % 2 ? 1.2 : 1.6;
        ctx.stroke();
      }
      if (activePtr) {
        const g = ctx.createRadialGradient(mx, my, 4, mx, my, 170);
        g.addColorStop(0, `rgba(${ar},${ag},${ab},0.14)`);
        g.addColorStop(1, `rgba(${ar},${ag},${ab},0)`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(mx, my, 170, 0, Math.PI * 2); ctx.fill();
      }
    } else if (mode === "matrix") {
      ctx.font = '11px "JetBrains Mono", Consolas, monospace';
      ctx.textAlign = "center";
      for (const s of bg.streams) {
        if (dt > 0) {
          s.y += s.speed * busyMul;
          if (s.y - s.len * 22 > innerHeight) s.y = -20;
        }
        const dx = activePtr ? (s.x - mx) : 9999;
        const near = Math.abs(dx) < 130 ? (1 - Math.abs(dx) / 130) : 0;
        const pushX = near > 0 ? Math.sign(dx || 1) * near * 14 : 0;
        const px = s.x + pushX;
        ctx.strokeStyle = `rgba(${ar},${ag},${ab},${0.03 + near * 0.08})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px, Math.max(0, s.y - s.len * 24)); ctx.lineTo(px, s.y); ctx.stroke();
        for (let k = 0; k < s.len; k++) {
          const gy = s.y - k * 20;
          if (gy < -12 || gy > innerHeight + 12) continue;
          const fade = (1 - k / s.len) * (0.14 + near * 0.34);
          ctx.fillStyle = k === 0 ? `rgba(${br},${bgc},${bb},${Math.min(0.7, fade * 1.6)})` : `rgba(${ar},${ag},${ab},${fade})`;
          ctx.fillText(s.glyph, px, gy);
        }
      }
    } else if (mode === "flow") {
      for (const p of bg.flow) {
        if (dt > 0) {
          let ang = Math.sin(p.x * 0.0045 + bg.t * 0.48) * Math.PI + Math.cos(p.y * 0.0045 - bg.t * 0.36) * Math.PI;
          let spd = p.speed * busyMul;
          if (activePtr) {
            const dx = p.x - mx, dy = p.y - my, d = Math.hypot(dx, dy);
            if (d < 210 && d > 1) {
              const k = 1 - d / 210;
              ang = Math.atan2(dy, dx) + Math.PI * 0.62 * k;
              spd *= 1 + k * 1.15;
            }
          }
          p.x += Math.cos(ang) * spd;
          p.y += Math.sin(ang) * spd;
          if (p.x < -16 || p.x > innerWidth + 16 || p.y < -16 || p.y > innerHeight + 16) {
            p.x = ((p.x % innerWidth) + innerWidth) % innerWidth;
            p.y = ((p.y % innerHeight) + innerHeight) % innerHeight;
            p.trail = [{ x: p.x, y: p.y }];
          } else {
            p.trail.push({ x: p.x, y: p.y });
            if (p.trail.length > 10) p.trail.shift();
          }
        }
        const col = p.alt ? `${br},${bgc},${bb}` : `${ar},${ag},${ab}`;
        if (p.trail.length > 1) {
          for (let i = 1; i < p.trail.length; i++) {
            const a = p.trail[i - 1], b = p.trail[i];
            if (Math.hypot(a.x - b.x, a.y - b.y) > 40) continue;
            ctx.strokeStyle = `rgba(${col},${(i / p.trail.length) * 0.24})`;
            ctx.lineWidth = p.w;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
        ctx.fillStyle = `rgba(${col},0.46)`;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.w * 1.15, 0, Math.PI * 2); ctx.fill();
      }
    } else if (mode === "coffee") {
      const nodes = bg.coffeeNodes, links = bg.coffeeLinks;
      if (dt > 0) {
        for (let i = 0; i < nodes.length; i++) {
          const n = nodes[i];
          let tx = n.hx, ty = n.hy;
          if (n.kind === "steam") {
            tx += Math.sin(bg.t * 1.9 + n.phase) * 7;
            ty += Math.cos(bg.t * 1.4 + n.phase) * 3;
          } else if (n.kind === "crema") {
            tx += Math.cos(bg.t * 0.9 + n.phase) * 3;
            ty += Math.sin(bg.t * 0.9 + n.phase) * 1.5;
          } else if (n.kind === "amb") {
            tx += Math.sin(bg.t * 0.5 + n.phase) * 10;
            ty += Math.cos(bg.t * 0.4 + n.phase) * 8;
          }
          n.vx += (tx - n.x) * 0.055;
          n.vy += (ty - n.y) * 0.055;
          if (activePtr) {
            const dx = n.x - mx, dy = n.y - my, d = Math.hypot(dx, dy);
            if (d < 130 && d > 0.5) {
              const push = (1 - d / 130) * (n.kind === "steam" ? 1.55 : 0.95);
              n.vx += (dx / d) * push; n.vy += (dy / d) * push;
            }
          }
          n.vx *= 0.84; n.vy *= 0.84;
          n.x += n.vx; n.y += n.vy;
        }
      }
      for (let i = 0; i < links.length; i++) {
        const [ia, ib] = links[i], a = nodes[ia], b = nodes[ib];
        if (!a || !b) continue;
        const dm = activePtr ? Math.hypot((a.x + b.x) * 0.5 - mx, (a.y + b.y) * 0.5 - my) : 9999;
        const glow = dm < 150 ? (1 - dm / 150) : 0;
        const fire = Math.sin(bg.t * 2.8 - i * 0.35) > 0.78 ? 0.14 : 0;
        const isCream = a.kind === "steam" || a.kind === "crema" || fire > 0;
        const col = isCream ? `${br},${bgc},${bb}` : `${ar},${ag},${ab}`;
        ctx.strokeStyle = `rgba(${col},${Math.min(0.68, (a.kind === "steam" ? 0.17 : 0.22) + glow * 0.36 + fire)})`;
        ctx.lineWidth = a.kind === "cup" || a.kind === "rim" ? 1.25 : 1.0;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const dm = activePtr ? Math.hypot(mx - n.x, my - n.y) : 9999;
        const glow = dm < 140 ? (1 - dm / 140) : 0;
        const isCrema = n.kind === "crema" || n.kind === "steam";
        const col = isCrema ? `${br},${bgc},${bb}` : `${ar},${ag},${ab}`;
        ctx.fillStyle = `rgba(${col},${Math.min(0.92, (n.kind === "amb" ? 0.18 : 0.38) + glow * 0.48)})`;
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r + glow * 1.7, 0, Math.PI * 2); ctx.fill();
      }
    } else if (mode === "lattice") {
      const step = 44;
      for (let gx = step * 0.5; gx < innerWidth; gx += step) {
        for (let gy = step * 0.5; gy < innerHeight; gy += step) {
          const wave = Math.sin(gx * 0.015 + bg.t * 1.2) * Math.cos(gy * 0.015 - bg.t * 0.9) * 3;
          let x = gx, y = gy + wave, glow = 0;
          if (activePtr) {
            const dx = gx - mx, dy = gy - my, d = Math.hypot(dx, dy);
            if (d < 170 && d > 0.1) { glow = 1 - d / 170; x += (dx / d) * glow * 12; y += (dy / d) * glow * 12; }
          }
          const r = 1.1 + glow * 1.9;
          ctx.fillStyle = glow > 0.35 ? `rgba(${br},${bgc},${bb},${0.16 + glow * 0.5})` : `rgba(${ar},${ag},${ab},${0.12 + glow * 0.42})`;
          ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
        }
      }
    } else {
      if (dt > 0) {
        for (const n of bg.nodes) {
          n.x += n.vx * busyMul; n.y += n.vy * busyMul;
          n.vx += (n.hx - n.x) * 0.0004; n.vy += (n.hy - n.y) * 0.0004;
          if (n.x < 0 || n.x > innerWidth) n.vx *= -1;
          if (n.y < 0 || n.y > innerHeight) n.vy *= -1;
          if (activePtr) {
            const dx = mx - n.x, dy = my - n.y, dist = Math.hypot(dx, dy);
            if (dist < 175 && dist > 0.1) { const f = dist < 48 ? -0.65 : 0.36; n.x += dx / dist * f; n.y += dy / dist * f; }
          }
        }
      }
      for (let i = 0; i < bg.nodes.length; i++) for (let j = i + 1; j < bg.nodes.length; j++) {
        const a = bg.nodes[i], b = bg.nodes[j], d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 130) { ctx.strokeStyle = `rgba(${ar},${ag},${ab},${(1 - d / 130) * 0.11})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
      }
      for (const n of bg.nodes) {
        const dm = activePtr ? Math.hypot(mx - n.x, my - n.y) : 9999;
        const glow = dm < 175 ? (1 - dm / 175) : 0;
        if (glow > 0.15 && dm > 10) {
          ctx.strokeStyle = `rgba(${br},${bgc},${bb},${glow * 0.14})`;
          ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(n.x, n.y); ctx.stroke();
        }
        ctx.fillStyle = `rgba(${ar},${ag},${ab},${0.18 + glow * 0.52})`;
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r + glow * 1.6, 0, Math.PI * 2); ctx.fill();
      }
    }
    if (dt > 0) drawRipples(br, bgc, bb);
  }
  function paintStatic() { drawScene(0); }
  function frame() {
    if (state.motion === "off" || state.bgMode === "off") { bg.raf = 0; return; }
    drawScene(0.016);
    bg.raf = requestAnimationFrame(frame);
  }
  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = innerWidth * dpr; cv.height = innerHeight * dpr;
    cv.style.width = innerWidth + "px"; cv.style.height = innerHeight + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seedObjects();
    if (reduced || state.motion !== "full") paintStatic();
  }
  resize(); addEventListener("resize", resize);
  addEventListener("mousemove", e => { bg.mouse.x = e.clientX; bg.mouse.y = e.clientY; });
  addEventListener("mousedown", e => {
    bgRipple(e.clientX, e.clientY, 190);
    if (state.bgMode === "neural") {
      for (const n of bg.nodes) {
        const dx = n.x - e.clientX, dy = n.y - e.clientY, d = Math.hypot(dx, dy);
        if (d < 150 && d > 1) { n.vx += (dx / d) * 1.4; n.vy += (dy / d) * 1.4; }
      }
    } else if (state.bgMode === "coffee") {
      for (const n of bg.coffeeNodes) {
        const dx = n.x - e.clientX, dy = n.y - e.clientY, d = Math.hypot(dx, dy);
        if (d < 170 && d > 1) { n.vx += (dx / d) * 2.2; n.vy += (dy / d) * 2.2; }
      }
    } else if (state.bgMode === "flow") {
      bg.flow.forEach((p, idx) => {
        if (idx % 4 === 0) { const a = (idx / bg.flow.length) * Math.PI * 2; p.x = e.clientX + Math.cos(a) * 12; p.y = e.clientY + Math.sin(a) * 12; p.trail = [{ x: p.x, y: p.y }]; }
      });
    }
  });
  addEventListener("mouseout", () => { bg.mouse.x = -9999; bg.mouse.y = -9999; });
  bgRecolor();
  bg.frame = frame; bg.paintStatic = paintStatic;
  bgSync();
}
function bgSync() {
  const osReduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mode = osReduced || state.bgMode === "off" ? "off" : state.motion;
  if (mode === "full") { if (!bg.raf && bg.frame) bg.raf = requestAnimationFrame(bg.frame); return; }
  if (bg.raf) { cancelAnimationFrame(bg.raf); bg.raf = 0; }
  if (mode === "calm") { bg.paintStatic?.(); return; }
  const cv = $("#bgNet"); cv?.getContext("2d").clearRect(0, 0, innerWidth, innerHeight);
}
function setBgMode(id) {
  state.bgMode = id;
  store.set("bgMode", id);
  bgSync();
  renderStylesGrids();
  const m = BG_MODES.find(x => x.id === id);
  toast("info", "Фон: " + (m ? m.name : id), m ? m.desc : "", null, 1800);
}
function syncBgDim() {
  const hasThread = !!($("#threadInner") && $("#threadInner").querySelector(".turn"));
  document.body.classList.toggle("bg-dim", state.view !== "session" || hasThread);
}
function applyEffects() {
  const root = document.documentElement;
  root.setAttribute("data-motion", state.motion);
  root.setAttribute("data-gradients", state.gradients ? "on" : "off");
  bgSync();
}

/* ============================================================ ENGINE (real WeightX via Tauri) */
function syncSendBtn() {
  const send = $("#sendBtn"); const inp = $("#composerInput");
  if (send) send.disabled = state.busy || !inp || !inp.value.trim();
}
function setBusy(b) {
  state.busy = b;
  $("#app").classList.toggle("is-busy", b);
  $$("[data-run], #clearSession").forEach(x => x.disabled = b);
  syncSendBtn();
  $("#modelChipDot").classList.toggle("is-pulse", b);
  $("#sbSandboxDot").classList.toggle("is-pulse", b);
}
function busyGuard() { if (state.busy) { toast("warning", "Пайплайн уже выполняется", "Дождитесь завершения текущего запуска", null, 2200); return true; } return false; }
function updateGuardUI() {
  const g = state.guards;
  $("#sbSandbox").textContent = "sandbox";
  $("#sbSandboxDot").className = "dot dot-success";
  $("#sbVespasian").textContent = g.vespasian ? "Vespasian активен" : "Vespasian выкл";
  $("#sbVespasian").style.color = g.vespasian ? "" : "var(--status-warning)";
}
function backendLabel(b) { return b === "torch" ? "PyTorch sidecar" : "Rust core"; }
function renderSidebarStatus() {
  const box = $("#sidebarStatus"); if (!box) return;
  const m = state.model;
  if (!m) {
    box.innerHTML = `<div class="ss-card"><div class="ss-row"><span class="dot dot-warning"></span><span class="ss-name">Модель не выбрана</span></div><div class="ss-actions"><button class="btn btn-sm btn-primary" id="ssDemoBtn">${icon("zap")} Демо</button><button class="btn btn-sm" id="ssCatBtn">Каталог</button></div></div>`;
    $("#ssDemoBtn").onclick = () => loadModel("minillama-demo");
    $("#ssCatBtn").onclick = () => navigate("models");
    return;
  }
  const d = (state.modelInfo && state.modelInfo.diff) || { changed_params: 0, total_params: 0 };
  const dirty = (d.changed_params || 0) > 0;
  const pct = dirty && d.total_params ? (d.changed_params / d.total_params * 100) : 0;
  const v = (state.modelInfo && state.modelInfo.vespasian) || {};
  box.innerHTML = `<div class="ss-card">
    <div class="ss-row"><span class="dot ${dirty ? "dot-warning" : "dot-success"}"></span><span class="ss-name" title="${esc(m.id)}">${esc(m.id)}</span></div>
    <div class="ss-meta"><span>${esc(m.params)}</span><span>·</span><span>${m.layers || "—"}L</span>${dirty ? `<span class="badge badge-warning" style="height:15px">Δ ${pct < 0.01 ? "<0.01" : pct.toFixed(2)}%</span>` : (v.best ? `<span>score ${fmt(v.best, 3)}</span>` : "")}</div>
    <div class="ss-actions">${dirty ? `<button class="btn btn-sm" id="ssDiffBtn">${icon("diff")} Diff</button><button class="btn btn-sm btn-primary" id="ssExpBtn">${icon("download")} Экспорт</button>` : `<button class="btn btn-sm" id="ssDiagBtn">${icon("activity")} Профиль</button><button class="btn btn-sm" id="ssCatBtn">Сменить</button>`}</div>
  </div>`;
  const db = $("#ssDiffBtn"); if (db) db.onclick = () => showDiff("workspace");
  const eb = $("#ssExpBtn"); if (eb) eb.onclick = () => exportCheckpoint();
  const pb = $("#ssDiagBtn"); if (pb) pb.onclick = () => navigate("diagnostics");
  const cb = $("#ssCatBtn"); if (cb) cb.onclick = () => navigate("models");
}
function updateNavBadges() {
  const tb = $("#navTensorsBadge");
  if (tb) {
    const modCount = state.tensors.filter(t => t.modified).length;
    if (modCount > 0) { tb.textContent = modCount + " Δ"; tb.className = "nav-badge is-dirty"; tb.hidden = false; }
    else if (state.model && state.tensors.length) { tb.textContent = String(state.tensors.length); tb.className = "nav-badge"; tb.hidden = false; }
    else { tb.hidden = true; }
  }
  const hb = $("#navHistoryBadge");
  if (hb) {
    const n = (state.history && state.history.iterations || []).length;
    if (n > 0) { hb.textContent = String(n); hb.hidden = false; }
    else { hb.hidden = true; }
  }
}
function updateModelChip() {
  const m = state.model;
  $("#modelChipName").textContent = m ? m.id : "модель не загружена";
  $("#modelChipParams").textContent = m ? `${m.params} · ${m.layers || "—"}L` : "";
  $("#sbParams").textContent = m ? `${fmtInt(m.paramsN)} параметров` : "—";
  const b = state.modelInfo && state.modelInfo.backend;
  $("#chipBackend").textContent = b === "torch" ? "PyTorch" : "Rust";
  $("#sbBackend").textContent = backendLabel(b);
  const d = (state.modelInfo && state.modelInfo.diff) || { changed_params: 0, total_params: 0 };
  const dirtyEl = $("#chipDirty");
  if (dirtyEl) {
    if (m && d.changed_params > 0 && d.total_params > 0) {
      const pct = d.changed_params / d.total_params * 100;
      dirtyEl.textContent = `Δ ${pct < 0.01 ? "<0.01" : pct.toFixed(2)}%`;
      dirtyEl.hidden = false;
    } else { dirtyEl.hidden = true; }
  }
  const sbp = $("#sbProvider"); if (sbp) sbp.textContent = "мозг: " + brainLabel(true);
  renderSidebarStatus();
  updateNavBadges();
  if (typeof renderEmptyReadiness === "function") renderEmptyReadiness();
}
/* Pull the real engine snapshot (model, backend, Vespasian, diff). */
async function refreshModelInfo() {
  try {
    const info = await invoke("model_info");
    state.modelInfo = info;
    if (info.loaded) {
      const e = info.entry || {};
      state.model = {
        id: info.model_id, params: e.params_human || "—", paramsN: info.total_params || 0,
        layers: e.num_layers || 0, dModel: e.d_model || 0, vram: e.vram_fp32_mb || 0,
        license: e.license_spdx || "—", desc: e.description || "", family: e.family || "",
        localCached: !!e.local_cached, backend: info.backend
      };
      if (info.pipeline) state.pipeline = { ...PIPELINE_DEFAULTS, ...info.pipeline };
    } else { state.model = null; }
    updateModelChip();
    if (typeof updateBrainBtn === "function") updateBrainBtn();
    if (typeof updateCurrentBrain === "function") updateCurrentBrain();
    if ($("#sessionModelLabel")) $("#sessionModelLabel").textContent = state.model ? state.model.id : "модель не загружена";
  } catch (e) { /* engine not ready yet */ }
}
async function refreshTensors() {
  try {
    const rows = await invoke("tensor_table");
    state.tensors = rows.map(r => ({
      name: r.name, shape: String(r.shape).split("x").map(Number), dtype: r.dtype,
      elements: r.numel, norm: r.l2_norm, modified: r.modified, role: tensorRole(r.name)
    }));
    if (!state.selectedTensor || !state.tensors.some(t => t.name === state.selectedTensor.name)) state.selectedTensor = state.tensors[0] || null;
  } catch { state.tensors = []; }
  updateNavBadges();
}
async function refreshDiagnostics(force) {
  if (state.diag && !force) return state.diag;
  if (state.diagLoading) return state.diag;
  state.diagLoading = true;
  try { state.diag = await invoke("diagnostics"); }
  catch (e) { state.diag = null; if (state.view === "diagnostics") toast("error", "Диагностика недоступна", String(e)); }
  finally { state.diagLoading = false; }
  return state.diag;
}
async function refreshHistory() {
  try { state.history = await invoke("history"); } catch { state.history = { iterations: [], baseline: 0, best: 0 }; }
  updateNavBadges();
  if (state.view === "history") renderHistory();
}
async function afterMutation() {
  state.diag = null;
  await Promise.all([refreshModelInfo(), refreshHistory(), refreshTensors()]);
  updateModelChip();
  if (state.view === "tensors") renderTensors();
  if (state.view === "diagnostics") renderDiagnostics();
  if (state.view === "history") renderHistory();
  if ($("#app").dataset.inspector === "open") renderInspector();
}
/* No implicit model loading: the user must pick a model explicitly. */
function requireModelUI() {
  if (state.modelInfo && state.modelInfo.loaded) return true;
  toast("warning", "Сначала загрузите модель", "Откройте каталог моделей и выберите цель для WSP", { label: "Модели", fn: () => navigate("models") }, 4200);
  navigate("models");
  return false;
}
/* Real model load: catalog ID, HF repo (downloads via hf-hub) or local .safetensors. */
async function loadModel(id, silent = false) {
  if (state.busy && !silent) { busyGuard(); return state.model; }
  setBusy(true);
  try {
    if (!silent) toast("info", isDownloaded(id) ? "Загрузка модели" : "Загрузка / скачивание с Hugging Face", id, null, 2600);
    const msg = await invoke("load_model", { modelId: id });
    store.set("model", id);
    state.diag = null; state.selectedTensor = null; state.selectedLayer = null;
    await Promise.all([refreshModelInfo(), refreshCatalog(), refreshTensors(), refreshHistory()]);
    updateModelChip();
    renderModels(); if (state.view === "tensors") renderTensors();
    if ($("#sessionModelLabel")) $("#sessionModelLabel").textContent = state.model ? state.model.id : id;
    if (!silent) toast("success", "Модель готова", msg);
    else toast("success", "Модель готова", msg, null, 3200);
  } catch (e) {
    toast("error", "Ошибка загрузки модели", String(e), null, 6000);
    throw e;
  } finally { setBusy(false); }
  return state.model;
}
async function refreshCatalog() { try { state.catalog = await invoke("catalog"); } catch { state.catalog = []; } }

/* ---- Live engine log → activity feed ---- */
let liveTurn = null;
function handleLog(msg) {
  if (!liveTurn) return;
  const m = msg.match(/^\[(\d+)\/(\d+)\]\s*(.+?)\s*—\s*(.*)$/);
  const running = liveTurn._events.find(e => e.status === "running");
  if (m) {
    if (running) { running.status = "done"; running.name = `[${m[1]}/${m[2]}] ${m[3]}`; running.output = m[4]; }
    else pushEvent(liveTurn, { kind: "tool", name: `[${m[1]}/${m[2]}] ${m[3]}`, output: m[4], status: "done" });
    pushEvent(liveTurn, { kind: "tool", name: "Следующий шаг пайплайна…", status: "running", output: "" });
  } else if (running) { running.output = msg; paintFeed(liveTurn); }
  else pushEvent(liveTurn, { kind: "tool", name: msg, status: "running", output: "" });
  scrollThread();
}
function finishLiveEvents(turn) {
  turn._events.forEach(e => { if (e.status === "running") { e.status = "done"; if (!e.output) e.output = "ok"; } });
}
function flattenMetrics(metrics) {
  const out = {};
  Object.entries(metrics || {}).forEach(([k, v]) => {
    out[k] = typeof v === "number" ? String(Math.round(v * 1e5) / 1e5) : String(v);
  });
  return out;
}

/* ---- Real YAML validation (weightx-core plan.rs operators) ---- */
const YAML_ASSERT_OPS = ["all", "any", "not", "exists", "shape", "dtype", "equal", "iszero", "finite", "norm", "rel_delta"];
const YAML_MUTATION_OPS = ["copy", "assign", "scale", "scale_", "fill", "fill_", "zeroes", "ones", "add", "add_", "subtract", "subtract_", "multiply", "multiply_", "clamp", "clamp_", "phlora_", "svd_low_rank_", "spectral_norm_cap_", "project_out_direction_", "abliterate_direction_", "lerp_", "slerp_", "ties_trim_", "prune_magnitude_", "rescale_outlier_rows_", "replace_nonfinite_", "rank_one_update_", "closed_form_ridge_", "eamit_batch_"];
function validateYaml(plan) {
  const errors = []; const asserts = [];
  if (!plan.trim()) return { ok: false, errors: ["План пуст"], asserts };
  if (!/transforms:/.test(plan)) errors.push("Нет секции transforms:");
  const lines = plan.split("\n");
  let mutationCount = 0;
  lines.forEach((ln, i) => {
    const t = ln.trim();
    if (t.startsWith("#")) return;
    const am = t.match(/^-\s*assert:\s*\{\s*(\w+):/);
    if (am) { const name = am[1]; const ok = YAML_ASSERT_OPS.includes(name); asserts.push({ name, line: i + 1, ok }); if (!ok) errors.push(`Строка ${i + 1}: неизвестный инвариант «${name}»`); return; }
    const om = t.match(/^-\s*([a-z_]+[a-z0-9_]*):/);
    if (om && om[1] !== "assert") {
      const op = om[1];
      if (YAML_MUTATION_OPS.includes(op)) mutationCount++;
      else if (!["dump", "diff", "transforms"].includes(op)) errors.push(`Строка ${i + 1}: неизвестная операция «${op}»`);
    }
  });
  if (mutationCount > 0 && asserts.length === 0) errors.push("Invariant Policy: план с правками обязан содержать хотя бы один assert");
  return { ok: errors.length === 0, errors, asserts };
}
async function applyYaml(plan, fisher) {
  if (busyGuard()) return;
  const v = validateYaml(plan); renderYamlValidation(v);
  if (!v.ok) { toast("error", "План не прошёл валидацию", v.errors[0]); return; }
  if (!requireModelUI()) return;
  const effFisher = fisher && state.guards.fisher;
  const verify = state.guards.vespasian;
  navigate("session"); setBusy(true);
  addUserTurn("YAML-план · " + (effFisher ? "Fisher guard" : "raw") + (verify ? "" : " · Vespasian выкл"), "BrainSurgery-WX");
  const turn = addAssistantTurn("Декларативная YAML-хирургия", null);
  pushEvent(turn, { kind: "plan", name: "BrainSurgery-WX", status: "done", output: "Парсинг → assertions → правки → Vespasian" });
  pushEvent(turn, { kind: "tool", name: "execute_yaml_plan", args: `fisher=${effFisher} · verify=${verify}`, status: "running", output: "" });
  liveTurn = turn;
  try {
    const out = await invoke("apply_yaml_ex", { plan, fisher: effFisher, verify });
    liveTurn = null; finishLiveEvents(turn);
    const rolled = /ROLLED BACK|откат|rolled_back\s*[:=]\s*true/i.test(out) && !/accepted\s*[:=]\s*true/i.test(out);
    pushEvent(turn, { kind: "result", name: rolled ? "ROLLED BACK" : "ВЫПОЛНЕНО", status: rolled ? "error" : "done", metrics: {}, output: out });
    state.lastYaml = plan;
    await afterMutation();
    toast(rolled ? "warning" : "success", rolled ? "Vespasian: откат выполнен" : "План применён", rolled ? "Инвариант нарушен · веса восстановлены" : "Отчёт в ленте чата", { label: "История", fn: () => navigate("history") });
  } catch (e) {
    liveTurn = null; finishLiveEvents(turn);
    pushEvent(turn, { kind: "result", name: "ОШИБКА", status: "error", metrics: {}, output: String(e) });
    toast("error", "План не применён", String(e));
  } finally { setBusy(false); }
}
async function runSkill(skillId) {
  if (busyGuard()) return; const sk = skillById(skillId); if (!sk) return;
  if (!requireModelUI()) return;
  navigate("session");
  addUserTurn(`/skill ${skillId}`, sk.pill_label); setBusy(true);
  const turn = addAssistantTurn(sk.pipeline_title, skillId);
  pushEvent(turn, { kind: "plan", name: sk.pipeline_title, status: "done", output: (sk.stages || []).join(" → ") });
  pushEvent(turn, { kind: "tool", name: "Запуск пайплайна…", status: "running", output: "" });
  liveTurn = turn;
  try {
    const res = await invoke("run_skill", { skillId, args: "" });
    liveTurn = null;
    turn._events = turn._events.filter(e => e.kind === "plan"); turn._open = new Set();
    (res.steps || []).forEach(st => pushEvent(turn, {
      kind: "tool", name: `[${st.index}/${st.total}] ${st.title}`, output: st.detail,
      status: st.status === "rollback" ? "error" : "done"
    }));
    pushEvent(turn, { kind: "result", name: res.accepted ? "COMMITTED" : "ROLLED BACK", status: res.accepted ? "done" : "error", metrics: flattenMetrics(res.metrics), output: res.summary });
    if (res.yaml_plan && res.yaml_plan.trim()) { state.lastYaml = res.yaml_plan; pushEvent(turn, { kind: "yaml", name: "BrainSurgery-WX · plan.yaml", status: "done", output: res.yaml_plan }); }
    await afterMutation();
    if (res.accepted) toast("success", "Пайплайн завершён", `${sk.pill_label} · COMMITTED`, { label: "Экспорт", fn: () => exportCheckpoint() });
    else toast("warning", "Vespasian: откат выполнен", `${sk.pill_label} · веса восстановлены bit-exact`, { label: "История", fn: () => navigate("history") });
  } catch (e) {
    liveTurn = null; finishLiveEvents(turn);
    pushEvent(turn, { kind: "result", name: "ОШИБКА", status: "error", metrics: {}, output: String(e) });
    toast("error", "Навык не выполнен", String(e));
  } finally { setBusy(false); }
}
async function runChat(goal) {
  if (busyGuard()) return;
  if (!requireModelUI()) return;
  navigate("session"); addUserTurn(goal, "WSP-цель"); setBusy(true);
  const turn = addAssistantTurn("5-стадийный WSP-цикл (Brain → Tools → Vespasian)", null);
  pushEvent(turn, { kind: "plan", name: "WSP-цикл", status: "done", output: ["Brain (" + brainLabel(true) + ")", "L1/L2 диагностика", "L3 правки в sandbox", "Vespasian", "Экспорт"].join(" → ") });
  pushEvent(turn, { kind: "tool", name: "Brain: планирование цели…", status: "running", output: "" });
  liveTurn = turn;
  try {
    const out = await invoke("run_action", { tag: "chat", prompt: goal });
    liveTurn = null; finishLiveEvents(turn);
    await afterMutation();
    const h = state.history;
    pushEvent(turn, { kind: "result", name: /Ошибка:/.test(out) ? "ОШИБКА" : "ЦИКЛ ЗАВЕРШЁН", status: /Ошибка:/.test(out) ? "error" : "done",
      metrics: { "held-out база": String(Math.round((h.baseline || 0) * 1e5) / 1e5), "held-out лучший": String(Math.round((h.best || 0) * 1e5) / 1e5), "итераций Vespasian": String((h.iterations || []).length) },
      output: out });
    toast("success", "WSP-цикл завершён", "Отчёт и diff доступны в истории", { label: "История", fn: () => navigate("history") });
  } catch (e) {
    liveTurn = null; finishLiveEvents(turn);
    pushEvent(turn, { kind: "result", name: "ОШИБКА", status: "error", metrics: {}, output: String(e) });
    toast("error", "Цикл не выполнен", String(e));
  } finally { setBusy(false); }
}
async function rollbackCheckpoint(id) {
  const ok = await confirmDialog({ title: "Восстановить исходные веса?", desc: `Vespasian/Workspace восстановит все изменённые тензоры к исходному состоянию загруженной модели (bit-exact). Текущие правки в sandbox будут потеряны. Запрошено из истории: ${id}.`, confirmLabel: "Восстановить", danger: true });
  if (!ok) return;
  setBusy(true);
  try { const msg = await invoke("rollback_weights"); await afterMutation(); toast("success", "Откат выполнен", msg, { label: "История", fn: () => navigate("history") }); }
  catch (e) { toast("error", "Откат не выполнен", String(e)); }
  finally { setBusy(false); }
}
async function exportCheckpoint(path) {
  const p = path || state.exportPath || "checkpoints/exported_model.safetensors";
  if (!state.model) { toast("warning", "Нет модели для экспорта", "Загрузите модель из каталога моделей"); return; }
  if (!/\.safetensors$/.test(p)) { toast("error", "Некорректный путь", "Ожидается файл .safetensors"); return; }
  setBusy(true);
  toast("info", "Экспорт запущен", `${p} · ${state.model.params} параметров`, null, 2000);
  try { const msg = await invoke("export_checkpoint", { path: p }); toast("success", "Экспорт завершён", msg, { label: "История", fn: () => navigate("history") }, 5200); }
  catch (e) { toast("error", "Экспорт не выполнен", String(e)); }
  finally { setBusy(false); }
}
/* Push real pipeline parameters to the engine (Fisher guard + Vespasian). */
let pipelineTimer = null;
function pushPipelineParams() {
  clearTimeout(pipelineTimer);
  pipelineTimer = setTimeout(async () => {
    const p = state.pipeline; if (!p || !state.modelInfo || !state.modelInfo.loaded) return;
    try { await invoke("set_pipeline", { fisherAlpha: p.fisher_alpha, fisherFreeze: p.fisher_freeze, improvementThreshold: p.improvement_threshold, maxKlDrift: p.max_kl_drift, maxChangedFraction: p.max_changed_fraction, practiceBudget: p.practice_budget }); }
    catch (e) { toast("error", "Параметры не применены", String(e)); }
  }, 350);
}

/* ============================================================ CHARTS */
function getHeatStops() { if (heatStopsCache) return heatStopsCache; const cs = getComputedStyle(document.documentElement);
  heatStopsCache = ["--data-1", "--data-2", "--data-3", "--data-4", "--data-5"].map(v => hexToRgb(cs.getPropertyValue(v) || "#7aaaff")); return heatStopsCache; }
function heatColor(v, min, max) { const t = Math.max(0, Math.min(1, (v - min) / (max - min || 1))); const stops = getHeatStops();
  const x = t * (stops.length - 1), i = Math.floor(x), f = x - i; const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  const c = a.map((av, k) => Math.round(av + (b[k] - av) * f)); return `rgb(${c[0]},${c[1]},${c[2]})`; }
/* Real SVD spectrum (top singular values from describe_tensor). */
function spectrumChart(vals) {
  const W = 520, H = 160, pad = 30, n = vals.length; const max = Math.max(...vals, 1e-9), bw = (W - pad * 2) / n; let bars = "", labels = "", grid = "";
  [0.25, 0.5, 0.75, 1.0].forEach(f => {
    const gy = H - pad - f * (H - pad - 18);
    grid += `<line x1="${pad}" y1="${gy.toFixed(1)}" x2="${W - pad}" y2="${gy.toFixed(1)}" stroke="var(--border-subtle)" stroke-dasharray="3 3"/>`;
  });
  vals.forEach((v, i) => { const h = Math.max(3, (v / max) * (H - pad - 18)); const x = pad + i * bw + 2, y = H - pad - h;
    const hue = i < 3 ? "var(--data-1)" : i < 8 ? "var(--data-2)" : i < 16 ? "var(--data-7)" : "var(--data-8)";
    bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(2, bw - 4).toFixed(1)}" height="${h.toFixed(1)}" rx="2.5" fill="${hue}" opacity="0.92" data-chart data-tip="σ${i + 1} = ${v.toFixed(3)}"></rect>`;
    if (i < 3) labels += `<text x="${(x + Math.max(2, bw - 4) / 2).toFixed(1)}" y="${(y - 5).toFixed(1)}" text-anchor="middle" font-size="9" font-family="var(--font-mono)" fill="var(--text-tertiary)">${v.toFixed(1)}</text>`; });
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="SVD-спектр">${grid}<line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="var(--border-default)"/>${bars}${labels}<text x="${pad}" y="${H - 8}" font-size="9" font-family="var(--font-mono)" fill="var(--text-tertiary)">σ1</text><text x="${W - pad}" y="${H - 8}" text-anchor="end" font-size="9" font-family="var(--font-mono)" fill="var(--text-tertiary)">σ${n}</text><text x="${pad - 6}" y="${pad - 4}" text-anchor="end" font-size="9" font-family="var(--font-mono)" fill="var(--text-tertiary)">${max.toFixed(1)}</text></svg>
  <div class="legend" style="margin-top:6px"><span class="lg"><span class="sw" style="background:var(--data-1)"></span>головные σ1–3</span><span class="lg"><span class="sw" style="background:var(--data-2)"></span>средние σ4–8</span><span class="lg"><span class="sw" style="background:var(--data-7)"></span>тело σ9–16</span><span class="lg"><span class="sw" style="background:var(--data-8)"></span>хвост σ17+</span></div>`;
}
/* Real weight block heatmap (rows × cols from tensor_block). */
function heatmapHTML(matrix, rows, cols, label) { const min = Math.min(...matrix), max = Math.max(...matrix), mid = (min + max) / 2; let cells = "";
  for (let i = 0; i < matrix.length; i++) { const r = Math.floor(i / cols), c = i % cols; cells += `<div class="cell" style="background:${heatColor(matrix[i], min, max)}" data-chart data-tip="${label} [${r},${c}] = ${matrix[i].toFixed(4)}"></div>`; }
  return `<div class="heatmap heatmap-weights" style="grid-template-columns:repeat(${cols},1fr)">${cells}</div><div class="legend" style="margin-top:10px"><span class="lg"><span class="sw" style="background:${heatColor(min, min, max)}"></span>${min.toFixed(3)}</span><span class="lg"><span class="sw" style="background:${heatColor(mid, min, max)}"></span>${mid.toFixed(3)}</span><span class="lg"><span class="sw" style="background:${heatColor(max, min, max)}"></span>${max.toFixed(3)}</span><span style="margin-left:auto">наведите</span></div>`; }
/* Real Knowledge Map: Logit-Lens per-layer stats (compact fixed-height rows). */
function knowledgeHeatmap(knowledge) {
  const ls = (knowledge && knowledge.logit_lens_layers) || [];
  if (!ls.length) return `<div class="field-hint">Logit Lens недоступен для этой модели (нет runtime-архитектуры). Загрузите модель семейства Llama/GPT-2.</div>`;
  const cols = ["target_prob_mean", "target_reciprocal_rank", "predictive_entropy"];
  const colLabels = ["P(target)", "1/rank", "Энтропия"];
  const vals = []; ls.forEach(l => cols.forEach(k => vals.push(Number(l[k]) || 0)));
  const min = Math.min(...vals), max = Math.max(...vals);
  const rowsHTML = ls.map(l => {
    const cells = cols.map((k, ci) => {
      const v = Number(l[k]) || 0;
      return `<div class="km-cell" style="background:${heatColor(v, min, max)}" data-chart data-tip="слой ${l.layer_idx} · ${colLabels[ci]} = ${v.toFixed(3)}"></div>`;
    }).join("");
    return `<div class="km-row"><span class="km-layer">L${l.layer_idx}</span>${cells}</div>`;
  }).join("");
  return `<div class="km-wrap"><div class="km-head"><span>Слой</span>${colLabels.map(c => `<span>${c}</span>`).join("")}</div><div class="km-rows">${rowsHTML}</div></div><div class="legend" style="margin-top:8px"><span class="lg"><span class="sw" style="background:${heatColor(min, min, max)}"></span>мин (${min.toFixed(2)})</span><span class="lg"><span class="sw" style="background:${heatColor(max, min, max)}"></span>макс (${max.toFixed(2)})</span><span style="margin-left:auto">${ls.length} слоёв</span></div>`;
}
/* Real held-out score trajectory from Vespasian iteration history. */
function scoreChart(history) {
  const its = (history && history.iterations) || [];
  const W = 520, H = 176, pad = 36;
  const pts = [{ x: 0, base: history.baseline || 0, best: history.baseline || 0 }];
  its.forEach((it, i) => pts.push({ x: i + 1, base: it.candidate_score, best: it.best_after, rolled: it.rolled_back }));
  if (pts.length < 2) return `<div class="field-hint">Запустите навык или WSP-цикл — динамика held-out score появится здесь (база ${fmt(history.baseline || 0, 4)}).</div>`;
  const ys = pts.flatMap(p => [p.base, p.best]); const min = Math.min(...ys), max = Math.max(...ys); const span = (max - min) || 1;
  const X = i => pad + i * (W - pad * 2) / (pts.length - 1); const Y = v => H - pad - (v - min) / span * (H - pad * 2);
  const path = key => pts.map((p, i) => (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(p[key]).toFixed(1)).join(" ");
  const bestArea = `${path("best")} L${X(pts.length - 1).toFixed(1)} ${(H - pad).toFixed(1)} L${X(0).toFixed(1)} ${(H - pad).toFixed(1)} Z`;
  let dots = "", grid = "", xticks = "";
  pts.forEach((p, i) => {
    dots += `<circle cx="${X(i)}" cy="${Y(p.base)}" r="3.5" fill="${p.rolled ? "var(--status-error)" : "var(--data-1)"}" stroke="var(--bg-raised)" stroke-width="1.5" data-chart data-tip="итерация ${i} · candidate: ${fmt(p.base, 4)}${p.rolled ? " · откат" : ""}"></circle>`;
    if (pts.length <= 12 || i === 0 || i === pts.length - 1 || i % Math.ceil(pts.length / 8) === 0) {
      xticks += `<text x="${X(i).toFixed(1)}" y="${H - 12}" text-anchor="middle" font-size="9" font-family="var(--font-mono)" fill="var(--text-ghost)">#${i}</text>`;
    }
  });
  [min, (min + max) / 2, max].forEach(g => grid += `<line x1="${pad}" y1="${Y(g)}" x2="${W - pad}" y2="${Y(g)}" stroke="var(--border-subtle)" stroke-dasharray="3 3"/><text x="${pad - 6}" y="${Y(g) + 3}" text-anchor="end" font-size="9" font-family="var(--font-mono)" fill="var(--text-tertiary)">${g.toFixed(3)}</text>`);
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Held-out score по итерациям"><defs><linearGradient id="scoreAreaGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="var(--data-3)" stop-opacity="0.18"/><stop offset="100%" stop-color="var(--data-3)" stop-opacity="0.0"/></linearGradient></defs>${grid}<path d="${bestArea}" fill="url(#scoreAreaGrad)"/><path d="${path("base")}" fill="none" stroke="var(--data-1)" stroke-width="2"/><path d="${path("best")}" fill="none" stroke="var(--data-3)" stroke-width="2" stroke-dasharray="6 3"/>${dots}${xticks}<text x="${X(pts.length - 1) - 4}" y="${Math.max(14, Y(pts[pts.length - 1].best) - 8)}" text-anchor="end" font-size="9" font-family="var(--font-mono)" fill="var(--data-3)">best ${fmt(pts[pts.length - 1].best, 4)}</text></svg>
  <div class="legend" style="margin-top:6px"><span class="lg"><span class="sw" style="background:var(--data-1)"></span>candidate score</span><span class="lg"><span class="sw" style="background:var(--data-3)"></span>лучший (best)</span><span class="lg"><span class="sw" style="background:var(--status-error)"></span>откат</span></div>`;
}
function gaugeSVG(value, label) { const r = 42, c = 2 * Math.PI * r, off = c * (1 - Math.max(0, Math.min(1, value)));
  return `<svg viewBox="0 0 110 110" width="110" height="110" role="img" aria-label="${label} ${value}"><circle cx="55" cy="55" r="${r}" fill="none" stroke="var(--bg-active)" stroke-width="9"/><circle cx="55" cy="55" r="${r}" fill="none" stroke="url(#gaugegrad)" stroke-width="9" stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${off}" transform="rotate(-90 55 55)"/><defs><linearGradient id="gaugegrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="var(--data-1)"/><stop offset="1" stop-color="var(--data-2)"/></linearGradient></defs><text x="55" y="52" text-anchor="middle" font-size="20" font-family="var(--font-mono)" font-weight="600" fill="var(--text-primary)">${value.toFixed(2)}</text><text x="55" y="68" text-anchor="middle" font-size="8" letter-spacing="1" fill="var(--text-tertiary)">${label}</text></svg>`; }
/* Model architecture visualization from real L2 layer rows (data-driven bar heights). */
function modelVizHTML(layers) {
  const sel = state.selectedLayer;
  const dModel = state.model ? state.model.dModel : 0;
  const maxInf = Math.max(...layers.map(l => Math.abs(Number(l.block_influence) || 0)), 1e-6);
  const maxRank = Math.max(...layers.map(l => Math.abs(Number(l.effective_rank) || 0)), 1e-6);
  let blocks = `<div class="mv-cap"><div class="mv-cap-box">embed<br>${dModel || "—"}d</div></div><div class="mv-arrow">${icon("arrow-right")}</div>`;
  layers.forEach(l => {
    const idx = l.layer_idx;
    const infH = Math.round(8 + (Math.abs(Number(l.block_influence) || 0) / maxInf) * 14);
    const rankH = Math.round(8 + (Math.abs(Number(l.effective_rank) || 0) / maxRank) * 14);
    const cls = `${l.isBottleneck ? "bn" : ""} ${l.isRedundant ? "rd" : ""} ${sel && sel.layer_idx === idx ? "is-selected" : ""}`;
    const tag = l.isBottleneck ? " · bottleneck" : l.isRedundant ? " · избыточный" : "";
    blocks += `<div class="mv-block"><div class="mv-node ${cls}" data-mvlayer="${idx}" role="button" tabindex="0" aria-label="Слой ${idx}" data-chart data-tip="слой ${idx}${tag} · influence ${fmt(l.block_influence, 3)} · rank ${fmt(l.effective_rank, 1)}"><div class="mv-bars"><div class="mv-attn" style="height:${infH}px"></div><div class="mv-mlp" style="height:${rankH}px"></div></div><div class="mv-idx">${idx}</div></div></div>`;
  });
  blocks += `<div class="mv-arrow">${icon("arrow-right")}</div><div class="mv-cap"><div class="mv-cap-box">norm<br>+ lm_head</div></div>`;
  return `<div class="model-viz">${blocks}</div><div class="legend" style="margin-top:6px"><span class="lg"><span class="sw" style="background:color-mix(in srgb, var(--data-1) 68%, var(--bg-raised))"></span>attention (influence)</span><span class="lg"><span class="sw" style="background:color-mix(in srgb, var(--data-7) 60%, var(--bg-raised))"></span>MLP (eff. rank)</span><span class="lg"><span class="sw" style="background:color-mix(in srgb, var(--status-warning) 80%, var(--bg-raised))"></span>bottleneck</span><span class="lg"><span class="sw" style="background:color-mix(in srgb, var(--data-2) 42%, var(--bg-raised))"></span>избыточный</span><span style="margin-left:auto">клик по блоку — инспектор</span></div>`;
}

/* ============================================================ UI CORE */
const VIEWS = ["session", "tensors", "diagnostics", "yaml", "history", "models", "provider", "settings", "hotkeys"];
const VIEW_LABEL = { session: "Чат", tensors: "Тензоры", diagnostics: "Диагностика", yaml: "YAML-план", history: "История", models: "Модели", provider: "Провайдер", settings: "Настройки", hotkeys: "Горячие клавиши" };
const VIEW_ICON = { session: "chat", tensors: "box", diagnostics: "activity", yaml: "file-code", history: "history", models: "layers", provider: "brain", settings: "settings", hotkeys: "keyboard" };
function navigate(view) {
  if (!VIEWS.includes(view)) return; state.view = view;
  $$(".nav-item[data-view]").forEach(b => { const on = b.dataset.view === view; b.classList.toggle("is-active", on); if (on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
  $$(".view").forEach(v => v.classList.toggle("is-active", v.id === "view-" + view));
  if (window.innerWidth <= 980) { $("#app").dataset.sidebar = "collapsed"; $("#sideBackdrop").classList.remove("is-in"); $("#app").dataset.inspector = "closed"; }
  if (view === "tensors") renderTensors();
  if (view === "diagnostics") renderDiagnostics();
  if (view === "history") renderHistory();
  if (view === "models") renderModelsView();
  if (view === "provider") renderProviderView();
  if (view === "settings") renderSettings();
  if (view === "hotkeys") renderHotkeys();
  if (view === "yaml") renderYamlValidation(validateYaml($("#yamlArea")?.value || state.lastYaml || ""));
  if ($("#app").dataset.inspector === "open") renderInspector();
  syncBgDim();
  closeStylesPop(); closeBrainPop(); closeSelectPop(); closeSlash();
}
/* Styles */
function setStyle(id) {
  const prev = state.style;
  state.style = id;
  if (id === "coffee") { state.bgMode = "coffee"; store.set("bgMode", "coffee"); bgSync(); }
  else if (prev === "coffee" && state.bgMode === "coffee") { state.bgMode = "neural"; store.set("bgMode", "neural"); bgSync(); }
  applyStyle();
  toast("info", "Стиль: " + (STYLE_BY_ID[id]?.name || id), id === "coffee" ? "Включён фон «Кофе из нейронов»" : "Оформление обновлено", null, 1800);
}
function renderStylesGrids() {
  const html = STYLES.map(s => {
    const pvBg = s.id === "coffee" ? "linear-gradient(135deg,#2e1b10 0%,#1c110b 65%,#3f2617 100%)" : s.bg;
    return `<button class="style-card ${state.style === s.id ? "is-active" : ""}" data-style-set="${s.id}"><div class="style-preview" style="background:${pvBg};border-radius:${s.radius ?? 6}px"><span class="sp-bar" style="background:${s.primary}"></span><span class="sp-dot" style="background:${s.accent};box-shadow:0 0 10px ${s.accent}"></span></div><div class="style-name">${esc(s.name)}</div><div class="style-desc">${esc(s.desc)}</div></button>`;
  }).join("");
  const pop = $("#stylesPopGrid"); if (pop) { pop.innerHTML = html; $$("[data-style-set]", pop).forEach(b => b.onclick = () => { setStyle(b.dataset.styleSet); }); }
  const set = $("#stylesGridSettings"); if (set) { set.innerHTML = html; $$("[data-style-set]", set).forEach(b => b.onclick = () => { setStyle(b.dataset.styleSet); renderSettings(); }); }
  const bgPop = $("#bgPopGrid");
  if (bgPop) {
    bgPop.innerHTML = BG_MODES.map(m => `<button type="button" class="bg-chip ${state.bgMode === m.id ? "is-active" : ""}" data-bg-set="${m.id}" data-tip="${esc(m.desc)}">${esc(m.name)}</button>`).join("");
    $$("[data-bg-set]", bgPop).forEach(b => b.onclick = () => setBgMode(b.dataset.bgSet));
  }
  const bgSet = $("#bgModesGridSettings");
  if (bgSet) {
    bgSet.innerHTML = BG_MODES.map(m => `<button type="button" class="bg-mode-card ${state.bgMode === m.id ? "is-active" : ""}" data-bg-set="${m.id}"><div class="bm-name">${esc(m.name)}</div><div class="bm-desc">${esc(m.desc)}</div></button>`).join("");
    $$("[data-bg-set]", bgSet).forEach(b => b.onclick = () => setBgMode(b.dataset.bgSet));
  }
}
function openStylesPop() { closeBrainPop(); closeSelectPop(); closeSlash(); renderStylesGrids(); $("#stylesPop").classList.add("is-open"); }
function closeStylesPop() { $("#stylesPop").classList.remove("is-open"); }
$("#styleBtn").onclick = (e) => { e.stopPropagation(); $("#stylesPop").classList.contains("is-open") ? closeStylesPop() : openStylesPop(); };
document.addEventListener("click", e => {
  if (!e.target.closest("#stylesPop") && !e.target.closest("#styleBtn")) closeStylesPop();
  if (!e.target.closest(".composer")) closeSlash();
  if (!e.target.closest("#selectPop") && !e.target.closest(".wx-select-btn")) closeSelectPop();
});
/* Brain model picker (composer) — real active brain + live provider model lists */
function updateBrainBtn() { const btn = $("#brainBtn"); if (!btn) return; const b = activeBrain();
  $("#brainBtnIcon").innerHTML = `<use href="#i-${(PROVIDER_META[b.kind] || {}).icon || "brain"}"/>`;
  $("#brainBtnLabel").textContent = b.kind === "autonomous" ? "Встроенный" : (b.model || "модель не указана");
  btn.dataset.tip = "Мозг WeightX · " + esc(brainLabel()); }
function brainModelsFor(kind) {
  const cache = state.providerModelsCache[kind] || {};
  const list = [];
  const custom = providerCfg(kind).model;
  if (custom) list.push({ id: custom, meta: "сохранена" });
  (cache.models || []).forEach(id => { if (!list.some(x => x.id === id)) list.push({ id, meta: "API" }); });
  return list;
}
async function fetchProviderModels(kind) {
  if (kind === "autonomous") return;
  const cfg = providerCfg(kind);
  state.providerModelsCache[kind] = { ...(state.providerModelsCache[kind] || {}), loading: true };
  if ($("#brainPop").classList.contains("is-open")) renderBrainList();
  try {
    const r = await invoke("provider_models", { kind: fullProviderKind(kind), baseUrl: cfg.baseUrl || "", apiKey: cfg.apiKey || "" });
    state.providerModelsCache[kind] = { models: r.models || [], source: r.source || "", note: r.note || "", loading: false };
  } catch (e) {
    state.providerModelsCache[kind] = { models: [], error: String(e), loading: false };
  }
  if ($("#brainPop").classList.contains("is-open")) renderBrainList();
}
function renderBrainList() { const q = ($("#brainSearch").value || "").trim().toLowerCase(); const b = activeBrain(); const list = $("#brainList"); let html = "";
  configuredProviders().forEach(p => {
    if (p.kind === "autonomous") {
      const sel = b.kind === "autonomous";
      if (q && !"autonomous встроенный".includes(q)) return;
      html += `<div class="bp-group">${icon(p.icon)}<span>Встроенный runtime</span><span class="dot dot-success"></span></div>`;
      html += `<button class="bp-item ${sel ? "is-selected" : ""}" data-bkind="autonomous" data-bmodel="">${icon("check", "icon bp-check")}<span class="bp-name">Autonomous WSP-оркестратор</span><span class="bp-meta">без LLM</span></button>`;
      return;
    }
    const cache = state.providerModelsCache[p.kind] || {};
    const models = brainModelsFor(p.kind).filter(m => !q || m.id.toLowerCase().includes(q));
    const isActive = b.kind === p.kind;
    const chk = state.providerChecks[p.kind];
    const dot = chk ? (chk.ok ? "dot-success" : "dot-error") : (isActive ? "dot-success" : "dot-info");
    const statusMeta = isActive ? "активен" : chk ? (chk.ok ? "доступен" : "не отвечает") : "";
    html += `<div class="bp-group">${icon(p.icon)}<span>${esc(p.label)}</span><span class="dot ${dot}"></span>${statusMeta ? `<span class="bp-meta">${statusMeta}</span>` : ""}</div>`;
    if (chk && !chk.ok) html += `<div class="bp-empty" style="padding:6px 12px">${esc(chk.detail)}</div>`;
    else if (cache.loading) html += `<div class="bp-empty" style="padding:6px 12px">загрузка моделей…</div>`;
    else if (cache.error) html += `<div class="bp-empty" style="padding:6px 12px">${esc(cache.error)}</div>`;
    models.forEach(m => { const sel = isActive && b.model === m.id;
      html += `<button class="bp-item ${sel ? "is-selected" : ""}" data-bkind="${p.kind}" data-bmodel="${esc(m.id)}">${icon("check", "icon bp-check")}<span class="bp-name">${esc(m.id)}</span><span class="bp-meta">${esc(m.meta || "")}</span></button>`; });
    if (!models.length && !cache.loading && !cache.error && !(chk && !chk.ok)) html += `<div class="bp-empty" style="padding:6px 12px">${esc(cache.note || "Нет сохранённой модели. Укажите её в разделе «Провайдер».")}</div>`;
  });
  list.innerHTML = html || `<div class="bp-empty">Ничего не найдено.<br>Настройте провайдеры в разделе «Провайдер».</div>`;
  $$(".bp-item", list).forEach(btn => btn.onclick = () => { selectBrain(btn.dataset.bkind, btn.dataset.bmodel); closeBrainPop(); }); }
function openBrainPop() { closeStylesPop(); closeSelectPop(); closeSlash(); $("#brainSearch").value = ""; renderBrainList(); const pop = $("#brainPop"); pop.classList.add("is-open"); $("#brainBtn").classList.add("is-open");
  configuredProviders().forEach(p => { if (p.kind !== "autonomous") { if (!state.providerModelsCache[p.kind]) fetchProviderModels(p.kind); if (!state.providerChecks[p.kind]) checkProvider(p.kind, providerCfg(p.kind)); } });
  const r = $("#brainBtn").getBoundingClientRect(); const w = pop.offsetWidth, h = pop.offsetHeight;
  const x = Math.max(8, Math.min(r.right - w, innerWidth - w - 8)); let y = r.top - h - 8; if (y < 8) y = Math.min(r.bottom + 8, innerHeight - h - 8);
  pop.style.left = x + "px"; pop.style.top = y + "px"; setTimeout(() => $("#brainSearch").focus(), 80); }
function closeBrainPop() { $("#brainPop").classList.remove("is-open"); const b = $("#brainBtn"); if (b) b.classList.remove("is-open"); }
/* Selecting a brain model switches the engine only after a real availability check. */
async function selectBrain(kind, model) {
  const prev = providerCfg(kind);
  const cfg = { model: kind === "autonomous" ? "" : model, apiKey: prev.apiKey || "", baseUrl: prev.baseUrl || "" };
  if (kind !== "autonomous") {
    const chk = await checkProvider(kind, cfg);
    if (!chk.ok) { toast("warning", "Провайдер недоступен — мозг не переключён", chk.detail, null, 6000); return; }
  }
  try {
    await invoke("set_provider", { kind: fullProviderKind(kind), model: cfg.model, apiKey: cfg.apiKey, baseUrl: cfg.baseUrl });
    state.provider = { kind, ...cfg };
    state.providerConfigs[kind] = { ...cfg };
    store.set("provider", state.provider); store.set("providerConfigs", state.providerConfigs);
    await refreshModelInfo();
    updateProviderLive();
    toast("success", "Мозг установлен: " + (kind === "autonomous" ? "Встроенный runtime" : model), brainLabel() + " · активен в движке", null, 2800);
  } catch (e) { toast("error", "Провайдер не установлен", String(e), null, 5000); }
}
$("#brainSearch").addEventListener("input", renderBrainList);
$("#brainManage").onclick = () => { closeBrainPop(); navigate("provider"); };
document.addEventListener("click", e => { if (!e.target.closest("#brainPop") && !e.target.closest("#brainBtn")) closeBrainPop(); });

/* Theme-native floating select dropdowns (avoids white WebView2 native popups) */
let activeSelectState = null;
function closeSelectPop() {
  const pop = $("#selectPop");
  if (pop) { pop.classList.remove("is-open"); pop.innerHTML = ""; }
  if (activeSelectState) {
    activeSelectState.btn.classList.remove("is-open");
    activeSelectState.btn.setAttribute("aria-expanded", "false");
    activeSelectState = null;
  }
}
function openSelectPop(sel, btn) {
  const pop = $("#selectPop"); if (!pop) return;
  if (activeSelectState && activeSelectState.btn === btn) { closeSelectPop(); return; }
  closeStylesPop(); closeBrainPop(); closeSlash(); closeSelectPop();
  const opts = Array.from(sel.options);
  let focusIdx = Math.max(0, sel.selectedIndex);
  const paintOpts = () => {
    pop.innerHTML = opts.map((o, i) => `<button type="button" class="select-opt ${o.value === sel.value ? "is-selected" : ""} ${i === focusIdx ? "is-focused" : ""}" data-opt-idx="${i}" role="option" aria-selected="${o.value === sel.value}">${icon("check", "icon so-check")}<span class="so-text">${esc(o.textContent)}</span></button>`).join("");
    $$(".select-opt", pop).forEach(item => {
      item.onmouseenter = () => { focusIdx = +item.dataset.optIdx; $$(".select-opt", pop).forEach((x, k) => x.classList.toggle("is-focused", k === focusIdx)); };
      item.onclick = (e) => {
        e.stopPropagation();
        const o = opts[+item.dataset.optIdx];
        if (!o) return;
        sel.value = o.value;
        sel.dispatchEvent(new Event("change", { bubbles: true }));
        sel._syncBtnLabel?.();
        closeSelectPop();
        btn.focus();
      };
    });
  };
  paintOpts();
  pop.classList.add("is-open");
  btn.classList.add("is-open");
  btn.setAttribute("aria-expanded", "true");
  activeSelectState = { sel, btn, getIdx: () => focusIdx, setIdx: (i) => { focusIdx = Math.max(0, Math.min(opts.length - 1, i)); paintOpts(); pop.querySelector(".select-opt.is-focused")?.scrollIntoView({ block: "nearest" }); }, opts };
  const r = btn.getBoundingClientRect();
  pop.style.minWidth = Math.max(180, Math.round(r.width)) + "px";
  const pw = pop.offsetWidth || Math.max(180, r.width);
  const ph = pop.offsetHeight || 220;
  const x = Math.max(8, Math.min(r.left, innerWidth - pw - 8));
  const belowOk = r.bottom + 6 + ph <= innerHeight - 8;
  const y = belowOk ? r.bottom + 4 : Math.max(8, r.top - ph - 4);
  pop.style.left = x + "px";
  pop.style.top = y + "px";
  pop.querySelector(".select-opt.is-selected")?.scrollIntoView({ block: "nearest" });
}
function enhanceSelects(root = document) {
  $$("select.select:not(.wx-select-native)", root).forEach(sel => {
    const wrap = document.createElement("span");
    wrap.className = "wx-select";
    sel.parentNode.insertBefore(wrap, sel);
    wrap.appendChild(sel);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "select wx-select-btn";
    if (sel.getAttribute("style")) btn.setAttribute("style", sel.getAttribute("style"));
    if (sel.getAttribute("aria-label")) btn.setAttribute("aria-label", sel.getAttribute("aria-label"));
    if (sel.dataset.tip) { btn.dataset.tip = sel.dataset.tip; delete sel.dataset.tip; }
    btn.setAttribute("aria-haspopup", "listbox");
    btn.setAttribute("aria-expanded", "false");
    btn.innerHTML = `<span class="ws-label"></span>`;
    const syncLabel = () => {
      const opt = sel.options[sel.selectedIndex] || sel.options[0];
      btn.querySelector(".ws-label").textContent = opt ? opt.textContent : "";
    };
    sel._syncBtnLabel = syncLabel;
    syncLabel();
    sel.classList.add("wx-select-native");
    sel.tabIndex = -1;
    sel.setAttribute("aria-hidden", "true");
    sel.addEventListener("change", () => setTimeout(syncLabel, 0));
    btn.onclick = (e) => { e.stopPropagation(); openSelectPop(sel, btn); };
    btn.addEventListener("keydown", e => {
      if (activeSelectState && activeSelectState.btn === btn) {
        if (e.key === "ArrowDown") { e.preventDefault(); activeSelectState.setIdx(activeSelectState.getIdx() + 1); return; }
        if (e.key === "ArrowUp") { e.preventDefault(); activeSelectState.setIdx(activeSelectState.getIdx() - 1); return; }
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#selectPop .select-opt.is-focused")?.click(); return; }
      } else if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
        e.preventDefault(); openSelectPop(sel, btn);
      }
    });
    wrap.appendChild(btn);
  });
}
window.addEventListener("resize", closeSelectPop);

/* Toasts */
function toast(type, title, desc, action, ttl = 5000) {
  const stack = $("#toastStack"); const el = document.createElement("div"); el.className = "toast toast-" + type;
  const ic = { success: "check-circ", error: "alert-circ", info: "info", warning: "alert-tri" }[type] || "info";
  el.innerHTML = `${icon(ic, "icon t-icon")}<div class="t-main"><div class="t-title">${esc(title)}</div>${desc ? `<div class="t-desc">${esc(desc)}</div>` : ""}${action ? `<div class="t-action"><button class="btn btn-sm">${esc(action.label)}</button></div>` : ""}</div><button class="toast-close" aria-label="Закрыть">${icon("x")}</button>`;
  stack.appendChild(el); requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add("is-in")));
  const remove = () => { el.classList.remove("is-in"); setTimeout(() => el.remove(), 280); };
  el.querySelector(".toast-close").onclick = remove;
  if (action) el.querySelector(".t-action button").onclick = () => { action.fn(); remove(); };
  let timer = setTimeout(remove, ttl);
  el.addEventListener("mouseenter", () => clearTimeout(timer));
  el.addEventListener("mouseleave", () => { clearTimeout(timer); timer = setTimeout(remove, ttl); });
  return el;
}
/* Tooltip */
const tipEl = $("#tooltip"); let tipTarget = null;
document.addEventListener("mouseover", e => { const t = e.target.closest("[data-tip]"); if (!t || t === tipTarget) return; tipTarget = t; tipEl.innerHTML = t.dataset.tip;
  const r = t.getBoundingClientRect(); tipEl.classList.add("is-in"); const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
  let x = r.left + r.width / 2 - tw / 2, y = r.bottom + 8; x = Math.max(8, Math.min(x, innerWidth - tw - 8)); if (y + th > innerHeight - 8) y = r.top - th - 8; tipEl.style.left = x + "px"; tipEl.style.top = y + "px"; });
document.addEventListener("mouseout", e => { if (e.target.closest("[data-tip]") === tipTarget) { tipTarget = null; tipEl.classList.remove("is-in"); } });
const chartTip = $("#chartTip");
document.addEventListener("mouseover", e => { const t = e.target.closest("[data-chart]"); if (!t) return; chartTip.textContent = t.dataset.tip || ""; chartTip.classList.add("is-in"); });
document.addEventListener("mousemove", e => { if (!chartTip.classList.contains("is-in")) return; chartTip.style.left = Math.min(e.clientX + 14, innerWidth - chartTip.offsetWidth - 8) + "px"; chartTip.style.top = (e.clientY + 16) + "px"; });
document.addEventListener("mouseout", e => { if (e.target.closest("[data-chart]")) chartTip.classList.remove("is-in"); });
/* Ripple */
document.addEventListener("click", e => { const b = e.target.closest(".btn, .pill, .send-btn"); if (!b) return;
  const r = b.getBoundingClientRect(); const d = Math.max(r.width, r.height); const s = document.createElement("span"); s.className = "ripple";
  s.style.width = s.style.height = d + "px"; s.style.left = (e.clientX - r.left - d / 2) + "px"; s.style.top = (e.clientY - r.top - d / 2) + "px";
  b.appendChild(s); setTimeout(() => s.remove(), 560); });

/* Overlays */
const overlayTimers = {}; let overlayReturnFocus = null;
function openOverlay(id) { clearTimeout(overlayTimers[id]); overlayReturnFocus = document.activeElement; const o = $("#" + id); o.style.display = "flex";
  requestAnimationFrame(() => requestAnimationFrame(() => { o.classList.add("is-open");
    const danger = id === "dialog" && o.querySelector("[data-dlg='1'].btn-destructive");
    const f = o.querySelector(".cmdk-input") || (danger ? o.querySelector("[data-dlg='0']") : o.querySelector("[data-dlg='1'], button, input")); f?.focus(); })); }
function closeOverlay(id) { const o = $("#" + id); if (!o.classList.contains("is-open") && o.style.display === "none") return; o.classList.remove("is-open");
  if (id === "dialog" && dialogResolve) { dialogResolve(false); dialogResolve = null; }
  clearTimeout(overlayTimers[id]); overlayTimers[id] = setTimeout(() => { o.style.display = "none"; }, 280);
  if (overlayReturnFocus?.focus) { overlayReturnFocus.focus(); overlayReturnFocus = null; } }
$$(".overlay").forEach(o => { o.style.display = "none"; });
$$("[data-close]").forEach(b => b.addEventListener("click", () => closeOverlay(b.dataset.close)));
document.addEventListener("keydown", e => { if (e.key !== "Tab") return; const open = $$(".overlay.is-open").pop(); if (!open) return;
  const foc = $$("button, input, select, textarea, [href], [tabindex]:not([tabindex='-1'])", open).filter(x => !x.disabled && x.offsetParent !== null); if (!foc.length) return;
  const first = foc[0], last = foc[foc.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); } });
let dialogResolve = null;
function confirmDialog({ title, desc, confirmLabel = "Подтвердить", danger = false, bodyHTML = "" }) {
  $("#dialogTitle").textContent = title; $("#dialogDesc").textContent = desc; $("#dialogBody").innerHTML = bodyHTML;
  $("#dialogFooter").innerHTML = `<button class="btn btn-ghost" data-dlg="0">Отмена</button><button class="btn ${danger ? "btn-destructive" : "btn-primary"}" data-dlg="1">${esc(confirmLabel)}</button>`;
  $$("#dialogFooter [data-dlg]").forEach(b => b.onclick = () => { const r = dialogResolve; dialogResolve = null; closeOverlay("dialog"); r?.(b.dataset.dlg === "1"); });
  openOverlay("dialog"); return new Promise(res => dialogResolve = res);
}
/* Command palette */
const cmdkItems = [];
function buildCmdk() { cmdkItems.length = 0;
  VIEWS.forEach((v, i) => cmdkItems.push({ group: "Навигация", icon: VIEW_ICON[v], label: VIEW_LABEL[v], sub: String(i + 1), run: () => navigate(v) }));
  state.skills.forEach(s => cmdkItems.push({ group: "Навыки WSP", icon: (SKILL_META[s.skill_id] || {}).icon || "zap", label: s.pill_label, sub: "/" + s.skill_id, run: () => runSkill(s.skill_id) }));
  state.catalog.slice(0, 12).forEach(m => cmdkItems.push({ group: "Модели", icon: "cpu", label: m.model_id, sub: m.params_human, run: () => loadModel(m.model_id) }));
  STYLES.forEach(s => cmdkItems.push({ group: "Стили", icon: "palette", label: s.name, sub: s.desc, run: () => setStyle(s.id) }));
  BG_MODES.forEach(m => cmdkItems.push({ group: "Интерактивный фон", icon: "layers", label: "Фон: " + m.name, sub: m.desc, run: () => setBgMode(m.id) }));
  cmdkItems.push({ group: "Действия", icon: "download", label: "Экспорт .safetensors", sub: "Ctrl E", run: () => exportCheckpoint() });
  cmdkItems.push({ group: "Действия", icon: "panel-right", label: "Инспектор", sub: "Ctrl I", run: toggleInspector });
  cmdkItems.push({ group: "Действия", icon: "shield", label: "Тест отката Vespasian", sub: "/vespasian-guard", run: () => runSkill("vespasian-guard") });
}
let cmdkFiltered = [], cmdkActive = 0;
function renderCmdk() { const q = $("#cmdkInput").value.trim().toLowerCase();
  cmdkFiltered = cmdkItems.filter(it => !q || it.label.toLowerCase().includes(q) || it.sub.toLowerCase().includes(q) || it.group.toLowerCase().includes(q)); cmdkActive = 0;
  const res = $("#cmdkResults"); if (!cmdkFiltered.length) { res.innerHTML = `<div class="cmdk-empty">Ничего не найдено по запросу «${esc(q)}»</div>`; return; }
  let html = "", lastGroup = "";
  cmdkFiltered.forEach((it, i) => { if (it.group !== lastGroup) { html += `<div class="cmdk-group">${esc(it.group)}</div>`; lastGroup = it.group; }
    html += `<button class="cmdk-item ${i === cmdkActive ? "is-active" : ""}" data-idx="${i}">${icon(it.icon)}<span class="ci-label">${esc(it.label)}</span>${it.sub ? `<span class="ci-sub">${esc(it.sub)}</span>` : ""}</button>`; });
  res.innerHTML = html; $$(".cmdk-item", res).forEach(b => { b.onclick = () => execCmdk(+b.dataset.idx); b.onmousemove = () => { cmdkActive = +b.dataset.idx; highlightCmdk(); }; }); }
function highlightCmdk() { $$(".cmdk-item").forEach(b => b.classList.toggle("is-active", +b.dataset.idx === cmdkActive)); }
function execCmdk(i) { const it = cmdkFiltered[i]; if (!it) return; closeOverlay("cmdk"); it.run(); }
function openCmdk() { buildCmdk(); $("#cmdkInput").value = ""; renderCmdk(); openOverlay("cmdk"); setTimeout(() => $("#cmdkInput").focus(), 60); }
$("#cmdkInput").addEventListener("input", renderCmdk);
$("#cmdkInput").addEventListener("keydown", e => { if (e.key === "ArrowDown") { e.preventDefault(); cmdkActive = Math.min(cmdkActive + 1, cmdkFiltered.length - 1); highlightCmdk(); scrollCmdk(); } else if (e.key === "ArrowUp") { e.preventDefault(); cmdkActive = Math.max(cmdkActive - 1, 0); highlightCmdk(); scrollCmdk(); } else if (e.key === "Enter") { e.preventDefault(); execCmdk(cmdkActive); } });
function scrollCmdk() { $(`.cmdk-item[data-idx="${cmdkActive}"]`)?.scrollIntoView({ block: "nearest" }); }
$("#cmdkTrigger").onclick = openCmdk;
/* Inspector / sidebar / models toggles */
function toggleInspector() { const app = $("#app"); app.dataset.inspector = app.dataset.inspector === "open" ? "closed" : "open"; if (app.dataset.inspector === "open") renderInspector(); }
$("#inspectorToggle").onclick = toggleInspector; $("#inspectorClose").onclick = toggleInspector;
$("#modelsToggle").onclick = () => navigate("models");
$("#modelChip").onclick = () => navigate("models");
$("#sidebarCollapse").onclick = () => { const app = $("#app"); if (innerWidth <= 980) { app.dataset.sidebar = app.dataset.sidebar === "mobile-open" ? "collapsed" : "mobile-open"; $("#sideBackdrop").classList.toggle("is-in", app.dataset.sidebar === "mobile-open"); } else { const toExpanded = app.dataset.sidebar === "collapsed" || app.dataset.sidebar === "hidden"; app.dataset.sidebar = toExpanded ? "expanded" : "collapsed"; if (toExpanded) setSidebarHidden(false); } };
$("#mobileNav").onclick = () => { const app = $("#app"); const open = app.dataset.sidebar === "mobile-open"; app.dataset.sidebar = open ? "collapsed" : "mobile-open"; if (!open) app.dataset.inspector = "closed"; $("#sideBackdrop").classList.toggle("is-in", !open); };
$("#sideBackdrop").onclick = () => { $("#app").dataset.sidebar = "collapsed"; $("#sideBackdrop").classList.remove("is-in"); };
$$(".nav-item[data-view]").forEach(b => b.onclick = () => { navigate(b.dataset.view); if ($("#app").dataset.sidebar === "hidden") setPeek(false); });
matchMedia("(min-width: 981px)").addEventListener?.("change", e => { const app = $("#app"); if (e.matches) { $("#sideBackdrop").classList.remove("is-in"); if (app.dataset.sidebar === "mobile-open") app.dataset.sidebar = "expanded"; } else if (app.dataset.sidebar === "hidden") { app.classList.remove("is-peek"); app.dataset.sidebar = "collapsed"; } });
/* Sidebar hidden mode with hover peek */
let peekTimer = null;
function setPeek(on) { const app = $("#app"); if (app.dataset.sidebar !== "hidden") return; clearTimeout(peekTimer);
  const g = $("#sideEdgeGrip");
  if (on) { app.classList.add("is-peek"); if (g) g.dataset.tip = "Закрепить сайдбар"; }
  else peekTimer = setTimeout(() => { app.classList.remove("is-peek"); if (g) g.dataset.tip = "Показать сайдбар"; }, 180); }
function setSidebarHidden(hidden, persist = true) { const app = $("#app"); if (innerWidth <= 980) return;
  clearTimeout(peekTimer); app.classList.remove("is-peek"); app.dataset.sidebar = hidden ? "hidden" : "expanded";
  if (persist) store.set("sidebarHidden", hidden);
  const lbl = $("#sidebarCollapse .nav-label"); if (lbl) lbl.textContent = hidden ? "Закрепить" : "Свернуть";
  $("#sidebarCollapse").dataset.tip = hidden ? "Закрепить сайдбар на месте" : "Свернуть <span class='tt-kbd'>Ctrl B</span>";
  const sw = $("#sidebarAutoHide"); if (sw) { sw.classList.toggle("is-on", hidden); sw.setAttribute("aria-checked", String(hidden)); } }
$("#sidebarHide").onclick = () => { if (innerWidth <= 980) { $("#app").dataset.sidebar = "collapsed"; $("#sideBackdrop").classList.remove("is-in"); } else setSidebarHidden(true); };
$("#sideEdgeGrip").onclick = (e) => { e.stopPropagation(); setSidebarHidden(false); };
$("#sideEdge").addEventListener("click", () => setSidebarHidden(false));
$("#sideEdge").addEventListener("mouseenter", () => setPeek(true));
$("#sideEdge").addEventListener("mouseleave", () => setPeek(false));
document.querySelector(".sidebar").addEventListener("mouseenter", () => setPeek(true));
document.querySelector(".sidebar").addEventListener("mouseleave", () => setPeek(false));
/* Keyboard */
document.addEventListener("keydown", e => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "");
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openCmdk(); return; }
  if (e.key === "Escape") { closeOverlay("cmdk"); closeOverlay("dialog"); closeStylesPop(); closeBrainPop(); closeSelectPop(); closeSlash(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") { e.preventDefault(); $("#sidebarCollapse").click(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "i") { e.preventDefault(); toggleInspector(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "e") { e.preventDefault(); exportCheckpoint(); return; }
  if (!typing && !e.ctrlKey && !e.metaKey && !e.altKey && /^[1-9]$/.test(e.key)) navigate(VIEWS[+e.key - 1]);
});

/* ============================================================ SESSION FEED */
function scrollThread() { const t = $("#thread"); if (t) t.scrollTop = t.scrollHeight; }
function addUserTurn(text, label) { const inner = $("#threadInner"); if (!inner) return; $("#sessionEmpty")?.remove(); inner.classList.remove("is-empty");
  const el = document.createElement("div"); el.className = "turn user";
  el.innerHTML = `<div class="turn-head"><span class="avatar user">${icon("terminal")}</span><span class="turn-name">${esc(label || "Вы")}</span><span class="turn-time">${nowTime()}</span></div><div class="turn-body">${esc(text)}</div>`;
  inner.appendChild(el); syncBgDim(); scrollThread(); }
function addAssistantTurn(title, skillId) { const inner = $("#threadInner"); if (!inner) return; $("#sessionEmpty")?.remove(); inner.classList.remove("is-empty");
  const el = document.createElement("div"); el.className = "turn";
  el.innerHTML = `<div class="turn-head"><span class="avatar">${icon("sparkles")}</span><span class="turn-name">WeightX Brain</span><span class="badge" data-tip="Модель Brain · выбор в composer">${icon("brain")} ${esc(brainLabel(true))}</span>${skillId ? `<span class="badge badge-accent">/${esc(skillId)}</span>` : ""}<span class="turn-time">${nowTime()}</span></div><div class="turn-body"><strong>${esc(title)}</strong></div><div class="turn-feed" style="margin-left:34px"></div>`;
  el._events = []; el._open = new Set();
  inner.appendChild(el); syncBgDim(); scrollThread(); return el; }
/* ---- Agent Activity Feed ---- */
function statusIcon(st) { return st === "running" ? icon("loader", "icon spin") : st === "error" ? icon("x") : st === "warn" ? icon("alert-tri") : icon("check"); }
function metricChipsHTML(metrics) { const entries = Object.entries(metrics || {}); if (!entries.length) return ""; return `<div class="metric-chips">${entries.map(([k, v]) => `<span class="metric-chip"><span class="mc-k">${esc(k)}</span><span class="mc-v">${esc(v)}</span></span>`).join("")}</div>`; }
function formatResultOutput(text) {
  const s = String(text || "").trim();
  if (!s) return "";
  const hasMd = /(^|\n)#{1,6}\s|\*\*[^*]+\*\*|```|^[-*]\s|\|.+\|/m.test(s);
  if (hasMd) return `<div class="feed-out feed-md">${mdToHtml(s)}</div>`;
  return `<div class="feed-out mono">${esc(s)}</div>`;
}
function feedBodyHTML(e) {
  if (e.kind === "result") {
    const out = String(e.output || "");
    return `<div class="feed-result-wrap ${e.status === "error" ? "is-error" : ""}">${metricChipsHTML(e.metrics)}${formatResultOutput(out)}${out ? `<div class="feed-result-actions"><button class="btn btn-sm btn-ghost" data-feed-copy data-copy="${esc(out)}">${icon("copy")} Копировать отчёт</button></div>` : ""}</div>`;
  }
  if (e.kind === "yaml") return `<pre class="feed-code">${esc(e.output || "")}</pre><button class="btn btn-sm" data-open-yaml data-yaml="${esc(e.output || "")}">${icon("file-code")} Открыть в редакторе</button>`;
  return `<div class="feed-out mono">${esc(e.output || "")}</div>`;
}
const FEED_RENDER = {
  tools(ev, open) { return ev.map(e => { const isOpen = open.has(e.id);
    return `<div class="feed-row ${e.status} ${isOpen ? "is-open" : ""}">
      <button class="feed-row-head" data-feed-toggle="${e.id}" aria-expanded="${isOpen}">
        <span class="feed-ic ${e.status}">${statusIcon(e.status)}</span>
        <span class="feed-name">${esc(e.name)}</span>
        ${e.args ? `<span class="feed-args mono">${esc(e.args)}</span>` : ""}
        <span class="feed-dur mono">${e.dur || (e.status === "running" ? "…" : "")}</span>
        <span class="feed-chev">${icon("chev-down")}</span>
      </button>
      ${isOpen ? `<div class="feed-row-body">${e.args ? `<div class="feed-kv mono">args · ${esc(e.args)}</div>` : ""}${feedBodyHTML(e)}</div>` : ""}
    </div>`; }).join(""); },
  timeline(ev, open) { return `<div class="feed-tl">` + ev.map(e => `<div class="feed-tl-item ${e.status}">
      <span class="feed-tl-dot">${statusIcon(e.status)}</span>
      <div class="feed-tl-main">
        <button class="feed-tl-head" data-feed-toggle="${e.id}" aria-expanded="${open.has(e.id)}"><span class="feed-tl-time mono">${e.ts}</span><span class="feed-name">${esc(e.name)}</span>${e.dur ? `<span class="feed-dur mono">${e.dur}</span>` : ""}</button>
        ${open.has(e.id) ? `<div class="feed-tl-body">${feedBodyHTML(e)}</div>` : ""}
      </div></div>`).join("") + `</div>`; },
  console(ev) { const lines = ev.map(e => { const tag = e.kind === "result" ? "RESULT" : e.kind === "yaml" ? "YAML" : e.kind === "plan" ? "PLAN" : "TOOL";
      const lvl = e.status === "error" ? "ERR" : e.status === "warn" ? "WARN" : e.status === "running" ? "RUN" : "OK";
      let l = `<div class="cl"><span class="cl-ts">${e.ts}</span><span class="cl-lvl ${e.status}">${lvl}</span><span class="cl-tag">${tag}</span><span class="cl-name">${esc(e.name)}</span>${e.dur ? `<span class="cl-dur">${e.dur}</span>` : ""}${e.status === "running" ? '<span class="stream-caret"></span>' : ""}</div>`;
      if (e.kind === "yaml") l += `<pre class="cl-code">${esc(e.output || "")}</pre>`;
      else if (e.kind === "result") l += `<div class="cl-out">${esc(Object.entries(e.metrics || {}).map(([k, v]) => k + "=" + v).join("  "))}</div><div class="cl-out">${esc(e.output || "")}</div>`;
      else if (e.output) l += `<div class="cl-out">${esc(e.output)}</div>`;
      return l; }).join("");
    return `<div class="feed-console"><div class="feed-console-head"><span class="mono">agent://weightspace</span><span style="flex:1"></span><button class="btn btn-sm btn-ghost" data-feed-copy data-copy="${esc(ev.map(e => "[" + e.ts + "] " + e.name + (e.output ? " · " + e.output : "")).join("\n"))}">${icon("copy")} Копировать</button></div><div class="feed-console-body">${lines}</div></div>`; },
  steps(ev, open) { const tools = ev.filter(e => e.kind === "tool"); const names = tools.map(t => t.name);
    const done = tools.filter(t => t.status !== "running").length; const fail = tools.some(t => t.status === "error");
    const step = `<ol class="stepper" style="flex-wrap:wrap">${names.map((n, i) => { let cls = "", dot = i + 1;
      if (fail && i === tools.findIndex(t => t.status === "error")) { cls = "is-fail"; dot = icon("x"); } else if (i < done) { cls = "is-done"; dot = icon("check"); } else if (i === done) { cls = "is-active"; dot = icon("loader", "icon spin"); }
      return `<li class="stepper-step ${cls}"><span class="stepper-dot">${dot}</span><span class="stepper-label">${esc(n)}</span></li>${i < names.length - 1 ? '<li class="stepper-sep"></li>' : ""}`; }).join("")}</ol>`;
    const cards = tools.map((t, i) => `<div class="tool-card"><div class="tool-head"><span class="dot ${t.status === "error" ? "dot-error" : t.status === "running" ? "dot-info is-pulse" : "dot-success"}"></span><span class="tool-name">[${i + 1}/${tools.length}] ${esc(t.name)}</span><span class="tool-dur">${t.dur || "…"}</span></div><div class="tool-body">${t.status === "running" ? 'выполняется<span class="stream-caret"></span>' : esc(t.output || "")}</div></div>`).join("");
    const res = ev.find(e => e.kind === "result"); const y = ev.find(e => e.kind === "yaml");
    return step + `<div style="display:flex;flex-direction:column;gap:8px;margin-top:10px">${cards}</div>` + (res ? `<div style="margin-top:10px" class="feed-result-wrap ${res.status === "error" ? "is-error" : ""}">${metricChipsHTML(res.metrics)}${formatResultOutput(res.output || "")}</div>` : "") + (y ? `<div class="tool-card" style="margin-top:10px"><div class="tool-head"><span class="dot dot-accent"></span><span class="tool-name">${esc(y.name)}</span><span class="tool-dur">YAML</span></div><div class="tool-body">${esc(y.output)}</div><div style="padding:8px 12px;border-top:1px solid var(--border-subtle)"><button class="btn btn-sm" data-open-yaml data-yaml="${esc(y.output)}">${icon("file-code")} Открыть в редакторе</button></div></div>` : ""); }
};
function paintFeed(el) { const feed = el.querySelector(".turn-feed"); if (!feed || !el._events) return;
  const v = FEED_RENDER[state.feedVariant] ? state.feedVariant : "timeline";
  feed.className = "turn-feed feed-" + v;
  feed.innerHTML = FEED_RENDER[v](el._events, el._open);
  $$("[data-feed-toggle]", feed).forEach(b => b.onclick = () => { const id = b.dataset.feedToggle; el._open.has(id) ? el._open.delete(id) : el._open.add(id); paintFeed(el); });
  $$("[data-open-yaml]", feed).forEach(b => b.onclick = () => { navigate("yaml"); setYamlEditor(b.dataset.yaml); });
  $$("[data-feed-copy]", feed).forEach(b => b.onclick = () => { try { navigator.clipboard?.writeText(b.dataset.copy); } catch {} toast("success", "Скопировано", null, null, 1400); });
  bindMdLinks(feed);
  if (v === "console") { const c = feed.querySelector(".feed-console-body"); if (c) c.scrollTop = c.scrollHeight; }
}
function repaintAllFeeds() { $$("#threadInner .turn").forEach(el => { if (el._events) paintFeed(el); }); }
function pushEvent(turn, e) { e.id = "e" + (turn._events.length + 1) + "-" + Math.random().toString(36).slice(2, 6); e.ts = nowTime(); turn._events.push(e); if (e.kind === "result") turn._open.add(e.id); paintFeed(turn); scrollThread(); return e; }

/* ============================================================ RENDER · Models */
/* Real catalog entries → unified model cards. */
function entryCard(e) { return { id: e.model_id, repo: e.hf_repo_id, params: e.params_human || "—", layers: e.num_layers || 0, vram: e.vram_fp32_mb || 0, license: e.license_spdx || "—", desc: e.description || "", family: e.family || "", dl: !!e.local_cached, builtin: !!e.is_builtin_synthetic, local: e.model_id.startsWith("local:"), thumbnail: "", tasks: e.recommended_tasks || [] }; }
function hfCard(h) { return { id: h.id, repo: h.id, params: h.params ? humanParams(h.params) : "—", paramsN: h.params || 0, layers: h.layers || 0, vram: h.vram || 0, license: h.license || "—", desc: h.desc || "Модель Hugging Face", family: h.family || "—", downloads: h.downloads, likes: h.likes, lastModified: h.last_modified || "", gated: !!h.gated, thumbnail: h.thumbnail || "", live: true }; }
function humanParams(n) { n = Number(n) || 0; if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + "B"; if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + "M"; if (n >= 1e3) return (n / 1e3).toFixed(0) + "K"; return String(n); }
function orgOf(id) { return String(id).includes("/") ? String(id).split("/")[0] : ""; }
function hfStaticCards() {
  const fromCatalog = state.catalog.filter(e => !e.model_id.startsWith("local:") && !e.is_builtin_synthetic).map(entryCard);
  const known = new Set(fromCatalog.map(c => c.repo.toLowerCase()));
  const extra = HF_FALLBACK.filter(h => !known.has(h.id.toLowerCase())).map(hfCard);
  return [...fromCatalog, ...extra];
}
const HF_TASKS = [["", "Автоподбор · LLM"], ["text-generation", "Text Generation"], ["text-classification", "Text Classification"], ["token-classification", "Token Classification"], ["sentence-similarity", "Sentence Similarity"], ["image-text-to-text", "Image-Text-to-Text"], ["text-to-image", "Text-to-Image"], ["automatic-speech-recognition", "Speech Recognition"], ["feature-extraction", "Feature Extraction"], ["any-to-any", "Any-to-Any"]];
const HF_LICENSES = [["", "Любая лицензия"], ["apache-2.0", "Apache 2.0"], ["mit", "MIT"], ["bsd-3-clause", "BSD 3-Clause"], ["cc-by-nc-4.0", "CC BY-NC 4.0"], ["llama3.2", "Llama 3.2"], ["llama3.1", "Llama 3.1"], ["gemma", "Gemma"], ["openrail", "OpenRAIL"], ["other", "Other"]];
const HF_SORTS = [["downloads", "По загрузкам"], ["likes", "По лайкам"], ["lastModified", "По дате обновления"], ["trendingScore", "В тренде"]];
function sysResourcesLabel() {
  const r = state.sysResources;
  if (!r) return "";
  const ramGb = r.ram_total ? (r.ram_total / 1073741824).toFixed(0) : "?";
  const diskGb = r.disk_free ? (r.disk_free / 1073741824).toFixed(0) : "?";
  const cap = state.maxParams > 0 ? ` · ≤ ${humanParams(state.maxParams)}` : "";
  return `<span class="sys-res-chip" data-tip="Ресурсы ПК для автоподбора моделей">${icon("cpu")} RAM ${ramGb} GB · Диск ${diskGb} GB${cap}</span>`;
}
function renderModelsView() {
  const pad = $("#modelsPad"); if (!pad) return;
  pad.innerHTML = `
    <div class="view-head"><div class="view-head-text"><h2 class="section-title">Модели</h2><span class="section-sub">Каталог Hugging Face и локальные модели для WSP-пайплайнов</span></div>${sysResourcesLabel()}</div>
    <div class="models-toolbar">
      <div class="segmented" id="modelSource">
        <button class="segmented-item ${state.modelSource === "hf" ? "is-active" : ""}" data-source="hf"><svg class="icon"><use href="#i-cloud"/></svg> Hugging Face</button>
        <button class="segmented-item ${state.modelSource === "local" ? "is-active" : ""}" data-source="local"><svg class="icon"><use href="#i-hard-drive"/></svg> Скаченные</button>
      </div>
      <div class="models-search"><svg class="icon"><use href="#i-search"/></svg><input class="input" id="modelSearch" placeholder="${state.modelSource === "local" ? "Поиск скаченных…" : "Поиск моделей HF…"}" value="${esc(state.modelSearch)}" autocomplete="off" /></div>
      <button class="btn btn-sm btn-ghost" id="modelRefresh" data-tip="Обновить список из реального источника">${icon("refresh")} Обновить</button>
      <span style="flex:1"></span>
      <span class="section-sub" id="modelCount"></span>
    </div>
    <div class="models-filters" id="hfFilters" style="${state.modelSource === "hf" ? "" : "display:none"}">
      <span class="mf-label">${icon("filter")} Фильтры</span>
      <select class="select" id="hfTask" aria-label="Задача" data-tip="Автоподбор: генерация текста + размер, который поместится в RAM и на диске вашего ПК">${HF_TASKS.map(([v, l]) => `<option value="${v}" ${state.hfTask === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      <select class="select" id="hfLicense" aria-label="Лицензия">${HF_LICENSES.map(([v, l]) => `<option value="${v}" ${state.hfLicense === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      <select class="select" id="hfSort" aria-label="Сортировка">${HF_SORTS.map(([v, l]) => `<option value="${v}" ${state.hfSort === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      <button class="btn btn-sm btn-ghost" id="hfReset">${icon("x")} Сбросить</button>
    </div>
    <div class="models-toolbar" style="margin-top:-6px">
      <div class="models-search" style="flex:1;max-width:520px"><svg class="icon"><use href="#i-folder"/></svg><input class="input" id="modelCustom" placeholder="HF repo (org/name) или путь к .safetensors" autocomplete="off" /></div>
      <button class="btn btn-sm" id="modelCustomLoad">${icon("download")} Загрузить</button>
    </div>
    <div class="models-grid" id="modelList"></div>
    <div class="load-more-wrap" id="modelMore"></div>
    <div class="mp-foot" id="modelFoot"><span class="dot dot-info"></span><span id="modelFootText">Каталог Hugging Face</span></div>`;
  $$("#modelSource [data-source]").forEach(b => b.onclick = () => {
    state.modelSource = b.dataset.source;
    $$("#modelSource [data-source]").forEach(x => x.classList.toggle("is-active", x === b));
    $("#modelSearch").placeholder = state.modelSource === "local" ? "Поиск скаченных…" : "Поиск моделей HF…";
    $("#hfFilters").style.display = state.modelSource === "hf" ? "" : "none";
    paintModelList();
    if (state.modelSource === "hf") doHFSearch();
    else refreshCatalog().then(paintModelList);
  });
  $("#modelRefresh").onclick = () => {
    if (state.modelSource === "hf") doHFSearch();
    else refreshCatalog().then(() => { paintModelList(); toast("info", "Кэш проверен", "Скаченные модели перечитаны с диска", null, 1800); });
  };
  $("#modelSearch").addEventListener("input", e => { state.modelSearch = e.target.value; paintModelList();
    if (state.modelSource === "hf") { clearTimeout(hfSearchTimer); hfSearchTimer = setTimeout(doHFSearch, 500); } });
  const onFilter = () => { state.hfTask = $("#hfTask").value; state.hfLicense = $("#hfLicense").value; state.hfSort = $("#hfSort").value; doHFSearch(); };
  $("#hfTask").onchange = onFilter; $("#hfLicense").onchange = onFilter; $("#hfSort").onchange = onFilter;
  $("#hfReset").onclick = () => { state.hfTask = ""; state.hfLicense = ""; state.hfSort = "downloads"; state.modelSearch = ""; $("#modelSearch").value = ""; renderModelsView(); };
  $("#modelCustomLoad").onclick = () => { const v = $("#modelCustom").value.trim(); if (!v) { toast("warning", "Укажите HF repo или путь", "Например: HuggingFaceTB/SmolLM2-135M"); return; } loadModel(v); };
  $("#modelCustom").addEventListener("keydown", e => { if (e.key === "Enter") $("#modelCustomLoad").click(); });
  // Infinite scroll: load the next HF page when nearing the bottom.
  const scroller = pad.closest(".view-scroll");
  if (scroller && !scroller._wxModelsScroll) {
    scroller._wxModelsScroll = true;
    scroller.addEventListener("scroll", () => {
      closeSelectPop();
      if (state.view !== "models" || state.modelSource !== "hf") return;
      if (scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 420) loadMoreHF();
    });
  }
  enhanceSelects(pad);
  paintModelList();
  // Connect to the real Hugging Face API immediately when the catalog opens.
  if (state.modelSource === "hf") doHFSearch();
  else refreshCatalog().then(paintModelList);
}
function modelThumbHTML(m) {
  const url = m.thumbnail || state.avatarCache[orgOf(m.id)] || "";
  if (url) return `<img class="mr-thumb" src="${esc(url)}" alt="" loading="lazy" onerror="this.style.visibility='hidden'" />`;
  return `<span class="mr-thumb-ph" data-thumb-for="${esc(m.id)}">${icon("box")}</span>`;
}
function paintModelList() {
  const list = $("#modelList"); if (!list) return;
  const q = state.modelSearch.trim().toLowerCase();
  let models;
  if (state.modelSource === "local") {
    models = localEntries().map(entryCard);
    $("#modelFootText").textContent = `${models.length} на компьютере · кэш HF + локальные .safetensors (проверено на диске)`;
    $("#modelFoot").querySelector(".dot").className = "dot dot-success";
  } else {
    models = state.hfResults.length ? state.hfResults : hfStaticCards();
    const autoHint = !state.hfTask && state.maxParams > 0 ? ` · LLM под ваш ПК (≤ ${humanParams(state.maxParams)})` : "";
    $("#modelFootText").textContent = state.hfLoading ? "Подключение к Hugging Face…" : state.hfLive ? (state.modelSearch.trim() ? "Hugging Face · онлайн · результаты поиска" : "Hugging Face · онлайн · каталог" + autoHint) : "Офлайн · встроенный каталог HF repos (поиск недоступен)";
    $("#modelFoot").querySelector(".dot").className = "dot " + (state.hfLoading ? "dot-info is-pulse" : state.hfLive ? "dot-success" : "dot-warning");
  }
  // Local search filters client-side; HF search/filter/sort happen in the real API.
  if (q && state.modelSource === "local") models = models.filter(m => m.id.toLowerCase().includes(q) || (m.family || "").toLowerCase().includes(q) || (m.repo || "").toLowerCase().includes(q));
  const count = $("#modelCount"); if (count) count.textContent = `${models.length} моделей`;
  renderLoadMore();
  if (!models.length) { list.innerHTML = `<div class="mp-empty">${state.modelSource === "local" ? "Нет скаченных моделей.<br>Загрузите модель из каталога Hugging Face." : state.hfLoading ? "Загрузка…" : "Ничего не найдено."}</div>`; return; }
  list.innerHTML = models.map(m => { const loaded = state.model && (state.model.id === m.id || state.model.id === m.repo); const dl = m.dl || isDownloaded(m.id) || isDownloaded(m.repo);
    const dlText = m.downloads ? (typeof m.downloads === "number" ? fmtInt(m.downloads) : m.downloads) : "";
    const likesText = m.likes ? fmtInt(m.likes) : "";
    const fam = m.family && m.family !== "—" ? `<span class="badge">${esc(m.family)}</span>` : "";
    return `<div class="model-row ${loaded ? "is-loaded" : ""}" data-model-row="${esc(m.id)}" data-tip="Открыть описание модели">
      <div class="mr-top">${modelThumbHTML(m)}<span class="mr-name">${esc(m.id)}</span>${loaded ? '<span class="badge badge-success">активна</span>' : dl ? '<span class="badge badge-accent">скачена</span>' : ""}${m.gated ? '<span class="badge badge-warning">gated</span>' : ""}</div>
      <div class="mr-meta"><span><b>${esc(m.params)}</b></span>${m.layers ? `<span>${m.layers}L</span>` : ""}${m.vram ? `<span>${m.vram} MB</span>` : ""}${dlText ? `<span>${icon("download", "icon")} ${dlText}</span>` : ""}${likesText ? `<span>${icon("heart", "icon")} ${likesText}</span>` : ""}<span>${esc(m.license || "")}</span></div>
      <div class="mr-desc">${esc(m.desc || "")}</div>
      <div class="mr-actions"><button class="btn btn-sm ${loaded ? "" : "btn-primary"}" data-load-model="${esc(m.id)}">${icon(dl ? "play" : "download")} ${loaded ? "Перезагрузить" : dl ? "Загрузить" : "Скачать"}</button><button class="btn btn-sm btn-ghost" data-open-card="${esc(m.id)}">${icon("eye")} Описание</button>${fam}</div>
    </div>`; }).join("");
  $$("[data-load-model]", list).forEach(b => b.onclick = (e) => { e.stopPropagation(); loadModel(b.dataset.loadModel); });
  $$("[data-open-card]", list).forEach(b => b.onclick = (e) => { e.stopPropagation(); const m = models.find(x => x.id === b.dataset.openCard); if (m) openModelCard(m); });
  $$(".model-row", list).forEach(row => row.onclick = () => { const m = models.find(x => x.id === row.dataset.modelRow); if (m) openModelCard(m); });
  lazyLoadAvatars(models);
}
/* Org avatars for cards without a card thumbnail (real HF API, cached). */
async function lazyLoadAvatars(models) {
  if (!hasTauri()) return;
  const orgs = [...new Set(models.map(m => orgOf(m.id)).filter(o => o && !state.avatarCache[o] && !models.some(m => orgOf(m.id) === o && m.thumbnail)))];
  for (const org of orgs.slice(0, 24)) {
    try {
      const url = await invoke("hf_avatar", { org });
      state.avatarCache[org] = url || "";
      if (url) $$(`[data-thumb-for]`, $("#modelList") || document).forEach(ph => {
        const row = ph.closest("[data-model-row]");
        if (row && orgOf(row.dataset.modelRow) === org) ph.outerHTML = `<img class="mr-thumb" src="${esc(url)}" alt="" loading="lazy" onerror="this.style.visibility='hidden'" />`;
      });
    } catch { state.avatarCache[org] = ""; }
  }
}
function renderLoadMore() {
  const box = $("#modelMore"); if (!box) return;
  if (state.modelSource !== "hf" || !state.hfLive) { box.innerHTML = ""; return; }
  if (state.hfLoadingMore) box.innerHTML = `<button class="btn" disabled>${icon("loader", "icon spin")} Загрузка…</button>`;
  else if (state.hfCursor) box.innerHTML = `<button class="btn" id="loadMoreBtn">${icon("chev-down")} Показать ещё</button><span class="section-sub">загружено ${state.hfResults.length}</span>`;
  else box.innerHTML = `<span class="section-sub">Все результаты загружены · ${state.hfResults.length}</span>`;
  const b = $("#loadMoreBtn"); if (b) b.onclick = loadMoreHF;
}
function renderModels() { if (!$("#modelList")) renderModelsView(); else paintModelList(); }
let hfSearchTimer = null;
/* Live HF browse/search: native Rust command (real filters, sort, cursor).
   Empty task = invisible base filter: text-generation LLMs that fit this PC
   (RAM/disk budget from system_resources, enforced in the native command). */
async function fetchHFSearch(query, cursor) {
  const res = await invoke("hf_search", { query, limit: 18, cursor: cursor || "", task: state.hfTask || "text-generation", license: state.hfLicense, sort: state.hfSort, maxParams: state.maxParams });
  return { cards: (res.models || []).map(h => hfCard(h)), nextCursor: res.next_cursor || "" };
}
async function doHFSearch() {
  const q = state.modelSearch.trim(); const reqId = ++state.hfReqId;
  state.hfLoading = true; state.hfCursor = "";
  const foot = $("#modelFootText"); if (foot) foot.textContent = q ? "Поиск на Hugging Face…" : "Загрузка каталога Hugging Face…";
  renderLoadMore();
  let live = null;
  try { live = await fetchHFSearch(q, ""); } catch { live = null; }
  if (reqId !== state.hfReqId) return; // stale response
  state.hfLoading = false;
  if (live && live.cards.length) { state.hfLive = true; state.hfResults = live.cards; state.hfCursor = live.nextCursor; }
  else { state.hfLive = false; state.hfResults = []; state.hfCursor = ""; }
  renderModels();
}
/* Next page via the real HF API cursor (button + infinite scroll). */
async function loadMoreHF() {
  if (state.modelSource !== "hf" || !state.hfLive || !state.hfCursor || state.hfLoadingMore || state.hfLoading) return;
  state.hfLoadingMore = true; renderLoadMore();
  const reqId = state.hfReqId;
  try {
    const page = await fetchHFSearch(state.modelSearch.trim(), state.hfCursor);
    if (reqId !== state.hfReqId) return; // filters changed meanwhile
    const known = new Set(state.hfResults.map(m => m.id));
    page.cards.forEach(c => { if (!known.has(c.id)) state.hfResults.push(c); });
    state.hfCursor = page.nextCursor;
  } catch (e) { toast("error", "Не удалось загрузить ещё", String(e), null, 3000); }
  finally { state.hfLoadingMore = false; if (state.view === "models") paintModelList(); }
}

/* ---- Model description dialog: real HF model card + README ---- */
function isHfRepo(m) { return !m.local && !m.builtin && String(m.repo || m.id).includes("/") && !String(m.repo).endsWith(".safetensors") && !String(m.repo).includes("\\"); }
function mcChipsHTML(pairs) { return `<div class="mc-chips">${pairs.filter(([, v]) => v !== "" && v != null).map(([k, v]) => `<span class="metric-chip"><span class="mc-k">${esc(k)}</span><span class="mc-v">${esc(v)}</span></span>`).join("")}</div>`; }
async function openModelCard(m) {
  const hf = isHfRepo(m);
  const img = m.thumbnail || state.avatarCache[orgOf(m.id)] || "";
  const chips = mcChipsHTML([
    ["параметры", m.params], ["слоёв", m.layers || ""], ["VRAM f32", m.vram ? m.vram + " MB" : ""],
    ["лицензия", m.license], ["загрузки", m.downloads ? (typeof m.downloads === "number" ? fmtInt(m.downloads) : m.downloads) : ""],
    ["лайки", m.likes ? fmtInt(m.likes) : ""], ["обновлена", m.lastModified ? String(m.lastModified).slice(0, 10) : ""], ["доступ", m.gated ? "gated (нужен запрос)" : "открытая"]
  ]);
  const body = `
    <div class="mc-head">${img ? `<img class="mc-hero" src="${esc(img)}" alt="" onerror="this.style.display='none'" />` : ""}<div style="flex:1;min-width:0">${chips}</div></div>
    ${hf ? `<div style="margin-bottom:10px;display:flex;gap:8px"><button class="btn btn-sm" data-open-href="https://huggingface.co/${esc(m.repo)}">${icon("cloud")} Открыть на Hugging Face</button></div>` : ""}
    <div class="md-body" id="mcReadme"><div class="empty" style="padding:24px"><div class="empty-icon">${icon("loader", "icon spin")}</div><div class="empty-title">${hf ? "Загрузка описания с Hugging Face…" : "Локальная модель"}</div></div></div>`;
  const p = confirmDialog({ title: m.id, desc: (m.family || "") + (hf ? " · карточка модели Hugging Face" : " · локальная модель"), confirmLabel: "Загрузить модель", bodyHTML: body });
  $$("#dialogBody [data-open-href]").forEach(b => b.onclick = () => invoke("open_external", { url: b.dataset.openHref }).catch(e => toast("error", "Не удалось открыть браузер", String(e))));
  const box = $("#mcReadme");
  if (hf) {
    try {
      const card = await invoke("hf_model_card", { repo: m.repo });
      const cm = card.model || {};
      const hero = $(".mc-hero");
      const newImg = cm.thumbnail || card.avatar || "";
      if (!hero && newImg) { $(".mc-head").insertAdjacentHTML("afterbegin", `<img class="mc-hero" src="${esc(newImg)}" alt="" onerror="this.style.display='none'" />`); }
      const tags = (cm.tags || []).slice(0, 24);
      box.innerHTML = `${tags.length ? `<div class="mc-tags">${tags.map(t => `<span class="badge">${esc(t)}</span>`).join("")}</div>` : ""}${card.readme ? mdToHtml(card.readme) : `<div class="field-hint">У модели нет README.md на Hugging Face.</div>`}`;
      bindMdLinks(box);
    } catch (e) {
      box.innerHTML = `<div class="alert alert-error">${icon("alert-circ")}<div><div class="alert-title">Описание недоступно</div><div class="alert-desc">${esc(String(e))}</div></div></div><div class="field-hint" style="margin-top:8px">${esc(m.desc || "")}</div>`;
    }
  } else {
    const tasks = (m.tasks || []).map(t => `<span class="badge badge-accent">${esc(t)}</span>`).join(" ");
    box.innerHTML = `<div class="md-body-inner"><p>${esc(m.desc || "Локальная модель WeightX.")}</p>${tasks ? `<p><b>Рекомендуемые задачи:</b></p><div class="mc-tags">${tasks}</div>` : ""}<p class="field-hint">Источник: ${esc(m.repo || m.id)}</p></div>`;
  }
  const ok = await p;
  if (ok) loadModel(m.id);
}
function bindMdLinks(root) { $$("a[data-open-href]", root).forEach(a => a.onclick = e => { e.preventDefault(); invoke("open_external", { url: a.dataset.openHref }).catch(err => toast("error", "Не удалось открыть браузер", String(err))); }); }
/* Compact markdown renderer for HF model cards (escaped, safe). */
function mdToHtml(src) {
  let text = String(src).replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, ""); // strip YAML frontmatter
  const escd = esc(text);
  const blocks = [];
  // fenced code blocks → placeholders
  let s = escd.replace(/```[^\n]*\n([\s\S]*?)```/g, (_, code) => { blocks.push(`<pre class="md-code">${code.replace(/\n$/, "")}</pre>`); return `\u0000B${blocks.length - 1}\u0000`; });
  const inline = (line) => {
    let out = line;
    const codes = [];
    out = out.replace(/`([^`]+)`/g, (_, c) => { codes.push(`<code>${c}</code>`); return `\u0000C${codes.length - 1}\u0000`; });
    out = out.replace(/!\[([^\]]*)\]\((https?:[^)\s]+)\)/g, '<img class="md-img" src="$2" alt="$1" loading="lazy" />');
    out = out.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="#" data-open-href="$2">$1</a>');
    out = out.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<i>$2</i>");
    out = out.replace(/\u0000C(\d+)\u0000/g, (_, i) => codes[+i]);
    return out;
  };
  const lines = s.split(/\r?\n/);
  let html = "", i = 0, para = [];
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  while (i < lines.length) {
    const ln = lines[i];
    const bm = ln.match(/^\u0000B(\d+)\u0000$/);
    if (bm) { flushPara(); html += blocks[+bm[1]]; i++; continue; }
    if (!ln.trim()) { flushPara(); i++; continue; }
    const hm = ln.match(/^(#{1,6})\s+(.*)$/);
    if (hm) { flushPara(); const lvl = Math.min(hm[1].length + 1, 6); html += `<h${lvl} class="md-h">${inline(hm[2])}</h${lvl}>`; i++; continue; }
    if (/^(\*{3,}|-{3,}|_{3,})$/.test(ln.trim())) { flushPara(); html += "<hr class='md-hr'/>"; i++; continue; }
    if (/^\|.*\|$/.test(ln.trim()) && i + 1 < lines.length && /^\|[\s:|-]+\|$/.test(lines[i + 1].trim())) {
      flushPara();
      const cells = r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => inline(c.trim()));
      const head = cells(ln); i += 2; let rows = "";
      while (i < lines.length && /^\|.*\|$/.test(lines[i].trim())) { rows += `<tr>${cells(lines[i]).map(c => `<td>${c}</td>`).join("")}</tr>`; i++; }
      html += `<div class="table-wrap md-table"><table class="table"><thead><tr>${head.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
      continue;
    }
    if (/^&gt;\s?/.test(ln)) { flushPara(); let q = []; while (i < lines.length && /^&gt;\s?/.test(lines[i])) { q.push(lines[i].replace(/^&gt;\s?/, "")); i++; } html += `<blockquote class="md-q">${inline(q.join(" "))}</blockquote>`; continue; }
    if (/^[-*+]\s+/.test(ln)) { flushPara(); let items = []; while (i < lines.length && /^[-*+]\s+/.test(lines[i])) { items.push(`<li>${inline(lines[i].replace(/^[-*+]\s+/, ""))}</li>`); i++; } html += `<ul class="md-ul">${items.join("")}</ul>`; continue; }
    if (/^\d+\.\s+/.test(ln)) { flushPara(); let items = []; while (i < lines.length && /^\d+\.\s+/.test(lines[i])) { items.push(`<li>${inline(lines[i].replace(/^\d+\.\s+/, ""))}</li>`); i++; } html += `<ol class="md-ul">${items.join("")}</ol>`; continue; }
    para.push(ln.trim()); i++;
  }
  flushPara();
  return html;
}

/* ============================================================ RENDER · Provider */
/* Real per-provider status: engine truth + live availability checks. */
function providerStatus(kind) {
  const active = activeBrain().kind === kind;
  const chk = state.providerChecks[kind];
  if (kind === "autonomous") return active ? { text: "активен в движке", dot: "dot-success" } : { text: "встроенный · без LLM", dot: "dot-info" };
  if (active) return chk && !chk.ok ? { text: "активен · не отвечает", dot: "dot-error" } : { text: "активен в движке", dot: "dot-success" };
  if (chk && !chk.ok) return { text: "не отвечает", dot: "dot-error" };
  if (chk && chk.ok) return { text: "настроен · доступен", dot: "dot-success" };
  if (hasConfig(kind)) return { text: "настроен · не активен", dot: "dot-info" };
  return { text: "не настроен", dot: "dot-warning" };
}
/* Real availability check (CLI binary on PATH / API endpoint ping), cached. */
async function checkProvider(kind, cfg) {
  try {
    state.providerChecks[kind] = await invoke("provider_check", { kind: fullProviderKind(kind), baseUrl: (cfg && cfg.baseUrl) || "", apiKey: (cfg && cfg.apiKey) || "" });
  } catch (e) { state.providerChecks[kind] = { ok: false, detail: String(e) }; }
  updateProviderLive();
  if ($("#brainPop") && $("#brainPop").classList.contains("is-open")) renderBrainList();
  return state.providerChecks[kind];
}
/* Save the config, then activate the brain ONLY if the provider really works. */
async function saveAndActivate(kind, cfg) {
  state.provider = { kind, model: cfg.model || "", apiKey: cfg.apiKey || "", baseUrl: cfg.baseUrl || "" };
  state.providerConfigs[kind] = { model: state.provider.model, apiKey: state.provider.apiKey, baseUrl: state.provider.baseUrl };
  store.set("provider", state.provider); store.set("providerConfigs", state.providerConfigs);
  delete state.providerModelsCache[kind];
  updateProviderLive();
  const chk = await checkProvider(kind, cfg);
  if (!chk.ok) {
    toast("warning", "Конфигурация сохранена, мозг не переключён", chk.detail, null, 6500);
    return false;
  }
  try {
    await invoke("set_provider", { kind: fullProviderKind(kind), model: cfg.model || "", apiKey: cfg.apiKey || "", baseUrl: cfg.baseUrl || "" });
    await refreshModelInfo();
    updateProviderLive();
    toast("success", "Провайдер активирован", brainLabel() + " · " + chk.detail, null, 3600);
    return true;
  } catch (e) {
    updateProviderLive();
    toast("error", "Провайдер не установлен", String(e), null, 6000);
    return false;
  }
}
/* Patch statuses/cards/lists in place — no full re-render, no flicker. */
function updateProviderLive() {
  updateCurrentBrain();
  updateProviderStatus();
  updateBrainBtn();
  const sbp = $("#sbProvider"); if (sbp) sbp.textContent = "мозг: " + brainLabel(true);
  $$("#providerPad [data-provider-kind]").forEach(card => {
    const kind = card.dataset.providerKind;
    card.classList.toggle("is-active", kind === state.provider.kind);
    const st = providerStatus(kind);
    const statusEl = card.querySelector(".pc-status");
    if (statusEl) statusEl.innerHTML = `<span class="dot ${st.dot}"></span>${st.text}`;
  });
  const sc = $("#savedConfigsBody"); if (sc) { sc.innerHTML = savedConfigsHTML(); bindSavedConfigs(sc); }
  const scs = $("#savedConfigsBodySettings"); if (scs) { scs.innerHTML = savedConfigsHTML(); bindSavedConfigs(scs); }
}
function currentBrainHTML() {
  const b = activeBrain();
  const prov = state.providerKinds.find(p => p.kind === b.kind);
  const label = prov ? prov.label : b.kind;
  const detail = b.kind === "autonomous" ? "автономный WSP-оркестратор (без LLM)" : (b.model ? esc(b.model) : "модель не указана");
  return `<div class="provider-status"><span class="dot dot-success"></span><span><b>${esc(label)}</b> · ${detail}</span></div><div class="field-hint" style="margin-top:8px">Данные прочитаны из движка (config провайдера). Меняются через «Сохранить и активировать» или выбор мозга в composer.</div>`;
}
function updateCurrentBrain() { const box = $("#currentBrainBody"); if (box) box.innerHTML = currentBrainHTML(); }
function savedConfigsHTML() {
  const rows = state.providerKinds.filter(p => p.kind !== "autonomous" && hasConfig(p.kind) && state.providerConfigs[p.kind]);
  if (!rows.length) return `<div class="field-hint">Сохранённых конфигураций пока нет. Выберите провайдера слева, заполните параметры и нажмите «Сохранить и активировать».</div>`;
  return rows.map(p => { const c = state.providerConfigs[p.kind]; const active = activeBrain().kind === p.kind;
    return `<div class="setting-row"><div class="sr-text"><div class="sr-name">${icon(p.icon)}${esc(p.label)}${active ? ' <span class="badge badge-success">активен</span>' : ""}</div><div class="sr-desc">${c.model ? "модель: " + esc(c.model) : "модель не указана"} · ${c.apiKey ? "API key сохранён" : "без ключа"}${c.baseUrl ? " · " + esc(c.baseUrl) : ""}</div></div><div class="sr-control" style="gap:6px;display:flex">${active ? "" : `<button class="btn btn-sm" data-activate="${p.kind}">${icon("play")} Активировать</button>`}<button class="btn btn-sm btn-ghost" data-forget="${p.kind}" data-tip="Удалить конфигурацию">${icon("trash")}</button></div></div>`; }).join("");
}
function bindSavedConfigs(root) {
  $$("[data-activate]", root).forEach(b => b.onclick = async () => { const kind = b.dataset.activate; const c = state.providerConfigs[kind] || {};
    await saveAndActivate(kind, c); });
  $$("[data-forget]", root).forEach(b => b.onclick = () => { const kind = b.dataset.forget; delete state.providerConfigs[kind]; delete state.providerChecks[kind]; store.set("providerConfigs", state.providerConfigs);
    updateProviderLive();
    toast("info", "Конфигурация удалена", kind, null, 1800); });
}
const PROVIDER_BASE_HINTS = {
  ollama: "http://localhost:11434",
  "lm-studio": "http://localhost:1234",
  vllm: "http://localhost:8000",
  openai: "https://api.openai.com",
  openrouter: "https://openrouter.ai/api",
  anthropic: "https://api.anthropic.com"
};
function syncProviderFormContext(kind) {
  const isAuto = kind === "autonomous";
  const isCli = kind.startsWith("cli:");
  const fModel = $("#pvFieldModel"); const fKey = $("#pvFieldKey"); const fUrl = $("#pvFieldUrl");
  const hintAuto = $("#pvAutoNote"); const fetchBtn = $("#pvFetchModels"); const urlInp = $("#pvBaseUrl");
  if (fModel) fModel.style.display = isAuto ? "none" : "";
  if (fKey) fKey.style.display = (isAuto || isCli) ? "none" : "";
  if (fUrl) fUrl.style.display = (isAuto || isCli) ? "none" : "";
  if (hintAuto) hintAuto.style.display = isAuto ? "" : "none";
  if (fetchBtn) fetchBtn.style.display = (isAuto || (isCli && kind !== "cli:opencode" && kind !== "cli:antigravity")) ? "none" : "";
  if (urlInp) urlInp.placeholder = PROVIDER_BASE_HINTS[kind] || "https://...";
}
function renderProviderView() {
  const pad = $("#providerPad"); if (!pad) return;
  const activeKind = state.provider.kind; const cfg = providerCfg(activeKind);
  const activeMeta = state.providerKinds.find(p => p.kind === activeKind) || { label: activeKind, icon: "brain" };
  const groups = [
    ["Встроенный", p => p.kind === "autonomous"],
    ["Локальные серверы", p => ["ollama", "lm-studio", "vllm"].includes(p.kind)],
    ["Облачные API", p => ["openai", "openrouter", "anthropic"].includes(p.kind)],
    ["CLI-агенты", p => p.kind.startsWith("cli:")]
  ];
  const used = new Set();
  let cards = "";
  const renderCard = pp => {
    used.add(pp.kind);
    const st = providerStatus(pp.kind);
    return `<button class="provider-card ${pp.kind === activeKind ? "is-active" : ""}" data-provider-kind="${pp.kind}">
      <span class="pc-icon">${icon(pp.icon)}</span>
      <span class="pc-main"><span class="pc-name">${esc(pp.label)}</span><span class="pc-status"><span class="dot ${st.dot}"></span>${st.text}</span></span>
      ${icon("check", "icon pc-check")}
    </button>`;
  };
  groups.forEach(([title, pred]) => {
    const items = state.providerKinds.filter(pred);
    if (!items.length) return;
    cards += `<div class="provider-group-label">${title}</div>` + items.map(renderCard).join("");
  });
  const rest = state.providerKinds.filter(p => !used.has(p.kind));
  if (rest.length) cards += `<div class="provider-group-label">Другие</div>` + rest.map(renderCard).join("");
  pad.innerHTML = `
    <div class="view-head"><div class="view-head-text"><h2 class="section-title">Провайдер</h2><span class="section-sub">Reasoning-мозг WeightX · реальные подключения к API и CLI</span></div></div>
    <div class="provider-layout">
      <div class="provider-list">${cards}</div>
      <div class="provider-side">
        <div class="card"><div class="card-header">${icon("brain")}<span class="card-title">Текущий мозг · из движка</span></div><div class="card-body" id="currentBrainBody">${currentBrainHTML()}</div></div>
        <div class="card"><div class="card-header">${icon("settings")}<span class="card-title" id="providerFormTitle">Настройка · ${esc(activeMeta.label)}</span></div><div class="card-body">
          <div class="provider-status"><span class="dot dot-warning" id="providerStatusDot"></span><span id="providerStatusText">не настроен</span></div>
          <div class="alert alert-info" id="pvAutoNote" style="margin-top:12px;display:none">${icon("hard-drive")}<div><div class="alert-title">Автономный WSP-оркестратор</div><div class="alert-desc">Работает локально без внешнего LLM API и ключей. Использует встроенные эвристики L1→L3, EAMIT и Vespasian.</div></div></div>
          <div class="provider-form-grid" style="margin-top:12px">
            <div class="field" id="pvFieldModel"><label class="field-label">Модель</label><input class="input" id="pvModel" placeholder="пусто = по умолчанию" value="${esc(cfg.model || "")}" /></div>
            <div class="field" id="pvFieldKey"><label class="field-label">API key</label><div class="input-wrap"><input class="input" id="pvApiKey" type="password" placeholder="или переменная окружения" value="${esc(cfg.apiKey || "")}" /><button type="button" class="input-eye" id="pvKeyEye" aria-label="Показать ключ" data-tip="Показать / скрыть">${icon("eye")}</button></div></div>
            <div class="field" id="pvFieldUrl"><label class="field-label">base_url</label><input class="input" id="pvBaseUrl" placeholder="https://..." value="${esc(cfg.baseUrl || "")}" /></div>
          </div>
          <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">
            <button class="btn btn-primary" id="pvSave">${icon("check")} Сохранить и активировать</button>
            <button class="btn" id="pvFetchModels">${icon("refresh")} Модели с API</button>
            <button class="btn btn-ghost" id="pvOpenSettings">${icon("settings")} Расширенные</button>
          </div>
          <div class="field-hint" style="margin-top:8px">«Сохранить и активировать» проверяет доступность и устанавливает мозг в движок.</div>
          <div id="pvModelsList" style="margin-top:8px"></div>
        </div></div>
        <div class="card"><div class="card-header">${icon("network")}<span class="card-title">Сохранённые конфигурации</span></div><div class="card-body">
          <div class="settings-card-sub">Конфигурации с моделью / ключом / base_url. Активация реально переключает мозг движка; в выборе Brain видны только настроенные провайдеры.</div>
          <div id="savedConfigsBody">${savedConfigsHTML()}</div>
        </div></div>
      </div>
    </div>`;
  $$("#providerPad [data-provider-kind]").forEach(b => b.onclick = () => {
    const kind = b.dataset.providerKind; const c = providerCfg(kind);
    state.provider = { kind, model: c.model || "", apiKey: c.apiKey || "", baseUrl: c.baseUrl || "" };
    store.set("provider", state.provider);
    const meta = state.providerKinds.find(p => p.kind === kind) || { label: kind };
    const t = $("#providerFormTitle"); if (t) t.textContent = "Настройка · " + meta.label;
    $("#pvModel").value = c.model || ""; $("#pvApiKey").value = c.apiKey || ""; $("#pvBaseUrl").value = c.baseUrl || "";
    $("#pvModelsList").innerHTML = "";
    syncProviderFormContext(kind);
    updateProviderLive();
  });
  $("#pvKeyEye").onclick = () => {
    const inp = $("#pvApiKey"); const show = inp.type === "password"; inp.type = show ? "text" : "password";
    $("#pvKeyEye").innerHTML = icon(show ? "eye-off" : "eye");
  };
  $("#pvSave").onclick = async () => { const kind = state.provider.kind;
    const c = { model: $("#pvModel").value.trim(), apiKey: $("#pvApiKey").value.trim(), baseUrl: $("#pvBaseUrl").value.trim() };
    await saveAndActivate(kind, c);
  };
  $("#pvFetchModels").onclick = async () => { const kind = state.provider.kind;
    const c = { model: $("#pvModel").value.trim(), apiKey: $("#pvApiKey").value.trim(), baseUrl: $("#pvBaseUrl").value.trim() };
    const box = $("#pvModelsList"); box.innerHTML = `<div class="field-hint">запрос моделей…</div>`;
    try {
      const r = await invoke("provider_models", { kind: fullProviderKind(kind), baseUrl: c.baseUrl, apiKey: c.apiKey });
      state.providerModelsCache[kind] = { models: r.models || [], source: r.source || "", note: r.note || "", loading: false };
      const models = r.models || [];
      box.innerHTML = models.length
        ? `<div class="field-hint">Источник: ${esc(r.source)} · ${models.length} моделей</div><div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px">${models.map(m => `<button class="btn btn-sm" data-pick-model="${esc(m)}">${esc(m)}</button>`).join("")}</div>`
        : `<div class="field-hint">${esc(r.note || "Провайдер вернул пустой список моделей.")}</div>`;
      $$("[data-pick-model]", box).forEach(b => b.onclick = () => { $("#pvModel").value = b.dataset.pickModel; toast("info", "Модель выбрана", b.dataset.pickModel + " · нажмите «Сохранить и активировать»", null, 2200); });
    } catch (e) { box.innerHTML = `<div class="alert alert-error">${icon("alert-circ")}<div><div class="alert-title">Список моделей недоступен</div><div class="alert-desc">${esc(String(e))}</div></div></div>`; }
  };
  $("#pvOpenSettings").onclick = () => { state.settingsTab = "provider"; navigate("settings"); };
  bindSavedConfigs(pad);
  syncProviderFormContext(activeKind);
  updateProviderStatus();
  // Live availability checks for every configured provider (real PATH/API probes).
  configuredProviders().forEach(p => { if (p.kind !== "autonomous") checkProvider(p.kind, providerCfg(p.kind)); });
}
function updateProviderStatus() { const kind = state.provider.kind; const st = providerStatus(kind);
  const txt = $("#providerStatusText"); const dot = $("#providerStatusDot");
  const isActive = activeBrain().kind === kind;
  const chk = state.providerChecks[kind];
  if (txt) txt.textContent = isActive
    ? (chk && !chk.ok ? "активен, но не отвечает: " + chk.detail : "активен в движке")
    : hasConfig(kind)
      ? (chk ? (chk.ok ? "настроен · доступен · нажмите «Сохранить и активировать»" : "не отвечает: " + chk.detail) : "настроен · нажмите «Сохранить и активировать»")
      : "не настроен";
  if (dot) dot.className = "dot " + st.dot; }

/* ============================================================ RENDER · Session */
const WX_EMBLEM_SVG = `<svg viewBox="0 0 256 256" aria-hidden="true">
  <line x1="62" y1="72" x2="95" y2="184" stroke="var(--text-tertiary)" stroke-width="9" stroke-linecap="round"/>
  <line x1="194" y1="72" x2="161" y2="184" stroke="var(--text-tertiary)" stroke-width="9" stroke-linecap="round"/>
  <line x1="95" y1="184" x2="128" y2="128" stroke="var(--text-tertiary)" stroke-width="9" stroke-linecap="round"/>
  <line x1="161" y1="184" x2="128" y2="128" stroke="var(--text-tertiary)" stroke-width="9" stroke-linecap="round"/>
  <line x1="62" y1="72" x2="161" y2="184" stroke="var(--text-primary)" stroke-width="15" stroke-linecap="round"/>
  <line x1="194" y1="72" x2="95" y2="184" stroke="var(--text-primary)" stroke-width="15" stroke-linecap="round"/>
  <circle cx="62" cy="72" r="11" fill="var(--text-primary)"/><circle cx="194" cy="72" r="11" fill="var(--text-primary)"/>
  <circle cx="95" cy="184" r="11" fill="var(--text-primary)"/><circle cx="161" cy="184" r="11" fill="var(--text-primary)"/>
  <circle cx="128" cy="128" r="21" fill="var(--bg-raised)"/><circle class="focal" cx="128" cy="128" r="16" fill="var(--accent-highlight)"/>
</svg>`;
function renderEmptyReadiness() {
  const box = $("#emptyReadiness"); if (!box) return;
  const m = state.model;
  if (!m) {
    box.innerHTML = `<span class="dot dot-warning"></span><span>Модель не загружена</span><span class="er-sep">·</span><button class="btn btn-sm btn-primary" id="heroDemoBtn" style="height:22px;padding:0 10px">${icon("zap")} Загрузить minillama-demo (215K)</button><button class="btn btn-sm btn-ghost" id="heroCatBtn" style="height:22px;padding:0 8px">Каталог</button>`;
    $("#heroDemoBtn").onclick = () => loadModel("minillama-demo");
    $("#heroCatBtn").onclick = () => navigate("models");
    return;
  }
  const v = (state.modelInfo && state.modelInfo.vespasian) || {};
  box.innerHTML = `<span class="dot dot-success"></span><span class="mono" style="color:var(--text-primary);font-weight:600">${esc(m.id)}</span><span class="er-sep">·</span><span>${esc(m.params)} · ${m.layers || "—"}L</span><span class="er-sep">·</span><span class="mono">held-out ${fmt(v.best || 0, 4)}</span>`;
}
function autoResizeComposer(el) {
  if (!el) return;
  el.style.height = "auto";
  el.style.height = Math.min(168, Math.max(40, el.scrollHeight)) + "px";
}
function renderSession() {
  const v = $("#view-session");
  v.innerHTML = `
    <div class="session-head">
      <span class="badge badge-accent">${icon("chat")} Чат</span>
      <span class="section-sub" id="sessionModelLabel">${state.model ? esc(state.model.id) : "модель не загружена"}</span>
      <span style="flex:1"></span>
      <span class="section-sub">5-стадийный цикл · Vespasian guard</span>
      <button class="btn btn-sm btn-ghost" id="clearSession">${icon("trash")} Очистить</button>
    </div>
    <div class="thread" id="thread"><div class="thread-inner is-empty" id="threadInner">
      <div class="empty" id="sessionEmpty">
        <div class="empty-emblem">${WX_EMBLEM_SVG}</div>
        <div class="empty-brand">WeightX</div>
        <div class="empty-readiness" id="emptyReadiness"></div>
        <div class="empty-skills">
          ${state.skills.map(s => { const meta = SKILL_META[s.skill_id] || { icon: "zap" }; return `<button class="pill" data-skill="${s.skill_id}" data-tip="${esc(s.pipeline_title)}">${icon(meta.icon)} ${esc(s.pill_label)}</button>`; }).join("")}
        </div>
      </div>
    </div></div>
    <div class="composer-wrap"><div class="composer" data-component="composer">
      <div class="slash-pop" id="slashPop" role="listbox" aria-label="Навыки WeightX"></div>
      <textarea class="composer-input" id="composerInput" rows="2" placeholder="Опишите цель хирургии или введите / для запуска навыка…"></textarea>
      <div class="composer-foot">
        <button type="button" class="slash-trigger" id="slashTrigger" aria-haspopup="listbox" aria-expanded="false" data-tip="Открыть меню навыков (/)">${icon("sparkles")}<span>Навыки</span></button>
        <span class="composer-hint"><kbd class="kbd">↵</kbd> запуск · <kbd class="kbd">Ctrl</kbd><kbd class="kbd">↵</kbd> перенос</span>
        <span class="spacer"></span>
        <button class="brain-btn" id="brainBtn" aria-haspopup="dialog" aria-label="Модель Brain: выбрать провайдера и модель"><svg class="icon bb-icon" id="brainBtnIcon"><use href="#i-hard-drive"/></svg><span class="bb-label" id="brainBtnLabel">Встроенный</span><svg class="icon bb-chev"><use href="#i-chev-down"/></svg></button>
        <button class="send-btn" id="sendBtn" data-run disabled aria-label="Запустить">${icon("arrow-up")}</button>
      </div>
    </div></div>`;
  renderEmptyReadiness();
  syncBgDim();
  $$("#view-session [data-skill]").forEach(b => b.onclick = () => runSkill(b.dataset.skill));
  $("#clearSession").onclick = () => { if (busyGuard()) return; state.messages = []; renderSession(); toast("info", "Сессия очищена", "История чекпоинтов сохранена", null, 1800); };
  const input = $("#composerInput");
  const submit = () => {
    const goal = input.value.trim();
    if (!goal) { input.focus(); return; }
    input.value = ""; autoResizeComposer(input); syncSendBtn(); closeSlash();
    const m = goal.match(/^\/(?:skill\s+)?([a-z0-9-]+)$/i);
    if (m && skillById(m[1].toLowerCase())) runSkill(m[1].toLowerCase());
    else runChat(goal);
  };
  $("#sendBtn").onclick = submit;
  const slashBtn = $("#slashTrigger");
  const slashPop = $("#slashPop");
  slashBtn.onmousedown = (e) => { e.preventDefault(); clearTimeout(slashBlurTimer); };
  slashPop.onmousedown = (e) => e.preventDefault();
  slashBtn.onclick = (e) => {
    e.stopPropagation();
    clearTimeout(slashBlurTimer);
    closeStylesPop(); closeBrainPop(); closeSelectPop();
    const pop = $("#slashPop");
    if (pop && pop.classList.contains("is-open")) {
      if (input.value === "/") { input.value = ""; autoResizeComposer(input); syncSendBtn(); }
      closeSlash();
      return;
    }
    if (!input.value.startsWith("/")) { input.value = "/"; autoResizeComposer(input); syncSendBtn(); }
    input.focus();
    renderSlash();
  };
  $("#brainBtn").onclick = (e) => { e.stopPropagation(); $("#brainPop").classList.contains("is-open") ? closeBrainPop() : openBrainPop(); };
  updateBrainBtn();
  input.addEventListener("focus", () => clearTimeout(slashBlurTimer));
  input.addEventListener("input", () => { autoResizeComposer(input); syncSendBtn(); renderSlash(); });
  input.addEventListener("blur", () => {
    clearTimeout(slashBlurTimer);
    slashBlurTimer = setTimeout(() => {
      const ae = document.activeElement;
      if (ae && ae.closest && ae.closest(".composer")) return;
      closeSlash();
    }, 160);
  });
  input.addEventListener("keydown", e => {
    const popOpen = $("#slashPop")?.classList.contains("is-open");
    if (popOpen && slashItems.length) {
      if (e.key === "ArrowDown") { e.preventDefault(); slashActive = (slashActive + 1) % slashItems.length; highlightSlash(); return; }
      if (e.key === "ArrowUp") { e.preventDefault(); slashActive = (slashActive - 1 + slashItems.length) % slashItems.length; highlightSlash(); return; }
      if (e.key === "Tab" || (e.key === "Enter" && !e.ctrlKey && !e.metaKey && !e.shiftKey)) { e.preventDefault(); pickSlash(slashItems[slashActive].skill_id); return; }
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeSlash(); return; }
    }
    if (popOpen && !slashItems.length && e.key === "Enter" && !e.ctrlKey && !e.metaKey && !e.shiftKey) { e.preventDefault(); return; }
    if (e.key === "Enter" && !e.ctrlKey && !e.metaKey && !e.shiftKey) { e.preventDefault(); submit(); return; }
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); input.setRangeText("\n", input.selectionStart, input.selectionEnd, "end"); input.dispatchEvent(new Event("input", { bubbles: true })); return; }
  });
}
/* ---- Slash command menu ---- */
let slashActive = 0, slashItems = [], slashBlurTimer = 0;
function slashQuery() { const el = $("#composerInput"); if (!el) return null; const v = el.value; if (!v.startsWith("/")) return null; const q = v.slice(1); if (/\s/.test(q)) return null; return q.toLowerCase(); }
function renderSlash() {
  clearTimeout(slashBlurTimer);
  const pop = $("#slashPop"); if (!pop) return;
  const q = slashQuery();
  if (q === null) { closeSlash(); return; }
  slashItems = state.skills.filter(s => !q || s.skill_id.includes(q) || s.pill_label.toLowerCase().includes(q) || (s.short_hint || "").toLowerCase().includes(q));
  if (slashActive >= slashItems.length) slashActive = 0;
  pop.innerHTML = `<div class="slash-head">${icon("sparkles")} Навыки WeightX</div>
    <div class="slash-list">${slashItems.length ? slashItems.map((s, i) => { const meta = SKILL_META[s.skill_id] || { icon: "zap" }; return `<button class="slash-item ${i === slashActive ? "is-active" : ""}" data-slash="${s.skill_id}" role="option" aria-selected="${i === slashActive}">
      <span class="si-ic">${icon(meta.icon)}</span>
      <span class="si-text"><span class="si-name">${esc(s.pill_label)}</span><span class="si-hint">${esc(s.short_hint || "")}</span></span>
      <span class="si-cmd mono">/${s.skill_id}</span></button>`; }).join("") : `<div class="cmdk-empty">Нет навыка «/${esc(q)}»</div>`}</div>
    <div class="slash-foot"><span><kbd class="kbd">↑</kbd><kbd class="kbd">↓</kbd> выбор</span><span><kbd class="kbd">↵</kbd> запуск</span><span><kbd class="kbd">Esc</kbd> закрыть</span></div>`;
  pop.classList.add("is-open");
  const st = $("#slashTrigger"); if (st) { st.classList.add("is-open"); st.setAttribute("aria-expanded", "true"); }
  $$(".slash-item", pop).forEach(b => {
    b.onmousemove = () => { const i = slashItems.findIndex(s => s.skill_id === b.dataset.slash); if (i >= 0 && i !== slashActive) { slashActive = i; highlightSlash(); } };
    b.onclick = () => pickSlash(b.dataset.slash);
  });
}
function highlightSlash() { $$("#slashPop .slash-item").forEach((b, i) => { b.classList.toggle("is-active", i === slashActive); b.setAttribute("aria-selected", String(i === slashActive)); });
  $(`#slashPop .slash-item[data-slash="${slashItems[slashActive]?.skill_id}"]`)?.scrollIntoView({ block: "nearest" }); }
function closeSlash() {
  clearTimeout(slashBlurTimer);
  $("#slashPop")?.classList.remove("is-open");
  const st = $("#slashTrigger"); if (st) { st.classList.remove("is-open"); st.setAttribute("aria-expanded", "false"); }
}
function pickSlash(id) { const el = $("#composerInput"); if (el) { el.value = ""; autoResizeComposer(el); syncSendBtn(); } closeSlash(); runSkill(id); }

/* ============================================================ RENDER · Tensors */
const ROLE_LABEL = { q: "Q", k: "K", v: "V", o: "O", mlp: "MLP", norm: "Norm", embed: "Embed", other: "·" };
function renderTensors() {
  const v = $("#view-tensors");
  if (!state.model) { v.innerHTML = `<div class="view-scroll"><div class="empty" style="height:100%"><div class="empty-icon">${icon("box")}</div><div class="empty-title">Модель не загружена</div><div class="empty-desc">Загрузите модель из каталога моделей, чтобы исследовать тензоры L3.</div><button class="btn btn-primary" id="tensorLoad">${icon("cloud")} Открыть модели</button></div></div>`; $("#tensorLoad").onclick = () => { navigate("models"); }; return; }
  if (!state.tensors.length) {
    if (state.tensorsLoading) return;
    state.tensorsLoading = true;
    v.innerHTML = `<div class="view-scroll"><div class="empty" style="height:100%"><div class="empty-icon">${icon("loader", "icon spin")}</div><div class="empty-title">Чтение тензоров…</div><div class="empty-desc">L3 workspace · ${esc(state.model.id)}</div></div></div>`;
    refreshTensors().then(() => { state.tensorsLoading = false; if (state.view === "tensors") renderTensors(); });
    return;
  }
  const { q, role, onlyModified } = state.tensorFilter;
  const modCount = state.tensors.filter(t => t.modified).length;
  const list = filteredTensors(); const maxNorm = Math.max(...state.tensors.map(t => t.norm), 1e-9); const sel = state.selectedTensor || list[0];
  v.innerHTML = `
    <div class="toolbar">
      <div style="position:relative;flex:1;min-width:160px;max-width:300px"><input class="input" id="tensorSearch" placeholder="Поиск тензора…" value="${esc(q)}" style="padding-left:28px" /><svg class="icon" style="position:absolute;left:8px;top:7px;width:14px;height:14px;color:var(--text-ghost)"><use href="#i-search"/></svg></div>
      <div class="segmented" id="roleFilter">${["all", "q", "k", "v", "o", "mlp", "norm", "embed"].map(r => `<button class="segmented-item ${role === r ? "is-active" : ""}" data-role="${r}">${r === "all" ? "Все" : ROLE_LABEL[r]}</button>`).join("")}</div>
      <button class="btn btn-sm btn-filter-mod ${onlyModified ? "is-active" : ""}" id="tensorModFilter" data-tip="Показать только изменённые в sandbox">${icon("diff")} Изменённые (${modCount})</button>
      <span style="flex:1"></span><span class="section-sub" id="tensorCount">${fmtInt(list.length)} / ${fmtInt(state.tensors.length)}</span>
      <button class="btn btn-sm btn-ghost" id="tensorRefresh">${icon("refresh")} Обновить</button>
    </div>
    <div class="split"><div class="split-left"><div class="tensor-list" id="tensorList">${tensorListHTML(list, sel, maxNorm)}</div></div><div class="split-right"><div class="detail-pad" id="tensorDetail"></div></div></div>`;
  $("#tensorSearch").oninput = e => { const caret = e.target.selectionStart; state.tensorFilter.q = e.target.value; paintTensorList(); e.target.focus(); e.target.setSelectionRange(caret, caret); };
  $$("#roleFilter [data-role]").forEach(b => b.onclick = () => { state.tensorFilter.role = b.dataset.role; renderTensors(); });
  $("#tensorModFilter").onclick = () => { state.tensorFilter.onlyModified = !state.tensorFilter.onlyModified; renderTensors(); };
  $("#tensorRefresh").onclick = async () => { await refreshTensors(); renderTensors(); toast("info", "Тензоры обновлены", fmtInt(state.tensors.length) + " тензоров", null, 1600); };
  bindTensorList(); renderTensorDetail(sel);
}
function filteredTensors() { const { q, role, onlyModified } = state.tensorFilter; return state.tensors.filter(t => (!onlyModified || t.modified) && (role === "all" || t.role === role) && (!q || t.name.toLowerCase().includes(q.toLowerCase()))); }
function tensorListHTML(list, sel, maxNorm) { return list.length ? list.map(t => `<button class="tensor-item ${sel && sel.name === t.name ? "is-selected" : ""}" data-tensor="${esc(t.name)}" title="${esc(t.name)}"><span class="ti-name">${esc(t.name)}${t.modified ? ' <span class="badge badge-warning" style="height:14px;font-size:9px">изменён</span>' : ""}</span><span class="ti-meta"><span class="badge" style="height:15px">${ROLE_LABEL[t.role] || "·"}</span><span class="mono">${t.shape.join("×")}</span><span class="ti-norm"><i style="width:${(t.norm / maxNorm * 100).toFixed(0)}%"></i></span></span></button>`).join("") : `<div class="empty"><div class="empty-icon">${icon("filter")}</div><div class="empty-title">Ничего не найдено</div><div class="empty-desc">Измените запрос или фильтр роли.</div></div>`; }
function bindTensorList() { $$("#tensorList [data-tensor]").forEach(b => b.onclick = () => { state.selectedTensor = state.tensors.find(t => t.name === b.dataset.tensor); $$("#tensorList .tensor-item").forEach(x => x.classList.toggle("is-selected", x.dataset.tensor === state.selectedTensor.name)); renderTensorDetail(state.selectedTensor); renderInspector(); }); }
function paintTensorList() { const list = filteredTensors(); const maxNorm = Math.max(...state.tensors.map(t => t.norm), 1e-9); $("#tensorList").innerHTML = tensorListHTML(list, state.selectedTensor, maxNorm); $("#tensorCount").textContent = `${fmtInt(list.length)} / ${fmtInt(state.tensors.length)}`; bindTensorList(); }
/* Real L3 describe_tensor + tensor_block + read_tensor_slice. */
function renderTensorDetail(t) {
  const box = $("#tensorDetail"); if (!box) return;
  if (!t) { box.innerHTML = `<div class="empty"><div class="empty-icon">${icon("box")}</div><div class="empty-title">Выберите тензор</div></div>`; return; }
  box.innerHTML = `<div class="empty" style="padding:32px"><div class="empty-icon">${icon("loader", "icon spin")}</div><div class="empty-title">describe_tensor…</div><div class="empty-desc mono">${esc(t.name)}</div></div>`;
  loadTensorDetail(t, box);
}
async function loadTensorDetail(t, box) {
  try {
    const [desc, block] = await Promise.all([
      invoke("describe_tensor", { name: t.name, topK: 24 }),
      invoke("tensor_block", { name: t.name, size: 16 }).catch(() => null)
    ]);
    if (!state.selectedTensor || state.selectedTensor.name !== t.name) return; // stale selection
    state.tensorDescCache = { name: t.name, desc };
    const spec = desc.top_singular_values || [];
    const shape = (desc.shape || t.shape).join("×");
    const minV = fmt(desc.min ?? 0, 3), maxV = fmt(desc.max ?? 0, 3);
    const kv = [
      ["Форма", shape], ["Элементов", fmtInt(desc.numel ?? t.elements)], ["Норма F", fmt(desc.l2_norm ?? t.norm)],
      ["Std", fmt(desc.std ?? 0, 4)], ["Среднее", fmt(desc.mean ?? 0, 4)], ["Диапазон", `${minV} .. ${maxV}`],
      ["Нулей %", fmt(desc.zeros_pct ?? 0, 2)], ["Эфф. ранг (90%)", desc.rank_90pct_energy || "—"]
    ];
    box.innerHTML = `
      <div><div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap"><span class="mono" style="font-size:15px;font-weight:600;word-break:break-all">${esc(t.name)}</span><span class="badge badge-accent">${ROLE_LABEL[t.role] || "·"}</span><span class="badge">${esc(desc.dtype || t.dtype)}</span>${t.modified ? '<span class="badge badge-warning">изменён</span>' : ""}</div>
        <div class="kv-grid">${kv.map(([k, val]) => `<div class="kv"><div class="k">${k}</div><div class="v">${val}</div></div>`).join("")}</div></div>
      <div class="tensor-charts-grid">
        <div class="chart-box"><div class="chart-title">${icon("activity")} SVD-спектр <span class="spacer"></span><span class="section-sub">${spec.length ? "σ1…σ" + spec.length : "недоступен"}</span></div>${spec.length ? spectrumChart(spec) : '<div class="field-hint">SVD вычисляется для матриц с минимальной стороной ≤2048. Для этого тензора спектр не рассчитан.</div>'}</div>
        <div class="chart-box"><div class="chart-title">${icon("box")} Карта весов <span class="spacer"></span><span class="section-sub">${block ? block.rows + "×" + block.cols : "16×16"}</span></div>${block && block.values.length ? heatmapHTML(block.values, block.rows, block.cols, t.name.split(".").pop()) : '<div class="field-hint">Блок недоступен.</div>'}</div>
      </div>
      <div class="chart-box"><div class="chart-title">${icon("terminal")} Чтение среза</div><div style="display:flex;gap:8px;margin-bottom:10px"><input class="input" id="sliceIndex" placeholder="срез, напр. 0:8, :16" style="flex:1" /><button class="btn btn-sm" id="sliceBtn">${icon("eye")} Читать</button><button class="btn btn-sm btn-ghost" id="copyTensor">${icon("copy")} Имя</button></div><div class="slice-out" id="sliceOut">загрузка среза…</div></div>`;
    const readSlice = async () => {
      const idx = $("#sliceIndex").value.trim() || ":";
      $("#sliceOut").textContent = "чтение…";
      try {
        const r = await invoke("read_tensor_slice", { name: t.name, index: idx, maxElements: 256 });
        $("#sliceOut").textContent = `${r.name}  ·  срез ${r.slice_expr}  ·  форма ${(r.slice_shape || []).join("×")}\n${r.values}\n· ${fmtInt(r.slice_numel)} элементов${r.truncated ? " (вывод обрезан до 256)" : ""}`;
      } catch (e) { $("#sliceOut").textContent = "Ошибка среза: " + e; }
    };
    $("#sliceBtn").onclick = readSlice;
    $("#sliceIndex").addEventListener("keydown", e => { if (e.key === "Enter") readSlice(); });
    $("#copyTensor").onclick = () => { navigator.clipboard?.writeText(t.name).then(() => toast("success", "Скопировано", t.name, null, 1600)).catch(() => toast("info", "Имя тензора", t.name, null, 2400)); };
    readSlice();
  } catch (e) {
    if (state.selectedTensor && state.selectedTensor.name === t.name) box.innerHTML = `<div class="empty"><div class="empty-icon">${icon("alert-circ")}</div><div class="empty-title">Не удалось прочитать тензор</div><div class="empty-desc">${esc(String(e))}</div></div>`;
  }
}

/* ============================================================ RENDER · Diagnostics */
async function renderDiagnostics() {
  const pad = $("#diagPad");
  if (!state.model) { pad.innerHTML = `<div class="empty" style="min-height:60vh"><div class="empty-icon">${icon("activity")}</div><div class="empty-title">Модель не загружена</div><div class="empty-desc">L2-диагностика доступна после загрузки модели.</div><button class="btn btn-primary" id="diagLoad">${icon("cloud")} Открыть модели</button></div>`; $("#diagLoad").onclick = () => { navigate("models"); }; return; }
  if (!state.diag) {
    pad.innerHTML = `<div class="empty" style="min-height:60vh"><div class="empty-icon">${icon("loader", "icon spin")}</div><div class="empty-title">L1 AutoArch + L2 Uni-Layer…</div><div class="empty-desc">Профилирование 13 метрик по слоям · ${esc(state.model.id)}</div></div>`;
    await refreshDiagnostics();
    if (state.view !== "diagnostics") return;
    if (!state.diag) { pad.innerHTML = `<div class="empty" style="min-height:60vh"><div class="empty-icon">${icon("alert-circ")}</div><div class="empty-title">Диагностика недоступна</div><div class="empty-desc">Движок не вернул профиль модели.</div><button class="btn btn-primary" id="diagRetry">${icon("refresh")} Повторить</button></div>`; $("#diagRetry").onclick = () => renderDiagnostics(); return; }
  }
  const d = state.diag;
  const l1 = d.l1 || {};
  const layers = (d.layers || []).map(l => ({ ...l, isBottleneck: (d.bottlenecks || []).includes(l.layer), isRedundant: (d.redundant || []).includes(l.layer) }));
  state.diagLayers = layers;
  if (!layers.length) { pad.innerHTML = `<div class="empty" style="min-height:60vh"><div class="empty-icon">${icon("activity")}</div><div class="empty-title">Слои не найдены</div><div class="empty-desc">Модель не содержит распознаваемых transformer-блоков.</div></div>`; return; }
  const bn = layers.filter(l => l.isBottleneck).length, rd = layers.filter(l => l.isRedundant).length;
  const kn = d.knowledge || {}; const kci = kn.kci_overall || 0;
  const v = (state.modelInfo && state.modelInfo.vespasian) || { baseline: 0, best: 0 };
  const scoreDelta = (v.best || 0) - (v.baseline || 0);
  const sorted = [...layers].sort((a, b) => ((a[state.sortDiag.key] || 0) - (b[state.sortDiag.key] || 0)) * state.sortDiag.dir);
  const cols = LAYER_METRICS.slice(0, 8);
  const carriers = (kn.logit_lens_layers || []).filter(l => (l.target_prob_mean || 0) > 0.5).length;
  const l1Chips = [
    l1.family ? ["семейство", l1.family] : null,
    l1.mlp_type ? ["MLP", l1.mlp_type] : null,
    l1.norm_type ? ["norm", l1.norm_type] : null,
    l1.d_model ? ["d_model", String(l1.d_model)] : null,
    l1.mlp_hidden_dim ? ["hidden", String(l1.mlp_hidden_dim)] : null,
    l1.confidence != null ? ["L1 conf", fmt(l1.confidence * 100, 0) + "%"] : null
  ].filter(Boolean);
  pad.innerHTML = `
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:12px;flex-wrap:wrap"><h2 class="section-title">L2-диагностика</h2><span class="section-sub">L1 AutoArch + Uni-Layer · 13 метрик · ${esc(state.model.id)}</span><span style="flex:1"></span><button class="btn btn-sm btn-ghost" id="diagRefresh">${icon("refresh")} Пересчитать</button></div>
    ${l1Chips.length ? `<div class="l1-strip">${l1Chips.map(([k, val]) => `<span class="metric-chip"><span class="mc-k">${esc(k)}</span><span class="mc-v">${esc(val)}</span></span>`).join("")}</div>` : ""}
    <div class="stat-grid" style="margin-bottom:20px">
      <div class="stat"><div class="stat-label">Bottleneck-слои</div><div class="stat-value" style="color:${bn ? "var(--status-warning)" : "inherit"}">${bn}</div><div class="stat-delta is-flat">из ${layers.length}</div></div>
      <div class="stat"><div class="stat-label">Избыточные</div><div class="stat-value">${rd}</div><div class="stat-delta is-flat">кандидаты на сжатие</div></div>
      <div class="stat"><div class="stat-label">KCI (Gini)</div><div class="stat-value">${kci.toFixed(2)}</div><div class="stat-delta is-flat">${esc(kn.concentration_regime || "концентрация знаний")}</div></div>
      <div class="stat"><div class="stat-label">Held-out score</div><div class="stat-value">${fmt(v.best || 0, 4)}</div><div class="stat-delta ${scoreDelta <= 0 ? "is-up" : "is-down"}">${scoreDelta > 0 ? "+" : ""}${fmt(scoreDelta, 4)} к базе</div></div>
    </div>
    <div class="chart-box" style="margin-bottom:16px"><div class="chart-title">${icon("network")} Архитектура модели <span class="spacer"></span><span class="section-sub">${layers.length} transformer-блоков · attention + MLP</span></div>${modelVizHTML(layers)}</div>
    <div class="diag-grid">
      <div class="card"><div class="card-header"><span class="dot dot-accent"></span><span class="card-title">Профиль слоёв</span><span class="spacer"></span><span class="section-sub">клик — инспектор</span></div>
        <div class="table-wrap" style="border:0;border-radius:0;max-height:440px;overflow-y:auto"><table class="table"><thead><tr><th class="sortable" data-sort="layer_idx">Слой ${icon("sort", "icon sort-ic")}</th>${cols.map(c => `<th class="sortable num" data-sort="${c.key}">${c.label} ${icon("sort", "icon sort-ic")}</th>`).join("")}<th>Статус</th></tr></thead>
        <tbody>${sorted.map(l => `<tr data-layer="${l.layer_idx}" class="${state.selectedLayer && state.selectedLayer.layer_idx === l.layer_idx ? "is-selected" : ""}"><td class="mono">${l.layer_idx}</td>${cols.map(c => `<td class="num">${fmt(l[c.key] || 0, c.d)}</td>`).join("")}<td>${l.isBottleneck ? '<span class="badge badge-warning">bottleneck</span>' : l.isRedundant ? '<span class="badge badge-info">redundant</span>' : '<span class="badge badge-success">ok</span>'}</td></tr>`).join("")}</tbody></table></div></div>
      <div style="display:flex;flex-direction:column;gap:16px">
        <div class="chart-box"><div class="chart-title">${icon("map")} Карта знаний · Logit Lens</div>${knowledgeHeatmap(kn)}</div>
        <div class="chart-box"><div class="chart-title">${icon("gauge")} Knowledge Concentration Index</div><div class="gauge-wrap">${gaugeSVG(kci, "KCI")}<div class="gauge-label"><div class="gl-v">${carriers} / ${(kn.logit_lens_layers || []).length || layers.length}</div><div class="gl-k">слоёв с P(target) &gt; 0.5</div><div class="gl-k" style="margin-top:6px">кристаллизация: ${fmt((kn.crystallization_fraction || 0) * 100, 0)}% · слой ${fmt(kn.avg_crystallization_layer || 0, 1)}</div></div></div></div>
      </div>
    </div>
    <div class="chart-box" style="margin-top:16px"><div class="chart-title">${icon("waveform")} Held-out score · история Vespasian <span class="spacer"></span><span class="section-sub">реальные итерации верификации</span></div>${scoreChart(state.history)}</div>
    ${d.summary ? `<div class="alert alert-info" style="margin-top:16px">${icon("info")}<div><div class="alert-title">L2 Uni-Layer summary</div><div class="alert-desc">${esc(d.summary)}</div></div></div>` : ""}`;
  $("#diagRefresh").onclick = async () => { state.diag = null; await renderDiagnostics(); toast("info", "Диагностика пересчитана", "L1 + L2 + Knowledge Map", null, 1600); };
  $$("#diagPad [data-sort]").forEach(th => th.onclick = () => { const k = th.dataset.sort; state.sortDiag.dir = state.sortDiag.key === k ? -state.sortDiag.dir : -1; state.sortDiag.key = k; renderDiagnostics(); });
  $$("#diagPad [data-layer]").forEach(tr => tr.onclick = () => { state.selectedLayer = layers.find(l => l.layer_idx === +tr.dataset.layer); renderDiagnostics(); renderInspector(); if ($("#app").dataset.inspector !== "open") toggleInspector(); });
  $$("#diagPad [data-mvlayer]").forEach(nd => { const act = () => { state.selectedLayer = layers.find(l => l.layer_idx === +nd.dataset.mvlayer); renderDiagnostics(); renderInspector(); if ($("#app").dataset.inspector !== "open") toggleInspector(); };
    nd.onclick = act; nd.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } }; });
}

/* ============================================================ RENDER · YAML */
/* Real BrainSurgery-WX plan format (single-key ops + assert invariants). */
const DEFAULT_YAML = "transforms:\n  - assert: { finite: { of: '*' } }\n  - assert: { exists: { of: '*proj*' } }\n# Пример правки (укажите тензор вашей модели):\n#  - scale_: { target: 'model.layers.0.mlp.up_proj.weight', by: 1.02 }\n#  - assert: { rel_delta: { of: 'model.layers.0.mlp.up_proj.weight', le: 0.06 } }";
function pickSampleTensor(rolePref) {
  const twoD = state.tensors.filter(t => t.shape.length === 2);
  const hit = twoD.find(t => t.role === rolePref) || twoD[0];
  return hit ? hit.name : "model.layers.0.mlp.down_proj.weight";
}
function yamlTemplatePlan(kind) {
  const tMlp = pickSampleTensor("mlp");
  const tAttn = pickSampleTensor("q");
  if (kind === "outlier") return `transforms:\n  - assert: { exists: { of: '${tMlp}' } }\n  - rescale_outlier_rows_: { target: '${tMlp}', max_ratio: 4.5 }\n  - assert: { finite: { of: '${tMlp}' } }\n  - assert: { rel_delta: { of: '${tMlp}', le: 0.10 } }`;
  if (kind === "svd") return `transforms:\n  - assert: { exists: { of: '${tMlp}' } }\n  - svd_low_rank_: { target: '${tMlp}', energy_fraction: 0.98 }\n  - ties_trim_: { target: '${tMlp}', keep_top_fraction: 0.85 }\n  - assert: { finite: { of: '${tMlp}' } }`;
  if (kind === "spectral") return `transforms:\n  - assert: { exists: { of: '${tAttn}' } }\n  - spectral_norm_cap_: { target: '${tAttn}', max_median_ratio: 2.5 }\n  - assert: { finite: { of: '${tAttn}' } }\n  - assert: { rel_delta: { of: '${tAttn}', le: 0.08 } }`;
  if (kind === "abliterate") return `transforms:\n  - assert: { exists: { of: '${tMlp}' } }\n  - project_out_direction_: { target: '${tMlp}', alpha: 0.25 }\n  - assert: { finite: { of: '${tMlp}' } }`;
  if (kind === "stress") return `transforms:\n  - assert: { exists: { of: '${tMlp}' } }\n  - scale_: { target: '${tMlp}', by: -50.0 }\n  - assert: { finite: { of: '${tMlp}' } }`;
  return DEFAULT_YAML;
}
function setYamlEditor(text) { const a = $("#yamlArea"); if (a) { a.value = text; syncGutter(); renderYamlValidation(validateYaml(text)); } }
function syncGutter() { const a = $("#yamlArea"), g = $("#yamlGutter"); if (!a || !g) return; const n = a.value.split("\n").length; g.innerHTML = Array.from({ length: n }, (_, i) => `<div>${i + 1}</div>`).join(""); g.scrollTop = a.scrollTop; }
function renderYaml() {
  const v = $("#view-yaml");
  v.innerHTML = `
    <div class="toolbar"><span class="badge badge-accent">${icon("file-code")} BrainSurgery-WX</span><span class="section-sub">декларативный план · строгие assertions</span><span style="flex:1"></span>
      <select class="select" id="yamlTemplate" style="width:auto;min-width:170px;height:24px;font-size:11px" aria-label="Шаблон плана">
        <option value="">Шаблон плана…</option>
        <option value="outlier">Очистка выбросов (rescale)</option>
        <option value="svd">SVD-сжатие + TIES trim</option>
        <option value="spectral">Spectral norm cap</option>
        <option value="abliterate">Project out direction</option>
        <option value="stress">Стресс-тест отката (-50x)</option>
      </select>
      <button class="btn btn-sm btn-ghost" id="yamlInsert" ${state.lastYaml ? "" : "disabled"}>${icon("sparkles")} План навыка</button><button class="btn btn-sm btn-ghost" id="yamlReset">${icon("refresh")} Сброс</button></div>
    <div class="yaml-split"><div class="editor"><div class="editor-gutter" id="yamlGutter"></div><textarea class="editor-area" id="yamlArea" spellcheck="false" wrap="off">${esc(state.lastYaml || DEFAULT_YAML)}</textarea></div>
      <div class="yaml-side">
        <div class="card"><div class="card-header"><span class="dot dot-accent"></span><span class="card-title">Валидация инвариантов</span></div><div class="card-body" id="yamlValidation"></div></div>
        <div class="card"><div class="card-header"><span class="dot dot-info"></span><span class="card-title">Применение</span></div><div class="card-body" style="display:flex;flex-direction:column;gap:8px"><button class="btn btn-primary" id="applyGuard" data-run>${icon("shield")} Применить с Fisher guard</button><button class="btn btn-destructive" id="applyRaw" data-run>${icon("alert-tri")} Применить без guard</button><div class="field-hint">Fisher guard и Vespasian auto-rollback настраиваются в Настройках → Безопасность. Инварианты (finite, norm, rel_delta, shape…) проверяются до commit; при нарушении Vespasian выполняет атомарный откат.</div></div></div>
        <div class="alert alert-info">${icon("info")}<div><div class="alert-title">Как это работает</div><div class="alert-desc">План описывает целевые тензоры и операции (scale_, clamp_, svd_low_rank_, project_out_direction_, eamit_batch_…). Каждый план с правками обязан содержать assert. Vespasian проверяет held-out регрессию и откатывает нарушения.</div></div></div>
      </div></div>`;
  const a = $("#yamlArea"); a.addEventListener("input", () => { syncGutter(); renderYamlValidation(validateYaml(a.value)); }); a.addEventListener("scroll", syncGutter);
  syncGutter(); renderYamlValidation(validateYaml(a.value));
  $("#yamlTemplate").onchange = (e) => { if (!e.target.value) return; setYamlEditor(yamlTemplatePlan(e.target.value)); e.target.value = ""; };
  $("#applyGuard").onclick = () => applyYaml(a.value, true);
  $("#applyRaw").onclick = () => applyYaml(a.value, false);
  $("#yamlReset").onclick = () => setYamlEditor(DEFAULT_YAML);
  $("#yamlInsert").onclick = () => { if (state.lastYaml) setYamlEditor(state.lastYaml); };
  enhanceSelects(v);
}
function renderYamlValidation(v) { const box = $("#yamlValidation"); if (!box) return;
  if (!v.asserts.length && v.ok) { box.innerHTML = `<div class="field-hint">Добавьте секцию assertions для проверки инвариантов.</div>`; return; }
  box.innerHTML = `${v.asserts.map(a => `<div class="assert-item">${a.ok ? '<span class="dot dot-success"></span>' : '<span class="dot dot-error"></span>'}<span class="ai-name">${esc(a.name)}</span><span class="badge ${a.ok ? "badge-success" : "badge-error"}">${a.ok ? "ok" : "unknown"}</span><span class="mono" style="color:var(--text-ghost);font-size:10px">L${a.line}</span></div>`).join("")}${v.errors.length ? `<div class="alert alert-error" style="margin-top:8px">${icon("alert-circ")}<div><div class="alert-title">Ошибки плана</div><div class="alert-desc">${v.errors.map(esc).join("<br>")}</div></div></div>` : `<div class="alert alert-success" style="margin-top:8px">${icon("check-circ")}<div><div class="alert-title">План валиден</div><div class="alert-desc">${v.asserts.length} инвариантов распознано</div></div></div>`}`; }

/* ============================================================ RENDER · History */
function renderHistory() {
  const pad = $("#historyPad"); const h = state.history; const cps = h.iterations || [];
  const d = (state.modelInfo && state.modelInfo.diff) || { changed_params: 0, total_params: 0 };
  const dirty = (d.changed_params || 0) > 0;
  const pct = dirty && d.total_params ? (d.changed_params / d.total_params * 100) : 0;
  const deltaTotal = (h.best || 0) - (h.baseline || 0);
  const committedCount = cps.filter(c => c.accepted && !c.rolled_back).length;
  pad.innerHTML = `
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:16px;flex-wrap:wrap"><h2 class="section-title">История и откаты</h2><span class="section-sub">Vespasian · атомарные bit-exact откаты · ${esc(h.model_id || state.model?.id || "")}</span><span style="flex:1"></span>${state.model ? `<button class="btn btn-sm" id="histDiffBtn">${icon("diff")} Diff весов</button>` : ""}${dirty ? `<button class="btn btn-sm btn-destructive" id="histRollbackAll">${icon("undo")} Откатить всё</button>` : ""}<button class="btn btn-sm btn-primary" id="exportBtn">${icon("download")} Экспорт .safetensors</button></div>
    ${state.model ? `<div class="stat-grid" style="margin-bottom:20px">
      <div class="stat"><div class="stat-label">Базовый held-out</div><div class="stat-value">${fmt(h.baseline || 0, 4)}</div><div class="stat-delta is-flat">исходный чекпоинт</div></div>
      <div class="stat"><div class="stat-label">Лучший held-out</div><div class="stat-value">${fmt(h.best || 0, 4)}</div><div class="stat-delta ${deltaTotal <= 0 ? "is-up" : "is-down"}">${deltaTotal > 0 ? "+" : ""}${fmt(deltaTotal, 4)} к базе</div></div>
      <div class="stat"><div class="stat-label">Итераций</div><div class="stat-value">${cps.length}</div><div class="stat-delta is-flat">${committedCount} принято · ${cps.length - committedCount} откатов</div></div>
      <div class="stat"><div class="stat-label">Изменено весов</div><div class="stat-value" style="color:${dirty ? "var(--status-warning)" : "inherit"}">${dirty ? (pct < 0.01 ? "<0.01" : pct.toFixed(2)) + "%" : "0%"}</div><div class="stat-delta is-flat">${fmtInt(d.changed_params || 0)} параметров</div></div>
    </div>` : ""}
    ${cps.length ? `<div class="timeline">${cps.map(cp => { const committed = cp.accepted && !cp.rolled_back; const delta = cp.delta || 0;
      return `<div class="tl-item ${committed ? "ok" : "rb"}"><div class="tl-card"><div class="tl-top"><span class="tl-title">${esc(cp.hypothesis || "итерация " + cp.iteration)}</span><span class="badge ${committed ? "badge-success" : "badge-error"}">${committed ? "COMMITTED" : "ROLLED BACK"}</span><span class="tl-time">${cp.id} · итерация ${cp.iteration}</span></div><div class="tl-metrics"><span>held-out <b>${fmt(cp.previous_best, 4)} → ${fmt(cp.candidate_score, 4)}</b></span><span>Δ <b style="color:${delta <= 0 ? "var(--status-success)" : "var(--status-error)"}">${delta > 0 ? "+" : ""}${fmt(delta, 4)}</b></span><span>KL <b>${fmt(cp.kl_drift, 4)}</b></span><span>изменено <b>${fmt(cp.changed_pct, 2)}%</b></span></div><div class="tl-actions"><button class="btn btn-sm" data-diff="${cp.id}">${icon("diff")} Diff весов</button><button class="btn btn-sm btn-ghost" data-export="${cp.id}">${icon("download")} Экспорт</button>${committed ? `<button class="btn btn-sm btn-destructive" data-rollback="${cp.id}">${icon("undo")} Откатить</button>` : `<span class="badge badge-error">веса восстановлены</span>`}</div>${cp.verdict ? `<div class="field-hint" style="margin-top:8px">${esc(cp.verdict)}</div>` : ""}</div></div>`; }).join("")}</div>` : `<div class="empty" style="min-height:46vh"><div class="empty-icon">${icon("history")}</div><div class="empty-title">История пуста</div><div class="empty-desc">Запустите навык, YAML-план или WSP-цикл — итерации Vespasian появятся здесь.</div></div>`}`;
  $("#exportBtn").onclick = () => exportCheckpoint();
  const hdb = $("#histDiffBtn"); if (hdb) hdb.onclick = () => showDiff("workspace");
  const hra = $("#histRollbackAll"); if (hra) hra.onclick = () => rollbackCheckpoint("workspace");
  $$("#historyPad [data-rollback]").forEach(b => b.onclick = () => rollbackCheckpoint(b.dataset.rollback));
  $$("#historyPad [data-export]").forEach(b => b.onclick = () => exportCheckpoint());
  $$("#historyPad [data-diff]").forEach(b => b.onclick = () => showDiff(b.dataset.diff));
}
/* Real workspace diff (changed tensors, L2/max deltas, SHA-256). */
async function showDiff(id) {
  let d;
  try { d = await invoke("diff"); } catch (e) { toast("error", "Diff недоступен", String(e)); return; }
  if (d.error) { toast("warning", "Модель не загружена", "Загрузите модель, чтобы увидеть diff"); return; }
  const rows = d.tensors || [];
  const body = `
    <div class="kv-grid" style="margin-bottom:12px">
      <div class="kv"><div class="k">Изменено параметров</div><div class="v">${fmtInt(d.changed_params)} / ${fmtInt(d.total_params)}</div></div>
      <div class="kv"><div class="k">Доля изменённых</div><div class="v">${fmt(d.changed_pct, 3)}%</div></div>
      <div class="kv"><div class="k">Тензоров изменено</div><div class="v">${d.tensors_changed}</div></div>
    </div>
    ${rows.length ? `<div class="table-wrap" style="max-height:320px;overflow-y:auto"><table class="table diff-table"><thead><tr><th>Тензор</th><th class="num">Элементов</th><th class="num">Изменено</th><th class="num">L2 Δ</th><th class="num">Max Δ</th><th class="num">Rel Δ</th><th>SHA-256</th></tr></thead><tbody>${rows.map(t => `<tr><td class="mono" style="font-size:11px">${esc(t.name)}</td><td class="num">${fmtInt(t.numel)}</td><td class="num">${fmtInt(t.changed_count)}</td><td class="num">${t.l2_delta}</td><td class="num">${t.max_abs_delta}</td><td class="num">${t.rel_delta}</td><td class="mono" style="font-size:10px">${esc(t.sha256_after)}</td></tr>`).join("")}</tbody></table></div>` : '<div class="field-hint">Весовые тензоры не изменены относительно исходной загрузки (sandbox чист).</div>'}`;
  confirmDialog({ title: "Diff весов · " + id, desc: `Текущее состояние workspace относительно исходных весов модели`, confirmLabel: "Закрыть", bodyHTML: body });
}

/* ============================================================ RENDER · Settings */
const SETTINGS_TABS = [["chat", "Чат", "chat"], ["provider", "Провайдер", "brain"], ["pipeline", "Пайплайн", "sliders"], ["safety", "Безопасность", "shield"], ["styles", "Стили", "palette"], ["data", "Данные", "database"], ["about", "О программе", "info"]];
const PIPELINE_DEFAULTS = { fisher_alpha: 0.15, fisher_freeze: 0.05, improvement_threshold: 0.01, max_kl_drift: 2.0, max_changed_fraction: 0.5, practice_budget: 16 };
function renderSettings() {
  const pad = $("#settingsPad"); const p = state.pipeline || PIPELINE_DEFAULTS;
  const engineReady = !!(state.modelInfo && state.modelInfo.loaded);
  const rangeRow = (id, label, desc, min, max, step, val, unit = "") => `<div class="setting-row"><div class="sr-text"><div class="sr-name">${label}</div><div class="sr-desc">${desc}</div></div><div class="sr-control" style="min-width:200px"><div class="range-row" style="flex:1"><input type="range" class="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}" ${engineReady ? "" : "disabled"} /><span class="range-val" id="${id}Val">${val}${unit}</span></div></div></div>`;
  const backend = state.modelInfo && state.modelInfo.backend;
  pad.innerHTML = `
    <div style="margin-bottom:16px"><h2 class="section-title">Настройки</h2><span class="section-sub">Все параметры WeightX в одном месте</span></div>
    <div class="settings-layout">
      <div class="settings-nav">${SETTINGS_TABS.map(([id, label, ic]) => `<button class="sn-item ${state.settingsTab === id ? "is-active" : ""}" data-stab="${id}">${icon(ic)} ${label}</button>`).join("")}</div>
      <div>
        <div class="settings-pane ${state.settingsTab === "chat" ? "is-active" : ""}" data-pane="chat">
          <div class="card"><div class="card-header">${icon("activity")}<span class="card-title">Лента активности агента</span></div><div class="card-body">
            <div class="settings-card-sub">Как отображаются выполняемые действия в чате. Выбор применяется ко всем сообщениям сразу и сохраняется.</div>
            <div class="feed-variant-grid" id="feedVariant" role="radiogroup" aria-label="Вариант ленты активности">
              ${FEED_VARIANTS.map(v => `<button class="feed-variant-card ${state.feedVariant === v.id ? "is-active" : ""}" data-feed="${v.id}" role="radio" aria-checked="${state.feedVariant === v.id}">
                <span class="fv-ic">${icon(v.icon)}</span>
                <span class="fv-text"><span class="fv-name">${esc(v.label)}${v.id === "timeline" ? ' <span class="badge badge-accent" style="margin-left:4px">по умолчанию</span>' : ""}</span><span class="fv-desc">${esc(v.hint)}</span></span>
                <span class="fv-check">${icon("check")}</span>
              </button>`).join("")}
            </div>
          </div></div>
          <div class="card"><div class="card-header">${icon("panel-left")}<span class="card-title">Сайдбар</span></div><div class="card-body">
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Автоскрытие сайдбара</div><div class="sr-desc">Сайдбар полностью скрыт. При наведении на левый край он плавно появляется; клик по пункту навигации снова скрывает его. Доступно на широких экранах.</div></div><div class="sr-control"><button class="switch ${$("#app").dataset.sidebar === "hidden" ? "is-on" : ""}" id="sidebarAutoHide" role="switch" aria-checked="${$("#app").dataset.sidebar === "hidden"}" aria-label="Автоскрытие сайдбара"></button></div></div>
          </div></div>
          <div class="card"><div class="card-header">${icon("keyboard")}<span class="card-title">Клавиши composer</span></div><div class="card-body">
            ${[["Запуск цели или навыка", ["↵"]], ["Новая строка", ["Ctrl", "↵"]], ["Меню навыков", ["/"]]].map(([n, keys]) => `<div class="setting-row"><div class="sr-text"><div class="sr-name">${n}</div></div><div class="sr-control">${keys.map(k => `<kbd class="kbd">${k}</kbd>`).join(" ")}</div></div>`).join("")}
          </div></div>
        </div>
        <div class="settings-pane ${state.settingsTab === "provider" ? "is-active" : ""}" data-pane="provider">
          <div class="card"><div class="card-header">${icon("brain")}<span class="card-title">Мозг (provider)</span></div><div class="card-body">
            <div class="settings-card-sub">Reasoning-модель, которая управляет WSP-инструментами. Основной выбор провайдера — в отдельном разделе «Провайдер».</div>
            <div class="field"><label class="field-label">Тип провайдера</label><select class="select" id="setProvider">${state.providerKinds.map(pp => `<option value="${pp.kind}" ${state.provider.kind === pp.kind ? "selected" : ""}>${esc(pp.label)}</option>`).join("")}</select></div>
            <div class="field"><label class="field-label">Модель</label><input class="input" id="setProviderModel" placeholder="пусто = по умолчанию" value="${esc(state.provider.model)}" /></div>
            <div class="field"><label class="field-label">API key</label><input class="input" id="setApiKey" type="password" placeholder="или переменная окружения" value="${esc(state.provider.apiKey)}" /></div>
            <div class="field"><label class="field-label">base_url</label><input class="input" id="setBaseUrl" placeholder="ollama / lm-studio / vllm" value="${esc(state.provider.baseUrl)}" /></div>
            <button class="btn btn-primary" id="setProviderBtn">${icon("check")} Сохранить провайдера</button>
          </div></div>
          <div class="card"><div class="card-header">${icon("network")}<span class="card-title">Сохранённые конфигурации</span></div><div class="card-body">
            <div class="settings-card-sub">Конфигурации с моделью / ключом / base_url. Активация реально переключает мозг движка; в выборе Brain видны только настроенные провайдеры.</div>
            <div id="savedConfigsBodySettings">${savedConfigsHTML()}</div>
          </div></div>
        </div>
        <div class="settings-pane ${state.settingsTab === "pipeline" ? "is-active" : ""}" data-pane="pipeline">
          <div class="card"><div class="card-header">${icon("sliders")}<span class="card-title">Параметры пайплайна</span></div><div class="card-body">
            <div class="settings-card-sub">Значения применяются к движку (Fisher guard + Vespasian) сразу после изменения.${engineReady ? "" : " Загрузите модель, чтобы управлять параметрами."}</div>
            ${rangeRow("pFisherAlpha", "Fisher damping α", "Демпфирование Fisher-карты важных весов (IWCKEdit)", 0, 1, 0.05, p.fisher_alpha)}
            ${rangeRow("pFisherFreeze", "Fisher freeze top", "Доля наиболее важных весов, замороженных при правках", 0, 0.5, 0.01, p.fisher_freeze)}
            ${rangeRow("pThreshold", "Порог улучшения held-out", "Минимальное улучшение score для commit (Vespasian)", 0, 0.1, 0.001, p.improvement_threshold)}
            ${rangeRow("pKl", "Max KL drift", "Максимальный KL-дрейф относительно контрольного корпуса", 0, 2, 0.05, p.max_kl_drift)}
            ${rangeRow("pChanged", "Max changed fraction", "Максимальная доля изменённых параметров", 0.01, 1, 0.01, p.max_changed_fraction)}
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Practice budget</div><div class="sr-desc">Бюджет вызовов practice-оценки за сессию</div></div><div class="sr-control"><div class="segmented" id="pBudget">${[8, 16, 32, 64].map(n => `<button class="segmented-item ${p.practice_budget === n ? "is-active" : ""}" data-budget="${n}" ${engineReady ? "" : "disabled"}>${n}</button>`).join("")}</div></div></div>
          </div></div>
          <div class="card"><div class="card-header">${icon("cpu")}<span class="card-title">Вычислительный backend</span></div><div class="card-body">
            <div class="setting-row"><div class="sr-text"><div class="sr-name">${backend === "torch" ? "PyTorch sidecar" : "Rust core"}</div><div class="sr-desc">Гибридный движок: нативный Rust для оркестрации и малых моделей; PyTorch sidecar (MKL/oneDNN + CUDA) автоматически подключается для моделей ≥5M параметров. Переопределяется переменной WEIGHTX_BACKEND=rust|torch|auto.</div></div><div class="sr-control"><span class="badge badge-accent">${backend === "torch" ? "torch · активен" : "rust · активен"}</span></div></div>
          </div></div>
        </div>
        <div class="settings-pane ${state.settingsTab === "safety" ? "is-active" : ""}" data-pane="safety">
          <div class="card"><div class="card-header">${icon("shield")}<span class="card-title">Контуры безопасности</span></div><div class="card-body">
            ${[["fisher", "Fisher guard", "Защита важных весов при closed-form правках (IWCKEdit)"], ["vespasian", "Vespasian auto-rollback", "Атомарный откат при нарушении инвариантов и регрессии"]].map(([k, n, d]) => `<div class="setting-row"><div class="sr-text"><div class="sr-name">${n}</div><div class="sr-desc">${d}</div></div><div class="sr-control"><button class="switch ${state.guards[k] ? "is-on" : ""}" data-guard="${k}" role="switch" aria-checked="${state.guards[k]}" aria-label="${n}"></button></div></div>`).join("")}
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Sandbox-изоляция</div><div class="sr-desc">Встроена в движок: правки применяются к живой копии весов, исходные тензоры сохраняются для bit-exact отката и diff. Экспорт на диск — только по команде.</div></div><div class="sr-control"><button class="switch is-on" disabled role="switch" aria-checked="true" aria-label="Sandbox-изоляция"></button></div></div>
            <div class="alert alert-warning" style="margin-top:12px">${icon("alert-tri")}<div><div class="alert-title">Отключение guard снижает безопасность</div><div class="alert-desc">Без Vespasian регрессии не откатываются автоматически. Рекомендуется оставить включённым.</div></div></div>
          </div></div>
        </div>
        <div class="settings-pane ${state.settingsTab === "styles" ? "is-active" : ""}" data-pane="styles">
          <div class="card"><div class="card-header">${icon("palette")}<span class="card-title">Стили оформления</span></div><div class="card-body">
            <div class="settings-card-sub">Стиль меняет акцентный цвет и поверхности всего интерфейса. Доступен также в титулбаре.</div>
            <div class="styles-grid" id="stylesGridSettings"></div>
          </div></div>
          <div class="card"><div class="card-header">${icon("layers")}<span class="card-title">Интерактивный фон</span></div><div class="card-body">
            <div class="settings-card-sub">Живой холст позади интерфейса: реагирует на движение курсора, клики и работу движка. Также доступен в меню стилей и Ctrl K.</div>
            <div class="bg-modes-grid" id="bgModesGridSettings"></div>
          </div></div>
          <div class="card"><div class="card-header">${icon("sparkles")}<span class="card-title">Анимации и градиенты</span></div><div class="card-body">
            <div class="settings-card-sub">Насыщенность оформления применяется мгновенно и сохраняется. Системная настройка «уменьшить движение» имеет приоритет.</div>
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Анимации</div><div class="sr-desc" id="motionDesc">${state.motion === "full" ? "Полные: фоновая сеть, aurora, живой градиент логотипа, входы экранов, sheen и пульсации." : state.motion === "calm" ? "Умеренные: только короткие функциональные переходы, без фоновых циклов." : "Выключены: статичный интерфейс, мгновенные отклики."}</div></div><div class="sr-control"><div class="segmented" id="motionLevel" role="radiogroup" aria-label="Уровень анимаций">${[["full", "Полные"], ["calm", "Умеренные"], ["off", "Выкл"]].map(([id, label]) => `<button class="segmented-item ${state.motion === id ? "is-active" : ""}" data-motion-set="${id}" role="radio" aria-checked="${state.motion === id}">${label}</button>`).join("")}</div></div></div>
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Градиенты</div><div class="sr-desc">Градиентные поверхности, акценты, кнопки и фоновое свечение. Выключение делает все заливки плоскими.</div></div><div class="sr-control"><button class="switch ${state.gradients ? "is-on" : ""}" id="gradToggle" role="switch" aria-checked="${state.gradients}" aria-label="Градиенты"></button></div></div>
          </div></div>
        </div>
        <div class="settings-pane ${state.settingsTab === "data" ? "is-active" : ""}" data-pane="data">
          <div class="card"><div class="card-header">${icon("database")}<span class="card-title">Данные и экспорт</span></div><div class="card-body">
            <div class="field"><label class="field-label">Путь экспорта по умолчанию</label><input class="input" id="exportPath" value="${esc(state.exportPath)}" /></div>
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Скаченные модели</div><div class="sr-desc">${localEntries().length} моделей в локальном кэше (HF hub + checkpoints)</div></div><div class="sr-control"><button class="btn btn-sm" id="openModelsBtn">${icon("cloud")} Открыть панель</button></div></div>
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Сбросить локальные настройки</div><div class="sr-desc">Стиль, провайдер, guards и параметры пайплайна</div></div><div class="sr-control"><button class="btn btn-sm btn-destructive" id="resetSettings">${icon("trash")} Сбросить</button></div></div>
          </div></div>
          <div class="alert alert-info">${icon("cpu")}<div><div class="alert-title">Нативный движок</div><div class="alert-desc">Данные поступают из weightx-core (Rust) через Tauri invoke/listen: реальные тензоры, L1/L2-профили, YAML-хирургия, EAMIT и откаты Vespasian. Тяжёлые вычисления делегируются PyTorch sidecar. Каталог Hugging Face подключён к реальному API (hf_search + hf-hub загрузка).</div></div></div>
        </div>
        <div class="settings-pane ${state.settingsTab === "about" ? "is-active" : ""}" data-pane="about">
          <div class="card"><div class="card-header">${icon("info")}<span class="card-title">О WeightX</span></div><div class="card-body">
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Версия</div><div class="sr-desc">WeightX v0.4.0 · hybrid Rust + PyTorch</div></div><div class="sr-control"><span class="badge">v0.4.0</span></div></div>
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Лицензия</div><div class="sr-desc">GNU General Public License v3.0 or later</div></div><div class="sr-control"><span class="badge">GPLv3</span></div></div>
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Архитектура</div><div class="sr-desc">Provider Layer → Tool Layer (L1/L2/L3 + YAML) → Execution & Verification (Vespasian)</div></div><div class="sr-control"><span class="badge badge-accent">3 слоя</span></div></div>
            <div class="setting-row"><div class="sr-text"><div class="sr-name">LLM-провайдеры</div><div class="sr-desc" id="aboutStackDesc">загрузка…</div></div><div class="sr-control"><button class="btn btn-sm" id="aboutStackLink">${icon("cloud")} cersei</button></div></div>
            <div class="setting-row"><div class="sr-text"><div class="sr-name">Горячие клавиши</div><div class="sr-desc">Полный список сочетаний</div></div><div class="sr-control"><button class="btn btn-sm" id="aboutHotkeys">${icon("keyboard")} Открыть</button></div></div>
          </div></div>
        </div>
      </div>
    </div>`;
  $$("#settingsPad [data-stab]").forEach(b => b.onclick = () => { state.settingsTab = b.dataset.stab; renderSettings(); });
  // chat: feed variant + sidebar auto-hide
  $$("#feedVariant [data-feed]").forEach(b => b.onclick = () => { state.feedVariant = b.dataset.feed; store.set("feedVariantV2", state.feedVariant);
    repaintAllFeeds(); renderSettings(); toast("info", "Лента: " + (FEED_VARIANTS.find(v => v.id === state.feedVariant)?.label || state.feedVariant), "Применено ко всем сообщениям чата", null, 1600); });
  $("#sidebarAutoHide").onclick = () => { setSidebarHidden($("#app").dataset.sidebar !== "hidden"); };
  // provider
  $("#setProviderBtn").onclick = async () => { const kind = $("#setProvider").value; const cfg = { model: $("#setProviderModel").value.trim(), apiKey: $("#setApiKey").value.trim(), baseUrl: $("#setBaseUrl").value.trim() };
    await saveAndActivate(kind, cfg);
  };
  $("#setProvider").onchange = () => { const c = providerCfg($("#setProvider").value); $("#setProviderModel").value = c.model || ""; $("#setApiKey").value = c.apiKey || ""; $("#setBaseUrl").value = c.baseUrl || ""; };
  bindSavedConfigs($("#settingsPad"));
  // pipeline sliders → real engine params
  const bindRange = (id, key, unit = "") => { const el = $("#" + id); if (!el) return; el.oninput = () => { state.pipeline = state.pipeline || { ...PIPELINE_DEFAULTS }; state.pipeline[key] = parseFloat(el.value); $("#" + id + "Val").textContent = el.value + unit; store.set("pipeline", state.pipeline); pushPipelineParams(); }; };
  bindRange("pFisherAlpha", "fisher_alpha"); bindRange("pFisherFreeze", "fisher_freeze"); bindRange("pThreshold", "improvement_threshold"); bindRange("pKl", "max_kl_drift"); bindRange("pChanged", "max_changed_fraction");
  $$("#pBudget [data-budget]").forEach(b => b.onclick = () => { state.pipeline = state.pipeline || { ...PIPELINE_DEFAULTS }; state.pipeline.practice_budget = +b.dataset.budget; store.set("pipeline", state.pipeline); pushPipelineParams(); renderSettings(); });
  // safety
  $$("#settingsPad [data-guard]").forEach(b => b.onclick = () => { const k = b.dataset.guard; state.guards[k] = !state.guards[k]; store.set("guards", state.guards); b.classList.toggle("is-on", state.guards[k]); b.setAttribute("aria-checked", state.guards[k]); updateGuardUI(); toast(state.guards[k] ? "success" : "warning", (state.guards[k] ? "Включено: " : "Выключено: ") + k, state.guards[k] ? null : "Постур безопасности снижен", null, 1800); });
  // styles
  renderStylesGrids();
  // effects: motion level + gradients
  $$("#motionLevel [data-motion-set]").forEach(b => b.onclick = () => { state.motion = b.dataset.motionSet; store.set("motion", state.motion); applyEffects(); renderSettings();
    toast("info", "Анимации: " + (b.textContent.trim()), state.motion === "off" ? "Интерфейс статичен" : state.motion === "calm" ? "Фоновые циклы отключены" : "Полное оформление включено", null, 1600); });
  $("#gradToggle").onclick = () => { state.gradients = !state.gradients; store.set("gradients", state.gradients); applyEffects();
    const sw = $("#gradToggle"); sw.classList.toggle("is-on", state.gradients); sw.setAttribute("aria-checked", String(state.gradients));
    toast("info", state.gradients ? "Градиенты включены" : "Градиенты выключены", state.gradients ? "Градиентные заливки восстановлены" : "Все поверхности плоские", null, 1600); };
  // data
  $("#exportPath").onchange = e => { const v = e.target.value.trim(); if (!/\.safetensors$/.test(v)) { toast("error", "Некорректный путь", "Ожидается файл .safetensors"); e.target.value = state.exportPath; return; } state.exportPath = v; store.set("exportPath", v); toast("success", "Путь экспорта сохранён", v, null, 1800); };
  $("#openModelsBtn").onclick = () => { navigate("models"); };
  $("#resetSettings").onclick = async () => { const ok = await confirmDialog({ title: "Сбросить настройки?", desc: "Стиль, фон, провайдер, guards и параметры пайплайна вернутся к значениям по умолчанию.", confirmLabel: "Сбросить", danger: true }); if (!ok) return; ["style", "bgMode", "provider", "guards", "pipeline", "providerConfigs", "feedVariantV2", "sidebarHidden", "motion", "gradients"].forEach(k => localStorage.removeItem("wx." + k)); state.style = "graphite"; state.bgMode = "neural"; state.provider = { kind: "autonomous", model: "", apiKey: "", baseUrl: "" }; state.providerConfigs = { autonomous: { model: "", apiKey: "", baseUrl: "" } }; state.providerModelsCache = {}; state.guards = { fisher: true, vespasian: true, sandbox: true }; state.pipeline = null; state.feedVariant = "timeline"; state.motion = "full"; state.gradients = true; setSidebarHidden(false, false); applyStyle(); applyEffects(); updateGuardUI(); updateProviderStatus(); updateBrainBtn(); await refreshModelInfo(); renderSettings(); toast("success", "Настройки сброшены", "Значения по умолчанию восстановлены"); };
  $("#aboutHotkeys").onclick = () => navigate("hotkeys");
  enhanceSelects(pad);
  // Real provider stack facts from the backend (cersei SDK usage).
  (async () => {
    try {
      const s = await invoke("provider_stack");
      const d = $("#aboutStackDesc");
      if (d) d.textContent = `${s.sdk} SDK v${s.version} · pacifio/cersei — используется для всех API-провайдеров (OpenAI, OpenRouter, Ollama, LM Studio, vLLM, Anthropic). CLI-агенты подключаются напрямую.`;
      const b = $("#aboutStackLink");
      if (b) b.onclick = () => invoke("open_external", { url: s.repository }).catch(e => toast("error", "Не удалось открыть браузер", String(e)));
    } catch { const d = $("#aboutStackDesc"); if (d) d.textContent = "сведения недоступны"; }
  })();
}

/* ============================================================ RENDER · Hotkeys */
const HOTKEYS = [
  ["Навигация", [["Чат", ["1"]], ["Тензоры", ["2"]], ["Диагностика", ["3"]], ["YAML-план", ["4"]], ["История", ["5"]], ["Модели", ["6"]], ["Провайдер", ["7"]], ["Настройки", ["8"]], ["Горячие клавиши", ["9"]]]],
  ["Действия", [["Командная палитра", ["Ctrl", "K"]], ["Запуск composer", ["↵"]], ["Новая строка в composer", ["Ctrl", "↵"]], ["Меню навыков", ["/"]], ["Экспорт .safetensors", ["Ctrl", "E"]], ["Свернуть панель", ["Ctrl", "B"]], ["Инспектор", ["Ctrl", "I"]], ["Закрыть оверлей", ["Esc"]]]],
  ["Палитра и меню навыков", [["Перемещение", ["↑", "↓"]], ["Выбор / выполнение", ["↵"]], ["Закрыть", ["Esc"]]]]
];
function renderHotkeys() {
  const pad = $("#hotkeysPad");
  pad.innerHTML = `<div style="margin-bottom:16px"><h2 class="section-title">Горячие клавиши</h2><span class="section-sub">Управление WeightX с клавиатуры</span></div>
    <div class="hotkeys-grid">${HOTKEYS.map(([group, rows]) => `<div class="card"><div class="card-header">${icon("keyboard")}<span class="card-title">${group}</span></div><div class="card-body"><table class="shortcuts-table"><tbody>${rows.map(([name, keys]) => `<tr><td>${name}</td><td>${keys.map(k => `<kbd class="kbd">${k}</kbd>`).join(" ")}</td></tr>`).join("")}</tbody></table></div></div>`).join("")}</div>
    <div class="alert alert-info" style="margin-top:16px">${icon("info")}<div><div class="alert-title">Подсказка</div><div class="alert-desc">Цифровые клавиши 1–9 переключают экраны, когда фокус не в поле ввода. Командная палитра (Ctrl K) даёт доступ ко всем действиям, навыкам, моделям, фонам и стилям.</div></div></div>`;
}

/* ============================================================ RENDER · Inspector */
function renderInspector() {
  const box = $("#inspectorBody"); if (!box) return;
  if (state.view === "tensors" && state.selectedTensor) { const t = state.selectedTensor;
    const cached = state.tensorDescCache && state.tensorDescCache.name === t.name ? state.tensorDescCache.desc : null;
    const spec = cached && cached.top_singular_values ? cached.top_singular_values : null;
    box.innerHTML = `<div class="panel-title" style="margin-bottom:10px">Тензор</div><div class="mono" style="font-size:12px;word-break:break-all;margin-bottom:12px">${esc(t.name)}</div>${[["Роль", ROLE_LABEL[t.role] || "·"], ["Форма", t.shape.join("×")], ["Dtype", t.dtype], ["Норма", fmt(t.norm)], ["Элементов", fmtInt(t.elements)], ["Изменён", t.modified ? "да" : "нет"]].map(([k, v]) => `<div class="setting-row"><div class="sr-text"><div class="sr-desc">${k}</div></div><span class="mono" style="font-size:12px">${v}</span></div>`).join("")}${spec ? `<div class="chart-box" style="margin-top:12px"><div class="chart-title">Спектр</div>${spectrumChart(spec)}</div>` : ""}`;
  } else if (state.view === "diagnostics" && state.selectedLayer) { const l = state.selectedLayer;
    box.innerHTML = `<div class="panel-title" style="margin-bottom:10px">Слой ${l.layer_idx}</div><div class="mono" style="font-size:11px;word-break:break-all;margin-bottom:10px">${esc(l.layer || "")}</div>${l.isBottleneck ? '<span class="badge badge-warning" style="margin-bottom:10px">bottleneck</span>' : l.isRedundant ? '<span class="badge badge-info" style="margin-bottom:10px">redundant</span>' : '<span class="badge badge-success" style="margin-bottom:10px">ok</span>'}${LAYER_METRICS.map(m => `<div class="setting-row"><div class="sr-text"><div class="sr-desc">${m.label}</div></div><span class="mono" style="font-size:12px">${fmt(l[m.key] || 0, m.d)}</span></div>`).join("")}`;
  } else {
    const v = (state.modelInfo && state.modelInfo.vespasian) || {};
    box.innerHTML = `<div class="empty" style="padding:24px"><div class="empty-icon">${icon("eye")}</div><div class="empty-title">Инспектор</div><div class="empty-desc">Выберите тензор или слой, чтобы увидеть детали. Все 13 метрик L2 доступны здесь.</div></div><div class="panel-title" style="margin:12px 0 8px">Сессия</div>${[["Модель", state.model ? state.model.id : "—"], ["Backend", backendLabel(state.modelInfo && state.modelInfo.backend)], ["Итераций Vespasian", String(v.iteration || 0)], ["Тензоров", state.model ? fmtInt(state.tensors.length) : "—"], ["Vespasian", state.guards.vespasian ? "активен" : "выкл"]].map(([k, val]) => `<div class="setting-row"><div class="sr-text"><div class="sr-desc">${k}</div></div><span class="mono" style="font-size:12px">${esc(val)}</span></div>`).join("")}`;
  }
}

/* ============================================================ INIT */
/* Re-apply user-saved pipeline params after a fresh model load (engine resets). */
function applySavedPipeline() {
  const saved = store.get("pipeline", null);
  if (saved && state.modelInfo && state.modelInfo.loaded) { state.pipeline = { ...PIPELINE_DEFAULTS, ...saved }; pushPipelineParams(); }
}
async function loadBackendData() {
  const [sks, kinds, cat] = await Promise.all([invoke("skills"), invoke("provider_kinds"), invoke("catalog")]);
  state.skills = sks;
  state.providerKinds = kinds.map(full => { const kind = full.split(" ")[0]; const meta = PROVIDER_META[kind] || { label: full, icon: "network" }; return { kind, full, label: meta.label, icon: meta.icon }; });
  state.catalog = cat;
  if (!state.providerKinds.some(p => p.kind === state.provider.kind)) state.provider = { kind: "autonomous", model: "", apiKey: "", baseUrl: "" };
  // Real host resources → invisible catalog budget (f32 weights must fit in
  // half the RAM and in the free disk space of the HF cache drive).
  try {
    const r = await invoke("system_resources");
    state.sysResources = r;
    const ramBudget = (r.ram_total || 0) * 0.5;
    const diskBudget = r.disk_free || 0;
    const budget = Math.min(ramBudget, diskBudget);
    state.maxParams = budget > 0 ? Math.floor(budget / 4) : 0;
  } catch { state.maxParams = 0; }
}
async function onModelReady(msg) {
  state.diag = null; state.selectedTensor = null; state.selectedLayer = null; state.tensorDescCache = null;
  await Promise.all([refreshModelInfo(), refreshCatalog(), refreshTensors(), refreshHistory()]);
  applySavedPipeline();
  updateModelChip();
  renderModels(); if (state.view === "tensors") renderTensors(); if (state.view === "diagnostics") renderDiagnostics(); if (state.view === "history") renderHistory();
  if ($("#app").dataset.inspector === "open") renderInspector();
  if (msg) toast("success", "Модель готова", msg, null, 3600);
}
function init() {
  applyStyle();
  matchMedia("(prefers-color-scheme: light)").addEventListener?.("change", () => { if (state.style === "system") applyStyle(); });
  bgInit();
  applyEffects();
  if (innerWidth <= 980) { $("#app").dataset.sidebar = "collapsed"; }
  else if (store.get("sidebarHidden", false)) { setSidebarHidden(true, false); }
  updateGuardUI();
  const tick = () => { $("#sbClock").textContent = new Date().toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" }); };
  tick(); setInterval(tick, 15000);
  // Real engine events
  listen("log", e => handleLog(String(e.payload)));
  listen("model-loaded", e => onModelReady(String(e.payload)));
  listen("model-error", e => toast("error", "Ошибка загрузки модели", String(e.payload), null, 6000));
  listen("status-changed", () => refreshModelInfo());
  (async () => {
    if (!hasTauri()) {
      renderSession(); renderYaml(); renderModelsView(); renderHistory(); renderHotkeys(); renderSettings();
      toast("warning", "Движок недоступен", "Откройте WeightX Desktop (wx desktop) — интерфейс работает только внутри нативного приложения.", null, 12000);
      navigate("session");
      return;
    }
    try { await loadBackendData(); }
    catch (e) { toast("error", "Не удалось получить данные движка", String(e), null, 8000); }
    renderProviderView();
    renderSession(); renderYaml(); renderModelsView(); renderHistory(); renderHotkeys(); renderSettings();
    await refreshModelInfo();
    if (state.modelInfo && state.modelInfo.loaded) { await Promise.all([refreshTensors(), refreshHistory()]); applySavedPipeline(); }
    updateModelChip(); renderInspector();
    navigate("session");
    setTimeout(() => $("#composerInput")?.focus(), 400);
  })();
}
init();