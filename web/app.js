const GB = 1e9;
const PAGE_FIND = 18;
const PAGE_EXPLORE = 40;

const QUANT_QUALITY = {
  F32: 1, F16: 0.99, BF16: 0.99, Q8_0: 0.96, Q6_K: 0.92,
  Q5_K_M: 0.88, Q5_K_S: 0.86, Q5_1: 0.85, Q5_0: 0.84,
  Q4_K_M: 0.8, Q4_K_S: 0.77, Q4_1: 0.75, Q4_0: 0.73,
  Q3_K_L: 0.66, Q3_K_M: 0.62, Q3_K_S: 0.57, Q2_K: 0.46,
};

const USECASES = {
  chat: { label: "chat and writing" },
  code: { label: "coding" },
  long: { label: "long documents" },
  vision: { label: "vision" },
  embedding: { label: "embeddings" },
};

const state = {
  mode: "find",
  gpu: "l4",
  ram: 32,
  ctx: 8192,
  usecase: "chat",
  priority: "balanced",
  allowPartial: false,
  q: "",
  fitFilter: "runnable",
  archFilter: "all",
  quantFilter: "all",
  sort: "recommended",
  sizes: [],
  sizeMenuOpen: false,
  visible: PAGE_FIND,
  open: null,
};

const SIZE_CLASSES = [
  { id: "micro", name: "Micro", hint: "<300M params", max: 3e8 },
  { id: "tiny", name: "Tiny", hint: "300M–1B params", max: 1e9 },
  { id: "small", name: "Small", hint: "1B–3B params", max: 3e9 },
  { id: "compact", name: "Compact", hint: "3B–7B params", max: 7e9 },
  { id: "medium", name: "Medium", hint: "7B–14B params", max: 14e9 },
  { id: "large", name: "Large", hint: "14B–32B params", max: 32e9 },
  { id: "xl", name: "XL", hint: "32B–70B params", max: 70e9 },
  { id: "xxl", name: "XXL", hint: "70B–120B params", max: 120e9 },
  { id: "huge", name: "Huge", hint: "120B–400B params", max: 400e9 },
  { id: "frontier", name: "Frontier", hint: ">400B params", max: Infinity },
  { id: "unknown", name: "Unknown", hint: "missing parameter count" },
];
let presentSizeIds = [];

function availableSizeClasses() {
  return SIZE_CLASSES.filter((cls) => presentSizeIds.includes(cls.id));
}

function defaultSizes() {
  return availableSizeClasses().filter((cls) => cls.id !== "unknown").map((cls) => cls.id);
}

function collectPresentSizes() {
  const seen = new Set(catalog.models.map((m) => sizeClassId(m.params)));
  presentSizeIds = SIZE_CLASSES.map((cls) => cls.id).filter((id) => seen.has(id));
}

let manifest;
let hardware;
let catalog;
let quality = { by_ref: {} };

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

function safeUrl(url) {
  const trimmed = String(url || "").trim().replace(/&amp;/g, "&");
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  if (trimmed.startsWith("/assets/") || trimmed.startsWith("/library/")) return `https://ollama.com${trimmed}`;
  return "";
}

function inlineMd(text) {
  let out = text;
  out = out.replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`);
  out = out.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, url) => {
    const href = safeUrl(url);
    return href ? `<img src="${esc(href)}" alt="${alt}" loading="lazy">` : "";
  });
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => {
    const href = safeUrl(url);
    return href ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${label}</a>` : label;
  });
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  return out;
}

function splitTableRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((cell) => cell.trim().replace(/\u000b/g, " "));
}

function isTableSep(cells) {
  return cells.length > 0 && cells.every((cell) => /^:?-{1,}:?$/.test(cell.replace(/\s/g, "")));
}

function isTableLine(line) {
  const trimmed = line.trim();
  return trimmed.startsWith("|") && trimmed.includes("|", 1);
}

function tableCellHtml(cell) {
  return inlineMd(cell.replace(/&lt;br\s*\/?&gt;/gi, "<br>"));
}

function renderTable(tableLines) {
  const rows = tableLines.map(splitTableRow);
  if (!rows.length) return "";
  let head = rows[0];
  let start = 1;
  if (rows[1] && isTableSep(rows[1])) start = 2;
  const body = rows.slice(start).filter((row) => !isTableSep(row) && row.some((cell) => cell));
  const cols = Math.max(head.length, ...body.map((row) => row.length), 1);
  const pad = (row) => {
    const next = row.slice(0, cols);
    while (next.length < cols) next.push("");
    return next;
  };
  head = pad(head);
  const thead = `<thead><tr>${head.map((cell) => `<th>${tableCellHtml(cell)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${body.map((row) =>
    `<tr>${pad(row).map((cell) => `<td>${tableCellHtml(cell)}</td>`).join("")}</tr>`
  ).join("")}</tbody>`;
  return `<div class="md-table"><table>${thead}${tbody}</table></div>`;
}

function mdToHtml(src) {
  const slots = [];
  const keep = (html) => {
    slots.push(html);
    return `\0${slots.length - 1}\0`;
  };
  let text = String(src || "").replace(/\r\n/g, "\n");
  text = text.replace(/```[^\n]*\n?([\s\S]*?)```/g, (_, code) =>
    keep(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`));
  text = text.replace(/<img\b([^>]*)\/?>/gi, (_, attrs) => {
    const srcMatch = /src=["']([^"']+)["']/i.exec(attrs);
    const altMatch = /alt=["']([^"']*)["']/i.exec(attrs);
    const href = srcMatch ? safeUrl(srcMatch[1]) : "";
    if (!href) return "";
    return keep(`<img src="${esc(href)}" alt="${esc(altMatch ? altMatch[1] : "")}" loading="lazy">`);
  });
  text = esc(text);
  const lines = text.split("\n");
  const out = [];
  let para = [];
  let list = null;
  const flushPara = () => {
    if (!para.length) return;
    out.push(`<p>${inlineMd(para.join("\n")).replace(/\n/g, "<br>")}</p>`);
    para = [];
  };
  const flushList = () => {
    if (!list) return;
    out.push(`<${list.tag}>${list.items.map((item) => `<li>${inlineMd(item)}</li>`).join("")}</${list.tag}>`);
    list = null;
  };
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (isTableLine(line)) {
      flushPara();
      flushList();
      const tableLines = [];
      while (i < lines.length && isTableLine(lines[i])) {
        tableLines.push(lines[i]);
        i += 1;
      }
      i -= 1;
      out.push(renderTable(tableLines));
      continue;
    }
    if (line.includes("\0")) {
      flushPara();
      flushList();
      out.push(line);
      continue;
    }
    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      flushPara();
      flushList();
      const level = heading[1].length;
      out.push(`<h${level}>${inlineMd(heading[2])}</h${level}>`);
      continue;
    }
    if (/^[-*]{3,}$/.test(line.trim())) {
      flushPara();
      flushList();
      out.push("<hr>");
      continue;
    }
    const quote = /^&gt;\s?(.*)$/.exec(line);
    if (quote) {
      flushPara();
      flushList();
      out.push(`<blockquote>${inlineMd(quote[1])}</blockquote>`);
      continue;
    }
    const ul = /^[-*]\s+(.+)$/.exec(line);
    if (ul) {
      flushPara();
      if (!list || list.tag !== "ul") {
        flushList();
        list = { tag: "ul", items: [] };
      }
      list.items.push(ul[1]);
      continue;
    }
    const ol = /^\d+\.\s+(.+)$/.exec(line);
    if (ol) {
      flushPara();
      if (!list || list.tag !== "ol") {
        flushList();
        list = { tag: "ol", items: [] };
      }
      list.items.push(ol[1]);
      continue;
    }
    if (!line.trim()) {
      flushPara();
      flushList();
      continue;
    }
    flushList();
    para.push(line);
  }
  flushPara();
  flushList();
  return out.join("").replace(/\0(\d+)\0/g, (_, i) => slots[Number(i)] || "");
}

function fmtPushed(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function paramHeadline(row) {
  if (row.ple_bytes && row.params > (row.active_params || 0) * 1.1) {
    return `${fmtParams(row.active_params)} effective · ${fmtParams(row.params)} with embeddings`;
  }
  return `${fmtParams(row.params)} parameters`;
}

function fmtParams(n) {
  if (!n) return "—";
  if (n >= 1e12) return `${(n / 1e12).toFixed(n >= 10e12 ? 0 : 1)}T`;
  if (n >= 1e9) {
    const value = n / 1e9;
    return `${value >= 10 ? value.toFixed(0) : value.toFixed(1).replace(".0", "")}B`;
  }
  return `${(n / 1e6).toFixed(0)}M`;
}

function fmtBytes(bytes) {
  if (bytes == null) return "—";
  const gb = bytes / GB;
  return gb < 1 ? `${Math.round(gb * 1000)} MB` : `${gb.toFixed(gb >= 10 ? 1 : 2)} GB`;
}

function fmtCtx(ctx) {
  if (!ctx) return "—";
  if (ctx < 1024) return String(ctx);
  if (ctx % 1024 === 0) return `${ctx / 1024}k`;
  const k = ctx / 1024;
  return `${Number.isInteger(k) ? k : k.toFixed(1)}k`;
}

function attnLabel(row) {
  if (row.n_global_layers && row.sliding_window) {
    return `hybrid ${fmtCtx(row.sliding_window)} SWA + global`;
  }
  if (row.sliding_window) return `${fmtCtx(row.sliding_window)} sliding window`;
  return "dense attention";
}

function fmtSpeed(speed) {
  if (speed == null || !Number.isFinite(speed)) return "Not modeled";
  return `${speed >= 100 ? speed.toFixed(0) : speed.toFixed(1)} tok/s`;
}

function gpuById(id) {
  return hardware.gpus.find((gpu) => gpu.id === id);
}

function groupName(gpu) {
  if (gpu.vendor === "apple") return "Apple silicon";
  if (gpu.vendor === "amd") return "AMD";
  if (gpu.vendor === "cpu") return "CPU only";
  if (gpu.id.startsWith("rtx") && !gpu.id.startsWith("rtxa") && gpu.id !== "rtx6000ada") {
    return "GeForce / RTX";
  }
  return "Datacenter / workstation";
}

function shortGpuName(gpu) {
  return gpu.name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").replace("Apple ", "");
}

function vramAt(model, ctx) {
  return model.vram_bytes_at_ctx[String(ctx)] ?? null;
}

function speedFor(model, gpu) {
  const estimate = gpu.estimate;
  if (!estimate || !model.active_weight_bytes || !model.layers) return null;
  const seconds = (
    model.active_weight_bytes / (estimate.bw_eff_gbs * GB)
    + model.layers * estimate.c_s_per_layer
    + estimate.d_s
  );
  const quantFactor = estimate.quant_factors[model.quant] ?? 1;
  return seconds > 0 ? quantFactor / seconds : null;
}

function fitFor(model, gpu, ctx) {
  if (!model.context_length || model.context_length < ctx) return "context";
  const vram = vramAt(model, ctx);
  if (vram == null) return "context";
  const usable = gpu.memory_gb * gpu.usable_fraction * GB;
  if (vram <= usable) return "full";
  const totalMemory = usable + state.ram * GB;
  if (model.weight_bytes + model.projector_bytes <= totalMemory) return "partial";
  return "none";
}

// Relative-capability half-life when Q_file is missing (years).
// Exponential form from arXiv:2603.28576 Tiered Super-Moore economy-tier fit
// (λ=0.629/yr → t½=1.10y). Capability leaderboards are largely calendar
// (arXiv:2608.29420). See data/quality/refs/capability-age-decay.md.
const QUALITY_AGE_HALF_LIFE_YEARS = 1.1;

function ageYears(pushedAt) {
  if (!pushedAt) return null;
  const ms = Date.parse(pushedAt);
  if (!Number.isFinite(ms)) return null;
  return Math.max(0, (Date.now() - ms) / (365.25 * 86400000));
}

function qualityAgeDecay(pushedAt) {
  const t = ageYears(pushedAt);
  if (t == null) return 1;
  return Math.exp(-Math.LN2 * t / QUALITY_AGE_HALF_LIFE_YEARS);
}

function qualityProxy(model) {
  const active = Math.max(1e8, model.active_params || model.params);
  const scale = Math.max(0, Math.min(1, (Math.log10(active) - 8) / 3));
  const quant = QUANT_QUALITY[model.quant] ?? 0.67;
  const sizePrior = scale * 0.72 + quant * 0.28;
  // Unmatched only: depreciate the size prior exponentially with age.
  return sizePrior * qualityAgeDecay(model.pushed_at);
}

function qualityFor(model) {
  const hit = quality.by_ref?.[model.ref];
  if (!hit || hit.q_file == null || !Number.isFinite(hit.q_file)) return null;
  return hit;
}

function capabilityFor(model) {
  const hit = qualityFor(model);
  if (hit) {
    return {
      capability: Math.max(0, Math.min(1, hit.q_file / 100)),
      qualitySource: "bench",
      evalId: hit.eval_id || null,
      qFile: hit.q_file,
      qBase: hit.q_base,
    };
  }
  // Keep unmatched tags sortable, but below typical measured Q_file values.
  return {
    capability: qualityProxy(model) * 0.45,
    qualitySource: "size_prior",
    evalId: null,
    qFile: null,
    qBase: null,
  };
}

function recencyScore(pushedAt) {
  if (!pushedAt) return 0;
  const ms = Date.parse(pushedAt);
  if (!Number.isFinite(ms)) return 0;
  const ageDays = Math.max(0, (Date.now() - ms) / 86400000);
  return Math.exp(-ageDays / 365);
}

function taskScoreFor(model) {
  const hit = qualityFor(model);
  const raw = hit?.q_tasks?.[state.usecase];
  if (raw != null && Number.isFinite(raw)) {
    return {
      match: Math.max(0, Math.min(1, raw / 100)),
      taskSource: "bench",
      qTask: raw,
    };
  }
  // Same conservative prior as unmatched intelligence.
  return {
    match: qualityProxy(model) * 0.45,
    taskSource: "size_prior",
    qTask: null,
  };
}

function viewModel(model) {
  const gpu = gpuById(state.gpu);
  const fit = fitFor(model, gpu, state.ctx);
  const rawSpeed = speedFor(model, gpu);
  const speed = fit === "full" ? rawSpeed : null;
  const usable = gpu.memory_gb * gpu.usable_fraction * GB;
  const vram = vramAt(model, state.ctx);
  const headroom = vram == null ? null : (usable - vram) / usable;
  const { capability, qualitySource, evalId, qFile, qBase } = capabilityFor(model);
  const { match, taskSource, qTask } = taskScoreFor(model);
  const recency = recencyScore(model.pushed_at);
  const speedScore = rawSpeed ? Math.max(0, Math.min(1, Math.log10(rawSpeed + 1) / 2.4)) : 0;
  const fitScore = fit === "full" ? 1 : fit === "partial" && state.allowPartial ? 0.3 : 0;
  let score;
  if (state.priority === "quality") score = capability * 60 + speedScore * 6 + match * 26 + fitScore * 18;
  else if (state.priority === "speed") score = capability * 15 + speedScore * 52 + match * 20 + fitScore * 18;
  else score = capability * 50 + speedScore * 14 + match * 26 + fitScore * 18;
  score += recency * 12;
  if (model.quant === "Q4_K_M") score += 3;
  if (fit === "partial") score -= 16;
  if (fit === "none" || fit === "context") score -= 100;
  return {
    ...model, fit, speed, rawSpeed, vram, usable, headroom, capability, match, recency, score,
    qualitySource, evalId, qFile, qBase, taskSource, qTask,
  };
}

function isUsecaseCandidate(row) {
  if (state.usecase === "vision" && !row.signals.includes("vision")) return false;
  if (state.usecase === "embedding" && !row.signals.includes("embedding")) return false;
  if (state.usecase !== "embedding" && row.signals.includes("embedding")) return false;
  return true;
}

function fitsSelection(row) {
  if (row.fit === "full") return true;
  return state.allowPartial && row.fit === "partial";
}

function searchMatches(row) {
  const q = state.q.trim().toLowerCase();
  if (!q) return true;
  const copy = familyCopy(row);
  return [
    row.ref, row.model, row.arch, row.quant, row.description, copy.description,
    ...row.aliases, ...row.signals,
  ].join(" ").toLowerCase().includes(q);
}

function sizeClassId(params) {
  if (!params) return "unknown";
  for (const cls of SIZE_CLASSES) {
    if (cls.id === "unknown") continue;
    if (params < cls.max) return cls.id;
  }
  return "frontier";
}

function sizeMatches(row) {
  return state.sizes.includes(sizeClassId(row.params));
}

function sizesAreDefault() {
  const known = defaultSizes();
  return state.sizes.length === known.length && known.every((id) => state.sizes.includes(id));
}

function renderSizeFilter() {
  const classes = availableSizeClasses();
  const selected = classes.filter((cls) => state.sizes.includes(cls.id));
  $("size-summary").textContent = sizesAreDefault()
    ? "All sizes"
    : selected.length
      ? selected.map((cls) => cls.name).join(", ")
      : "No sizes";
  $("size-menu").innerHTML = classes.map((cls) => `
    <label class="size-option">
      <input type="checkbox" data-size="${esc(cls.id)}" ${state.sizes.includes(cls.id) ? "checked" : ""}>
      <span>${esc(cls.name)}${cls.hint ? `<small>${esc(cls.hint)}</small>` : ""}</span>
    </label>
  `).join("");
}

function setSizeMenuOpen(open) {
  state.sizeMenuOpen = open;
  $("size-toggle").setAttribute("aria-expanded", String(open));
  $("size-menu").hidden = !open;
}

function applySharedFilters(rows) {
  if (state.fitFilter === "runnable") rows = rows.filter((r) => fitsSelection(r));
  else if (state.fitFilter !== "all") rows = rows.filter((r) => r.fit === state.fitFilter);
  if (state.archFilter !== "all") rows = rows.filter((r) => r.arch === state.archFilter);
  if (state.quantFilter !== "all") rows = rows.filter((r) => r.quant === state.quantFilter);
  return rows;
}

function rowsForView() {
  let rows = catalog.models.filter(searchMatches).filter(sizeMatches).map(viewModel);
  if (state.mode === "find") rows = rows.filter(isUsecaseCandidate);
  rows = applySharedFilters(rows);
  const sort = state.sort;
  rows.sort((a, b) => {
    if (sort === "newest") return (b.pushed_at || "").localeCompare(a.pushed_at || "");
    if (sort === "name") return a.ref.localeCompare(b.ref);
    if (sort === "speed") return (b.rawSpeed || -1) - (a.rawSpeed || -1);
    if (sort === "vram") return (a.vram ?? Infinity) - (b.vram ?? Infinity);
    if (sort === "context") return b.context_length - a.context_length;
    if (sort === "capability") return b.capability - a.capability;
    return b.score - a.score
      || (b.pushed_at || "").localeCompare(a.pushed_at || "")
      || a.ref.localeCompare(b.ref);
  });
  if (state.mode === "find") {
    const families = new Set();
    rows = rows.filter((row) => {
      if (families.has(row.model)) return false;
      families.add(row.model);
      return true;
    });
  }
  return rows;
}

function signalLabel(signal) {
  return {
    code: "Code-tuned signal",
    vision: "Vision input",
    embedding: "Embedding model",
    reasoning: "Reasoning-tuned signal",
  }[signal] || signal;
}

function fitLabel(fit) {
  return { full: "Full GPU", partial: "Partial offload", none: "Doesn’t fit", context: "Context too short" }[fit];
}

function rationale(row) {
  const parts = [];
  if (row.fit === "full") {
    const spare = Math.max(0, row.usable - row.vram);
    parts.push(`${fmtBytes(spare)} GPU memory left at ${fmtCtx(state.ctx)} context`);
  } else if (row.fit === "partial") {
    parts.push("fits only by moving weights into system RAM");
  }
  if (state.usecase === "code" && row.taskSource === "bench" && row.qTask != null) {
    parts.push(`code task score ${row.qTask.toFixed(0)} from coding benches`);
  }
  if (state.usecase === "vision" && row.taskSource === "bench" && row.qTask != null) {
    parts.push(`vision task score ${row.qTask.toFixed(0)} (MMMU-weighted)`);
  }
  if (state.usecase === "embedding" && row.taskSource === "bench" && row.qTask != null) {
    parts.push(`embedding task score ${row.qTask.toFixed(0)} (MTEB)`);
  }
  if (state.usecase === "long" && row.taskSource === "bench" && row.qTask != null) {
    parts.push(`long-doc task score ${row.qTask.toFixed(0)} from general/reasoning benches`);
  }
  if (state.usecase === "chat" && row.taskSource === "bench" && row.qTask != null
      && !(row.qualitySource === "bench" && row.qFile != null)) {
    parts.push(`chat task score ${row.qTask.toFixed(0)} from stored benches`);
  }
  if (state.priority === "speed" && row.rawSpeed) parts.push(`about ${fmtSpeed(row.rawSpeed)} when fully resident`);
  if (state.priority === "quality") {
    if (row.qualitySource === "bench" && row.qFile != null) {
      parts.push(`intelligence score ${row.qFile.toFixed(0)} from stored benches (${row.evalId || "matched"})`);
    } else {
      const age = ageYears(row.pushed_at);
      if (age != null) {
        parts.push(`${fmtParams(row.active_params)} size prior with ${age.toFixed(1)}y exponential age decay (t½=1.1y)`);
      } else {
        parts.push(`${fmtParams(row.active_params)} active parameters as a size prior`);
      }
    }
  }
  if (row.quant === "Q4_K_M") parts.push("uses Ollama’s common balanced quant");
  if (!parts.length) {
    if (row.fit === "context") return `trained context is below the requested ${fmtCtx(state.ctx)}.`;
    if (row.fit === "none") return "model weights exceed the selected GPU and system-memory budget.";
    return "shown for technical comparison; no recommendation claim is attached.";
  }
  return `${parts.slice(0, 2).join("; ")}.`;
}

function renderGpuSelect() {
  const groups = new Map();
  for (const gpu of hardware.gpus) {
    const group = groupName(gpu);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(gpu);
  }
  $("gpu").innerHTML = [...groups].map(([name, gpus]) => `
    <optgroup label="${esc(name)}">
      ${gpus.map((gpu) => `<option value="${esc(gpu.id)}">${esc(shortGpuName(gpu))} (${gpu.memory_gb} GB)${gpu.estimate.calibrated ? " — calibrated" : ""}</option>`).join("")}
    </optgroup>
  `).join("");
  $("gpu").value = state.gpu;
  $("ctx").innerHTML = hardware.ctx_points.map((ctx) =>
    `<option value="${ctx}">${fmtCtx(ctx)} tokens</option>`
  ).join("");
  $("ctx").value = String(state.ctx);

  const arches = [...new Set(catalog.models.map((m) => m.arch))].sort();
  $("arch-filter").innerHTML += arches.map((arch) => `<option value="${esc(arch)}">${esc(arch)}</option>`).join("");
  const quants = [...new Set(catalog.models.map((m) => m.quant).filter((q) => q && q !== "unknown"))].sort();
  $("quant-filter").innerHTML += quants.map((quant) => `<option value="${esc(quant)}">${esc(quant)}</option>`).join("");
}

function renderBudget() {
  const gpu = gpuById(state.gpu);
  const usable = gpu.memory_gb * gpu.usable_fraction;
  $("budget-value").textContent = `${usable.toFixed(1)} GB usable`;
  $("budget-fill").style.width = `${gpu.usable_fraction * 100}%`;
  $("budget-note").textContent = state.allowPartial
    ? `Plus ${state.ram} GB system RAM for partial offload.`
    : `${gpu.memory_gb} GB installed; recommendations stay fully on GPU.`;
}

function renderCalibration() {
  const gpu = gpuById(state.gpu);
  const calibrated = gpu.estimate.calibrated;
  $("calibration-note").className = `notice${calibrated ? "" : " warning"}`;
  $("calibration-note").innerHTML = calibrated
    ? `<strong>L4-calibrated speed model.</strong> Estimates use measured L4 coefficients; individual tags were not all benchmarked. VRAM confidence is shown per result.`
    : `<strong>Experimental speed estimate.</strong> ${esc(shortGpuName(gpu))} has not been benchmark-calibrated. Fit uses the VRAM model; speed transfers L4-derived coefficients to hardware specs.`;
}

function familyCopy(row) {
  const fam = catalog.library && catalog.library[row.model];
  if (fam) return fam;
  return { description: row.description || "", readme: "" };
}

// Brand marks, traced from Simple Icons (CC0). Single-path, fill="currentColor", viewBox 0 0 24 24.
const SOURCE_ICONS = {
  ollama: '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M16.361 10.26a.894.894 0 0 0-.558.47l-.072.148.001.207c0 .193.004.217.059.353.076.193.152.312.291.448.24.238.51.3.872.205a.86.86 0 0 0 .517-.436.752.752 0 0 0 .08-.498c-.064-.453-.33-.782-.724-.897a1.06 1.06 0 0 0-.466 0zm-9.203.005c-.305.096-.533.32-.65.639a1.187 1.187 0 0 0-.06.52c.057.309.31.59.598.667.362.095.632.033.872-.205.14-.136.215-.255.291-.448.055-.136.059-.16.059-.353l.001-.207-.072-.148a.894.894 0 0 0-.565-.472 1.02 1.02 0 0 0-.474.007Zm4.184 2c-.131.071-.223.25-.195.383.031.143.157.288.353.407.105.063.112.072.117.136.004.038-.01.146-.029.243-.02.094-.036.194-.036.222.002.074.07.195.143.253.064.052.076.054.255.059.164.005.198.001.264-.03.169-.082.212-.234.15-.525-.052-.243-.042-.28.087-.355.137-.08.281-.219.324-.314a.365.365 0 0 0-.175-.48.394.394 0 0 0-.181-.033c-.126 0-.207.03-.355.124l-.085.053-.053-.032c-.219-.13-.259-.145-.391-.143a.396.396 0 0 0-.193.032zm.39-2.195c-.373.036-.475.05-.654.086-.291.06-.68.195-.951.328-.94.46-1.589 1.226-1.787 2.114-.04.176-.045.234-.045.53 0 .294.005.357.043.524.264 1.16 1.332 2.017 2.714 2.173.3.033 1.596.033 1.896 0 1.11-.125 2.064-.727 2.493-1.571.114-.226.169-.372.22-.602.039-.167.044-.23.044-.523 0-.297-.005-.355-.045-.531-.288-1.29-1.539-2.304-3.072-2.497a6.873 6.873 0 0 0-.855-.031zm.645.937a3.283 3.283 0 0 1 1.44.514c.223.148.537.458.671.662.166.251.26.508.303.82.02.143.01.251-.043.482-.08.345-.332.705-.672.957a3.115 3.115 0 0 1-.689.348c-.382.122-.632.144-1.525.138-.582-.006-.686-.01-.853-.042-.57-.107-1.022-.334-1.35-.68-.264-.28-.385-.535-.45-.946-.03-.192.025-.509.137-.776.136-.326.488-.73.836-.963.403-.269.934-.46 1.422-.512.187-.02.586-.02.773-.002zm-5.503-11a1.653 1.653 0 0 0-.683.298C5.617.74 5.173 1.666 4.985 2.819c-.07.436-.119 1.04-.119 1.503 0 .544.064 1.24.155 1.721.02.107.031.202.023.208a8.12 8.12 0 0 1-.187.152 5.324 5.324 0 0 0-.949 1.02 5.49 5.49 0 0 0-.94 2.339 6.625 6.625 0 0 0-.023 1.357c.091.78.325 1.438.727 2.04l.13.195-.037.064c-.269.452-.498 1.105-.605 1.732-.084.496-.095.629-.095 1.294 0 .67.009.803.088 1.266.095.555.288 1.143.503 1.534.071.128.243.393.264.407.007.003-.014.067-.046.141a7.405 7.405 0 0 0-.548 1.873c-.062.417-.071.552-.071.991 0 .56.031.832.148 1.279L3.42 24h1.478l-.05-.091c-.297-.552-.325-1.575-.068-2.597.117-.472.25-.819.498-1.296l.148-.29v-.177c0-.165-.003-.184-.057-.293a.915.915 0 0 0-.194-.25 1.74 1.74 0 0 1-.385-.543c-.424-.92-.506-2.286-.208-3.451.124-.486.329-.918.544-1.154a.787.787 0 0 0 .223-.531c0-.195-.07-.355-.224-.522a3.136 3.136 0 0 1-.817-1.729c-.14-.96.114-2.005.69-2.834.563-.814 1.353-1.336 2.237-1.475.199-.033.57-.028.776.01.226.04.367.028.512-.041.179-.085.268-.19.374-.431.093-.215.165-.333.36-.576.234-.29.46-.489.822-.729.413-.27.884-.467 1.352-.561.17-.035.25-.04.569-.04.319 0 .398.005.569.04a4.07 4.07 0 0 1 1.914.997c.117.109.398.457.488.602.034.057.095.177.132.267.105.241.195.346.374.43.14.068.286.082.503.045.343-.058.607-.053.943.016 1.144.23 2.14 1.173 2.581 2.437.385 1.108.276 2.267-.296 3.153-.097.15-.193.27-.333.419-.301.322-.301.722-.001 1.053.493.539.801 1.866.708 3.036-.062.772-.26 1.463-.533 1.854a2.096 2.096 0 0 1-.224.258.916.916 0 0 0-.194.25c-.054.109-.057.128-.057.293v.178l.148.29c.248.476.38.823.498 1.295.253 1.008.231 2.01-.059 2.581a.845.845 0 0 0-.044.098c0 .006.329.009.732.009h.73l.02-.074.036-.134c.019-.076.057-.3.088-.516.029-.217.029-1.016 0-1.258-.11-.875-.295-1.57-.597-2.226-.032-.074-.053-.138-.046-.141.008-.005.057-.074.108-.152.376-.569.607-1.284.724-2.228.031-.26.031-1.378 0-1.628-.083-.645-.182-1.082-.348-1.525a6.083 6.083 0 0 0-.329-.7l-.038-.064.131-.194c.402-.604.636-1.262.727-2.04a6.625 6.625 0 0 0-.024-1.358 5.512 5.512 0 0 0-.939-2.339 5.325 5.325 0 0 0-.95-1.02 8.097 8.097 0 0 1-.186-.152.692.692 0 0 1 .023-.208c.208-1.087.201-2.443-.017-3.503-.19-.924-.535-1.658-.98-2.082-.354-.338-.716-.482-1.15-.455-.996.059-1.8 1.205-2.116 3.01a6.805 6.805 0 0 0-.097.726c0 .036-.007.066-.015.066a.96.96 0 0 1-.149-.078A4.857 4.857 0 0 0 12 3.03c-.832 0-1.687.243-2.456.698a.958.958 0 0 1-.148.078c-.008 0-.015-.03-.015-.066a6.71 6.71 0 0 0-.097-.725C8.997 1.392 8.337.319 7.46.048a2.096 2.096 0 0 0-.585-.041Zm.293 1.402c.248.197.523.759.682 1.388.03.113.06.244.069.292.007.047.026.152.041.233.067.365.098.76.102 1.24l.002.475-.12.175-.118.178h-.278c-.324 0-.646.041-.954.124l-.238.06c-.033.007-.038-.003-.057-.144a8.438 8.438 0 0 1 .016-2.323c.124-.788.413-1.501.696-1.711.067-.05.079-.049.157.013zm9.825-.012c.17.126.358.46.498.888.28.854.36 2.028.212 3.145-.019.14-.024.151-.057.144l-.238-.06a3.693 3.693 0 0 0-.954-.124h-.278l-.119-.178-.119-.175.002-.474c.004-.669.066-1.19.214-1.772.157-.623.434-1.185.68-1.382.078-.062.09-.063.159-.012z"/></svg>',
  // Simple Icons' huggingface trace is a single flat-fill path (no separate eyes/mouth
  // tone) -- looked like a blank yellow blob. The real mark people recognize is the
  // 🤗 emoji itself, which renders its own shading natively, so use that directly.
  huggingface: '<span class="source-emoji" role="img" aria-hidden="true">\u{1F917}</span>',
  github: '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>',
  homepage: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10Z"/></svg>',
  paper: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M9 13h6M9 17h6M9 9h1"/></svg>',
};

const SOURCE_LABELS = {
  huggingface: (l) => `Hugging Face — ${l.repo}`,
  github: (l) => `GitHub — ${l.repo}`,
  homepage: () => "Project homepage",
  paper: () => "Paper",
};

// Verified badge, traced from Lucide badge-check (ISC). Filled blue seal + white check.
const VERIFIED_BADGE = '<svg viewBox="0 0 24 24" width="12" height="12"><path fill="var(--blue)" stroke="none" d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z"/><path fill="none" stroke="var(--white)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" d="m16 9-5.5 5.5L8 12"/></svg>';

function sourceIconHtml(kind, href, label, verified) {
  const mark = verified ? `<i class="source-badge" aria-hidden="true">${VERIFIED_BADGE}</i>` : "";
  return `<a class="source-icon" href="${esc(href)}" target="_blank" rel="noopener noreferrer" title="${esc(label)}" aria-label="${esc(label)}">${SOURCE_ICONS[kind]}${mark}</a>`;
}

// links.hf is release-scoped: {release, family}, either possibly null. A family can bundle
// genuinely different upstream releases at different sizes (llava:7b/13b/34b are different
// checkpoints entirely), so a release-specific pick is preferred whenever one exists; the
// family-wide guess is kept as a fallback, not silently dropped, but flagged as such.
function resolvedHf(row) {
  const hf = (row.links || {}).hf;
  if (!hf) return null;
  const link = hf.release || hf.family;
  return link ? { ...link, isFamilyOnly: !hf.release && !!hf.family } : null;
}

function sourceConfidenceHint(key, link) {
  if (!link) return "";
  if (key !== "hf") {
    // github/homepage/paper have no hash-verification concept -- they're always "found
    // somewhere", the only question is how directly.
    return link.method === "github_llm" ? "found via the model's GitHub repo"
      : link.method === "homepage_llm" ? "found via the model's homepage"
      : "linked from the Ollama readme";
  }
  const base = link.confidence === "verified" ? "byte-identical file match"
    : link.method === "github_llm" ? "found via the model's GitHub repo, not hash-verified"
    : link.method === "homepage_llm" ? "found via the model's homepage, not hash-verified"
    : link.method === "readme_verified" ? "named in the readme, LLM-confirmed but not hash-verified"
    : "named in the readme, not hash-verified";
  return link.isFamilyOnly ? `${base}, family-wide guess, not confirmed for this exact size` : base;
}

function sourceIconsHtml(row) {
  const items = [sourceIconHtml("ollama", `https://ollama.com/library/${encodeURIComponent(row.model)}`, "Ollama library page")];
  const links = row.links || {};
  for (const kind of ["huggingface", "github", "homepage", "paper"]) {
    const key = kind === "huggingface" ? "hf" : kind;
    const link = key === "hf" ? resolvedHf(row) : links[key];
    if (!link || !link.url) continue;
    const verified = link.confidence === "verified";
    const label = `${SOURCE_LABELS[kind](link)} (${sourceConfidenceHint(key, link)})`;
    items.push(sourceIconHtml(kind, link.url, label, verified));
  }
  return `<div class="source-icons">${items.join("")}</div>`;
}

function sourceLinksHtml(row) {
  const parts = [`<a href="https://ollama.com/library/${encodeURIComponent(row.model)}" target="_blank" rel="noopener noreferrer">Ollama library page</a>`];
  const links = row.links || {};
  const labels = { hf: "Hugging Face", github: "GitHub", homepage: "Homepage", paper: "Paper" };
  for (const key of ["hf", "github", "homepage", "paper"]) {
    const link = key === "hf" ? resolvedHf(row) : links[key];
    if (!link || !link.url) continue;
    const hint = sourceConfidenceHint(key, link);
    const unverified = key === "hf" && link.confidence !== "verified";
    parts.push(`<a href="${esc(link.url)}" target="_blank" rel="noopener noreferrer" title="${esc(hint)}">${labels[key]}${unverified ? " (unverified)" : ""} ↗</a>`);
  }
  return `<p class="library-link">${parts.join(" · ")}</p>`;
}

function linkContentHtml(row) {
  const content = catalog.linkContent[row.model];
  if (!content) return "";
  const titles = { hf: "Hugging Face model card", github: "GitHub readme", homepage: "Homepage" };
  const hfRepo = (resolvedHf(row) || {}).repo;
  const contentFor = (key) => (key === "hf" ? (hfRepo && (content.hf || {})[hfRepo]) : content[key]);
  return ["hf", "github", "homepage"]
    .filter((key) => contentFor(key) && contentFor(key).content)
    .map((key) => `<details class="library-readme"><summary>${esc(titles[key])}</summary>${mdToHtml(contentFor(key).content)}</details>`)
    .join("");
}

function technicalHtml(row) {
  if (state.open !== row.ref) return "";
  const readme = familyCopy(row).readme || "";
  const extra = linkContentHtml(row);
  if (!readme && !extra) {
    return sourceLinksHtml(row);
  }
  return `
    <div class="library-readme">${readme ? mdToHtml(readme) : ""}
      ${sourceLinksHtml(row)}
    </div>
    ${extra}`;
}

function cardHtml(row, index) {
  const command = `ollama run ${row.ref}`;
  const headroomPct = row.headroom == null ? 0 : Math.max(0, Math.min(100, row.headroom * 100));
  const speedSub = row.fit === "partial" ? "partial speed unavailable" : (
    gpuById(state.gpu).estimate.calibrated ? "calibrated formula" : "experimental"
  );
  const vramSub = row.vram_digest_calibrated ? "digest-calibrated" : "global formula";
  const signals = row.signals.map((signal) => `<span class="signal">${esc(signalLabel(signal))}</span>`).join("");
  const description = familyCopy(row).description || row.description || "";
  const open = state.open === row.ref;
  return `
    <article class="model-card${open ? " is-open" : ""}" data-ref="${esc(row.ref)}" aria-expanded="${open}">
      <div>
        <div class="card-heading">
          <span class="rank">${index + 1}</span>
          <div class="model-title">
            <h3>${esc(row.ref)}</h3>
            <p>${paramHeadline(row)} · ${esc(row.quant || "—")} · ${esc(row.arch)}${row.pushed_at ? ` · ${fmtPushed(row.pushed_at)}` : ""}</p>
          </div>
          ${sourceIconsHtml(row)}
          <span class="fit-badge ${row.fit}">${esc(fitLabel(row.fit))}</span>
        </div>
        ${description ? `<p class="library-desc">${esc(description)}</p>` : ""}
        <p class="why">${esc(rationale(row))}</p>
        <div class="signals">
          ${signals}
          ${row.qualitySource === "bench" ? `<span class="signal">Intelligence ${row.qFile.toFixed(0)}</span>` : ""}
          ${row.taskSource === "bench" && row.qTask != null && state.usecase !== "chat"
            ? `<span class="signal">${esc(USECASES[state.usecase].label)} ${row.qTask.toFixed(0)}</span>`
            : ""}
          ${row.vram_digest_calibrated ? `<span class="signal">Measured VRAM offset</span>` : ""}
        </div>
      </div>
      <div>
        <div class="metrics">
          <div class="metric"><span>VRAM @ ${fmtCtx(state.ctx)}</span><strong>${fmtBytes(row.vram)}</strong><small>${vramSub}</small><div class="headroom"><i style="width:${headroomPct}%"></i></div></div>
          <div class="metric"><span>Decode speed</span><strong>${fmtSpeed(row.speed)}</strong><small>${speedSub}</small></div>
          <div class="metric"><span>Trained context</span><strong>${fmtCtx(row.context_length)}</strong><small>${attnLabel(row)}</small></div>
          <div class="metric"><span>Active size</span><strong>${fmtParams(row.active_params)}</strong><small>${row.ple_bytes ? "effective params; PLE embeddings in RAM" : row.experts ? `${row.expert_used}/${row.experts} experts active` : "dense model"}</small></div>
        </div>
        <div class="card-actions">
          <div class="run-command"><code>${esc(command)}</code><button type="button" data-copy="${esc(command)}">Copy</button></div>
          <span class="confidence">${row.vram_digest_calibrated ? "Higher VRAM confidence" : "Estimated VRAM"}</span>
        </div>
      </div>
      ${technicalHtml(row)}
    </article>`;
}

function renderResults() {
  const rows = rowsForView();
  const visibleRows = rows.slice(0, state.visible);
  const usecase = USECASES[state.usecase].label;
  $("results-label").textContent = state.mode === "find" ? "Recommendations" : "Full catalog";
  $("results-title").textContent = state.mode === "find"
    ? `${rows.length.toLocaleString()} fits for ${usecase}`
    : `${rows.length.toLocaleString()} models match`;
  $("results").innerHTML = visibleRows.map(cardHtml).join("");
  $("empty").hidden = rows.length !== 0;
  $("load-more").hidden = visibleRows.length >= rows.length;
  $("result-count").textContent = rows.length
    ? `Showing ${visibleRows.length.toLocaleString()} of ${rows.length.toLocaleString()} unique weight files`
    : "";
  $("results").closest(".results-panel").setAttribute("aria-busy", "false");
}

function renderMode() {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.mode === state.mode));
  });
  $("search-wrap").hidden = state.mode !== "explore";
}

function writeUrl() {
  const url = new URL(location.href);
  const values = {
    mode: state.mode, gpu: state.gpu, ram: state.ram, ctx: state.ctx,
    usecase: state.usecase, priority: state.priority,
    fit: state.fitFilter, arch: state.archFilter, quant: state.quantFilter, sort: state.sort,
  };
  for (const [key, value] of Object.entries(values)) url.searchParams.set(key, String(value));
  if (state.allowPartial) url.searchParams.set("partial", "1");
  else url.searchParams.delete("partial");
  if (sizesAreDefault()) url.searchParams.delete("size");
  else url.searchParams.set("size", state.sizes.join(","));
  history.replaceState(null, "", url);
}

function readUrl() {
  const params = new URL(location.href).searchParams;
  if (["find", "explore"].includes(params.get("mode"))) state.mode = params.get("mode");
  if (hardware.gpus.some((g) => g.id === params.get("gpu"))) state.gpu = params.get("gpu");
  if ([8, 16, 32, 64, 128, 256].includes(Number(params.get("ram")))) state.ram = Number(params.get("ram"));
  if (hardware.ctx_points.includes(Number(params.get("ctx")))) state.ctx = Number(params.get("ctx"));
  if (USECASES[params.get("usecase")]) state.usecase = params.get("usecase");
  if (["balanced", "quality", "speed"].includes(params.get("priority"))) state.priority = params.get("priority");
  state.allowPartial = params.get("partial") === "1";
  if (["runnable", "full", "partial", "all"].includes(params.get("fit"))) state.fitFilter = params.get("fit");
  if (params.get("arch")) state.archFilter = params.get("arch");
  if (params.get("quant")) state.quantFilter = params.get("quant");
  if (["recommended", "capability", "speed", "vram", "context", "newest", "name"].includes(params.get("sort"))) {
    state.sort = params.get("sort");
  }
  if (params.get("size")) {
    const wanted = params.get("size").split(",").filter((id) => presentSizeIds.includes(id));
    if (wanted.length) state.sizes = wanted;
  }
}

function syncControls() {
  $("gpu").value = state.gpu;
  $("ram").value = String(state.ram);
  $("ctx").value = String(state.ctx);
  $("allow-partial").checked = state.allowPartial;
  $("fit-filter").value = state.fitFilter;
  $("arch-filter").value = state.archFilter;
  $("quant-filter").value = state.quantFilter;
  $("sort").value = state.sort;
  document.querySelectorAll("[data-usecase]").forEach((button) =>
    button.setAttribute("aria-pressed", String(button.dataset.usecase === state.usecase)));
  document.querySelectorAll("[data-priority]").forEach((button) =>
    button.setAttribute("aria-pressed", String(button.dataset.priority === state.priority)));
  renderSizeFilter();
}

function render() {
  renderMode();
  renderBudget();
  renderCalibration();
  renderSizeFilter();
  renderResults();
  writeUrl();
}

function resetVisible() {
  state.visible = state.mode === "find" ? PAGE_FIND : PAGE_EXPLORE;
  state.open = null;
}

function bind() {
  document.querySelector(".mode-switch").addEventListener("click", (event) => {
    const button = event.target.closest("[data-mode]");
    if (!button) return;
    state.mode = button.dataset.mode;
    resetVisible();
    render();
  });
  $("gpu").addEventListener("change", (event) => { state.gpu = event.target.value; resetVisible(); render(); });
  $("ram").addEventListener("change", (event) => { state.ram = Number(event.target.value); resetVisible(); render(); });
  $("ctx").addEventListener("change", (event) => { state.ctx = Number(event.target.value); resetVisible(); render(); });
  $("allow-partial").addEventListener("change", (event) => { state.allowPartial = event.target.checked; resetVisible(); render(); });
  $("usecase-group").addEventListener("click", (event) => {
    const button = event.target.closest("[data-usecase]");
    if (!button) return;
    state.usecase = button.dataset.usecase;
    resetVisible();
    syncControls();
    render();
  });
  $("priority-group").addEventListener("click", (event) => {
    const button = event.target.closest("[data-priority]");
    if (!button) return;
    state.priority = button.dataset.priority;
    resetVisible();
    syncControls();
    render();
  });
  $("q").addEventListener("input", (event) => { state.q = event.target.value; resetVisible(); renderResults(); });
  for (const [id, key] of [
    ["fit-filter", "fitFilter"], ["arch-filter", "archFilter"],
    ["quant-filter", "quantFilter"], ["sort", "sort"],
  ]) {
    $(id).addEventListener("change", (event) => {
      state[key] = event.target.value;
      resetVisible();
      renderResults();
      writeUrl();
    });
  }
  $("load-more").addEventListener("click", () => {
    state.visible += state.mode === "find" ? PAGE_FIND : PAGE_EXPLORE;
    renderResults();
  });
  const toggleCard = (card) => {
    if (!card) return;
    const ref = card.dataset.ref;
    state.open = state.open === ref ? null : ref;
    renderResults();
  };
  $("results").addEventListener("click", async (event) => {
    const copy = event.target.closest("[data-copy]");
    if (copy) {
      event.stopPropagation();
      const previous = copy.textContent;
      try {
        await navigator.clipboard.writeText(copy.dataset.copy);
        copy.textContent = "Copied";
      } catch {
        copy.textContent = "Copy failed";
      }
      setTimeout(() => { copy.textContent = previous; }, 1200);
      return;
    }
    if (event.target.closest("a, button, .library-readme")) return;
    toggleCard(event.target.closest(".model-card"));
  });
  $("size-toggle").addEventListener("click", (event) => {
    event.stopPropagation();
    setSizeMenuOpen(!state.sizeMenuOpen);
  });
  $("size-menu").addEventListener("change", (event) => {
    const box = event.target.closest("[data-size]");
    if (!box) return;
    const id = box.dataset.size;
    if (box.checked && !state.sizes.includes(id)) state.sizes.push(id);
    if (!box.checked) state.sizes = state.sizes.filter((s) => s !== id);
    resetVisible();
    $("size-summary").textContent = sizesAreDefault()
      ? "All sizes"
      : state.sizes.length
        ? availableSizeClasses().filter((cls) => state.sizes.includes(cls.id)).map((cls) => cls.name).join(", ")
        : "No sizes";
    renderResults();
    writeUrl();
  });
  document.addEventListener("click", (event) => {
    if (!state.sizeMenuOpen) return;
    if (event.target.closest("#size-filter")) return;
    setSizeMenuOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setSizeMenuOpen(false);
  });
  $("reset").addEventListener("click", () => {
    Object.assign(state, {
      mode: "find", gpu: hardware.gpus.some((g) => g.id === "l4") ? "l4" : hardware.gpus[0].id,
      ram: 32, ctx: 8192, usecase: "chat", priority: "balanced", allowPartial: false,
      q: "", fitFilter: "runnable", archFilter: "all", quantFilter: "all",
      sort: "recommended", sizes: defaultSizes(), sizeMenuOpen: false,
    });
    $("q").value = "";
    setSizeMenuOpen(false);
    resetVisible();
    syncControls();
    render();
  });
}

async function loadData() {
  const base = String(window.MODELINDEX_DATA_BASE || "../prod/data").replace(/\/$/, "");
  const [manifestResponse, gpuResponse, modelResponse, libraryResponse, qualityResponse, linkContentResponse] = await Promise.all([
    fetch(`${base}/manifest.json`),
    fetch(`${base}/gpus.json`),
    fetch(`${base}/models.json`),
    fetch(`${base}/library.json`),
    fetch(`${base}/quality.json`),
    fetch(`${base}/link_content.json`),
  ]);
  for (const response of [manifestResponse, gpuResponse, modelResponse]) {
    if (!response.ok) throw new Error(`${response.url} returned ${response.status}`);
  }
  [manifest, hardware, catalog] = await Promise.all([
    manifestResponse.json(), gpuResponse.json(), modelResponse.json(),
  ]);
  catalog.library = {};
  if (libraryResponse.ok) {
    const library = await libraryResponse.json();
    if (library.schema_version && library.schema_version !== manifest.schema_version) {
      throw new Error("Production data schema versions do not match");
    }
    catalog.library = library.families || {};
  }
  catalog.linkContent = {};
  if (linkContentResponse.ok) {
    const linkContent = await linkContentResponse.json();
    if (linkContent.schema_version && linkContent.schema_version !== manifest.schema_version) {
      throw new Error("Production data schema versions do not match");
    }
    catalog.linkContent = linkContent.families || {};
  }
  if (qualityResponse.ok) {
    const payload = await qualityResponse.json();
    if (payload.schema_version && payload.schema_version !== manifest.schema_version) {
      throw new Error("Production data schema versions do not match");
    }
    quality = { by_ref: payload.by_ref || {} };
  } else {
    quality = { by_ref: {} };
  }
  if (manifest.schema_version !== hardware.schema_version || manifest.schema_version !== catalog.schema_version) {
    throw new Error("Production data schema versions do not match");
  }
}

async function main() {
  try {
    await loadData();
    collectPresentSizes();
    state.sizes = defaultSizes();
    readUrl();
    renderGpuSelect();
    syncControls();
    bind();
    render();
    $("data-status").textContent = `${manifest.unique_models.toLocaleString()} unique models · ${manifest.gpu_count} hardware profiles${manifest.quality_refs ? ` · ${manifest.quality_refs.toLocaleString()} with intelligence scores` : ""}`;
    $("footer-meta").textContent = `Data ${manifest.generated.slice(0, 10)} · VRAM ${manifest.vram_formula} · schema v${manifest.schema_version}`;
  } catch (error) {
    $("data-status").textContent = "Catalog unavailable";
    $("results-title").textContent = "Could not load production data";
    $("calibration-note").className = "notice warning";
    $("calibration-note").textContent = `${error.message}. Serve the repository root over HTTP and open /web/.`;
    $("results").closest(".results-panel").setAttribute("aria-busy", "false");
  }
}

main();
