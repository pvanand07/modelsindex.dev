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

function qualityProxy(model) {
  const active = Math.max(1e8, model.active_params || model.params);
  const scale = Math.max(0, Math.min(1, (Math.log10(active) - 8) / 3));
  const quant = QUANT_QUALITY[model.quant] ?? 0.67;
  return scale * 0.72 + quant * 0.28;
}

function recencyScore(pushedAt) {
  if (!pushedAt) return 0;
  const ms = Date.parse(pushedAt);
  if (!Number.isFinite(ms)) return 0;
  const ageDays = Math.max(0, (Date.now() - ms) / 86400000);
  return Math.exp(-ageDays / 365);
}

function usecaseMatch(model) {
  const signals = new Set(model.signals);
  const name = model.ref.toLowerCase();
  if (state.usecase === "vision") return signals.has("vision") ? 1 : 0;
  if (state.usecase === "embedding") return signals.has("embedding") ? 1 : 0;
  if (state.usecase === "code") return signals.has("code") ? 1 : signals.has("embedding") ? 0 : 0.2;
  if (state.usecase === "long") {
    const contextScore = Math.min(1, Math.log2(Math.max(model.context_length, 2048) / 2048) / 6);
    return signals.has("embedding") ? 0.15 : 0.45 + contextScore * 0.55;
  }
  if (signals.has("embedding") || name.includes("guard")) return 0.02;
  if (signals.has("code")) return 0.2;
  if (/(?:^|[-_:])(base|text)(?:$|[-_:])/.test(name)) return 0.25;
  if (name.includes("instruct") || name.includes("chat") || model.tag === "latest") return 1;
  return signals.has("vision") ? 0.65 : 0.7;
}

function viewModel(model) {
  const gpu = gpuById(state.gpu);
  const fit = fitFor(model, gpu, state.ctx);
  const rawSpeed = speedFor(model, gpu);
  const speed = fit === "full" ? rawSpeed : null;
  const usable = gpu.memory_gb * gpu.usable_fraction * GB;
  const vram = vramAt(model, state.ctx);
  const headroom = vram == null ? null : (usable - vram) / usable;
  const capability = qualityProxy(model);
  const match = usecaseMatch(model);
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
  return { ...model, fit, speed, rawSpeed, vram, usable, headroom, capability, match, recency, score };
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
  if (state.usecase === "code" && row.signals.includes("code")) parts.push("catalog name indicates code tuning");
  if (state.usecase === "vision") parts.push("includes a vision projector or multimodal architecture");
  if (state.usecase === "embedding") parts.push("identified as an embedding architecture");
  if (state.usecase === "long") parts.push(`supports up to ${fmtCtx(row.context_length)} trained context`);
  if (state.priority === "speed" && row.rawSpeed) parts.push(`about ${fmtSpeed(row.rawSpeed)} when fully resident`);
  if (state.priority === "quality") parts.push(`${fmtParams(row.active_params)} active parameters as a capability proxy`);
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

function technicalHtml(row) {
  if (state.open !== row.ref) return "";
  const readme = familyCopy(row).readme || "";
  if (!readme) {
    return `<p class="library-link"><a href="https://ollama.com/library/${encodeURIComponent(row.model)}" target="_blank" rel="noopener noreferrer">Ollama library page</a></p>`;
  }
  return `
    <div class="library-readme">${mdToHtml(readme)}
      <p class="library-link"><a href="https://ollama.com/library/${encodeURIComponent(row.model)}" target="_blank" rel="noopener noreferrer">Ollama library page</a></p>
    </div>`;
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
          <span class="fit-badge ${row.fit}">${esc(fitLabel(row.fit))}</span>
        </div>
        ${description ? `<p class="library-desc">${esc(description)}</p>` : ""}
        <p class="why">${esc(rationale(row))}</p>
        <div class="signals">
          ${signals}
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
  const [manifestResponse, gpuResponse, modelResponse, libraryResponse] = await Promise.all([
    fetch(`${base}/manifest.json`),
    fetch(`${base}/gpus.json`),
    fetch(`${base}/models.json`),
    fetch(`${base}/library.json`),
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
    $("data-status").textContent = `${manifest.unique_models.toLocaleString()} unique models · ${manifest.gpu_count} hardware profiles`;
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
