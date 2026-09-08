const QUANT_FACTOR = {
  F16: 1, Q8_0: 0.97, Q4_K_M: 0.92, Q4_0: 0.86,
  Q3_K_L: 0.78, Q3_K_M: 0.74, Q3_K_S: 0.68, Q2_K: 0.55,
};

const METRICS = [
  { id: "pick", label: "Recommended" },
  { id: "speed", label: "Speed" },
  { id: "latency", label: "Latency" },
  { id: "vram", label: "VRAM" },
  { id: "maxctx", label: "Max context" },
];

const SIZE_FILTERS = [
  { id: "all", label: "Size: all", test: () => true },
  { id: "lt3", label: "< 3B", test: (m) => m.params < 3e9 },
  { id: "3to8", label: "3–8B", test: (m) => m.params >= 3e9 && m.params < 8e9 },
  { id: "8to32", label: "8–32B", test: (m) => m.params >= 8e9 && m.params < 32e9 },
  { id: "gt32", label: "32B+", test: (m) => m.params >= 32e9 },
];

const KIND_FILTERS = [
  { id: "all", label: "Arch: all", test: () => true },
  { id: "dense", label: "Dense", test: (m) => !m.experts },
  { id: "moe", label: "MoE", test: (m) => m.experts > 0 },
];

const FIT_FILTERS = [
  { id: "runnable", label: "Runnable", test: (r) => r.fit === "full" || r.fit === "partial" },
  { id: "full", label: "Full GPU", test: (r) => r.fit === "full" },
  { id: "partial", label: "Partial offload", test: (r) => r.fit === "partial" },
  { id: "none", label: "OOM", test: (r) => r.fit === "none" },
  { id: "all", label: "All", test: () => true },
];

const GPU_GROUPS = [
  ["GeForce / RTX", (g) => g.vendor === "nvidia" && g.id.startsWith("rtx") && !g.id.startsWith("rtxa") && g.id !== "rtx6000ada"],
  ["Datacenter / workstation", (g) => g.vendor === "nvidia" && !(g.id.startsWith("rtx") && !g.id.startsWith("rtxa") && g.id !== "rtx6000ada")],
  ["AMD", (g) => g.vendor === "amd"],
  ["Apple", (g) => g.vendor === "apple"],
  ["CPU only", (g) => g.vendor === "cpu"],
];

const state = {
  gpu: "rtx4090",
  ctx: 4096,
  metric: "pick",
  q: "",
  family: "all",
  quant: "all",
  size: "all",
  kind: "all",
  fit: "runnable",
  unique: true,
  view: "list",
  sortKey: "pick",
  sortDir: "desc",
  open: null,
};

let DATA = null;
let chart = null;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

function fmtParams(n) {
  if (n >= 1e12) return `${(n / 1e12).toFixed(n >= 1e13 ? 0 : 2)}T`;
  if (n >= 1e9) {
    const v = n / 1e9;
    if (v >= 10) return `${v.toFixed(0)}B`;
    if (Math.abs(v - Math.round(v)) < 0.05) return `${Math.round(v)}B`;
    return `${v.toFixed(1)}B`;
  }
  return `${(n / 1e6).toFixed(0)}M`;
}
function fmtCtx(n) {
  if (!n) return "—";
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k`;
  if (n >= 1000) return `${Math.round(n / 1000)}k`;
  return String(n);
}
function fmtGb(bytes) {
  if (bytes == null) return "—";
  const gb = bytes / 1e9;
  if (gb < 1) return `${(gb * 1000).toFixed(0)} MB`;
  return gb >= 10 ? `${gb.toFixed(1)} GB` : `${gb.toFixed(2)} GB`;
}
function fmtTps(n) {
  if (n == null || Number.isNaN(n)) return "—";
  if (n >= 100) return `${n.toFixed(0)} tok/s`;
  return `${n.toFixed(1)} tok/s`;
}
function fmtSec(n) {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n < 0.05) return "<0.1s";
  if (n < 1) return `${n.toFixed(1)}s`;
  if (n < 10) return `${n.toFixed(1)}s`;
  return `${n.toFixed(0)}s`;
}
function qualityOf(q) {
  return { id: q || "other", label: q || "—", hint: q === "Q4_K_M" ? "Ollama default" : "" };
}
function fitLabel(fit) {
  if (fit === "full") return "Full";
  if (fit === "partial") return "Partial";
  return "OOM";
}
function familyName(arch) {
  if (arch === "qwen3moe") return "Qwen3 MoE";
  if (arch === "qwen3") return "Qwen3";
  if (arch === "gemma3") return "Gemma 3";
  if (arch === "llama") return "Llama";
  return arch;
}
function gpuShort(g) {
  return g.name
    .replace("NVIDIA GeForce ", "")
    .replace("NVIDIA ", "")
    .replace("Apple ", "");
}
function gpuOf(id) {
  return DATA.gpus.find((g) => g.id === id);
}
function vramAt(m, ctx) {
  const key = String(ctx);
  if (m.vram[key] != null) return m.vram[key];
  const keys = Object.keys(m.vram).map(Number).sort((a, b) => a - b);
  let best = keys[0];
  for (const k of keys) if (k <= ctx) best = k;
  return m.vram[String(best)] ?? null;
}
function fitAt(m, gpuId, ctx) {
  const g = m.gpu[gpuId];
  if (!g) return "none";
  return g.f[String(ctx)] || g.f[Object.keys(g.f).pop()] || "none";
}

function uniqueByDigest(models) {
  const map = new Map();
  for (const m of models) {
    const cur = map.get(m.digest);
    if (!cur) {
      map.set(m.digest, { ...m, aliases: [] });
      continue;
    }
    const prefer = m.ref.length < cur.ref.length || (m.ref.length === cur.ref.length && !/-/.test(m.tag) && /-/.test(cur.tag));
    if (prefer) {
      cur.aliases.push(cur.ref);
      Object.assign(cur, m, { aliases: cur.aliases });
    } else {
      cur.aliases.push(m.ref);
    }
  }
  return [...map.values()];
}

function pickScore(r) {
  if (r.fit === "none") return -1e6;
  const size = Math.log10(r.params);
  const qf = QUANT_FACTOR[r.quant] ?? 0.8;
  const fitN = r.fit === "full" ? 1 : 0.4;
  const everyday = r.quant === "Q4_K_M" ? 0.08 : 0;
  return (size * (0.7 + 0.3 * qf) + everyday) * fitN;
}

function lede(r) {
  const moe = r.experts ? ` MoE, ${fmtParams(r.active)} active of ${fmtParams(r.params)}` : ` ${fmtParams(r.params)}`;
  const fit = r.fit === "full"
    ? `full GPU at ${fmtCtx(state.ctx)}`
    : r.fit === "partial"
      ? `partial offload at ${fmtCtx(state.ctx)}`
      : `OOM at ${fmtCtx(state.ctx)}`;
  return `${familyName(r.arch)}${moe}, ${r.quant || "—"}. ${fmtGb(r.weight)} weights, ${fmtGb(r.vram)} VRAM, ${fit}. Speed ${fmtTps(r.decode)}.`;
}

const TTFT_PROMPT = 128; // same default as bench.py --prompt-tokens

function ttftS(prefill, decode, n = TTFT_PROMPT) {
  if (!prefill || prefill <= 0) return null;
  return n / prefill + (decode > 0 ? 1 / decode : 0);
}

function viewRow(m) {
  const g = m.gpu[state.gpu];
  const vram = vramAt(m, state.ctx);
  const gpu = gpuOf(state.gpu);
  const usable = gpu ? gpu.mem * gpu.use * 1e9 : null;
  const decode = g?.d ?? null;
  const prefill = g?.p ?? null;
  const fit = fitAt(m, state.gpu, state.ctx);
  const row = {
    ...m,
    decode,
    prefill,
    ceiling: g?.c ?? null,
    maxctx: g?.m ?? 0,
    fit,
    vramMap: m.vram,
    vram,
    occ: usable && vram ? vram / usable : null,
    ttft: ttftS(prefill, decode),
    quality: qualityOf(m.quant),
    download: m.weight,
  };
  row.pick = pickScore(row);
  return row;
}

function metricValue(row) {
  if (state.metric === "pick") return row.pick;
  if (state.metric === "speed") return row.decode;
  if (state.metric === "latency") return row.ttft;
  if (state.metric === "vram") return row.vram;
  if (state.metric === "maxctx") return row.maxctx;
  return row.pick;
}

function filteredRows() {
  const base = state.unique ? uniqueByDigest(DATA.models) : DATA.models.map((m) => ({ ...m, aliases: [] }));
  const size = SIZE_FILTERS.find((f) => f.id === state.size);
  const kind = KIND_FILTERS.find((f) => f.id === state.kind);
  const fit = FIT_FILTERS.find((f) => f.id === state.fit);
  const q = state.q.trim().toLowerCase();
  return base
    .map(viewRow)
    .filter((m) => (state.family === "all" ? true : m.model === state.family))
    .filter((m) => (state.quant === "all" ? true : m.quant === state.quant))
    .filter((m) => size.test(m))
    .filter((m) => kind.test(m))
    .filter((m) => fit.test(m))
    .filter((m) => {
      if (!q) return true;
      const blob = `${m.ref} ${m.arch} ${m.quant} ${m.model} ${familyName(m.arch)} ${m.quality.label} ${(m.aliases || []).join(" ")}`.toLowerCase();
      return blob.includes(q);
    });
}

function rankedRows() {
  const rows = filteredRows();
  const key = state.sortKey;
  const dir = state.sortDir === "asc" ? 1 : -1;
  const fitRank = { full: 2, partial: 1, none: 0 };
  rows.sort((a, b) => {
    let av;
    let bv;
    if (key === "fit") {
      av = fitRank[a.fit] ?? 0;
      bv = fitRank[b.fit] ?? 0;
    } else if (key === "ref") {
      return dir * a.ref.localeCompare(b.ref);
    } else if (key === "quality") {
      av = QUANT_FACTOR[a.quant] ?? 0;
      bv = QUANT_FACTOR[b.quant] ?? 0;
    } else {
      av = a[key];
      bv = b[key];
    }
    if (av == null && bv == null) return a.ref.localeCompare(b.ref);
    if (av == null) return 1;
    if (bv == null) return -1;
    if (av === bv) return a.ref.localeCompare(b.ref);
    return av > bv ? dir : -dir;
  });
  return rows;
}

function names(rows, n = 2) {
  return rows.slice(0, n).map((r) => r.ref);
}

function blurbFor(rows) {
  const gpu = gpuShort(gpuOf(state.gpu) || { name: state.gpu });
  const ctx = fmtCtx(state.ctx);
  if (!rows.length) return `No models match on ${gpu} at ${ctx}. Try All or a larger GPU.`;
  const [a, b] = names(rows);
  if (state.metric === "pick") return `${a} and ${b} rank highest that still run on ${gpu} at ${ctx}.`;
  if (state.metric === "speed") return `${a} and ${b} have the highest decode tok/s on ${gpu}.`;
  if (state.metric === "latency") return `${a} and ${b} have the lowest TTFT @ ${TTFT_PROMPT} on ${gpu}.`;
  if (state.metric === "vram") return `${a} and ${b} use the least VRAM at ${ctx}.`;
  return `${a} and ${b} stay fully resident to the longest context on ${gpu}.`;
}

function insightCopy(metricId) {
  const saved = { metric: state.metric, sortKey: state.sortKey, sortDir: state.sortDir };
  applyMetric(metricId);
  const rows = rankedRows();
  state.metric = saved.metric;
  state.sortKey = saved.sortKey;
  state.sortDir = saved.sortDir;
  const [a, b] = names(rows);
  if (!a) return "No models in this filter.";
  if (metricId === "pick") return `${a} and ${b} are the top recommended fits.`;
  if (metricId === "speed") return `${a} and ${b} lead decode tok/s.`;
  if (metricId === "latency") return `${a} and ${b} have the lowest TTFT.`;
  if (metricId === "vram") return `${a} and ${b} use the least VRAM.`;
  return `${a} and ${b} have the highest max full-GPU context.`;
}

function populateSelects() {
  $("gpu").innerHTML = GPU_GROUPS.map(([label, test]) => {
    const opts = DATA.gpus.filter(test).map((g) =>
      `<option value="${esc(g.id)}"${g.id === state.gpu ? " selected" : ""}>${esc(gpuShort(g))} (${g.mem} GB)</option>`
    ).join("");
    return `<optgroup label="${esc(label)}">${opts}</optgroup>`;
  }).join("");
  $("ctx").innerHTML = DATA.ctx_points.map((c) =>
    `<option value="${c}"${c === state.ctx ? " selected" : ""}>${fmtCtx(c)}</option>`
  ).join("");
}

function renderRig() {
  const g = gpuOf(state.gpu);
  if (!g) return;
  const usable = (g.mem * g.use).toFixed(1);
  const cal = DATA.calibrated_gpus.includes(g.id);
  $("rig-spec").innerHTML = cal
    ? `<strong>${esc(usable)} GB</strong> usable · calibrated`
    : `<strong>${esc(usable)} GB</strong> usable of ${esc(g.mem)} GB · uncalibrated estimates`;
}

function renderInsights() {
  $("insights").innerHTML = METRICS.map((m) => `
    <button type="button" class="insight" data-metric="${m.id}" aria-selected="${state.metric === m.id}">
      <span class="pill">${esc(m.label)}</span>
      <p>${esc(insightCopy(m.id))}</p>
    </button>
  `).join("");
}

function renderTabs() {
  $("tabs").innerHTML = METRICS.map((m) =>
    `<button type="button" class="tab" role="tab" data-metric="${m.id}" aria-selected="${state.metric === m.id}">${esc(m.label)}</button>`
  ).join("");
  $("view-list").setAttribute("aria-selected", state.view === "list" ? "true" : "false");
  $("view-table").setAttribute("aria-selected", state.view === "table" ? "true" : "false");
}

function optionList(values, current, allLabel, nameFn) {
  return values.map((id) => {
    const label = id === "all" ? allLabel : (nameFn ? nameFn(id) : id);
    return `<option value="${esc(id)}"${current === id ? " selected" : ""}>${esc(label)}</option>`;
  }).join("");
}

function renderFilters() {
  const families = ["all", ...[...new Set(DATA.models.map((m) => m.model))]];
  const quants = ["all", ...[...new Set(DATA.models.map((m) => m.quant).filter(Boolean))]];
  const uniqueChip = `<button type="button" class="chip" data-filter="unique" data-id="unique" aria-pressed="${state.unique}">${state.unique ? "Unique weights" : "All tags"}</button>`;
  const familySel = `<select data-filter="family" aria-label="Family">${optionList(families, state.family, "Family: all", (id) => id)}</select>`;
  const quantSel = `<select data-filter="quant" aria-label="Quant">${optionList(quants, state.quant, "Quant: all")}</select>`;
  const sizeChips = SIZE_FILTERS.map((f) =>
    `<button type="button" class="chip" data-filter="size" data-id="${f.id}" aria-pressed="${state.size === f.id}">${esc(f.label)}</button>`
  );
  const kindChips = KIND_FILTERS.map((f) =>
    `<button type="button" class="chip" data-filter="kind" data-id="${f.id}" aria-pressed="${state.kind === f.id}">${esc(f.label)}</button>`
  );
  const fitChips = FIT_FILTERS.map((f) =>
    `<button type="button" class="chip" data-filter="fit" data-id="${f.id}" aria-pressed="${state.fit === f.id}">${esc(f.label)}</button>`
  );
  $("filters").innerHTML = [uniqueChip, familySel, quantSel, ...fitChips, ...sizeChips, ...kindChips].join("");
}

function renderChart(rows) {
  const wrap = $("chart-wrap");
  wrap.hidden = state.view !== "table";
  if (state.view !== "table") {
    if (chart) { chart.destroy(); chart = null; }
    return;
  }
  const top = rows.slice(0, 10);
  const labels = top.map((r) => (r.ref.length > 26 ? `${r.ref.slice(0, 24)}…` : r.ref));
  const data = top.map((r) => {
    const v = metricValue(r);
    if (state.metric === "vram") return v == null ? 0 : v / 1e9;
    if (state.metric === "pick") return Math.max(0, v);
    if (state.metric === "latency") return r.ttft ?? 0;
    return v ?? 0;
  });
  if (chart) chart.destroy();
  if (!window.Chart) return;
  chart = new Chart($("rank-chart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ data, backgroundColor: "rgba(15, 110, 106, 0.72)", borderWidth: 0, borderRadius: 2 }],
    },
    options: {
      indexAxis: "y",
      animation: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 280 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => {
              const row = top[c.dataIndex];
              if (state.metric === "vram") return fmtGb(row.vram);
              if (state.metric === "maxctx") return fmtCtx(row.maxctx);
              if (state.metric === "pick") return familyName(row.arch);
              if (state.metric === "latency") return fmtSec(row.ttft);
              return fmtTps(row.decode);
            },
          },
        },
      },
      scales: {
        x: { grid: { color: "#e6ebf0" }, ticks: { font: { family: "IBM Plex Sans", size: 11 } } },
        y: { grid: { display: false }, ticks: { font: { family: "IBM Plex Sans", size: 11 } } },
      },
    },
  });
}

function sortAria(key) {
  if (state.sortKey !== key) return "none";
  return state.sortDir === "asc" ? "ascending" : "descending";
}

function ladder(m) {
  return DATA.ctx_points.map((c) => {
    const f = m.gpu[state.gpu]?.f[String(c)] || "none";
    const title = `${fmtCtx(c)}: ${fitLabel(f)}`;
    return `<i class="tick ${f}" title="${esc(title)}"></i>`;
  }).join("");
}

function meter(value, max, text) {
  const pct = max > 0 && value != null ? Math.max(2, Math.min(100, (value / max) * 100)) : 0;
  return `<div class="meter"><span class="fill" style="width:${pct}%"></span><span class="val">${esc(text)}</span></div>`;
}

function vramBars(r, maxVram) {
  return DATA.ctx_points.map((c) => {
    const b = r.vramMap[String(c)] ?? vramAt(r, c);
    const h = maxVram ? Math.max(8, (b / maxVram) * 100) : 8;
    const f = r.gpu[state.gpu]?.f[String(c)] || "";
    const color = f === "full" ? "var(--full)" : f === "partial" ? "var(--partial)" : "var(--none)";
    return `<div class="vram-col"><i style="height:${h}%;background:${color}"></i><span>${fmtCtx(c)}</span></div>`;
  }).join("");
}

function renderList(rows) {
  const maxVram = Math.max(0, ...rows.map((r) => r.vram || 0));
  $("list").hidden = state.view !== "list";
  if (state.view !== "list") {
    $("list").innerHTML = "";
    return;
  }
  $("list").innerHTML = rows.map((r, i) => {
    const open = state.open === r.ref;
    const alias = r.aliases?.length ? ` · ${r.aliases.length} alias${r.aliases.length > 1 ? "es" : ""}` : "";
    const cmd = `ollama run ${r.ref}`;
    const detail = open ? `
      <div class="card-detail">
        <dl class="detail-grid">
          <div><dt>Weights</dt><dd>${fmtGb(r.weight)}</dd></div>
          <div><dt>${r.experts ? "Active params" : "Params"}</dt><dd>${fmtParams(r.active)}${r.experts ? ` / ${fmtParams(r.params)}` : ""}</dd></div>
          <div><dt>Max full ctx</dt><dd>${r.maxctx ? fmtCtx(r.maxctx) : "—"}</dd></div>
          <div><dt>Quant</dt><dd>${esc(r.quant || "—")}${r.quality.hint ? ` · ${esc(r.quality.hint)}` : ""}</dd></div>
          <div><dt>TTFT @ ${TTFT_PROMPT}</dt><dd>${fmtSec(r.ttft)}${r.prefill && r.decode ? ` (${(TTFT_PROMPT / r.prefill).toFixed(2)}s prefill + ${(1 / r.decode).toFixed(2)}s decode)` : ""}</dd></div>
        </dl>
        <div class="vram-bars">${vramBars(r, maxVram)}</div>
      </div>` : "";
    return `
      <article class="card${open ? " is-open" : ""}" data-ref="${esc(r.ref)}" tabindex="0">
        <div class="card-top">
          <div>
            <h3>${esc(r.ref)}</h3>
            <p class="family-line">${esc(familyName(r.arch))} · #${i + 1}${esc(alias)}</p>
          </div>
          <span class="badge ${r.fit}">${esc(fitLabel(r.fit))}</span>
        </div>
        <p class="lede">${esc(lede(r))}</p>
        <ul class="pills">
          <li>${fmtGb(r.weight)}</li>
          <li>${fmtTps(r.decode)}</li>
          <li>${fmtSec(r.ttft)} TTFT</li>
          <li>${fmtGb(r.vram)} VRAM</li>
          <li>${esc(r.quant || "—")}</li>
          <li>${fmtCtx(r.ctx)} ctx</li>
        </ul>
        <div class="run">
          <code>${esc(cmd)}</code>
          <button type="button" class="copy" data-copy="${esc(cmd)}">Copy</button>
        </div>
        ${detail}
      </article>`;
  }).join("");
}

function renderTable(rows) {
  const wrap = $("table-wrap");
  wrap.hidden = state.view !== "table";
  if (state.view !== "table") return;
  const maxDecode = Math.max(0, ...rows.map((r) => r.decode || 0));
  const maxTtft = Math.max(0, ...rows.map((r) => r.ttft || 0));
  const maxVram = Math.max(0, ...rows.map((r) => r.vram || 0));
  const maxCtx = Math.max(0, ...rows.map((r) => r.maxctx || 0));

  $("grid").querySelector("thead").innerHTML = `
    <tr>
      <th class="sortable" data-sort="ref" aria-sort="${sortAria("ref")}">Model</th>
      <th>Family</th>
      <th class="sortable" data-sort="quality" aria-sort="${sortAria("quality")}">Quant</th>
      <th class="sortable num" data-sort="download" aria-sort="${sortAria("download")}">Weights</th>
      <th class="group sortable num" data-sort="decode" aria-sort="${sortAria("decode")}">Speed</th>
      <th class="sortable num" data-sort="ttft" aria-sort="${sortAria("ttft")}">Latency</th>
      <th class="group sortable num" data-sort="vram" aria-sort="${sortAria("vram")}">VRAM @ ${fmtCtx(state.ctx)}</th>
      <th class="sortable" data-sort="fit" aria-sort="${sortAria("fit")}">Fit</th>
      <th class="sortable num" data-sort="maxctx" aria-sort="${sortAria("maxctx")}">Max full ctx</th>
    </tr>
  `;

  $("grid").querySelector("tbody").innerHTML = rows.map((r, i) => {
    const open = state.open === r.ref;
    const detail = open ? `
      <tr class="detail"><td colspan="9">
        <p class="lede">${esc(lede(r))}</p>
        <div class="run"><code>ollama run ${esc(r.ref)}</code>
          <button type="button" class="copy" data-copy="ollama run ${esc(r.ref)}">Copy</button></div>
        <div class="vram-bars">${vramBars(r, maxVram)}</div>
      </td></tr>` : "";
    return `
      <tr data-ref="${esc(r.ref)}" class="${open ? "is-open" : ""}">
        <td class="model">
          <span class="ref">${esc(r.ref)}</span>
          <span class="sub">${fmtParams(r.params)} · #${i + 1}</span>
        </td>
        <td><span class="family"><i class="dot ${esc(r.arch)}"></i>${esc(familyName(r.arch))}</span></td>
        <td>${esc(r.quant || "—")}</td>
        <td class="num">${fmtGb(r.weight)}</td>
        <td class="group num">${meter(r.decode, maxDecode, fmtTps(r.decode))}</td>
        <td class="num">${meter(maxTtft && r.ttft != null ? maxTtft - r.ttft : null, maxTtft, fmtSec(r.ttft))}</td>
        <td class="group num">${meter(r.vram, maxVram, fmtGb(r.vram))}</td>
        <td>
          <span class="fit ${r.fit}">${esc(fitLabel(r.fit))}</span>
          <div class="ladder" aria-hidden="true">${ladder(r)}</div>
        </td>
        <td class="num">${meter(r.maxctx, maxCtx, r.maxctx ? fmtCtx(r.maxctx) : "—")}</td>
      </tr>${detail}`;
  }).join("");
}

function applyMetric(id) {
  state.metric = id;
  if (id === "speed") {
    state.sortKey = "decode";
    state.sortDir = "desc";
  } else if (id === "latency") {
    state.sortKey = "ttft";
    state.sortDir = "asc";
  } else {
    state.sortKey = id === "maxctx" ? "maxctx" : id;
    state.sortDir = id === "vram" ? "asc" : "desc";
  }
}

function readUrl() {
  const u = new URL(location.href);
  const gpu = u.searchParams.get("gpu");
  const ctx = Number(u.searchParams.get("ctx"));
  const metric = u.searchParams.get("metric");
  const view = u.searchParams.get("view");
  if (gpu) state.gpu = gpu;
  if (DATA.ctx_points.includes(ctx)) state.ctx = ctx;
  const mapped = { decode: "speed", prefill: "latency" }[metric] || metric;
  if (METRICS.some((m) => m.id === mapped)) applyMetric(mapped);
  if (view === "list" || view === "table") state.view = view;
}

function writeUrl() {
  const u = new URL(location.href);
  u.searchParams.set("gpu", state.gpu);
  u.searchParams.set("ctx", String(state.ctx));
  u.searchParams.set("metric", state.metric);
  u.searchParams.set("view", state.view);
  history.replaceState(null, "", u);
}

function render() {
  const wrap = document.querySelector(".table-wrap");
  const scrollY = wrap ? wrap.scrollTop : 0;
  const rows = rankedRows();
  renderRig();
  renderInsights();
  renderTabs();
  $("blurb").textContent = blurbFor(rows);
  renderFilters();
  renderChart(rows);
  renderList(rows);
  renderTable(rows);
  $("count").textContent = `${rows.length} models · context ${fmtCtx(state.ctx)}`;
  $("foot-note").textContent = DATA.calibrated_gpus.includes(state.gpu)
    ? "Speed (decode tok/s) and latency (TTFT) on this GPU are measured."
    : "Speed is decode tok/s from memory bandwidth. Latency is TTFT @ 128 tok (n/prefill + 1/decode). Uncalibrated until a bench run.";
  writeUrl();
  if (wrap) wrap.scrollTop = scrollY;
}

function copyText(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const prev = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = prev; }, 1200);
  }).catch(() => {
    btn.textContent = "Copy failed";
  });
}

function bind() {
  $("gpu").addEventListener("change", (e) => { state.gpu = e.target.value; state.open = null; render(); });
  $("ctx").addEventListener("change", (e) => { state.ctx = Number(e.target.value); state.open = null; render(); });
  $("q").addEventListener("input", (e) => { state.q = e.target.value; render(); });
  $("insights").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-metric]");
    if (!btn) return;
    applyMetric(btn.dataset.metric);
    render();
  });
  $("tabs").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-metric]");
    if (!btn) return;
    applyMetric(btn.dataset.metric);
    render();
  });
  document.querySelector(".view-toggle").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-view]");
    if (!btn) return;
    state.view = btn.dataset.view;
    render();
  });
  $("filters").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-filter]");
    if (!btn) return;
    const key = btn.dataset.filter;
    if (key === "unique") state.unique = !state.unique;
    else state[key] = btn.dataset.id;
    state.open = null;
    render();
  });
  $("filters").addEventListener("change", (e) => {
    const el = e.target.closest("select[data-filter]");
    if (!el) return;
    state[el.dataset.filter] = el.value;
    state.open = null;
    render();
  });
  const onCopy = (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    copyText(btn.dataset.copy, btn);
  };
  $("list").addEventListener("click", (e) => {
    if (e.target.closest("[data-copy]")) return onCopy(e);
    const card = e.target.closest("[data-ref]");
    if (!card) return;
    state.open = state.open === card.dataset.ref ? null : card.dataset.ref;
    render();
  });
  $("list").addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const card = e.target.closest("[data-ref]");
    if (!card) return;
    e.preventDefault();
    state.open = state.open === card.dataset.ref ? null : card.dataset.ref;
    render();
  });
  $("grid").addEventListener("click", (e) => {
    if (e.target.closest("[data-copy]")) return onCopy(e);
    const th = e.target.closest("th.sortable");
    if (th) {
      const key = th.dataset.sort;
      if (state.sortKey === key) state.sortDir = state.sortDir === "desc" ? "asc" : "desc";
      else {
        state.sortKey = key;
        state.sortDir = key === "vram" || key === "ttft" || key === "ref" ? "asc" : "desc";
      }
      render();
      return;
    }
    const tr = e.target.closest("tbody tr[data-ref]");
    if (!tr) return;
    state.open = state.open === tr.dataset.ref ? null : tr.dataset.ref;
    render();
  });
}

async function main() {
  const res = await fetch("data.json");
  if (!res.ok) {
    $("blurb").textContent = "Could not load data.json. Serve this folder over HTTP.";
    return;
  }
  DATA = await res.json();
  if (!gpuOf(state.gpu)) state.gpu = DATA.gpus[0].id;
  readUrl();
  populateSelects();
  $("gpu").value = state.gpu;
  $("ctx").value = String(state.ctx);
  bind();
  render();
}

main().catch((err) => {
  $("blurb").textContent = `Failed to start: ${err.message}`;
});
