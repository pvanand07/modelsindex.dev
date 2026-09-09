# MODELINDEX — Implementation Plan, Gates and Pass Criteria

Companion to [modal-test-plan.md](modal-test-plan.md). That document says *what* and *why*;
this one says *in what order*, *what must be true before moving on*, and *which script does it*.

## Repository layout

```
MODELINDEX_APP/
  docs/spec/modal-test-plan.md         project spec (sources, formulas, Modal test matrix)
  docs/spec/implementation-plan.md      this file
  scripts/gguf_header.py               Range-fetch + parse a GGUF header; tensor-table byte sums
  scripts/crawl.py                     enumerate library, fetch manifests/config/headers -> models.jsonl
  scripts/bench.py                     measurement protocol against an Ollama server -> measurements.jsonl
  scripts/modal_app.py                Modal App: CPU crawl/pull + GPU bench (replaces run_pod*.sh)
  scripts/fit.py                       fit constants from measurements -> constants.json + gate report
  scripts/build_index.py               models.jsonl x gpus.csv x constants.json -> index.json / index.csv
  data/gpus.csv                        static GPU table (seed; extend from dbgpu)
  data/cache/                          crawl cache, keyed by digest (gitignore)
  data/out/                            models.jsonl, measurements/*.jsonl, constants.json, index.*
  requirements.txt
```

`scripts/run_pod.sh` and `scripts/run_pod_crawl.sh` are leftover from the RunPod draft.
Do not use them. Delete once `modal_app.py` is in and G3 has passed once.

Dependencies: Python 3.10+. `crawl.py`, `gguf_header.py`, `bench.py` are stdlib only.
`fit.py` and `build_index.py` need `numpy`. `bench.py` uses `torch` only if present
(bandwidth test; the Modal GPU Image installs it). `scripts/modal_app.py` needs the
`modal` package locally (`pip install modal`) plus `modal token new` / `modal setup`.

## Phases, gates and pass criteria

Each phase ends at a gate. A gate is a list of checks that must all pass before the next phase
starts. Every check is something a script prints, not a judgment call.

### Phase 0 — Header parser and scaffold

Build `scripts/gguf_header.py` and `data/gpus.csv`.

**Gate G0**

| # | Check | Pass |
|---|---|---|
| G0.1 | `python scripts/gguf_header.py --selftest` | exits 0 |
| G0.2 | llama3.2:3b header via Range | `param_count == 3212749888`, `n_layer == 28`, `n_head_kv == 8`, `key_length == 128` |
| G0.3 | bytes fetched for that header | under 32 MB |
| G0.4 | tensor byte sums | `sum(groups) == weight_bytes` within 0.01% (offset method) |
| G0.5 | `data/gpus.csv` | loads, no empty bandwidth or memory cells; rows exist for `l4`, `a10`, `l40s`, `a100-80-sxm` (or pcie) |

### Phase 1 — Crawl

Build `scripts/crawl.py`. Run a small crawl first, then the full one. `crawl.py` is stdlib
Python doing HTTP GET/Range requests — no GPU. Run the full crawl as a **Modal CPU Function**
(`modal run scripts/modal_app.py --action crawl`) rather than a GPU Function: Modal CPU is
~$0.047/physical-core-hour plus memory, so a 4-core / 8 GiB job for under an hour is well
under a dollar, versus wasting GPU-second billing on pure network I/O. Persist cache and
`models.jsonl` on Volume `modelindex-data` (`/data/cache`, `/data/out`) so Phase 3/5 GPU
Functions remount it and do not re-fetch ~55 GB of headers. A laptop crawl still works for
the smoke (`--models llama3.2,qwen3,gemma3`); only the full library needs the CPU Function.

**Gate G1**

| # | Check | Pass |
|---|---|---|
| G1.1 | `crawl.py --models llama3.2,qwen3,gemma3` | all tags produce a row; `header_ok` true on every unique digest |
| G1.2 | MoE row (qwen3:30b-a3b) | `expert_count > 0`, `active_weight_bytes < 0.25 × weight_bytes` |
| G1.3 | SWA row (gemma3:4b) | `sliding_window > 0` |
| G1.3b | hybrid row (gemma4:e2b) | `n_swa_layers > 0`, `n_global_layers > 0`, `shared_kv_layers > 0`, `key_length_swa < key_length` |
| G1.4 | full crawl `crawl.py` (CPU Function) | ≥ 95% of tags have `header_ok`; summary printed with dedupe ratio |
| G1.5 | re-run with warm Volume | zero header fetches (cache hit on every digest) |

### Phase 2 — Local bench smoke test

Build `scripts/bench.py`. Run it on this laptop against the local Ollama (RTX 3060 Laptop 6 GB,
Ollama 0.33.3) with a reduced sweep. The point is to validate the script, not the numbers.

**Gate G2**

| # | Check | Pass |
|---|---|---|
| G2.1 | `bench.py --tier local --ctx 2048,4096 --speed-ctx 4096 --prompt-tokens 128 --runs 1 --num-predict 64` | exits 0, writes `env`, `model`, `vram`, `speed` records |
| G2.2 | `vram` record for llama3.2:3b at ctx 4096 | `ps_size` within 5% of what `ollama ps` shows (≈ 2.6 GB) and `fully_on_gpu == true` |
| G2.3 | `speed` record | `decode_tps`, `prefill_tps` numeric; `prompt_eval_count` ≥ 90% of requested tokens on every run (no prompt-cache hit) |
| G2.4 | throttle detection | on this laptop the run is `flagged: true` with mem clock ratio < 0.9 (known throttled GPU) |
| G2.5 | unload between ctx points | `/api/ps` empty before each load (script asserts) |

### Phase 3 — Modal P0 (L4)

Warm the Volume first, then bench. Budget: ~10 min CPU pull + 45 min L4, ≈ $0.80.

```
modal run scripts/modal_app.py --action pull --tier 24gb
modal run scripts/modal_app.py --action bench --gpu L4 --tier 24gb --kv-q8
```

**Gate G3**

| # | Check | Pass |
|---|---|---|
| G3.1 | all 24 GB-tier models complete | no model missing its `vram` and `speed` records |
| G3.2 | unflagged share | ≥ 90% of speed runs `flagged: false` |
| G3.3 | bandwidth test | `bw_measured_gbs` between 60% and 100% of spec (L4 spec = 300 GB/s) |
| G3.4 | SKU pin | `env` GPU name is an L4, not an unexpected substitute |
| G3.5 | export | `measurements/l4-<date>-main.jsonl` (and kvq8 jsonl) plus `server-*.log` on the Volume and copied locally via `modal volume get` |

If G3.2 fails, the host is power-limited: re-run the Function (new container, new host).
If G3.4 fails, the request was upgraded or mis-placed: fix `gpu=` and re-run.

### Phase 4 — Fit on P0 data

Build `scripts/fit.py`. Run it locally on the L4 JSONL joined with `models.jsonl`.

**Gate G4** (these are the spec's pass criteria, section 5.5)

| # | Check | Pass |
|---|---|---|
| G4.1 | decode fit `t = W_active/BW + n_layer×c + d` | R² > 0.95, n ≥ 8 |
| G4.2 | VRAM v2 (`capped_kv_graph_v2`, `constants_version: 2`) | global max relative error ≤ 5%; grouped structural holdout ≤ 5%; digest forward-context ≤ 2% |
| G4.3 | MoE | active-bytes prediction within 30% of measured; total-bytes prediction off by ≥ 5× |
| G4.4 | gemma3 capped-KV + graph | residuals ≤ 5% (replaces naive `kv_ratio` slope gate) |
| G4.4b | gemma4 hybrid KV + graph | residuals ≤ 5% when gemma4 measurements exist |
| G4.5 | quant factors | one factor per quant, all within 0.4–1.3 |
| G4.6 | prefill derate | one value per GPU, between 0.05 and 0.8 |
| G4.7 | partial offload | ≥ 3 points on the fraction-on-GPU vs tps curve |

`scripts/build_index.py` rejects legacy `constants_version: 1` artifacts. Prefer digest
offsets when present; otherwise use the validated global baseline coefficients.

If G4.1 fails, check G3.2 first (throttled runs leaking into the fit), then whether small
models dominate the residual (add a `n_layer²` term only if that fixes it).

### Phase 5 — Remaining GPUs

Pull the 48 GB-tier blobs onto the Volume (CPU), then bench L40S (`48gb`), A10 (`24gb`,
`--skip-pull` against the 24 GB blobs already there), then A100-80GB (`48gb` / 80 GB
card). Fit per GPU. Optional extras (not required for G5): `RTX-PRO-6000`, `H100!`, `B200`.

```
modal run scripts/modal_app.py --action pull --tier 48gb
modal run scripts/modal_app.py --action bench --gpu L40S --tier 48gb
modal run scripts/modal_app.py --action bench --gpu A10 --tier 24gb --include-partial
modal run scripts/modal_app.py --action bench --gpu A100-80GB --tier 48gb
```

Run these sequentially (one `modal run` at a time) unless the App is `modal deploy`ed.
Add `a10` to `data/gpus.csv` before the A10 fit if the row is missing.

**Gate G5**

| # | Check | Pass |
|---|---|---|
| G5.1 | per-GPU gates | G3 and G4 pass on every GPU in the core four (L4, L40S, A10, A100-80GB) |
| G5.2 | cross-GPU consistency | `c` and `d` within ±30% across GPUs of the same generation (L4 vs L40S Ada) |
| G5.3 | 70B fully resident (A100-80GB) | `fully_on_gpu` at ctx ≤ 32k; decode within 20% of fit |
| G5.4 | spend | total Modal billed compute ≤ $15 (planned ≈ $6; Starter grant is $30/mo) |

### Phase 6 — Build the index

Build `scripts/build_index.py`. Run it on the full crawl, the GPU table and `constants.json`.

**Gate G6**

| # | Check | Pass |
|---|---|---|
| G6.1 | coverage | every `header_ok` model × every GPU row present in `index.json` |
| G6.2 | spot check | for 5 measured (model, GPU, ctx) points, index VRAM within 5% and decode within 25% |
| G6.3 | fit status | matches measured `fully_on_gpu` on every 24 GB-card measurement |
| G6.4 | uncalibrated GPUs | rows carry `calibrated: false` and use documented defaults |

### Phase 7 — Refresh and publish

**Gate G7**

| # | Check | Pass |
|---|---|---|
| G7.1 | incremental crawl | new tags only; unchanged digests not refetched |
| G7.2 | G0 selftest | passes in CI on every change to the parser |

## Order of work and estimated effort

| Phase | Effort | Cost |
|---|---|---|
| 0 parser + scaffold | 2 h | 0 |
| 1 crawl | 3 h + ~30 min full crawl | ≈ $0.15 CPU |
| 2 local smoke | 1 h | 0 |
| 3 L4 run | 1 h + `modal_app.py` | ≈ $0.80 |
| 4 fit | 3 h | 0 |
| 5 remaining GPUs | 4 h wall, mostly waiting | ≈ $5 |
| 6 index | 2 h | 0 |
| 7 refresh | 1 h | 0 |

## Status (2026-09-08)

G0–G2 and G1.4 pass (full crawl 98.3% header_ok). G1.5 not separately re-run. Next: Modal pull smoke (llama3.2 1b+3b) then 24 GB-tier pull, then L4 bench (G3).

## How to run

```bash
# Phase 0
python scripts/gguf_header.py --selftest

# Phase 1 — Modal CPU. Default: smoke (llama3.2,qwen3,gemma3) then full library.
modal run scripts/modal_app.py --action crawl
# smoke only / full only:
# modal run scripts/modal_app.py --action crawl --smoke-only
# modal run scripts/modal_app.py --action crawl --skip-smoke

# Phase 2 (local Ollama must be running)
python scripts/bench.py --tier local --ctx 2048,4096 --speed-ctx 4096 --prompt-tokens 128 --runs 1 --num-predict 64 --out data/out/measurements/local.jsonl

# Phase 3 / 5 (Modal). Smoke pulls/benches llama3.2:1b+3b, then the full tier.
# One modal run at a time.
modal run scripts/modal_app.py --action pull --tier 24gb
modal run scripts/modal_app.py --action bench --gpu L4 --tier 24gb --kv-q8
modal run scripts/modal_app.py --action pull --tier 48gb
modal run scripts/modal_app.py --action bench --gpu L40S --tier 48gb
modal run scripts/modal_app.py --action bench --gpu A10 --tier 24gb --include-partial
modal run scripts/modal_app.py --action bench --gpu A100-80GB --tier 48gb

modal volume get modelindex-data /out/measurements ./data/out/measurements

# Phase 4 / 5
python scripts/fit.py --measurements data/out/measurements/*.jsonl --models data/out/models.jsonl --gpus data/gpus.csv --out data/out/constants.json

# Phase 6
python scripts/build_index.py --models data/out/models.jsonl --gpus data/gpus.csv --constants data/out/constants.json --out data/out/index.json
```
