# MODELINDEX — Implementation Plan, Gates and Pass Criteria

Companion to [runpod-test-plan.md](runpod-test-plan.md). That document says *what* and *why*;
this one says *in what order*, *what must be true before moving on*, and *which script does it*.

## Repository layout

```
MODELINDEX_APP/
  docs/spec/runpod-test-plan.md        project spec (sources, formulas, test matrix)
  docs/spec/implementation-plan.md     this file
  scripts/gguf_header.py               Range-fetch + parse a GGUF header; tensor-table byte sums
  scripts/crawl.py                     enumerate library, fetch manifests/config/headers -> models.jsonl
  scripts/bench.py                     measurement protocol against an Ollama server -> measurements.jsonl
  scripts/run_pod.sh                   RunPod GPU-pod bootstrap: install Ollama, set env, run bench.py
  scripts/run_pod_crawl.sh             RunPod CPU-pod bootstrap: run crawl.py (no GPU needed)
  scripts/fit.py                       fit constants from measurements -> constants.json + gate report
  scripts/build_index.py               models.jsonl x gpus.csv x constants.json -> index.json / index.csv
  data/gpus.csv                        static GPU table (seed; extend from dbgpu)
  data/cache/                          crawl cache, keyed by digest (gitignore)
  data/out/                            models.jsonl, measurements/*.jsonl, constants.json, index.*
  requirements.txt
```

Dependencies: Python 3.10+. `crawl.py`, `gguf_header.py`, `bench.py` are stdlib only.
`fit.py` and `build_index.py` need `numpy`. `bench.py` uses `torch` only if present (bandwidth test).

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
| G0.5 | `data/gpus.csv` | loads, no empty bandwidth or memory cells |

### Phase 1 — Crawl

Build `scripts/crawl.py`. Run a small crawl first, then the full one. `crawl.py` is stdlib
Python doing HTTP GET/Range requests — no GPU. Run the full crawl on a **RunPod CPU-only pod**
(`scripts/run_pod_crawl.sh`) rather than a GPU pod: CPU pods bill by the vCPU-hour (roughly
$0.03–0.04/vCPU-hr), so a 4-vCPU pod for under an hour costs well under a dollar, versus wasting
GPU-hour billing on pure network I/O. Attach a RunPod Network Volume at `/workspace` so the
header/manifest cache persists and can be re-mounted by the Phase 3/5 GPU pods, avoiding a
second ~55 GB of header downloads. Any GPU-free VPS works the same way if you'd rather not use
RunPod credits for this step.

**Gate G1**

| # | Check | Pass |
|---|---|---|
| G1.1 | `crawl.py --models llama3.2,qwen3,gemma3` | all tags produce a row; `header_ok` true on every unique digest |
| G1.2 | MoE row (qwen3:30b-a3b) | `expert_count > 0`, `active_weight_bytes < 0.25 × weight_bytes` |
| G1.3 | SWA row (gemma3:4b) | `sliding_window > 0` |
| G1.4 | full crawl `crawl.py` | ≥ 95% of tags have `header_ok`; summary printed with dedupe ratio |
| G1.5 | re-run with warm cache | zero header fetches (cache hit on every digest) |

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

### Phase 3 — RunPod P0 (RTX 4090)

Run `scripts/run_pod.sh 24gb` on one Community Cloud 4090. Budget: 45 min, ≈ $0.26.

**Gate G3**

| # | Check | Pass |
|---|---|---|
| G3.1 | all 24 GB-tier models complete | no model missing its `vram` and `speed` records |
| G3.2 | unflagged share | ≥ 90% of speed runs `flagged: false` |
| G3.3 | bandwidth test | `bw_measured_gbs` between 60% and 100% of spec |
| G3.4 | export | `measurements/rtx4090-<date>.jsonl` and `server.log` copied off the pod before termination |

If G3.2 fails, the host is power-limited: terminate and rent a different 4090.

### Phase 4 — Fit on P0 data

Build `scripts/fit.py`. Run it on the 4090 JSONL joined with `models.jsonl`.

**Gate G4** (these are the spec's pass criteria, section 5.5)

| # | Check | Pass |
|---|---|---|
| G4.1 | decode fit `t = W_active/BW + n_layer×c + d` | R² > 0.95, n ≥ 8 |
| G4.2 | VRAM model | max relative error ≤ 5% across every (model, ctx) point |
| G4.3 | MoE | active-bytes prediction within 30% of measured; total-bytes prediction off by ≥ 5× |
| G4.4 | gemma3 KV slope | measured slope / naive slope is stable across 4b and 12b (±15%) and < 1 |
| G4.5 | quant factors | one factor per quant, all within 0.4–1.3 |
| G4.6 | prefill derate | one value per GPU, between 0.05 and 0.8 |
| G4.7 | partial offload | ≥ 3 points on the fraction-on-GPU vs tps curve |

If G4.1 fails, check G3.2 first (throttled runs leaking into the fit), then whether small
models dominate the residual (add a `n_layer²` term only if that fixes it).

### Phase 5 — Remaining GPUs

Run `run_pod.sh 48gb` on A40, then `24gb` on L4 and 5090, then `48gb` on A100. Fit per GPU.

**Gate G5**

| # | Check | Pass |
|---|---|---|
| G5.1 | per-GPU gates | G3 and G4 pass on every GPU |
| G5.2 | cross-GPU consistency | `c` and `d` within ±30% across GPUs of the same generation |
| G5.3 | 70B fully resident (A100) | `fully_on_gpu` at ctx ≤ 32k; decode within 20% of fit |
| G5.4 | spend | total RunPod spend ≤ $10 |

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
| 1 crawl | 3 h + ~30 min full crawl | 0 |
| 2 local smoke | 1 h | 0 |
| 3 4090 run | 1 h | $0.26 |
| 4 fit | 3 h | 0 |
| 5 remaining GPUs | 4 h wall, mostly waiting | ≈ $3 |
| 6 index | 2 h | 0 |
| 7 refresh | 1 h | 0 |

## Status (2026-09-07)

G0, G1.1–G1.3 and G2 have been run and pass (see the session log: selftest PASS; 18/18 and 58/58 tags parsed; local bench records written, throttle flagged). G1.4–G1.5 (full crawl) and G3 onward are pending.

## How to run

```bash
# Phase 0
python scripts/gguf_header.py --selftest

# Phase 1
python scripts/crawl.py --models llama3.2,qwen3,gemma3
python scripts/crawl.py                       # full library

# Phase 2 (local Ollama must be running)
python scripts/bench.py --tier local --ctx 2048,4096 --speed-ctx 4096 --prompt-tokens 128 --runs 1 --num-predict 64 --out data/out/measurements/local.jsonl
# or, against models already pulled (no download):
python scripts/bench.py --models llama3.2 --skip-pull --ctx 2048,4096 --speed-ctx 4096 --prompt-tokens 128 --runs 2 --num-predict 64 --server-log "$LOCALAPPDATA/Ollama/server.log" --out data/out/measurements/local.jsonl

# Phase 3 / 5 (on the pod)
bash scripts/run_pod.sh 24gb                  # or 48gb; add --kv-q8 for the 4090 extra pass

# Phase 4 / 5
python scripts/fit.py --measurements data/out/measurements/*.jsonl --models data/out/models.jsonl --gpus data/gpus.csv --out data/out/constants.json

# Phase 6
python scripts/build_index.py --models data/out/models.jsonl --gpus data/gpus.csv --constants data/out/constants.json --out data/out/index.json
```
