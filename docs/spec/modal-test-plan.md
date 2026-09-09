# MODELINDEX — Project Spec and Modal Calibration Test Plan

Status: draft v2, 2026-09-08. v1 targeted RunPod Community Cloud. This rewrite
targets Modal. All registry facts in section 2 were verified live on 2026-09-07.
GPU SKUs and prices in section 5 are Modal on-demand as of 2026-09-08.

## 1. Goal

Build a local model index for Ollama library models: for every `model:tag` and every GPU in a
static hardware table, publish VRAM required as a function of context length, an estimated
decode tokens/sec, an estimated prefill tokens/sec, and a fit status (`full | partial | none`).
The dataset is built **without downloading any model weights**, from registry manifests and
GGUF headers fetched by HTTP Range request, joined against a static GPU spec table. The
estimation constants are calibrated once against a small measured set collected on Modal
(section 5).

## 2. Data sources (verified 2026-09-07)

| Source | Test | Result |
|---|---|---|
| Registry manifest | `GET https://registry.ollama.ai/v2/library/llama3.2/manifests/3b` | 200, JSON, no auth. Exact `size` in bytes per layer. |
| Config blob | `GET .../blobs/{config.digest}` (~560 B) | `model_type: "3.2B"`, `file_type: "Q4_K_M"`, `model_family: "llama"` |
| Weight blob, Range | `curl -r 0-63 .../blobs/{weight digest}` | 307 to Cloudflare R2 presigned URL, then **206 Partial Content**, `Accept-Ranges: bytes` |
| GGUF header size | parse header of llama3.2:3b via Range | **7.8 MB** (KV metadata ends at byte 7,822,528; tensor table ends at 7,837,658) |
| Param count from tensor table | sum of tensor dims | 3,212,749,888 for llama3.2:3b |
| Registry tags list | `GET .../v2/library/llama3.2/tags/list` | **404** |
| `https://ollama.com/api/tags` | GET | 200 but only **19 featured models**, not the library |
| `https://ollama.com/library` | GET HTML | **239 models** via `href="/library/{name}"` |
| `https://ollama.com/library/llama3.2/tags` | GET HTML | 63 tags, each with on-disk size text |

### 2.1 Registry endpoints

```
GET https://registry.ollama.ai/v2/library/{model}/manifests/{tag}
GET https://registry.ollama.ai/v2/library/{model}/blobs/{digest}        # follow 307; supports Range
```

Manifest layer media types that matter:

- `application/vnd.ollama.image.model` — the GGUF weight file. Its `size` is the ground-truth weight bytes.
- `application/vnd.ollama.image.projector` — vision projector (mmproj). Add to VRAM when present.
- `application/vnd.docker.container.image.v1+json` — the config blob. Use it for param size, quant and family instead of parsing tag strings.

### 2.2 GGUF header

The header is large because the tokenizer vocab (128K strings for Llama 3) lives in the
metadata. Fetch the first 16–24 MB in a single Range request, parse until the tensor table
ends, and cache by digest forever (digests are immutable). The header gives:

- `general.architecture`, `general.file_type`, `general.size_label`
- `{arch}.block_count`, `.context_length`, `.embedding_length`, `.feed_forward_length`
- `{arch}.attention.head_count`, `.attention.head_count_kv`, `.attention.key_length`, `.attention.value_length`
- `{arch}.vocab_size`
- MoE models: `{arch}.expert_count`, `{arch}.expert_used_count`
- SWA models: `{arch}.attention.sliding_window`
- Hybrid SWA (gemma4): `{arch}.attention.sliding_window_pattern`,
  `.attention.shared_kv_layers`, `.attention.key_length_swa` /
  `.value_length_swa`, per-layer `.attention.head_count_kv`
- The **full tensor table**: name, dims, dtype for every tensor. This yields an exact
  parameter count and exact per-tensor-group byte sums (embeddings, output, attention,
  dense FFN, expert FFN) without touching the weights.

### 2.3 Library enumeration

There is no JSON listing API. Scrape:

1. `https://ollama.com/library` → model names (239 as of 2026-09-07).
2. `https://ollama.com/library/{name}/tags` → every tag with its size.
3. For each tag, fetch the manifest. **Dedupe by weight digest** before fetching headers;
   many tags alias one blob (e.g. `latest`, `8b`, `8b-instruct-q4_K_M`).

### 2.4 GPU hardware table

Static, checked into the repo.

- Discrete GPUs: derive from [dbgpu](https://github.com/painebenjamin/dbgpu) or
  [RightNow GPU Database](https://github.com/RightNow-AI/RightNow-GPU-Database)
  (both TechPowerUp-derived): VRAM, memory bandwidth GB/s, FP16 tensor TFLOPS, generation.
- **Manual additions** (missing from those DBs): Apple Silicon M-series (unified memory,
  usable cap ≈ 75% of system RAM, bandwidth per chip), CPU-only rows for DDR4 and DDR5
  dual-channel (~50 and ~90 GB/s), and laptop GPU variants with their lower power limits.
- **Modal SKUs used for calibration** must exist as rows (`l4`, `a10`, `l40s`,
  `a100-80-sxm` or `a100-80-pcie`, `h100-80-sxm`). Add `a10`, `rtx-pro-6000`, `b200`
  if missing before Phase 5. The published index still covers consumer cards the
  user does not rent (4090, 5090, …) via spec BW + generation defaults; those rows
  stay `calibrated: false` until measured.

## 3. Formulas

### 3.1 Weights

```
weight_bytes        = manifest weight layer size                         (exact)
projector_bytes     = manifest projector layer size, else 0              (exact)
active_weight_bytes = non_expert_bytes + expert_bytes × (expert_used_count / expert_count)
```

`non_expert_bytes` and `expert_bytes` are summed from the tensor table; expert tensors are
the `ffn_*_exps` names. Dense models: `active_weight_bytes = weight_bytes`.
**VRAM uses total bytes. Decode speed uses active bytes.** Using total bytes for a MoE such as
qwen3:30b-a3b under-predicts speed by roughly 10×.

### 3.2 KV cache

```
kv_bytes_per_token = n_layer × n_head_kv × (key_length + value_length) × bytes_per_elem
effective_ctx      = min(num_ctx, sliding_window) when 0 < sliding_window < num_ctx, else num_ctx
kv_cache_bytes     = kv_bytes_per_token × effective_ctx
```

Uniform SWA (gemma3): cap every layer at `sliding_window`. Dense models: no cap.

Hybrid attention (gemma4): do **not** apply one window to every layer. Parse
`sliding_window_pattern` (local vs global), `shared_kv_layers` (trailing layers
alias a donor cache), `key_length_swa` / `value_length_swa`, and per-layer
`head_count_kv`. Only layers `[0, n_layer - shared_kv_layers)` allocate:

```
kv_cache = kv_local_per_tok × min(ctx, sliding_window) + kv_global_per_tok × ctx
```

Local layers use the SWA head dim; global layers use the full head dim and grow
with context. `kv_bytes_per_token_f16` on a hybrid row is `local + global` at one
token (not a slope to multiply by `effective_ctx`).

E2B/E4B Per-Layer Embedding tables (`per_layer_token_embd`) are host-RAM lookups,
not GPU decode weights. Subtract `ple_bytes` from GPU VRAM and from
`active_weight_bytes` / `active_params`. The Ollama `gemma4:e2b` Q4 blob is ~7.2 GB
because it ships those embeddings in F16 next to Q4 transformer weights.

- Use the header's `key_length` / `value_length`; do not derive head_dim from `embedding_length / head_count`.
- `bytes_per_elem` = 2 (f16, Ollama default). q8_0 (1 byte) only when the user sets
  `OLLAMA_KV_CACHE_TYPE=q8_0` **and** flash attention is enabled; for q8 KV, halve only
  `kv_cache_bytes` (graph and baseline stay unchanged).
- Sliding-window models (gemma3) cap KV growth at `sliding_window`; do not use a naive
  full-context KV slope for them.

### 3.3 Compute graph overhead and baseline

Graph scratch scales with heads and context (not a flat 500 MB). Baseline absorbs
tensor-table / output-tensor residency that is not in weights+KV+graph. Fitted constants
live in `constants.json` under `constants_version: 2` / formula `capped_kv_graph_v2`
(`scripts/vram_model.py`). Prefer a per-digest residual offset when calibrated; otherwise
use the global baseline coefficients.

```
graph_bytes(ctx) = 2048 × (n_head + 1) × num_ctx
                 # gemma4 hybrid: 2048 × (n_head + 1) ×
                 #   (n_swa_layers × min(ctx, sw) + n_global_layers × ctx) / n_layer
baseline         = b0 + b_tensor × tensor_count + b_output × output_tensor_bytes
                   (or digest_offset when known)
total_vram(ctx)  = weight_bytes + projector_bytes + kv_cache_bytes(ctx)
                   + graph_bytes(ctx) + baseline
```

Target is Ollama `ps_size` (full residency). Partial-offload points are excluded from the
full-demand fit. Report VRAM as a curve at 2k / 4k / 8k / 16k / 32k / 64k / 128k, capped at
the model's `context_length`.

### 3.4 Decode speed

Roofline ceiling (bandwidth-bound, batch size 1):

```
decode_tps_ceiling = BW_bytes_per_sec / active_weight_bytes
```

Calibrated estimate, fitted per GPU family:

```
t_token          = active_weight_bytes / BW_measured + n_layer × c + d
decode_tps_est   = 1 / t_token
```

`c` is per-layer kernel-launch overhead and `d` is a fixed per-token cost. Without them the
roofline overshoots badly for small models (< 3B). Publish the ceiling and the estimate as
separate columns.

### 3.5 Prefill speed

Compute-bound. `prefill_tps ≈ (FP16_TFLOPS × derate) / (2 × active_params)`, with one derate
per GPU generation fitted in section 5.

### 3.6 Fit status

Per (model, GPU, ctx): `full` if `total_vram ≤ usable_vram`, `partial` if weights alone
exceed it but Ollama can offload some layers, `none` otherwise. Partial-offload speed is a
curve versus fraction of layers on GPU and is characterized, not derived.

### 3.7 Why measurement is mandatory

Local test on this machine, llama3.2:3b q4_K_M, 29/29 layers on an RTX 3060 Laptop 6 GB:

| Source | Decode tokens/sec |
|---|---|
| Roofline (336 GB/s ÷ 2.02 GB) | ≈ 165 |
| Measured (`eval_count / eval_duration`) | **6.2** |

The GPU was pinned in P8 at 210 MHz SM / 405 MHz memory, 100% utilization, throttle reasons
`0x24` (SW power cap + SW thermal). A static index cannot see this, and a crowdsourced number
like it would poison calibration. **Every measurement must carry clock and throttle context.**

## 4. Dataset schema

```
model | tag | digest | param_count | quant | weight_bytes | active_weight_bytes | projector_bytes
| n_layer | n_head | n_head_kv | key_length | value_length | context_length | vocab_size
| expert_count | expert_used_count | sliding_window
| [gpu -> { fit_status, vram_at_ctx{2k,4k,8k,16k,32k,64k,128k}, decode_tps_ceiling, decode_tps_est, prefill_tps_est }]
```

## 5. Modal calibration test plan

Budget: $20 of billed compute. Planned spend ≈ $6. Reserve ~$6 for re-runs; $8 untouched.
Starter plan includes **$30/month free credit**, so the planned matrix should land inside
the grant. Modal bills per second for GPU, CPU cores, and memory; prices below are
on-demand as of 2026-09-08 from [modal.com/pricing](https://modal.com/pricing). Volume
storage is $0.09/GiB-month with **1 TiB/month free** — the Ollama blob cache (~150 GB)
fits in the free allowance.

v1 rented RunPod Community Cloud 4090 / A40 / 5090 / A100 PCIe pods and SSHed in.
Modal does not offer those consumer SKUs. The matrix below is remapped to Modal's
catalog while keeping the same scientific questions (section 5.5).

### 5.0 Design decisions (RunPod → Modal)

1. **Job shape: Modal Functions, not SSH pods.** v1 used `runpod/pytorch` + `run_pod.sh`.
   Modal's matching pattern is a Python App (`scripts/modal_app.py`): bake Ollama and
   `scripts/` into an Image, mount a Volume, run `bench.py` inside `@app.function`.
   Interactive GPU sandboxes (the `modal-gpu-dev` skill) are for debugging `bench.py`
   only. Do not leave a sandbox billed while the protocol runs.

2. **GPU catalog is datacenter, not consumer.** Modal has T4, L4, A10, L40S, A100-40/80,
   RTX PRO 6000, H100/H200, B200/B300. There is no 4090, 3090, 5090, or A40. The index
   still *publishes* those consumer rows from spec BW; only the calibration set changes.
   Local RTX 3060 Laptop remains the throttled-laptop datapoint (Phase 2).

3. **P0 is L4, not 4090.** L4 is Ada (same generation as 4090), 24 GB, 300 GB/s, and the
   cheapest Modal GPU that fits the 24 GB tier ($0.80/hr). Lower bandwidth is a *stronger*
   test of the decode roofline than a 1008 GB/s 4090. Ada constants `c`/`d` then transfer
   to L40S (also Ada) and, with a generation derate, toward unpublished 4090 rows.

4. **Pin exact SKUs.** Modal may upgrade `gpu="H100"` → H200 and `gpu="A100"` → A100-80GB
   at no extra cost. Silent upgrades poison a calibration run. Request `H100!`,
   `A100-80GB`, never a fallback list, never `B200+`.

5. **Pull weights on CPU, bench on GPU.** `ollama pull` is network I/O. On RunPod that
   happened on a rented GPU. On Modal, a CPU Function writing into the same Volume is
   ~$0.05–0.20 for 55–150 GB, then every GPU Function runs with `--skip-pull`. First GPU
   no longer eats 10–20 minutes of GPU-hour billing on downloads.

6. **One Volume, three directories.** `modelindex-data` at `/data`:
   `cache/` (GGUF headers), `ollama-models/` (`OLLAMA_MODELS`), `out/` (jsonl + server
   logs). Call `volume.commit()` before every Function returns. Export with
   `modal volume get`, not `scp`.

7. **Do not `modal run` the same App concurrently.** Ephemeral Apps with the same name kill
   each other. Phase 5 either runs GPUs sequentially from one `local_entrypoint` that
   `.spawn()`s after the Volume is warm, or `modal deploy`s once and calls the deployed
   Function (sub-agents skill). Sequential is the default; wall-clock is not the bottleneck.

8. **Datacenter cards throttle less than Community Cloud.** Keep the clock/throttle
   protocol anyway. If G3.2 fails, re-run; Modal will place the next container on a
   different host.

### 5.1 GPU matrix

Prices = Modal on-demand $/sec × 3600. Planned minutes include bench only (pull is a
separate CPU Function).

| GPU | Modal `gpu=` | VRAM | Spec BW | FP16 TFLOPS | $/hr | Planned | Cost | Role |
|---|---|---|---|---|---|---|---|---|
| L4 | `L4` | 24 GB | 300 GB/s | 121 | 0.80 | 45 min | $0.60 | **P0 anchor**, Ada, low BW |
| A10 | `A10` | 24 GB | 600 GB/s | 125 | 1.10 | 45 min | $0.83 | Ampere, GDDR6 |
| L40S | `L40S` | 48 GB | 864 GB/s | 362 | 1.95 | 75 min | $1.44 | Ada 48 GB, 27B–70B, MoE at long ctx |
| A100 80GB | `A100-80GB` | 80 GB | 2039 GB/s | 312 | 2.50 | 60 min | $2.50 | HBM, 70B fully resident |
| RTX PRO 6000 | `RTX-PRO-6000` | 96 GB | 1792 GB/s | — | 3.03 | 45 min | $2.27 | Optional, Blackwell GDDR7 |
| H100 SXM | `H100!` | 80 GB | 3350 GB/s | 989 | 3.95 | 30 min | $1.98 | Optional, Hopper; pin no H200 upgrade |
| B200 | `B200` | 192 GB | 8000 GB/s | — | 6.25 | 30 min | $3.13 | Optional, Blackwell HBM |

T4 (16 GB) is skipped: the 24 GB-tier list does not fit.

Priority if time-limited: **L4 → L40S → A10 + A100-80GB**. Those four answer every
question in 5.5. Optional cards are generation extras, not blockers.

Core four planned GPU spend ≈ **$5.37**, plus CPU crawl/pull ≈ $0.30, plus L4 kv-q8
pass ≈ $0.20 → **≈ $6**.

### 5.2 Model matrix

Grouped by the single variable each group isolates. All are existing library tags.

| Group | Tags | Isolates |
|---|---|---|
| Size ladder | llama3.2:1b, llama3.2:3b, llama3.1:8b, llama3.3:70b (all q4_K_M) | per-layer floor `c`, bandwidth slope |
| Quant ladder | llama3.1:8b at q4_0, q4_K_M, q8_0, fp16 | per-quant efficiency; fp16 @ 32k on 24 GB = controlled partial offload |
| MoE | qwen3:30b-a3b, gpt-oss:20b | active-bytes hypothesis |
| Sliding window | gemma3:4b, gemma3:12b, gemma3:27b | SWA KV slope |
| Cross-family | qwen2.5:14b, qwen2.5:32b, phi4:14b | constants generalize across arch and vocab |
| Forced partial | llama3.3:70b on a 24 GB card | offload cliff curve |

Tiers: **24 GB** (`L4`, `A10`) = everything except the 27b / 32b / 70b tags (~55 GB on
the Volume). **48 GB** (`L40S`) and **80 GB** (`A100-80GB`, `H100!`) = everything
(~150 GB on the Volume). Container overlay disk is not used for weights;
`OLLAMA_MODELS=/data/ollama-models` on the Volume.

### 5.3 App and image

One Modal App, two Images, one Volume. No SSH, no `run_pod.sh`.

```python
# scripts/modal_app.py — sketch; see implementation-plan.md for gates
import modal

app = modal.App("modelindex")
vol = modal.Volume.from_name("modelindex-data", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("curl", "ca-certificates")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("torch")          # bandwidth test in bench.py
    .add_local_dir("scripts", remote_path="/root/scripts")
)
```

CPU Function (crawl + pull): `cpu=4`, `memory=8192`, `timeout=60*60`, no `gpu=`.
GPU Function (bench): `timeout=90*60`, `gpu=` set at the call site with
`.with_options(gpu=...)`. Env inside the bench container:

```
OLLAMA_MODELS=/data/ollama-models
OLLAMA_NUM_PARALLEL=1
OLLAMA_FLASH_ATTENTION=0
OLLAMA_KV_CACHE_TYPE=f16
```

`OLLAMA_NUM_PARALLEL=1` is critical: the default can allocate KV for several slots, which
multiplies the cache and makes the context slope look several times too steep.

Start `ollama serve` in the Function body (same pattern as `run_pod.sh`: kill, env,
background, wait for `/api/version`). The kv-q8 pass restarts the server in-process with
`OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0`. Do not use `@modal.enter` for the
server if the kv-q8 pass needs a restart — keep start/stop in the method.

Retries: **off** for bench (`retries=0`). A mid-sweep retry would duplicate JSONL and
waste GPU time. Crawl may retry on network errors.

### 5.4 Protocol

One `bench.py` per GPU Function, one JSONL line per measurement. Results under
`/data/out/measurements/{gpu}-{stamp}-{label}.jsonl` plus `server-*.log`. Commit the
Volume before return.

1. **Environment capture (once).** GPU name, driver, Ollama version, power limit, max SM and
   memory clocks from `nvidia-smi -q`. Then a 10 s torch copy benchmark on a 1 GB tensor →
   **achieved bandwidth**. This is the `BW_measured` the fit divides by, not the spec number.
   Record the exact Modal SKU (`nvidia-smi` name) so an unexpected H200 upgrade is obvious.
2. **VRAM sweep, per model.** Load at `num_ctx` ∈ {2k, 4k, 8k, 16k, 32k}, plus 64k and 128k
   on the 48/80 GB cards, never above the header `context_length`. At each point record the
   `ollama ps` SIZE and PROCESSOR columns, the nvidia-smi memory delta, and the server-log
   line `offloaded N/N layers to GPU`. Unload (keep_alive 0) between points.
3. **Speed sweep, per model.** `num_ctx` ∈ {4k, 32k}. Prompts of 128, 1024 and 4096 tokens.
   `num_predict=256`, `temperature=0`, fixed `seed`. One discarded warm-up, then 3 runs.
   `decode_tps = eval_count / eval_duration × 1e9`,
   `prefill_tps = prompt_eval_count / prompt_eval_duration × 1e9`. Keep the median.
4. **Clock sampling.** Poll every 1 s during every run: pstate, SM clock, memory clock, power
   draw, `clocks_throttle_reasons.active`. Flag any run whose memory clock is below 90% of
   max and **exclude flagged runs from all fits**.
5. **KV-quant pass (L4 only).** Restart the server with `OLLAMA_FLASH_ATTENTION=1
   OLLAMA_KV_CACHE_TYPE=q8_0` and repeat step 2 for llama3.1:8b and qwen3:30b-a3b.
6. **Export.** `volume.commit()`, then from the laptop:
   `modal volume get modelindex-data /out/measurements ./data/out/measurements`.

Time per model ≈ 3 min plus (already-finished) pull, which yields the per-GPU minutes in 5.1.

### 5.5 Pass criteria

| Question | Fit | Pass threshold |
|---|---|---|
| VRAM vs context | `resident = W+proj + capped_KV + graph(n_head,ctx) + baseline` (`constants_version: 2`) | global max relative error ≤ 5%; structural holdout max ≤ 5%; digest-offset forward-context max ≤ 2%; no partial points in the full-demand fit |
| Decode speed | `t = W_active / BW_measured + n_layer × c + d` per GPU | R² > 0.95 |
| MoE active bytes | measured vs active-bytes vs total-bytes prediction | active within 30%; total off by ≥ 5× |
| SWA / capped KV | gemma3 residuals under capped-KV + graph | max relative error ≤ 5% |
| Quant efficiency | tps ratio across q4_0 / q4_K_M / q8_0 / fp16 | one factor per quant per GPU |
| Prefill | pp tps vs FP16 TFLOPS | monotone; one derate per GPU generation |
| Partial offload | tps vs fraction of layers on GPU | a curve, not a constant |

Before broad publication, treat L40S / A10 / A100 runs as out-of-sample checks that digest
offsets and the graph coefficient are GPU-independent; fall back to global metadata
coefficients if portability fails.

### 5.6 Pitfalls

- Modal may auto-upgrade H100→H200 and A100→A100-80GB. Pin `H100!` / `A100-80GB`. Treat a
  mismatch between requested SKU and `nvidia-smi` name as a failed run.
- Do not specify GPU fallback lists for calibration.
- Concurrent `modal run` of the same App name aborts the other. Sequential, or deploy-once.
- Forget `volume.commit()` and the JSONL is gone when the container exits.
- Overlay disk is too small for 55–150 GB of GGUF. Weights live only on the Volume.
- Image build: `add_local_dir("scripts", ...)` must use `copy=True` if any
  `.run_commands()` follows it.
- Ollama reloads the runner whenever `num_ctx` changes. Order the sweep by ctx **within** a
  model, not by model within a ctx.
- `num_ctx` above the model's trained context is silently capped; the recorded KV size will
  not match the requested ctx. Always cap at the header value.
- Function timeout defaults to 300 s. Bench needs 60–90 min. Set it explicitly.
- Community-Cloud-style power-limited hosts are rarer on Modal, but step 4 still catches
  them; do not average over flagged runs.

## 6. Next steps

1. `scripts/modal_app.py` is built. Authenticate (`modal setup`), then:
   `modal run scripts/modal_app.py --action crawl` (G1.4–G1.5),
   `modal run scripts/modal_app.py --action pull --tier 24gb`,
   `modal run scripts/modal_app.py --action bench --gpu L4 --tier 24gb --kv-q8` (G3).
2. `scripts/fit.py` — fit `c`, `d`, quant factors, SWA factor and graph overhead model from
   the JSONL; emit the constants table consumed by the index builder. Runs locally. Already built.
3. Remaining GPUs (G5), then `scripts/build_index.py` and publish the section 4 schema.

## 7. Sources

- Ollama registry and layers: https://deepwiki.com/ollama/ollama/4.2-model-registry-and-layers
- How Ollama stores models: https://medium.com/@enisbaskapan/how-ollama-stores-models-11fc47f48955
- Remote GGUF parsing via Range: https://huggingface.co/docs/huggingface.js/gguf/README and https://github.com/hyparam/hyllama
- GPU spec databases: https://github.com/painebenjamin/dbgpu , https://github.com/RightNow-AI/RightNow-GPU-Database
- Crowdsourced Ollama tokens/sec: https://ollamatps.com/ , https://github.com/MinhNgyuen/llm-benchmark
- llama.cpp standardized `llama-bench` tables: Apple Silicon discussion #4167 and the NVIDIA performance discussions on github.com/ggml-org/llama.cpp
- LLM inference roofline survey: https://arxiv.org/pdf/2402.16363
- Modal GPUs: https://modal.com/docs/guide/gpu
- Modal Volumes: https://modal.com/docs/guide/volumes
- Modal pricing: https://modal.com/pricing
- Modal dynamic GPU config: https://modal.com/docs/guide/dynamic-function-config
