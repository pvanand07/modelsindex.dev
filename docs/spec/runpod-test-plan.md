
Set up Runpod for me: fetch https://docs.runpod.io/agent-setup.md and follow it.

# MODELINDEX — Project Spec and RunPod Calibration Test Plan

Status: draft v1, 2026-09-07. All registry facts in section 2 were verified live on that date.

## 1. Goal

Build a local model index for Ollama library models: for every `model:tag` and every GPU in a
static hardware table, publish VRAM required as a function of context length, an estimated
decode tokens/sec, an estimated prefill tokens/sec, and a fit status (`full | partial | none`).
The dataset is built **without downloading any model weights**, from registry manifests and
GGUF headers fetched by HTTP Range request, joined against a static GPU spec table. The
estimation constants are calibrated once against a small measured set collected on RunPod
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
kv_cache_bytes     = kv_bytes_per_token × num_ctx
```

- Use the header's `key_length` / `value_length`; do not derive head_dim from `embedding_length / head_count`.
- `bytes_per_elem` = 2 (f16, Ollama default). q8_0 (1 byte) only when the user sets
  `OLLAMA_KV_CACHE_TYPE=q8_0` **and** flash attention is enabled.
- Sliding-window models (gemma3) are overestimated by this formula because llama.cpp shrinks
  the SWA layers' cache. Calibrate a per-architecture factor in section 5.

### 3.3 Compute graph overhead

Not a flat 500 MB. It scales with context length and vocab size (logits buffer, attention
scratch). Local datapoint: llama3.2:3b q4_K_M, 2.02 GB weights → **2.6 GB resident** at
`num_ctx=4096` per `ollama ps`. Preferred approach: port Ollama's own estimator from
`llm/memory.go`, since Ollama's decision about full offload is what users experience, and
validate/fit its constants with section 5 data.

```
total_vram(ctx) = weight_bytes + projector_bytes + kv_cache_bytes(ctx) + graph_bytes(ctx, vocab)
```

Report VRAM as a curve at 2k / 4k / 8k / 16k / 32k / 64k / 128k, capped at the model's
`context_length`.

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

## 5. RunPod calibration test plan

Budget: $20. Planned spend ≈ $4. Reserve $6 for re-runs and failed pulls; $10 untouched.
RunPod bills per second; prices below are Community Cloud on-demand as of 2026-09-07.

### 5.1 GPU matrix

| GPU | VRAM | Spec BW | $/hr | Planned | Cost | Role |
|---|---|---|---|---|---|---|
| RTX A5000 | 24 GB | 768 GB/s | 0.16 | 45 min | $0.12 | Ampere, low power |
| RTX 3090 | 24 GB | 936 GB/s | 0.22 | 45 min | $0.17 | Ampere consumer |
| RTX 4090 | 24 GB | 1008 GB/s | 0.34 | 45 min | $0.26 | **P0 anchor**, Ada |
| L4 | 24 GB | 300 GB/s | 0.44 | 45 min | $0.33 | Low bandwidth, high compute ratio |
| RTX 5090 | 32 GB | 1792 GB/s | 0.69 | 45 min | $0.52 | Blackwell, GDDR7 |
| A40 | 48 GB | 696 GB/s | 0.35 | 75 min | $0.44 | 27B–70B dense, MoE at long ctx |
| A100 PCIe | 80 GB | 2039 GB/s | 1.19 | 60 min | $1.19 | HBM, 70B fully resident |
| H100 PCIe | 80 GB | 2000 GB/s | 1.99 | 30 min | $1.00 | Optional, Hopper |

Priority if time-limited: **4090 → A40 → L4 + 5090 → A100**. Those four answer every
question in 5.5.

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

Tiers: **24 GB** = everything except the 27b / 32b / 70b tags (~55 GB pulled; 80 GB
container disk). **48 / 80 GB** = everything (150 GB container disk).

### 5.3 Pod setup

Use the `runpod/pytorch` template (Python, nvidia-smi, torch present), not the Ollama image.

```bash
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=0 OLLAMA_KV_CACHE_TYPE=f16
ollama serve > /workspace/server.log 2>&1 &
```

`OLLAMA_NUM_PARALLEL=1` is critical: the default can allocate KV for several slots, which
multiplies the cache and makes the context slope look several times too steep.

### 5.4 Protocol

One `bench.py` per pod, one JSONL line per measurement.

1. **Environment capture (once).** GPU name, driver, Ollama version, power limit, max SM and
   memory clocks from `nvidia-smi -q`. Then a 10 s torch copy benchmark on a 1 GB tensor →
   **achieved bandwidth**. This is the `BW_measured` the fit divides by, not the spec number.
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
5. **KV-quant pass (4090 only).** Restart the server with `OLLAMA_FLASH_ATTENTION=1
   OLLAMA_KV_CACHE_TYPE=q8_0` and repeat step 2 for llama3.1:8b and qwen3:30b-a3b.
6. **Export.** `scp` the JSONL and `server.log` out over the pod's SSH before terminating.

Time per model ≈ 3 min plus pull, which yields the per-GPU minutes in 5.1.

### 5.5 Pass criteria

| Question | Fit | Pass threshold |
|---|---|---|
| VRAM vs context | `resident = weights + kv(ctx) + graph(ctx, vocab)` | within 5% at every ctx point |
| Decode speed | `t = W_active / BW_measured + n_layer × c + d` per GPU | R² > 0.95 |
| MoE active bytes | measured vs active-bytes vs total-bytes prediction | active within 30%; total off by ≥ 5× |
| SWA KV slope | gemma3 measured slope vs naive formula | naive overestimates by a measurable, stable factor |
| Quant efficiency | tps ratio across q4_0 / q4_K_M / q8_0 / fp16 | one factor per quant per GPU |
| Prefill | pp tps vs FP16 TFLOPS | monotone; one derate per GPU generation |
| Partial offload | tps vs fraction of layers on GPU | a curve, not a constant |

### 5.6 Pitfalls

- Community Cloud hosts occasionally run power-limited cards. Step 4 catches this; do not
  average over flagged runs.
- Ollama reloads the runner whenever `num_ctx` changes. Order the sweep by ctx **within** a
  model, not by model within a ctx.
- `num_ctx` above the model's trained context is silently capped; the recorded KV size will
  not match the requested ctx. Always cap at the header value.
- Model load time on Community Cloud disks is noisy but does not affect `eval_duration`.

## 6. Next steps

1. `scripts/crawl.py` — walk the 239 library pages, dedupe by weight digest, cache
   manifests, config blobs and GGUF headers under `data/cache/{digest}/`.
2. `scripts/bench.py` + `scripts/run_pod.sh` — implement 5.3–5.4 with `--tier 24gb|48gb`
   model lists and JSONL output.
3. `scripts/fit.py` — fit `c`, `d`, quant factors, SWA factor and graph overhead model from
   the JSONL; emit the constants table consumed by the index builder.
4. Build and publish the dataset in the section 4 schema.

## 7. Sources

- Ollama registry and layers: https://deepwiki.com/ollama/ollama/4.2-model-registry-and-layers
- How Ollama stores models: https://medium.com/@enisbaskapan/how-ollama-stores-models-11fc47f48955
- Remote GGUF parsing via Range: https://huggingface.co/docs/huggingface.js/gguf/README and https://github.com/hyparam/hyllama
- GPU spec databases: https://github.com/painebenjamin/dbgpu , https://github.com/RightNow-AI/RightNow-GPU-Database
- Crowdsourced Ollama tokens/sec: https://ollamatps.com/ , https://github.com/MinhNgyuen/llm-benchmark
- llama.cpp standardized `llama-bench` tables: Apple Silicon discussion #4167 and the NVIDIA performance discussions on github.com/ggml-org/llama.cpp
- LLM inference roofline survey: https://arxiv.org/pdf/2402.16363
- RunPod pricing: https://www.runpod.io/pricing , https://gpus.io/en/providers/runpod , https://computeprices.com/providers/runpod
