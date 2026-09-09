#!/usr/bin/env python3
"""
Measurement protocol against a running Ollama server (spec section 5.4).

Writes one JSON line per record to --out:
  kind=env    once: GPU, driver, Ollama version, achieved bandwidth (torch, optional)
  kind=model  per model: digest, size, /api/show details and scalar model_info
  kind=vram   per (model, ctx): resident bytes from /api/ps, nvidia-smi delta, offload line
  kind=speed  per (model, ctx, prompt_len): median decode/prefill tok/s, TTFT, clock/throttle

    python scripts/bench.py --tier local --ctx 2048,4096 --speed-ctx 4096 --prompt-tokens 128 --runs 1 --num-predict 64
    python scripts/bench.py --tier 24gb --out /data/out/measurements/l4.jsonl --server-log /data/out/measurements/server.log

Stdlib only; uses torch for the bandwidth test if importable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from progress import Bar, ProgressFile  # noqa: E402

# --------------------------------------------------------------------------- model tiers
SIZE_LADDER = ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"]
QUANT_LADDER = ["llama3.1:8b-instruct-q4_0", "llama3.1:8b-instruct-q4_K_M",
                "llama3.1:8b-instruct-q8_0", "llama3.1:8b-instruct-fp16"]
MOE = ["gpt-oss:20b", "qwen3:30b-a3b"]
SWA = ["gemma3:4b", "gemma3:12b"]
CROSS = ["qwen2.5:14b", "phi4:14b"]
BIG = ["gemma3:27b", "qwen2.5:32b", "llama3.3:70b"]
TIERS = {
    "local": ["llama3.2:1b", "llama3.2:3b"],
    "24gb": SIZE_LADDER + QUANT_LADDER + MOE + SWA + CROSS,
    "48gb": SIZE_LADDER + QUANT_LADDER + MOE + SWA + CROSS + BIG,
}
PARTIAL_EXTRA = ["llama3.3:70b"]  # forced partial offload on a 24 GB card (--include-partial)

DEFAULT_CTX = [2048, 4096, 8192, 16384, 32768]
BIG_CTX = [65536, 131072]
NVSMI_FIELDS = ("pstate,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu,"
                "clocks_throttle_reasons.active,memory.used")
NVSMI_STATIC = "name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem"

BASE_TEXT = (
    "The bicycle emerged in the early nineteenth century as a wooden running machine without pedals, "
    "and over the following decades inventors added cranks, chains, pneumatic tyres and gears until the "
    "safety bicycle of the 1880s set the pattern still used today. Cities built lanes, clubs organised "
    "tours, and factories in Coventry, Saint-Etienne and Chicago turned out millions of frames. The "
    "machine changed how people worked, courted and travelled, and it did so at a price a clerk could "
    "afford. Later the motor car pushed it aside in many countries, but the oil shocks and the climate "
    "debate brought it back, and today the bicycle is again a serious part of urban transport policy."
)

HOST = "http://localhost:11434"


# --------------------------------------------------------------------------- helpers
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def api(method: str, path: str, payload: dict | None = None, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(HOST + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def nvsmi(fields: str) -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    vals = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
    return dict(zip(fields.split(","), vals))


def to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class ClockSampler(threading.Thread):
    """Polls nvidia-smi every `interval` s while a run is in progress."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict] = []
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            s = nvsmi(NVSMI_FIELDS)
            if s:
                s["t"] = time.time()
                self.samples.append(s)
            self._stop_event.wait(self.interval)

    def stop(self) -> list[dict]:
        self._stop_event.set()
        self.join(timeout=5)
        return self.samples


def summarize_clocks(samples: list[dict], max_mem_clock: float | None) -> dict:
    if not samples:
        return {"n": 0, "flagged": None}
    mem = [to_num(s.get("clocks.mem")) for s in samples]
    mem = [m for m in mem if m is not None]
    sm = [to_num(s.get("clocks.sm")) for s in samples]
    sm = [m for m in sm if m is not None]
    power = [to_num(s.get("power.draw")) for s in samples]
    power = [p for p in power if p is not None]
    reasons = sorted({s.get("clocks_throttle_reasons.active", "") for s in samples})
    pstates = sorted({s.get("pstate", "") for s in samples})
    ratio = (min(mem) / max_mem_clock) if (mem and max_mem_clock) else None
    return {
        "n": len(samples),
        "mem_clock_min": min(mem) if mem else None,
        "mem_clock_max": max(mem) if mem else None,
        "mem_clock_ratio_to_max": round(ratio, 3) if ratio is not None else None,
        "sm_clock_min": min(sm) if sm else None,
        "power_draw_max": max(power) if power else None,
        "pstates": pstates,
        "throttle_reasons": reasons,
        "flagged": (ratio is not None and ratio < 0.9),
    }


def torch_bandwidth_gbs() -> float | None:
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    n = 256 * 1024 * 1024  # 1 GB of fp32
    a = torch.empty(n, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del a, b
    torch.cuda.empty_cache()
    return round(2 * n * 4 * iters / dt / 1e9, 1)  # read + write


def make_prompt(n_tokens: int, nonce: str) -> str:
    words = BASE_TEXT.split()
    n_words = max(8, int((n_tokens - 45) / 1.3))  # ~45 tokens of nonce + instruction; ~1.3 tok/word
    body = " ".join(words[i % len(words)] for i in range(n_words))
    return f"[{nonce}] Read the following text, then continue it in the same style.\n\n{body}"


def loaded_models() -> list[dict]:
    return api("GET", "/api/ps").get("models", [])


def unload_all(timeout: float = 90) -> None:
    for m in loaded_models():
        try:
            api("POST", "/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=30)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            pass
    t0 = time.time()
    while loaded_models():
        if time.time() - t0 > timeout:
            raise RuntimeError("models still loaded after unload request")
        time.sleep(0.25)
    time.sleep(0.25)


def load_model(model: str, ctx: int) -> dict:
    return api("POST", "/api/generate", {
        "model": model, "prompt": "Hi", "stream": False, "keep_alive": "30m",
        "options": {"num_ctx": ctx, "num_predict": 1},
    }, timeout=300)


def generate(model: str, prompt: str, ctx: int, num_predict: int, seed: int) -> dict:
    """Stream the completion so a hung Ollama cannot sit silent for the full HTTP timeout."""
    payload = {
        "model": model, "prompt": prompt, "stream": True, "keep_alive": "30m",
        "options": {"num_ctx": ctx, "num_predict": num_predict, "temperature": 0, "seed": seed},
    }
    req = urllib.request.Request(
        HOST + "/api/generate",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    last: dict = {}
    with urllib.request.urlopen(req, timeout=60) as r:
        while True:
            line = r.readline()
            if not line:
                break
            line = line.decode("utf-8", "replace").strip()
            if not line:
                continue
            rec = json.loads(line)
            last = rec
            if rec.get("done"):
                return rec
    if last.get("done"):
        return last
    raise TimeoutError("generate stream ended without a done packet")


def completed_models(paths: list[Path], skip_speed: bool) -> set[str]:
    """Models that already have a model row, VRAM points, and (unless skip_speed) speed points."""
    stats: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = rec.get("model")
            if not model:
                continue
            slot = stats.setdefault(model, {"model": False, "vram": 0, "speed": 0})
            kind = rec.get("kind")
            if kind == "model":
                slot["model"] = True
            elif kind == "vram":
                slot["vram"] += 1
            elif kind == "speed":
                slot["speed"] += 1
    return {
        m for m, s in stats.items()
        if s["model"] and s["vram"] >= 1 and (skip_speed or s["speed"] >= 1)
    }


def offload_line(server_log: str | None) -> dict | None:
    if not server_log or not os.path.exists(server_log):
        return None
    last = None
    size = os.path.getsize(server_log)
    with open(server_log, encoding="utf-8", errors="replace") as f:
        if size > 256 * 1024:
            f.seek(size - 256 * 1024)
            f.readline()
        for line in f:
            if "offloaded" in line and "layers" in line:
                last = line.strip()
    if not last:
        return None
    m = re.search(r"offloaded (\d+)/(\d+) layers", last)
    return {"line": last[-200:], "layers_gpu": int(m.group(1)) if m else None,
            "layers_total": int(m.group(2)) if m else None}


def scalar_model_info(info: dict) -> dict:
    return {k: v for k, v in info.items() if not isinstance(v, (list, dict))}


# --------------------------------------------------------------------------- main protocol
def main(argv=None) -> int:
    global HOST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--tier", choices=sorted(TIERS), default="local")
    ap.add_argument("--models", help="comma-separated explicit list (overrides --tier)")
    ap.add_argument("--include-partial", action="store_true", help="add the forced partial-offload model")
    ap.add_argument("--ctx", default=",".join(map(str, DEFAULT_CTX)), help="VRAM sweep contexts")
    ap.add_argument("--ctx-big", action="store_true", help="also 65536 and 131072 (48/80 GB cards)")
    ap.add_argument("--speed-ctx", default="4096,32768", help="contexts for the speed sweep")
    ap.add_argument("--prompt-tokens", default="128,1024,4096")
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--skip-speed", action="store_true", help="VRAM sweep only (KV-quant pass)")
    ap.add_argument("--no-bandwidth", action="store_true")
    ap.add_argument("--server-log", help="path to the ollama serve log (offload lines)")
    ap.add_argument("--label", default="main")
    ap.add_argument("--out", default="data/out/measurements/local.jsonl")
    ap.add_argument("--resume", action="store_true",
                    help="skip models that already have vram+speed rows in --out or sibling *-{label}.jsonl")
    ap.add_argument("--progress", help="atomic JSON progress file (default: <out-dir>/job_progress.json)")
    a = ap.parse_args(argv)
    HOST = a.host.rstrip("/")

    models = a.models.split(",") if a.models else list(TIERS[a.tier])
    if a.include_partial:
        models += [m for m in PARTIAL_EXTRA if m not in models]
    ctxs = sorted({int(x) for x in a.ctx.split(",")} | (set(BIG_CTX) if a.ctx_big else set()))
    speed_ctxs = sorted({int(x) for x in a.speed_ctx.split(",")})
    prompt_lens = sorted({int(x) for x in a.prompt_tokens.split(",")})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    already: set[str] = set()
    if a.resume:
        siblings = [p for p in out.parent.glob(f"*-{a.label}.jsonl") if p.resolve() != out.resolve()]
        already = completed_models([out, *siblings], skip_speed=a.skip_speed)
        if already:
            print(f"[resume] skipping {len(already)} complete models: {sorted(already)}", flush=True)
    fout = open(out, "a", encoding="utf-8")
    prog = ProgressFile(a.progress or (out.parent / "job_progress.json"), job="bench")
    prog.update(label=a.label, out=str(out), models_total=len(models), models_done=[],
                models_remaining=len(models) - len(already), gpu=None)
    prog.start()

    gpu_holder = {"name": None}

    def emit(rec: dict) -> None:
        rec["ts"] = now_iso()
        rec["label"] = a.label
        rec.setdefault("gpu", gpu_holder["name"])
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    def speed_at_ctx(model: str, ctx: int) -> None:
        for plen in prompt_lens:
            if plen + a.num_predict + 64 > ctx:
                continue
            prog.update(phase="speed", model=model, ctx=ctx, prompt=plen, emit_log=True)
            sampler = ClockSampler()
            sampler.start()
            runs = []
            try:
                for i in range(a.runs + 1):  # first run is warm-up
                    prog.update(phase="generate", model=model, ctx=ctx, prompt=plen, run=i,
                                phase_t0=time.time(), emit_log=True)
                    nonce = f"{model}-{ctx}-{plen}-{i}-{time.time_ns()}"
                    r = generate(model, make_prompt(plen, nonce), ctx, a.num_predict, a.seed)
                    pe, ped = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0)
                    ec, ed = r.get("eval_count", 0), r.get("eval_duration", 0)
                    prefill_ms = ped / 1e6 if ped else None
                    tok_ms = (ed / ec / 1e6) if (ed and ec) else None
                    ttft_ms = (prefill_ms + tok_ms) if (prefill_ms is not None and tok_ms is not None) else None
                    runs.append({
                        "warmup": i == 0, "prompt_eval_count": pe, "eval_count": ec,
                        "prefill_tps": round(pe / ped * 1e9, 2) if ped else None,
                        "decode_tps": round(ec / ed * 1e9, 2) if ed else None,
                        "prompt_eval_ms": round(prefill_ms, 2) if prefill_ms is not None else None,
                        "eval_ms": round(ed / 1e6) if ed else None,
                        "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                        "total_ms": round(r.get("total_duration", 0) / 1e6),
                        "load_ms": round(r.get("load_duration", 0) / 1e6),
                    })
            except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as e:
                sampler.stop()
                emit({"kind": "error", "model": model, "ctx": ctx, "prompt_tokens_requested": plen,
                      "error": f"generate: {type(e).__name__}: {e}"})
                print(f"[skip] generate ctx={ctx} prompt={plen}: {e}", flush=True)
                continue
            samples = sampler.stop()
            real = [x for x in runs if not x["warmup"] and x["decode_tps"]]
            clocks = summarize_clocks(samples, max_mem_clock)
            rec = {
                "kind": "speed", "model": model, "ctx": ctx, "gpu": gpu_name,
                "prompt_tokens_requested": plen, "num_predict": a.num_predict,
                "prompt_eval_count": int(statistics.median(x["prompt_eval_count"] for x in real)) if real else None,
                "eval_count": int(statistics.median(x["eval_count"] for x in real)) if real else None,
                "decode_tps": round(statistics.median(x["decode_tps"] for x in real), 2) if real else None,
                "prefill_tps": round(statistics.median(x["prefill_tps"] for x in real if x["prefill_tps"]), 2) if any(x["prefill_tps"] for x in real) else None,
                "ttft_ms": round(statistics.median(x["ttft_ms"] for x in real if x["ttft_ms"] is not None), 2) if any(x.get("ttft_ms") is not None for x in real) else None,
                "prompt_eval_ms": round(statistics.median(x["prompt_eval_ms"] for x in real if x.get("prompt_eval_ms") is not None), 2) if any(x.get("prompt_eval_ms") is not None for x in real) else None,
                "runs": runs, "clocks": clocks, "flagged": clocks.get("flagged"),
            }
            emit(rec)
            print(f"[speed] ctx={ctx:>6} prompt={rec['prompt_eval_count']:>5} "
                  f"decode={rec['decode_tps']} t/s prefill={rec['prefill_tps']} t/s "
                  f"ttft={rec['ttft_ms']} ms "
                  f"memclk_ratio={clocks.get('mem_clock_ratio_to_max')} flagged={rec['flagged']}",
                  flush=True)

    # ---- env
    version = api("GET", "/api/version").get("version")
    static = nvsmi(NVSMI_STATIC) or {}
    gpu_name = static.get("name") or platform.processor() or "unknown"
    gpu_holder["name"] = gpu_name
    max_mem_clock = to_num(static.get("clocks.max.mem"))
    env = {
        "kind": "env", "host": socket.gethostname(), "platform": platform.platform(),
        "ollama_version": version, "gpu": gpu_name, "nvidia": static,
        "ollama_env": {k: v for k, v in os.environ.items() if k.startswith("OLLAMA_")},
        "bw_measured_gbs": None if a.no_bandwidth else torch_bandwidth_gbs(),
        "args": vars(a),
    }
    emit(env)
    print(f"[env] ollama {version} on {gpu_name}; bw_measured={env['bw_measured_gbs']} GB/s")
    prog.update(gpu=gpu_name, bw_measured_gbs=env["bw_measured_gbs"], emit_log=True)

    tag_index = {m["name"]: m for m in api("GET", "/api/tags").get("models", [])}
    seen_digests: dict[str, str] = {}
    remaining = [m for m in models if m not in already]
    done_models = [m for m in models if m in already]
    models_bar = Bar("bench models", len(models), done=len(models) - len(remaining))
    prog.update(models_done=done_models, models_remaining=len(remaining), phase="models")

    for model in remaining:
        try:
            print(f"\n=== {model} ===", flush=True)
            prog.update(phase="model", model=model, ctx=None, prompt=None, emit_log=True)
            if not a.skip_pull:
                t0 = time.time()
                api("POST", "/api/pull", {"model": model, "stream": False}, timeout=7200)
                print(f"[pull] {model} ready in {time.time() - t0:.0f}s")
                tag_index = {m["name"]: m for m in api("GET", "/api/tags").get("models", [])}
            entry = tag_index.get(model) or tag_index.get(model + ":latest") or {}
            digest = entry.get("digest")
            if digest and digest in seen_digests:
                print(
                    f"[skip] {model} same ollama digest as {seen_digests[digest]} "
                    f"(identical weights, {digest[:12]})",
                    flush=True,
                )
                continue
            if digest:
                seen_digests[digest] = model
            try:
                show = api("POST", "/api/show", {"model": model})
            except urllib.error.HTTPError as e:
                print(f"[skip] {model}: /api/show -> HTTP {e.code} (not pulled?)")
                emit({"kind": "error", "model": model, "error": f"show HTTP {e.code}"})
                continue
            info = scalar_model_info(show.get("model_info", {}))
            arch = info.get("general.architecture", "")
            trained_ctx = int(info.get(f"{arch}.context_length", 0) or 0)
            emit({"kind": "model", "model": model, "digest": digest, "size": entry.get("size"),
                  "details": show.get("details"), "model_info": info, "capabilities": show.get("capabilities"),
                  "trained_context_length": trained_ctx})

            model_ctxs = [c for c in ctxs if not trained_ctx or c <= trained_ctx] or [min(ctxs)]
            speed_ok = {c for c in speed_ctxs if c in model_ctxs}
            ran_speed: set[int] = set()

            for ctx in model_ctxs:
                unload_all()
                base = nvsmi("memory.used")
                base_used = to_num(base.get("memory.used")) if base else None
                t0 = time.time()
                try:
                    prog.update(phase="load", model=model, ctx=ctx, emit_log=True)
                    r = load_model(model, ctx)
                except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as e:
                    emit({"kind": "error", "model": model, "ctx": ctx, "error": f"load: {type(e).__name__}: {e}"})
                    print(f"[skip] load ctx={ctx}: {e}", flush=True)
                    break
                load_s = time.time() - t0
                ps = next((m for m in loaded_models() if m["name"] in (model, model + ":latest")), None)
                during = nvsmi("memory.used")
                during_used = to_num(during.get("memory.used")) if during else None
                rec = {
                    "kind": "vram", "model": model, "ctx": ctx, "gpu": gpu_name,
                    "ps_size": ps.get("size") if ps else None,
                    "ps_size_vram": ps.get("size_vram") if ps else None,
                    "ps_context_length": ps.get("context_length") if ps else None,
                    "fully_on_gpu": (ps is not None and ps.get("size_vram") == ps.get("size")),
                    "nvsmi_used_delta_mib": (during_used - base_used) if (during_used is not None and base_used is not None) else None,
                    "load_duration_ms": r.get("load_duration", 0) / 1e6,
                    "wall_load_s": round(load_s, 2),
                    "offload": offload_line(a.server_log),
                }
                emit(rec)
                print(f"[vram] ctx={ctx:>6} size={rec['ps_size']} vram={rec['ps_size_vram']} "
                      f"delta={rec['nvsmi_used_delta_mib']} MiB full={rec['fully_on_gpu']}")
                prog.update(phase="vram", model=model, ctx=ctx, fully_on_gpu=rec["fully_on_gpu"])
                if not a.skip_speed and ctx in speed_ok:
                    speed_at_ctx(model, ctx)
                    ran_speed.add(ctx)

            if not a.skip_speed:
                for ctx in speed_ok - ran_speed:
                    unload_all()
                    try:
                        load_model(model, ctx)
                    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as e:
                        emit({"kind": "error", "model": model, "ctx": ctx, "error": f"load: {type(e).__name__}: {e}"})
                        print(f"[skip] load ctx={ctx}: {e}", flush=True)
                        continue
                    speed_at_ctx(model, ctx)
        finally:
            done_models.append(model)
            prog.update(models_done=done_models, models_remaining=max(0, len(models) - len(done_models)))
            models_bar.update(item=model)
    models_bar.finish()

    unload_all()
    print(f"\n[done] wrote {out}", flush=True)
    fout.close()
    prog.close(status="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
