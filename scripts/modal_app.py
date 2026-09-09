#!/usr/bin/env python3
"""
Modal App for the calibration protocol (spec section 5.3 / 5.4).

CPU Functions crawl the library and hydrate GGUF blobs onto a Volume
(parallel registry downloads into Ollama's on-disk layout).
GPU Functions start Ollama and run bench.py against those blobs.

Each action runs a real-data smoke (sample crawl / 1b+3b pull / short L4 process)
and only then the full job. `--smoke-only` stops after smoke. `--skip-smoke` is full only.

    modal run scripts/modal_app.py --action crawl
    modal run scripts/modal_app.py --action pull --tier 24gb
    modal run scripts/modal_app.py --action bench --gpu L4 --tier 24gb --kv-q8
    modal run scripts/modal_app.py --action crawl --smoke-only
    modal run scripts/modal_app.py --action pull --tier 24gb --skip-smoke

    modal volume get --force modelindex-data /out/measurements ./data/out

One `modal run` at a time (ephemeral Apps with the same name abort each other).
Pin exact SKUs: L4, A10, L40S, A100-80GB, H100!, RTX-PRO-6000, B200.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "modelindex"
VOL_NAME = "modelindex-data"
DATA = "/data"
SCRIPTS = "/root/scripts"
OLLAMA_HOST = "127.0.0.1:11434"
ALLOWED_GPUS = ("L4", "A10", "L40S", "A100-80GB", "A100-40GB", "H100!", "RTX-PRO-6000", "B200")
TIERS = ("24gb", "48gb")
# Real-data smoke samples (not fixtures). Crawl hits MoE + SWA; pull/bench use tiny GGUFs.
SMOKE_CRAWL_MODELS = "llama3.2,qwen3,gemma3"
SMOKE_PULL_MODELS = "llama3.2:1b,llama3.2:3b"
SMOKE_BENCH_MODELS = "llama3.2:1b,llama3.2:3b"
SMOKE_BENCH_ARGS = [
    "--models", SMOKE_BENCH_MODELS,
    "--ctx", "2048,4096",
    "--speed-ctx", "4096",
    "--prompt-tokens", "128",
    "--runs", "1",
    "--num-predict", "64",
]

LOCAL_SCRIPTS = Path(__file__).resolve().parent

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

_ENV = {
    "MODELINDEX_DATA": DATA,
    "OLLAMA_MODELS": f"{DATA}/ollama-models",
    "OLLAMA_HOST": OLLAMA_HOST,
    "OLLAMA_NUM_PARALLEL": "1",
    "PYTHONUNBUFFERED": "1",
}


def _scripts_layer(image: modal.Image) -> modal.Image:
    return image.add_local_dir(
        str(LOCAL_SCRIPTS), remote_path=SCRIPTS, ignore=["**/__pycache__/**", "**/*.pyc"]
    )


def _with_ollama(image: modal.Image) -> modal.Image:
    return _scripts_layer(
        image.apt_install("curl", "ca-certificates", "procps", "zstd")
        .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    )


# Crawl is HTTP-only; skip Ollama so the smoke is not blocked on the GPU Image.
crawl_image = _scripts_layer(modal.Image.debian_slim(python_version="3.12"))
cpu_image = _with_ollama(modal.Image.debian_slim(python_version="3.12"))
gpu_image = _with_ollama(
    modal.Image.from_registry("nvidia/cuda:12.6.0-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("torch")
)


def _mkdirs() -> None:
    Path(DATA, "cache").mkdir(parents=True, exist_ok=True)
    Path(DATA, "out", "measurements").mkdir(parents=True, exist_ok=True)
    Path(os.environ.get("OLLAMA_MODELS", f"{DATA}/ollama-models")).mkdir(parents=True, exist_ok=True)


def _api(method: str, path: str, payload: dict | None = None, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"http://{OLLAMA_HOST}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def _wait_ollama(timeout: int = 60) -> None:
    t0 = time.time()
    last: Exception | None = None
    while time.time() - t0 < timeout:
        try:
            _api("GET", "/api/version", timeout=2)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1)
    raise RuntimeError(f"ollama did not start: {last}")


_ollama_log = None
_ollama_log_tmp = None
_ollama_log_dest = None


def _copy_ollama_log() -> None:
    if _ollama_log_tmp and _ollama_log_dest and _ollama_log_tmp.exists():
        _ollama_log_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_ollama_log_tmp, _ollama_log_dest)


def _stop_ollama() -> None:
    global _ollama_log
    subprocess.run(["pkill", "-f", "ollama serve"], check=False)
    time.sleep(2)
    if _ollama_log is not None:
        try:
            _ollama_log.close()
        except OSError:
            pass
        _ollama_log = None
    _copy_ollama_log()


def _start_ollama(log_path: Path, flash_attn: str, kv_type: str) -> None:
    global _ollama_log, _ollama_log_tmp, _ollama_log_dest
    _stop_ollama()
    _ollama_log_dest = log_path
    _ollama_log_tmp = Path("/tmp") / log_path.name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OLLAMA_HOST": OLLAMA_HOST,
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_FLASH_ATTENTION": flash_attn,
        "OLLAMA_KV_CACHE_TYPE": kv_type,
        "OLLAMA_MODELS": os.environ.get("OLLAMA_MODELS", f"{DATA}/ollama-models"),
    })
    _ollama_log = open(_ollama_log_tmp, "ab", buffering=0)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=_ollama_log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    _wait_ollama()


def _tier_models(tier: str) -> list[str]:
    sys.path.insert(0, SCRIPTS)
    from bench import TIERS as BENCH_TIERS  # noqa: E402
    if tier not in BENCH_TIERS:
        raise ValueError(f"tier must be one of {sorted(BENCH_TIERS)}")
    return list(BENCH_TIERS[tier])


def _gpu_slug(gpu: str) -> str:
    return gpu.lower().replace("!", "").replace("_", "-")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_crawl_smoke(out: Path) -> dict:
    rows = _load_jsonl(out)
    if not rows:
        raise RuntimeError(f"smoke crawl wrote no rows: {out}")
    by_ref = {r.get("ref"): r for r in rows}
    unique = {r.get("weight_digest"): r for r in rows if r.get("weight_digest")}
    bad = [d[:16] for d, r in unique.items() if not r.get("header_ok")]
    moe = by_ref.get("qwen3:30b-a3b")
    swa = by_ref.get("gemma3:4b")
    if bad:
        raise RuntimeError(f"smoke crawl header_ok failed on digests: {bad}")
    if not moe or not (moe.get("expert_count") or 0) > 0:
        raise RuntimeError("smoke crawl: qwen3:30b-a3b missing or expert_count==0")
    if moe.get("active_weight_bytes") and moe.get("weight_bytes"):
        if not moe["active_weight_bytes"] < 0.25 * moe["weight_bytes"]:
            raise RuntimeError("smoke crawl: MoE active_weight_bytes not << weight_bytes")
    if not swa or not (swa.get("sliding_window") or 0) > 0:
        raise RuntimeError("smoke crawl: gemma3:4b missing or sliding_window==0")
    report = {
        "rows": len(rows),
        "unique_ok": len(unique),
        "moe_experts": moe.get("expert_count"),
        "swa": swa.get("sliding_window"),
        "gate": "PASS",
    }
    print("[smoke crawl] " + json.dumps(report), flush=True)
    return report


def _check_pull_smoke(models: list[str]) -> dict:
    tags = {m["name"] for m in _api("GET", "/api/tags").get("models", [])}
    missing = [m for m in models if m not in tags and f"{m}:latest" not in tags]
    if missing:
        raise RuntimeError(f"smoke pull missing from ollama tags: {missing}")
    report = {"pulled": models, "gate": "PASS"}
    print("[smoke pull] " + json.dumps(report), flush=True)
    return report


def _check_bench_smoke(out: Path) -> dict:
    recs = _load_jsonl(out)
    kinds = {r.get("kind") for r in recs}
    env = next((r for r in recs if r.get("kind") == "env"), None)
    vram = [r for r in recs if r.get("kind") == "vram"]
    speed = [r for r in recs if r.get("kind") == "speed"]
    if not env:
        raise RuntimeError("smoke bench: no env record")
    if "vram" not in kinds or not vram:
        raise RuntimeError("smoke bench: no vram records")
    if "speed" not in kinds or not speed:
        raise RuntimeError("smoke bench: no speed records")
    if not any(s.get("decode_tps") for s in speed):
        raise RuntimeError("smoke bench: decode_tps missing")
    report = {
        "gpu": env.get("gpu"),
        "bw_measured_gbs": env.get("bw_measured_gbs"),
        "vram_n": len(vram),
        "speed_n": len(speed),
        "gate": "PASS",
    }
    print("[smoke bench] " + json.dumps(report), flush=True)
    return report


def _assert_gpu_sku(requested: str) -> dict:
    """Fail the run if Modal placed a different SKU (H100→H200, A100→80GB, etc.)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"nvidia-smi failed: {out.stderr}")
    name, mem = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",", 1)]
    mem_mib = float(mem)
    u = name.upper().replace("_", " ")

    def token(tok: str) -> bool:
        return bool(re.search(rf"(^|[^A-Z0-9]){re.escape(tok)}([^A-Z0-9]|$)", u))

    ok = False
    if requested == "L4":
        ok = token("L4") and "L40" not in u
    elif requested == "A10":
        ok = (token("A10") or token("A10G")) and "A100" not in u
    elif requested == "L40S":
        ok = token("L40S") or "L40 S" in u
    elif requested == "A100-80GB":
        ok = token("A100") and mem_mib >= 70_000
    elif requested == "A100-40GB":
        ok = token("A100") and mem_mib < 70_000
    elif requested == "H100!":
        ok = token("H100") and "H200" not in u
    elif requested == "RTX-PRO-6000":
        ok = "6000" in u and ("RTX" in u or "PRO" in u)
    elif requested == "B200":
        ok = token("B200")
    if not ok:
        raise RuntimeError(
            f"SKU pin failed: requested gpu={requested!r}, got {name!r} ({mem_mib:.0f} MiB). "
            f"Re-run; do not use GPU fallback lists."
        )
    return {"name": name, "memory_mib": mem_mib, "requested": requested}


def _commit_loop(stop: threading.Event, interval: int = 20) -> None:
    while not stop.wait(interval):
        try:
            _copy_ollama_log()
            vol.commit()
            print("[volume] commit", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[volume] commit failed: {e}", flush=True)


# --------------------------------------------------------------------------- CPU: crawl
@app.function(
    image=crawl_image,
    timeout=60 * 60 * 2,
    cpu=4,
    memory=8192,
    volumes={DATA: vol},
    env=_ENV,
    retries=modal.Retries(max_retries=2),
)
def crawl_library(models: str = "", workers: int = 16, smoke: bool = False) -> dict:
    _mkdirs()
    vol.reload()
    out = f"{DATA}/out/models-smoke.jsonl" if smoke else f"{DATA}/out/models.jsonl"
    cmd = [
        sys.executable, f"{SCRIPTS}/crawl.py",
        "--workers", str(workers),
        "--out", out,
    ]
    if models:
        cmd += ["--models", models]
    print("== crawl:", " ".join(cmd), flush=True)
    stop = threading.Event()
    ticker = threading.Thread(target=_commit_loop, args=(stop,), daemon=True)
    ticker.start()
    try:
        subprocess.run(cmd, check=True)
    finally:
        stop.set()
        vol.commit()
    summary_path = Path(DATA, "out", "crawl_summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if smoke:
        summary["smoke"] = _check_crawl_smoke(Path(out))
    print("[crawl] " + json.dumps(summary), flush=True)
    return summary


# --------------------------------------------------------------------------- CPU: pull GGUF blobs onto the Volume
@app.function(
    image=cpu_image,
    timeout=60 * 60 * 3,
    cpu=8,
    memory=8192,
    volumes={DATA: vol},
    env=_ENV,
)
def pull_models(tier: str = "24gb", models: str = "", smoke: bool = False, workers: int = 8) -> dict:
    _mkdirs()
    _stop_ollama()
    vol.reload()
    wanted = [m.strip() for m in models.split(",") if m.strip()] if models else _tier_models(tier)
    ckpt = Path(DATA, "out", f"pulled-{('smoke' if smoke else tier)}.json")
    already: list[str] = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else []
    remaining = [m for m in wanted if m not in already]
    print(f"[pull] {len(already)} already pulled, {len(remaining)} remaining", flush=True)
    if not remaining:
        return {"tier": "smoke" if smoke else tier, "pulled": [], "n": 0, "already": already}
    sys.path.insert(0, SCRIPTS)
    from hydrate_ollama import hydrate  # noqa: E402
    ollama_models = Path(os.environ.get("OLLAMA_MODELS", f"{DATA}/ollama-models"))

    def after_batch(ready: list[str]) -> None:
        merged = list(dict.fromkeys(already + ready))
        ckpt.write_text(json.dumps(merged), encoding="utf-8")
        vol.commit()
        print(f"[volume] commit ({len(merged)} tags)", flush=True)

    print(f"[pull] hydrate {len(remaining)} tags with {workers} workers", flush=True)
    hyd = hydrate(remaining, ollama_models, workers=max(1, workers), on_batch=after_batch)
    already = list(dict.fromkeys(already + hyd["refs"]))
    ckpt.write_text(json.dumps(already), encoding="utf-8")
    log = Path(DATA, "out", "measurements", "pull-smoke.log" if smoke else f"pull-{tier}.log")
    _start_ollama(log, flash_attn="0", kv_type="f16")
    try:
        result = {
            "tier": "smoke" if smoke else tier,
            "pulled": hyd["refs"],
            "n": hyd["fetched"],
            "blobs": hyd["blobs"],
            "bytes": hyd["bytes"],
        }
        result["verify"] = _check_pull_smoke(wanted if smoke else remaining)
        if smoke:
            result["smoke"] = result["verify"]
    finally:
        _stop_ollama()
        vol.commit()
    return result


# --------------------------------------------------------------------------- GPU: bench
@app.function(
    image=gpu_image,
    gpu="L4",
    timeout=60 * 60 * 3,
    retries=0,
    volumes={DATA: vol},
    env=_ENV,
)
def run_bench(
    gpu_requested: str = "L4",
    tier: str = "24gb",
    kv_q8: bool = False,
    include_partial: bool = False,
    smoke: bool = False,
    continue_full: bool = False,
) -> dict:
    _mkdirs()
    _stop_ollama()
    vol.reload()
    sku = _assert_gpu_sku(gpu_requested)
    slug = _gpu_slug(gpu_requested)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meas = Path(DATA, "out", "measurements")
    extra = ["--include-partial"] if include_partial else []

    def bench(label: str, args: list[str], flash: str, kv: str, is_smoke: bool, pass_extra: bool = False) -> str:
        dest_log = meas / f"{slug}-{stamp}-server-{label}.log"
        out = meas / f"{slug}-{stamp}-{label}.jsonl"
        _start_ollama(dest_log, flash_attn=flash, kv_type=kv)
        cmd = [
            sys.executable, f"{SCRIPTS}/bench.py",
            "--tier", "local" if is_smoke else tier,
            "--skip-pull",
            "--resume",
            "--out", str(out),
            "--server-log", str(Path("/tmp") / dest_log.name),
            "--progress", f"{DATA}/out/job_progress.json",
            "--label", label,
            *(["--ctx-big"] if (tier == "48gb" and not is_smoke) else []),
            *(extra if pass_extra else []),
            *(list(SMOKE_BENCH_ARGS) if is_smoke else []),
            *args,
        ]
        print("== bench:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        _copy_ollama_log()
        return str(out)

    stop = threading.Event()
    ticker = threading.Thread(target=_commit_loop, args=(stop, 45), daemon=True)
    ticker.start()
    smoke_out = None
    main_out = None
    kvq8_out = None
    try:
        if smoke:
            print(f"== smoke pass: requested={gpu_requested} placed={sku['name']} tier={tier}", flush=True)
            smoke_out = bench("smoke", [], flash="0", kv="f16", is_smoke=True)
            _check_bench_smoke(Path(smoke_out))
            if not continue_full:
                result = {"sku": sku, "smoke": True, "main": smoke_out, "kvq8": None}
                print("[bench] " + json.dumps(result), flush=True)
                return result
            print("== continuing to full in the same container", flush=True)

        label_prefix = "main"
        print(f"== {label_prefix} pass: requested={gpu_requested} placed={sku['name']} tier={tier}", flush=True)
        main_out = bench(label_prefix, [], flash="0", kv="f16", is_smoke=False, pass_extra=True)
        if kv_q8:
            print("== kv-q8 pass", flush=True)
            kv_models = "llama3.1:8b,qwen3:30b-a3b"
            kvq8_out = bench(
                "kvq8",
                ["--models", kv_models, "--skip-speed", "--no-bandwidth"],
                flash="1",
                kv="q8_0",
                is_smoke=False,
            )
        result = {"sku": sku, "smoke": smoke, "smoke_out": smoke_out, "main": main_out, "kvq8": kvq8_out}
        print("[bench] " + json.dumps(result), flush=True)
        return result
    finally:
        stop.set()
        _stop_ollama()
        vol.commit()


@app.local_entrypoint()
def main(
    action: str = "bench",
    gpu: str = "L4",
    tier: str = "24gb",
    kv_q8: bool = False,
    include_partial: bool = False,
    models: str = "",
    workers: int = 16,
    smoke_only: bool = False,
    skip_smoke: bool = False,
) -> None:
    """Default: real-data smoke, then full. --smoke-only stops after smoke; --skip-smoke is full only."""
    if smoke_only and skip_smoke:
        raise SystemExit("use only one of --smoke-only / --skip-smoke")
    if tier not in TIERS and action != "crawl":
        raise SystemExit(f"--tier must be {'|'.join(TIERS)}")

    def _crawl(do_smoke: bool) -> None:
        m = SMOKE_CRAWL_MODELS if do_smoke else models
        print(crawl_library.remote(models=m, workers=workers, smoke=do_smoke))

    def _pull(do_smoke: bool) -> None:
        m = SMOKE_PULL_MODELS if do_smoke else models
        print(pull_models.remote(tier=tier, models=m, smoke=do_smoke, workers=min(workers, 8)))

    run = {"crawl": _crawl, "pull": _pull}.get(action)
    if action == "bench":
        if gpu not in ALLOWED_GPUS:
            raise SystemExit(f"--gpu must be one of {ALLOWED_GPUS}")
        if skip_smoke:
            print("== FULL bench", flush=True)
            print(run_bench.with_options(gpu=gpu).remote(
                gpu_requested=gpu, tier=tier, kv_q8=kv_q8,
                include_partial=include_partial, smoke=False, continue_full=False,
            ))
        elif smoke_only:
            print("== SMOKE bench", flush=True)
            print(run_bench.with_options(gpu=gpu).remote(
                gpu_requested=gpu, tier=tier, kv_q8=False,
                include_partial=include_partial, smoke=True, continue_full=False,
            ))
        else:
            print("== SMOKE then FULL bench (one container)", flush=True)
            print(run_bench.with_options(gpu=gpu).remote(
                gpu_requested=gpu, tier=tier, kv_q8=kv_q8,
                include_partial=include_partial, smoke=True, continue_full=True,
            ))
        return
    if run is None:
        raise SystemExit("--action must be crawl | pull | bench")
    if not skip_smoke:
        print(f"== SMOKE {action}", flush=True)
        run(True)
        if smoke_only:
            print("== smoke-only; skipping full", flush=True)
            return
    print(f"== FULL {action}", flush=True)
    run(False)
