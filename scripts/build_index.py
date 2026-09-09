#!/usr/bin/env python3
"""
Build the model index: models.jsonl x gpus.csv x constants.json -> index.json + index.csv.

    python scripts/build_index.py --models data/out/models.jsonl --gpus data/gpus.csv \
        --constants data/out/constants.json --out data/out/index.json

Per (model, GPU):
  vram(ctx)     = W + proj + capped_KV(ctx) + graph(n_head,ctx) + baseline
                  (digest offset when calibrated; else global tensor/output coeffs)
  fit_status    = full | partial | none per ctx point, and max_ctx_full
  decode ceiling= bw_spec / W_active
  decode est    = quant_factor / (W_active / bw_eff + n_layer*c + d)
  prefill est   = fp16_tflops * derate * 1e12 / (2 * active_params)
GPUs with a calibrated entry in constants.json use their own constants; others use the
global medians (or documented defaults when nothing is calibrated) and carry calibrated=false.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_header import normalize_quant  # noqa: E402
from vram_model import (  # noqa: E402
    CONSTANTS_VERSION,
    DEFAULT_BASELINE,
    FORMULA_NAME,
    load_vram_model,
    predict_from_model_row,
)

GB = 1e9
CTX_POINTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
SYSTEM_RAM_ASSUMED = 32 * GB  # for the partial/none boundary
DEFAULTS = {
    "bw_eff_fraction": 0.75,
    "c_s_per_layer": 30e-6,
    "d_s": 1.5e-3,
    "quant_factors": {},
    "prefill_derate": 0.3,
    "calibrated": False,
}


def load_models(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_gpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("memory_gb", "usable_fraction", "bandwidth_gbs", "fp16_tflops"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
    return rows


def gpu_constants(gpu: dict, constants: dict) -> dict:
    g = dict(DEFAULTS)
    glob_ = constants.get("global") or {}
    g.update({k: v for k, v in glob_.items() if v is not None and k in (
        "bw_eff_fraction", "c_s_per_layer", "d_s", "quant_factors", "prefill_derate", "calibrated",
    )})
    g["calibrated"] = False
    for name, c in (constants.get("gpus") or {}).items():
        if c.get("gpu_id") == gpu["id"] and "a_s_per_byte" in c.get("decode_fit", {}):
            df = c["decode_fit"]
            g.update({
                "bw_eff_gbs": df["bw_eff_gbs"],
                "c_s_per_layer": df["c_s_per_layer"],
                "d_s": df["d_s"],
                "quant_factors": {**g.get("quant_factors", {}), **c.get("quant_factors", {})},
                "prefill_derate": c.get("prefill_derate") or g["prefill_derate"],
                "calibrated": True,
                "calibration_source": name,
            })
            break
    if "bw_eff_gbs" not in g:
        g["bw_eff_gbs"] = gpu["bandwidth_gbs"] * g["bw_eff_fraction"]
    return g


def resolve_vram_model(constants: dict) -> dict:
    vm = load_vram_model(constants)
    if vm.get("legacy"):
        raise SystemExit(
            f"constants_version must be {CONSTANTS_VERSION} with formula {FORMULA_NAME}; "
            f"got version={constants.get('constants_version')!r}. Re-run scripts/fit.py."
        )
    if not vm.get("coeffs"):
        vm = dict(vm)
        vm["coeffs"] = dict(DEFAULT_BASELINE)
    return vm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="data/out/models.jsonl")
    ap.add_argument("--gpus", default="data/gpus.csv")
    ap.add_argument("--constants", default="data/out/constants.json")
    ap.add_argument("--out", default="data/out/index.json")
    a = ap.parse_args(argv)

    models = [m for m in load_models(a.models) if m.get("header_ok")]
    gpus = load_gpus(a.gpus)
    constants = {}
    if Path(a.constants).exists():
        with open(a.constants, encoding="utf-8") as f:
            constants = json.load(f)
    if not constants:
        raise SystemExit(f"missing constants file: {a.constants}")
    vram_model = resolve_vram_model(constants)

    gpu_consts = {g["id"]: gpu_constants(g, constants) for g in gpus}
    out_models = []
    csv_rows = []
    for m in models:
        W_disk, proj = m["weight_bytes"], m.get("projector_bytes") or 0
        W = m.get("gpu_weight_bytes") or W_disk
        ple = m.get("ple_bytes") or 0
        W_active, n_layer = m["active_weight_bytes"], m["n_layer"]
        kv = m["kv_bytes_per_token_f16"]
        file_type = normalize_quant((m.get("config") or {}).get("file_type"), m.get("tag"))
        ctxs = [c for c in CTX_POINTS if not m.get("context_length") or c <= m["context_length"]] or [CTX_POINTS[0]]
        vram_at = {c: predict_from_model_row(m, c, vram_model) for c in ctxs}
        dig = m.get("weight_digest")
        used_digest = bool(dig and dig in (vram_model.get("digest_offsets") or {}))
        per_gpu = {}
        for g in gpus:
            gc = gpu_consts[g["id"]]
            usable = g["memory_gb"] * GB * g["usable_fraction"]
            status = {}
            for c, v in vram_at.items():
                if v <= usable:
                    status[c] = "full"
                elif W + ple + proj <= usable + SYSTEM_RAM_ASSUMED:
                    status[c] = "partial"
                else:
                    status[c] = "none"
            full_ctxs = [c for c, s in status.items() if s == "full"]
            t = W_active / (gc["bw_eff_gbs"] * GB) + n_layer * gc["c_s_per_layer"] + gc["d_s"]
            qf = gc["quant_factors"].get(file_type, 1.0) if file_type else 1.0
            decode_est = (qf / t) if t > 0 else None
            prefill = (
                g["fp16_tflops"] * gc["prefill_derate"] * 1e12 / (2 * m["active_params"])
            ) if (g["fp16_tflops"] and m["active_params"]) else None
            entry = {
                "fit_status": {str(c): s for c, s in status.items()},
                "max_ctx_full": max(full_ctxs) if full_ctxs else 0,
                "decode_tps_ceiling": round(g["bandwidth_gbs"] * GB / W_active, 1),
                "decode_tps_est": round(decode_est, 1) if decode_est else None,
                "prefill_tps_est": round(prefill, 0) if prefill else None,
                "calibrated": gc["calibrated"],
            }
            per_gpu[g["id"]] = entry
            csv_rows.append({
                "ref": m["ref"], "gpu": g["id"], "arch": m["arch"], "quant": file_type,
                "param_count": m["param_count"], "weight_gb": round(W_disk / GB, 3), "active_gb": round(W_active / GB, 3),
                **{f"vram_gb_{c}": round(vram_at[c] / GB, 2) if c in vram_at else "" for c in CTX_POINTS},
                "max_ctx_full": entry["max_ctx_full"], "decode_tps_ceiling": entry["decode_tps_ceiling"],
                "decode_tps_est": entry["decode_tps_est"], "prefill_tps_est": entry["prefill_tps_est"],
                "calibrated": entry["calibrated"],
            })
        out_models.append({
            "ref": m["ref"], "model": m["model"], "tag": m["tag"], "digest": m["weight_digest"], "arch": m["arch"],
            "quant": file_type, "param_count": m["param_count"], "active_params": m["active_params"],
            "weight_bytes": W_disk, "active_weight_bytes": W_active, "projector_bytes": proj,
            "ple_bytes": ple, "gpu_weight_bytes": W,
            "n_layer": n_layer, "n_head": m["n_head"], "n_head_kv": m["n_head_kv"],
            "key_length": m["key_length"], "value_length": m["value_length"], "context_length": m["context_length"],
            "vocab_size": m["vocab_size"], "expert_count": m["expert_count"], "expert_used_count": m["expert_used_count"],
            "sliding_window": m["sliding_window"], "kv_bytes_per_token_f16": kv,
            "kv_local_bytes_per_token_f16": m.get("kv_local_bytes_per_token_f16"),
            "kv_global_bytes_per_token_f16": m.get("kv_global_bytes_per_token_f16"),
            "shared_kv_layers": m.get("shared_kv_layers") or 0,
            "n_swa_layers": m.get("n_swa_layers") or 0,
            "n_global_layers": m.get("n_global_layers") or 0,
            "n_kv_alloc_layers": m.get("n_kv_alloc_layers") or 0,
            "key_length_swa": m.get("key_length_swa") or 0,
            "value_length_swa": m.get("value_length_swa") or 0,
            "vram_formula": FORMULA_NAME,
            "vram_digest_calibrated": used_digest,
            "vram_bytes_at_ctx": {str(c): v for c, v in vram_at.items()},
            "pushed_at": m.get("pushed_at"),
            "description": m.get("description") or "",
            "per_gpu": per_gpu,
        })

    index = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constants_source": a.constants,
        "constants_version": constants.get("constants_version"),
        "vram_formula": FORMULA_NAME,
        "calibrated_gpus": [gid for gid, gc in gpu_consts.items() if gc["calibrated"]],
        "ctx_points": CTX_POINTS,
        "gpus": [{k: v for k, v in g.items()} for g in gpus],
        "models": out_models,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f)
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["ref"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"[index] {len(out_models)} models x {len(gpus)} gpus -> {out} and {csv_path}; "
          f"calibrated gpus: {index['calibrated_gpus'] or 'none (defaults)'}; "
          f"vram={FORMULA_NAME} v{constants.get('constants_version')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
