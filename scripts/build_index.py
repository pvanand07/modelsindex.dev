#!/usr/bin/env python3
"""
Build the model index: models.jsonl x gpus.csv x constants.json -> index.json + index.csv.

    python scripts/build_index.py --models data/out/models.jsonl --gpus data/gpus.csv \
        --constants data/out/constants.json --out data/out/index.json

Per (model, GPU):
  vram(ctx)     = W + proj + g0_arch + kv_bytes_per_token * kv_ratio_arch * ctx
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
import sys
from datetime import datetime, timezone
from pathlib import Path

GB = 1e9
CTX_POINTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
SYSTEM_RAM_ASSUMED = 32 * GB  # for the partial/none boundary
DEFAULTS = {"bw_eff_fraction": 0.75, "c_s_per_layer": 30e-6, "d_s": 1.5e-3, "g0_bytes": 150e6,
            "kv_ratio": 1.0, "quant_factors": {}, "prefill_derate": 0.3, "calibrated": False}


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
    g.update({k: v for k, v in glob_.items() if v is not None})
    g["calibrated"] = False
    for name, c in (constants.get("gpus") or {}).items():
        if c.get("gpu_id") == gpu["id"] and "a_s_per_byte" in c.get("decode_fit", {}):
            df = c["decode_fit"]
            g.update({"bw_eff_gbs": df["bw_eff_gbs"], "c_s_per_layer": df["c_s_per_layer"], "d_s": df["d_s"],
                      "quant_factors": {**g.get("quant_factors", {}), **c.get("quant_factors", {})},
                      "prefill_derate": c.get("prefill_derate") or g["prefill_derate"], "calibrated": True,
                      "calibration_source": name})
            break
    if "bw_eff_gbs" not in g:
        g["bw_eff_gbs"] = gpu["bandwidth_gbs"] * g["bw_eff_fraction"]
    return g


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
    arch_c = constants.get("arch") or {}
    glob_ = {**DEFAULTS, **(constants.get("global") or {})}

    gpu_consts = {g["id"]: gpu_constants(g, constants) for g in gpus}
    out_models = []
    csv_rows = []
    for m in models:
        W, proj = m["weight_bytes"], m.get("projector_bytes") or 0
        W_active, n_layer = m["active_weight_bytes"], m["n_layer"]
        kv = m["kv_bytes_per_token_f16"]
        ac = arch_c.get(m["arch"], {})
        g0 = ac.get("g0_bytes", glob_["g0_bytes"])
        kv_ratio = ac.get("kv_ratio", glob_["kv_ratio"])
        file_type = (m.get("config") or {}).get("file_type")
        ctxs = [c for c in CTX_POINTS if not m.get("context_length") or c <= m["context_length"]] or [CTX_POINTS[0]]
        vram_at = {c: int(W + proj + g0 + kv * kv_ratio * c) for c in ctxs}
        per_gpu = {}
        for g in gpus:
            gc = gpu_consts[g["id"]]
            usable = g["memory_gb"] * GB * g["usable_fraction"]
            status = {}
            for c, v in vram_at.items():
                if v <= usable:
                    status[c] = "full"
                elif W + proj <= usable + SYSTEM_RAM_ASSUMED:
                    status[c] = "partial"
                else:
                    status[c] = "none"
            full_ctxs = [c for c, s in status.items() if s == "full"]
            t = W_active / (gc["bw_eff_gbs"] * GB) + n_layer * gc["c_s_per_layer"] + gc["d_s"]
            qf = gc["quant_factors"].get(file_type, 1.0) if file_type else 1.0
            decode_est = (qf / t) if t > 0 else None
            prefill = (g["fp16_tflops"] * gc["prefill_derate"] * 1e12 / (2 * m["active_params"])) if (g["fp16_tflops"] and m["active_params"]) else None
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
                "param_count": m["param_count"], "weight_gb": round(W / GB, 3), "active_gb": round(W_active / GB, 3),
                **{f"vram_gb_{c}": round(vram_at[c] / GB, 2) if c in vram_at else "" for c in CTX_POINTS},
                "max_ctx_full": entry["max_ctx_full"], "decode_tps_ceiling": entry["decode_tps_ceiling"],
                "decode_tps_est": entry["decode_tps_est"], "prefill_tps_est": entry["prefill_tps_est"],
                "calibrated": entry["calibrated"],
            })
        out_models.append({
            "ref": m["ref"], "model": m["model"], "tag": m["tag"], "digest": m["weight_digest"], "arch": m["arch"],
            "quant": file_type, "param_count": m["param_count"], "active_params": m["active_params"],
            "weight_bytes": W, "active_weight_bytes": W_active, "projector_bytes": proj,
            "n_layer": n_layer, "n_head": m["n_head"], "n_head_kv": m["n_head_kv"],
            "key_length": m["key_length"], "value_length": m["value_length"], "context_length": m["context_length"],
            "vocab_size": m["vocab_size"], "expert_count": m["expert_count"], "expert_used_count": m["expert_used_count"],
            "sliding_window": m["sliding_window"], "kv_bytes_per_token_f16": kv,
            "vram_bytes_at_ctx": {str(c): v for c, v in vram_at.items()}, "per_gpu": per_gpu,
        })

    index = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constants_source": a.constants if constants else "defaults",
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
          f"calibrated gpus: {index['calibrated_gpus'] or 'none (defaults)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
