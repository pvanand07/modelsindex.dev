#!/usr/bin/env python3
"""
Fit calibration constants from bench.py measurements and print the gate report (spec 5.5, plan G4).

    python scripts/fit.py --measurements data/out/measurements/*.jsonl \
        --models data/out/models.jsonl --gpus data/gpus.csv --out data/out/constants.json

Per GPU:
  decode   t_token = a*W_active + c*n_layer + d        (a = 1/BW_eff)   -> a, c, d, R2
  vram v2  W+proj + capped_KV + graph(n_head,ctx) + baseline(tensor,output)
  quant    measured / predicted per file_type
  prefill  derate = prefill_tps * 2 * active_params / (TFLOPS*1e12)
  moe      measured vs active-bytes and total-bytes predictions
  partial  (fraction on GPU, tps / predicted-full-tps)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vram_model import (  # noqa: E402
    CONSTANTS_VERSION,
    FORMULA_NAME,
    fit_vram_v2,
    predict_vram,
    structural_signature,
)

GB = 1e9


# --------------------------------------------------------------------------- loading
def load_jsonl(paths: list[str]) -> list[dict]:
    recs = []
    for pat in paths:
        for p in glob.glob(pat):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
    return recs


def load_models(path: str | None) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["ref"]] = r
    return out


def norm_gpu(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("nvidia", "").replace("geforce", ""))


def load_gpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_norm"] = norm_gpu(r["name"])
        for k in ("memory_gb", "usable_fraction", "bandwidth_gbs", "fp16_tflops"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
    return rows


def match_gpu(name: str, gpus: list[dict]) -> dict | None:
    n = norm_gpu(name)
    exact = [g for g in gpus if g["_norm"] == n]
    if exact:
        return exact[0]
    cands = [g for g in gpus if g["_norm"] and (g["_norm"] in n or n in g["_norm"])]
    return max(cands, key=lambda g: len(g["_norm"])) if cands else None


def model_fields(ref: str, models: dict[str, dict], bench_model: dict | None) -> dict | None:
    """Fields for the formulas: from models.jsonl if present, else from bench /api/show info."""
    row = models.get(ref) or models.get(ref + ":latest") or (models.get(ref.split(":")[0] + ":latest") if ":" not in ref else None)
    if row and row.get("header_ok"):
        groups = row.get("tensor_groups_bytes") or {}
        return {
            "arch": row["arch"],
            "digest": row.get("weight_digest"),
            "weight_bytes": row.get("gpu_weight_bytes") or row["weight_bytes"],
            "projector_bytes": row.get("projector_bytes", 0),
            "active_weight_bytes": row["active_weight_bytes"],
            "n_layer": row["n_layer"],
            "n_head": row.get("n_head") or 0,
            "n_head_kv": row.get("n_head_kv") or 0,
            "key_length": row.get("key_length") or 0,
            "value_length": row.get("value_length") or 0,
            "kv_bytes_per_token": row["kv_bytes_per_token_f16"],
            "active_params": row["active_params"],
            "expert_count": row.get("expert_count", 0),
            "sliding_window": row.get("sliding_window", 0),
            "shared_kv_layers": row.get("shared_kv_layers") or 0,
            "n_swa_layers": row.get("n_swa_layers") or 0,
            "n_global_layers": row.get("n_global_layers") or 0,
            "kv_local_bytes_per_token_f16": row.get("kv_local_bytes_per_token_f16"),
            "kv_global_bytes_per_token_f16": row.get("kv_global_bytes_per_token_f16"),
            "tensor_count": row.get("gpu_tensor_count") or row.get("tensor_count") or 0,
            "output_tensor_bytes": float(groups.get("output") or 0),
            "param_count": row.get("param_count") or row.get("active_params") or 0,
            "file_type": (row.get("config") or {}).get("file_type"),
            "source": "models.jsonl",
        }
    if not bench_model:
        return None
    info = bench_model.get("model_info", {})
    arch = info.get("general.architecture")
    if not arch:
        return None
    n_layer = int(info.get(f"{arch}.block_count", 0) or 0)
    n_head = int(info.get(f"{arch}.attention.head_count", 0) or 0)
    n_kv = int(info.get(f"{arch}.attention.head_count_kv", n_head) or n_head)
    n_embd = int(info.get(f"{arch}.embedding_length", 0) or 0)
    hd = (n_embd // n_head) if n_head else 0
    kl = int(info.get(f"{arch}.attention.key_length", hd) or hd)
    vl = int(info.get(f"{arch}.attention.value_length", hd) or hd)
    size = bench_model.get("size") or 0
    return {
        "arch": arch,
        "digest": bench_model.get("digest"),
        "weight_bytes": size,
        "projector_bytes": 0,
        "active_weight_bytes": size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_head_kv": n_kv,
        "key_length": kl,
        "value_length": vl,
        "kv_bytes_per_token": n_layer * n_kv * (kl + vl) * 2,
        "active_params": int(info.get("general.parameter_count", 0) or 0),
        "expert_count": int(info.get(f"{arch}.expert_count", 0) or 0),
        "sliding_window": int(info.get(f"{arch}.attention.sliding_window", 0) or 0),
        "tensor_count": 0,
        "output_tensor_bytes": 0.0,
        "param_count": int(info.get("general.parameter_count", 0) or 0),
        "file_type": (bench_model.get("details") or {}).get("quantization_level"),
        "source": "bench_show",
    }


# --------------------------------------------------------------------------- fits
def r2(y, yhat) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


def fit_decode(points: list[dict], bw_measured_gbs: float | None) -> dict:
    """points: {W_active, n_layer, decode_tps}. Returns a, c, d, r2, n, mode."""
    n = len(points)
    if n == 0:
        return {"n": 0}
    y = np.array([1.0 / p["decode_tps"] for p in points])
    W = np.array([p["W_active"] for p in points], float)
    L = np.array([p["n_layer"] for p in points], float)
    out: dict = {"n": n}
    if n >= 4:
        X = np.column_stack([W, L, np.ones(n)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, c, d = (float(v) for v in coef)
        out.update({"a_s_per_byte": a, "c_s_per_layer": c, "d_s": d, "r2": r2(y, X @ coef), "mode": "free"})
        if a > 0:
            out["bw_eff_gbs"] = 1.0 / a / GB
    if ("bw_eff_gbs" not in out) and bw_measured_gbs and n >= 2:
        a = 1.0 / (bw_measured_gbs * GB)
        X = np.column_stack([L, np.ones(n)])
        coef, *_ = np.linalg.lstsq(X, y - a * W, rcond=None)
        c, d = (float(v) for v in coef)
        out.update({"a_s_per_byte": a, "c_s_per_layer": c, "d_s": d, "r2": r2(y, a * W + X @ coef),
                    "mode": "anchored_to_measured_bw", "bw_eff_gbs": bw_measured_gbs})
    if "a_s_per_byte" not in out:
        out["mode"] = "insufficient"
    return out


def predict_tps(fit: dict, W: float, n_layer: int) -> float | None:
    if "a_s_per_byte" not in fit:
        return None
    t = fit["a_s_per_byte"] * W + fit["c_s_per_layer"] * n_layer + fit["d_s"]
    return 1.0 / t if t > 0 else None


def swa_validation(points: list[dict], vfit: dict) -> dict:
    """G4.4: capped-KV + graph residuals for gemma3 models stay within 5%."""
    gemma = [p for p in points if p.get("arch") == "gemma3"]
    if not gemma:
        return {"n": 0, "max_rel_err": None, "models": {}}
    coeffs = vfit.get("coeffs") or {}
    by_model: dict[str, list[float]] = defaultdict(list)
    for p in gemma:
        pred = predict_vram(
            weight_bytes=p["W"], projector_bytes=p.get("proj") or 0,
            kv_bytes_per_token_f16=p["kv_per_token"], n_head=p["n_head"], ctx=p["ctx"],
            sliding_window=p.get("sliding_window") or 0, tensor_count=p.get("tensor_count") or 0,
            output_tensor_bytes=p.get("output_tensor_bytes") or 0,
            kv_elem_factor=p.get("kv_elem_factor") or 1.0, coeffs=coeffs,
            kv_local_bytes_per_token_f16=p.get("kv_local_bytes_per_token_f16"),
            kv_global_bytes_per_token_f16=p.get("kv_global_bytes_per_token_f16"),
            n_layer=p.get("n_layer") or 0,
            n_swa_layers=p.get("n_swa_layers") or 0,
            n_global_layers=p.get("n_global_layers") or 0,
        )
        by_model[p["model"]].append(abs(pred - p["resident"]) / p["resident"])
    model_max = {m: float(max(errs)) for m, errs in by_model.items()}
    return {
        "n": len(gemma),
        "max_rel_err": float(max(model_max.values())) if model_max else None,
        "models": model_max,
        "ok": bool(model_max) and float(max(model_max.values())) <= 0.05,
    }


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measurements", nargs="+", required=True, help="jsonl paths or globs")
    ap.add_argument("--models", help="data/out/models.jsonl (crawl output)")
    ap.add_argument("--gpus", default="data/gpus.csv")
    ap.add_argument("--out", default="data/out/constants.json")
    ap.add_argument("--report", help="markdown gate report path (default: alongside --out)")
    a = ap.parse_args(argv)

    recs = load_jsonl(a.measurements)
    models = load_models(a.models)
    gpus = load_gpus(a.gpus)
    if not recs:
        print("no measurements found", file=sys.stderr)
        return 1

    by_gpu: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_gpu[r.get("gpu") or "unknown"].append(r)
    envs = [r for r in recs if r["kind"] == "env"]

    constants: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constants_version": CONSTANTS_VERSION,
        "gpus": {},
        "vram_model": {},
    }
    report_lines = ["# Calibration gate report", "", f"generated {constants['generated']}",
                    f"constants_version {CONSTANTS_VERSION} formula {FORMULA_NAME}", ""]
    all_vpts: list[dict] = []

    for gpu_name, rs_all in by_gpu.items():
        if gpu_name == "unknown":
            continue
        # Smoke and kv-q8 passes must not mix into the main VRAM/decode fit.
        rs = [r for r in rs_all if (r.get("label") or "main") in ("main", "local")]
        if not rs:
            rs = rs_all
        env = next(
            (e for e in envs if e.get("gpu") == gpu_name and (e.get("label") or "main") in ("main", "local")),
            None,
        ) or next((e for e in envs if e.get("gpu") == gpu_name), {})
        g = match_gpu(gpu_name, gpus)
        bw_spec = g["bandwidth_gbs"] if g else None
        tflops = g["fp16_tflops"] if g else None
        bench_models = {r["model"]: r for r in recs if r["kind"] == "model" and r.get("gpu") in (None, gpu_name)
                        and (r.get("label") or "main") in ("main", "local", None)}
        if not bench_models:
            bench_models = {r["model"]: r for r in recs if r["kind"] == "model" and r.get("gpu") in (None, gpu_name)}
        fields = {m: model_fields(m, models, bm) for m, bm in bench_models.items()}
        vram_by = {(r["model"], r["ctx"]): r for r in rs if r["kind"] == "vram"}

        # ---- decode points: unflagged, fully on GPU, shortest prompt, smallest speed ctx per model
        speed = [r for r in rs if r["kind"] == "speed" and r.get("decode_tps") and r.get("flagged") is not True]
        best: dict[str, dict] = {}
        for r in speed:
            v = vram_by.get((r["model"], r["ctx"]))
            if v and v.get("fully_on_gpu") is False:
                continue
            key = r["model"]
            cur = best.get(key)
            if cur is None or (r["ctx"], r["prompt_tokens_requested"]) < (cur["ctx"], cur["prompt_tokens_requested"]):
                best[key] = r
        pts = []
        for m, r in best.items():
            f = fields.get(m)
            if not f or not f["n_layer"]:
                continue
            pts.append({"model": m, "W_active": f["active_weight_bytes"], "n_layer": f["n_layer"],
                        "decode_tps": r["decode_tps"], "file_type": f["file_type"], "arch": f["arch"],
                        "expert_count": f["expert_count"], "W_total": f["weight_bytes"]})
        dense_pts = [p for p in pts if not p["expert_count"]]
        dfit = fit_decode(dense_pts, env.get("bw_measured_gbs"))

        # ---- quant factors
        quant: dict[str, list[float]] = defaultdict(list)
        for p in dense_pts:
            pred = predict_tps(dfit, p["W_active"], p["n_layer"])
            if pred and p["file_type"]:
                quant[p["file_type"]].append(p["decode_tps"] / pred)
        quant_factors = {k: float(statistics.median(v)) for k, v in quant.items()}

        # ---- MoE check
        moe = []
        for p in pts:
            if not p["expert_count"]:
                continue
            pa = predict_tps(dfit, p["W_active"], p["n_layer"])
            pt = predict_tps(dfit, p["W_total"], p["n_layer"])
            moe.append({"model": p["model"], "measured": p["decode_tps"], "pred_active": pa, "pred_total": pt,
                        "ratio_active": (p["decode_tps"] / pa) if pa else None,
                        "ratio_total": (p["decode_tps"] / pt) if pt else None})

        # ---- VRAM v2 (full residency only)
        vpts = []
        for (m, ctx), v in vram_by.items():
            f = fields.get(m)
            if not f or not v.get("ps_size") or v.get("fully_on_gpu") is False:
                continue
            if not f.get("n_head"):
                continue
            pt = {
                "model": m,
                "digest": f.get("digest"),
                "arch": f["arch"],
                "ctx": ctx,
                "resident": v["ps_size"],
                "W": f["weight_bytes"],
                "proj": f["projector_bytes"],
                "kv_per_token": f["kv_bytes_per_token"],
                "kv_elem_factor": 1.0,
                "n_head": f["n_head"],
                "n_head_kv": f.get("n_head_kv") or 0,
                "n_layer": f["n_layer"],
                "key_length": f.get("key_length") or 0,
                "value_length": f.get("value_length") or 0,
                "sliding_window": f.get("sliding_window") or 0,
                "shared_kv_layers": f.get("shared_kv_layers") or 0,
                "n_swa_layers": f.get("n_swa_layers") or 0,
                "n_global_layers": f.get("n_global_layers") or 0,
                "kv_local_bytes_per_token_f16": f.get("kv_local_bytes_per_token_f16"),
                "kv_global_bytes_per_token_f16": f.get("kv_global_bytes_per_token_f16"),
                "tensor_count": f.get("tensor_count") or 0,
                "output_tensor_bytes": f.get("output_tensor_bytes") or 0,
                "expert_count": f.get("expert_count") or 0,
                "weight_bytes": f["weight_bytes"],
                "param_count": f.get("param_count") or f.get("active_params") or 0,
            }
            pt["signature"] = structural_signature(pt)
            vpts.append(pt)
        all_vpts.extend(vpts)
        vfit = fit_vram_v2(vpts)
        swa = swa_validation(vpts, vfit)

        # ---- prefill derate
        derates = []
        for r in speed:
            f = fields.get(r["model"])
            if not f or not r.get("prefill_tps") or not tflops or not f["active_params"]:
                continue
            if r["prompt_tokens_requested"] < 1024:
                continue
            derates.append(r["prefill_tps"] * 2 * f["active_params"] / (tflops * 1e12))
        prefill_derate = float(statistics.median(derates)) if derates else None

        # ---- partial offload
        partial = []
        for r in [r for r in rs if r["kind"] == "speed" and r.get("decode_tps")]:
            v = vram_by.get((r["model"], r["ctx"]))
            f = fields.get(r["model"])
            if not v or v.get("fully_on_gpu") is not False or not f or not v.get("ps_size"):
                continue
            pred_full = predict_tps(dfit, f["active_weight_bytes"], f["n_layer"])
            partial.append({"model": r["model"], "ctx": r["ctx"], "fraction_on_gpu": v["ps_size_vram"] / v["ps_size"],
                            "measured_tps": r["decode_tps"], "tps_ratio_to_full": (r["decode_tps"] / pred_full) if pred_full else None})

        # ---- gates
        flagged_total = [r for r in rs if r["kind"] == "speed"]
        unflagged_share = (sum(1 for r in flagged_total if r.get("flagged") is not True) / len(flagged_total)) if flagged_total else None
        hold = vfit.get("grouped_holdout") or {}
        fwd = vfit.get("forward_context") or {}
        gates = {
            "G3.2_unflagged_share_ge_0.9": None if unflagged_share is None else unflagged_share >= 0.9,
            "G4.1_decode_r2_gt_0.95_n_ge_8": (dfit.get("r2", 0) > 0.95 and dfit.get("n", 0) >= 8) if "r2" in dfit else None,
            "G4.2_vram_max_rel_err_le_0.05": (
                vfit.get("max_rel_err") is not None
                and vfit["max_rel_err"] <= 0.05
                and (hold.get("max_rel_err") is None or hold["max_rel_err"] <= 0.05)
            ),
            "G4.3_moe_active_within_30pct_total_off_5x": (
                all(x["ratio_active"] and 0.7 <= x["ratio_active"] <= 1.3 and x["ratio_total"] and x["ratio_total"] >= 5 for x in moe)
            ) if moe else None,
            "G4.4_swa_capped_kv_graph": (
                bool(swa.get("ok"))
            ) if swa.get("n") else None,
            "G4.5_quant_factors_in_0.4_1.3": (all(0.4 <= v <= 1.3 for v in quant_factors.values())) if quant_factors else None,
            "G4.6_prefill_derate_in_0.05_0.8": (0.05 <= prefill_derate <= 0.8) if prefill_derate is not None else None,
            "G4.7_partial_points_ge_3": (len(partial) >= 3) if partial else None,
        }

        kvq8_vs_f16 = []
        for r in rs_all:
            if r.get("label") != "kvq8" or r.get("kind") != "vram" or not r.get("ps_size"):
                continue
            main_v = vram_by.get((r["model"], r["ctx"]))
            if main_v and main_v.get("ps_size"):
                kvq8_vs_f16.append({
                    "model": r["model"], "ctx": r["ctx"],
                    "ratio": round(r["ps_size"] / main_v["ps_size"], 4),
                })

        constants["gpus"][gpu_name] = {
            "gpu_id": g["id"] if g else None,
            "bw_spec_gbs": bw_spec,
            "bw_measured_gbs": env.get("bw_measured_gbs"),
            "decode_fit": dfit,
            "quant_factors": quant_factors,
            "prefill_derate": prefill_derate,
            "vram_fit": {
                "formula": vfit.get("formula"),
                "max_rel_err": vfit.get("max_rel_err"),
                "n_points": vfit.get("n_points"),
                "train_global": vfit.get("train_global"),
                "train_digest": vfit.get("train_digest"),
                "forward_context": fwd,
                "grouped_holdout": hold,
            },
            "moe": moe,
            "swa_validation": swa,
            "partial": partial,
            "unflagged_share": unflagged_share,
            "kvq8_vs_f16": kvq8_vs_f16,
            "gates": gates,
        }

        report_lines += [f"## {gpu_name}  (spec {bw_spec} GB/s, measured {env.get('bw_measured_gbs')} GB/s)", "",
                         "| gate | result | detail |", "|---|---|---|"]
        detail = {
            "G3.2_unflagged_share_ge_0.9": f"{unflagged_share}",
            "G4.1_decode_r2_gt_0.95_n_ge_8": f"R2={dfit.get('r2')} n={dfit.get('n')} mode={dfit.get('mode')} bw_eff={dfit.get('bw_eff_gbs')} c={dfit.get('c_s_per_layer')} d={dfit.get('d_s')}",
            "G4.2_vram_max_rel_err_le_0.05": (
                f"global_max={vfit.get('max_rel_err')} holdout_max={hold.get('max_rel_err')} "
                f"forward_max={fwd.get('max_rel_err')} digest_max={(vfit.get('train_digest') or {}).get('max_rel_err')} "
                f"n={vfit.get('n_points')} coeffs={json.dumps(vfit.get('coeffs'))}"
            ),
            "G4.3_moe_active_within_30pct_total_off_5x": json.dumps(moe),
            "G4.4_swa_capped_kv_graph": json.dumps(swa),
            "G4.5_quant_factors_in_0.4_1.3": json.dumps(quant_factors),
            "G4.6_prefill_derate_in_0.05_0.8": f"{prefill_derate}",
            "G4.7_partial_points_ge_3": f"{len(partial)} points",
        }
        for k, v in gates.items():
            res = "PASS" if v else ("FAIL" if v is False else "N/A (insufficient data)")
            report_lines.append(f"| {k} | {res} | {detail[k][:300]} |")
        report_lines.append("")

    # ---- global VRAM model (fit across all full-residency points from all GPUs)
    global_vfit = fit_vram_v2(all_vpts) if all_vpts else fit_vram_v2([])
    constants["vram_model"] = {
        "formula": FORMULA_NAME,
        "graph_bytes_per_head_token": global_vfit.get("graph_bytes_per_head_token"),
        "coeffs": global_vfit.get("coeffs"),
        "digest_offsets": global_vfit.get("digest_offsets"),
        "fit_mode": global_vfit.get("fit_mode"),
        "train_global": global_vfit.get("train_global"),
        "train_digest": global_vfit.get("train_digest"),
        "forward_context": global_vfit.get("forward_context"),
        "grouped_holdout": global_vfit.get("grouped_holdout"),
        "n_points": global_vfit.get("n_points"),
        "n_digests": global_vfit.get("n_digests"),
    }

    # ---- global decode constants for uncalibrated hardware
    fits = [c["decode_fit"] for c in constants["gpus"].values() if "a_s_per_byte" in c["decode_fit"]]
    fracs = [c["decode_fit"]["bw_eff_gbs"] / c["bw_spec_gbs"] for c in constants["gpus"].values()
             if c.get("bw_spec_gbs") and c["decode_fit"].get("bw_eff_gbs")]
    qf: dict[str, list[float]] = defaultdict(list)
    for c in constants["gpus"].values():
        for k, v in c["quant_factors"].items():
            qf[k].append(v)
    derates = [c["prefill_derate"] for c in constants["gpus"].values() if c.get("prefill_derate")]
    constants["global"] = {
        "bw_eff_fraction": float(statistics.median(fracs)) if fracs else 0.75,
        "c_s_per_layer": float(statistics.median(f["c_s_per_layer"] for f in fits)) if fits else 30e-6,
        "d_s": float(statistics.median(f["d_s"] for f in fits)) if fits else 1.5e-3,
        "quant_factors": {k: float(statistics.median(v)) for k, v in qf.items()},
        "prefill_derate": float(statistics.median(derates)) if derates else 0.3,
        "calibrated": bool(fits),
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(constants, f, indent=2)
    rep = Path(a.report) if a.report else out.with_name("gate_report.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    print(f"[fit] wrote {out} and {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
