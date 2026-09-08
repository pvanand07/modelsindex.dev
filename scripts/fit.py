#!/usr/bin/env python3
"""
Fit calibration constants from bench.py measurements and print the gate report (spec 5.5, plan G4).

    python scripts/fit.py --measurements data/out/measurements/*.jsonl \
        --models data/out/models.jsonl --gpus data/gpus.csv --out data/out/constants.json

Per GPU:
  decode   t_token = a*W_active + c*n_layer + d        (a = 1/BW_eff)   -> a, c, d, R2
  vram     resident = W + proj + g0 + kv_bytes_per_token*kv_ratio*ctx   -> per-arch g0, kv_ratio
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
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
        return {
            "arch": row["arch"], "weight_bytes": row["weight_bytes"], "projector_bytes": row.get("projector_bytes", 0),
            "active_weight_bytes": row["active_weight_bytes"], "n_layer": row["n_layer"],
            "kv_bytes_per_token": row["kv_bytes_per_token_f16"], "active_params": row["active_params"],
            "expert_count": row.get("expert_count", 0), "sliding_window": row.get("sliding_window", 0),
            "file_type": (row.get("config") or {}).get("file_type"), "source": "models.jsonl",
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
        "arch": arch, "weight_bytes": size, "projector_bytes": 0, "active_weight_bytes": size,
        "n_layer": n_layer, "kv_bytes_per_token": n_layer * n_kv * (kl + vl) * 2,
        "active_params": int(info.get("general.parameter_count", 0) or 0),
        "expert_count": int(info.get(f"{arch}.expert_count", 0) or 0),
        "sliding_window": int(info.get(f"{arch}.attention.sliding_window", 0) or 0),
        "file_type": (bench_model.get("details") or {}).get("quantization_level"), "source": "bench_show",
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


def fit_vram(points: list[dict]) -> dict:
    """points: {model, arch, ctx, resident, W, proj, kv_per_token}. Per-arch g0 and kv_ratio."""
    per_model: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        per_model[p["model"]].append(p)
    model_fits = {}
    for m, ps in per_model.items():
        if len(ps) < 2:
            continue
        ctx = np.array([p["ctx"] for p in ps], float)
        over = np.array([p["resident"] - p["W"] - p["proj"] for p in ps], float)
        X = np.column_stack([ctx, np.ones(len(ps))])
        coef, *_ = np.linalg.lstsq(X, over, rcond=None)
        slope, g0 = float(coef[0]), float(coef[1])
        kv = ps[0]["kv_per_token"]
        model_fits[m] = {"arch": ps[0]["arch"], "g0_bytes": g0, "slope_bytes_per_token": slope,
                         "kv_ratio": (slope / kv) if kv else None, "n": len(ps)}
    arch_fits: dict[str, dict] = {}
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for mf in model_fits.values():
        by_arch[mf["arch"]].append(mf)
    for arch, mfs in by_arch.items():
        ratios = [x["kv_ratio"] for x in mfs if x["kv_ratio"] is not None]
        arch_fits[arch] = {
            "g0_bytes": float(statistics.median(x["g0_bytes"] for x in mfs)),
            "kv_ratio": float(statistics.median(ratios)) if ratios else 1.0,
            "n_models": len(mfs),
        }
    # error check
    errs = []
    for p in points:
        af = arch_fits.get(p["arch"])
        if not af:
            continue
        pred = p["W"] + p["proj"] + af["g0_bytes"] + p["kv_per_token"] * af["kv_ratio"] * p["ctx"]
        errs.append(abs(pred - p["resident"]) / p["resident"])
    return {"per_model": model_fits, "per_arch": arch_fits,
            "max_rel_err": float(max(errs)) if errs else None, "n_points": len(errs)}


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

    constants: dict = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gpus": {}, "arch": {}}
    report_lines = ["# Calibration gate report", "", f"generated {constants['generated']}", ""]

    for gpu_name, rs in by_gpu.items():
        if gpu_name == "unknown":
            continue
        env = next((e for e in envs if e.get("gpu") == gpu_name), {})
        g = match_gpu(gpu_name, gpus)
        bw_spec = g["bandwidth_gbs"] if g else None
        tflops = g["fp16_tflops"] if g else None
        # model records from this GPU's group; older files wrote them without a gpu field
        bench_models = {r["model"]: r for r in recs if r["kind"] == "model" and r.get("gpu") in (None, gpu_name)}
        fields = {m: model_fields(m, models, bm) for m, bm in bench_models.items()}
        vram_by = {(r["model"], r["ctx"]): r for r in rs if r["kind"] == "vram"}
        kv_elem_factor = {"kv-q8": 0.5}.get(env.get("label", "main"), 1.0)

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

        # ---- quant factors (all dense points, ratio measured/predicted)
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

        # ---- VRAM
        vpts = []
        for (m, ctx), v in vram_by.items():
            f = fields.get(m)
            if not f or not v.get("ps_size") or v.get("fully_on_gpu") is False:
                continue
            vpts.append({"model": m, "arch": f["arch"], "ctx": ctx, "resident": v["ps_size"],
                         "W": f["weight_bytes"], "proj": f["projector_bytes"],
                         "kv_per_token": f["kv_bytes_per_token"] * kv_elem_factor})
        vfit = fit_vram(vpts)

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

        # ---- SWA
        swa = {m: mf["kv_ratio"] for m, mf in vfit["per_model"].items()
               if fields.get(m) and fields[m]["sliding_window"] and mf["kv_ratio"] is not None}

        # ---- gates
        flagged_total = [r for r in rs if r["kind"] == "speed"]
        unflagged_share = (sum(1 for r in flagged_total if r.get("flagged") is not True) / len(flagged_total)) if flagged_total else None
        gates = {
            "G3.2_unflagged_share_ge_0.9": None if unflagged_share is None else unflagged_share >= 0.9,
            "G4.1_decode_r2_gt_0.95_n_ge_8": (dfit.get("r2", 0) > 0.95 and dfit.get("n", 0) >= 8) if "r2" in dfit else None,
            "G4.2_vram_max_rel_err_le_0.05": (vfit["max_rel_err"] <= 0.05) if vfit["max_rel_err"] is not None else None,
            "G4.3_moe_active_within_30pct_total_off_5x": (all(x["ratio_active"] and 0.7 <= x["ratio_active"] <= 1.3 and x["ratio_total"] and x["ratio_total"] >= 5 for x in moe)) if moe else None,
            "G4.4_swa_kv_ratio_lt_1_stable": (all(v < 1 for v in swa.values()) and (max(swa.values()) / min(swa.values()) <= 1.15 if len(swa) > 1 else True)) if swa else None,
            "G4.5_quant_factors_in_0.4_1.3": (all(0.4 <= v <= 1.3 for v in quant_factors.values())) if quant_factors else None,
            "G4.6_prefill_derate_in_0.05_0.8": (0.05 <= prefill_derate <= 0.8) if prefill_derate is not None else None,
            "G4.7_partial_points_ge_3": (len(partial) >= 3) if partial else None,
        }

        constants["gpus"][gpu_name] = {
            "gpu_id": g["id"] if g else None, "bw_spec_gbs": bw_spec, "bw_measured_gbs": env.get("bw_measured_gbs"),
            "decode_fit": dfit, "quant_factors": quant_factors, "prefill_derate": prefill_derate,
            "vram_fit": {"per_arch": vfit["per_arch"], "max_rel_err": vfit["max_rel_err"], "n_points": vfit["n_points"]},
            "moe": moe, "swa_kv_ratio": swa, "partial": partial, "unflagged_share": unflagged_share, "gates": gates,
        }
        for arch, af in vfit["per_arch"].items():
            constants["arch"].setdefault(arch, []).append(af)

        report_lines += [f"## {gpu_name}  (spec {bw_spec} GB/s, measured {env.get('bw_measured_gbs')} GB/s)", "",
                         "| gate | result | detail |", "|---|---|---|"]
        detail = {
            "G3.2_unflagged_share_ge_0.9": f"{unflagged_share}",
            "G4.1_decode_r2_gt_0.95_n_ge_8": f"R2={dfit.get('r2')} n={dfit.get('n')} mode={dfit.get('mode')} bw_eff={dfit.get('bw_eff_gbs')} c={dfit.get('c_s_per_layer')} d={dfit.get('d_s')}",
            "G4.2_vram_max_rel_err_le_0.05": f"max_rel_err={vfit['max_rel_err']} n={vfit['n_points']} per_arch={json.dumps(vfit['per_arch'])}",
            "G4.3_moe_active_within_30pct_total_off_5x": json.dumps(moe),
            "G4.4_swa_kv_ratio_lt_1_stable": json.dumps(swa),
            "G4.5_quant_factors_in_0.4_1.3": json.dumps(quant_factors),
            "G4.6_prefill_derate_in_0.05_0.8": f"{prefill_derate}",
            "G4.7_partial_points_ge_3": f"{len(partial)} points",
        }
        for k, v in gates.items():
            res = "PASS" if v else ("FAIL" if v is False else "N/A (insufficient data)")
            report_lines.append(f"| {k} | {res} | {detail[k][:300]} |")
        report_lines.append("")

    # ---- global constants (medians across calibrated GPUs) for uncalibrated hardware
    fits = [c["decode_fit"] for c in constants["gpus"].values() if "a_s_per_byte" in c["decode_fit"]]
    fracs = [c["decode_fit"]["bw_eff_gbs"] / c["bw_spec_gbs"] for c in constants["gpus"].values()
             if c.get("bw_spec_gbs") and c["decode_fit"].get("bw_eff_gbs")]
    arch_global = {arch: {"g0_bytes": float(statistics.median(x["g0_bytes"] for x in lst)),
                          "kv_ratio": float(statistics.median(x["kv_ratio"] for x in lst))}
                   for arch, lst in constants["arch"].items()}
    constants["arch"] = arch_global
    qf: dict[str, list[float]] = defaultdict(list)
    for c in constants["gpus"].values():
        for k, v in c["quant_factors"].items():
            qf[k].append(v)
    derates = [c["prefill_derate"] for c in constants["gpus"].values() if c.get("prefill_derate")]
    constants["global"] = {
        "bw_eff_fraction": float(statistics.median(fracs)) if fracs else 0.75,
        "c_s_per_layer": float(statistics.median(f["c_s_per_layer"] for f in fits)) if fits else 30e-6,
        "d_s": float(statistics.median(f["d_s"] for f in fits)) if fits else 1.5e-3,
        "g0_bytes": float(statistics.median(x["g0_bytes"] for x in arch_global.values())) if arch_global else 150e6,
        "kv_ratio": float(statistics.median(x["kv_ratio"] for x in arch_global.values())) if arch_global else 1.0,
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
