#!/usr/bin/env python3
"""VRAM demand formula (constants_version 2).

    effective_ctx = min(ctx, sliding_window) when 0 < sliding_window < ctx, else ctx
    kv_bytes      = kv_local × min(ctx, sw) + kv_global × ctx
                   (legacy: one slope × effective_ctx)
    graph_bytes   = GRAPH_BYTES_PER_HEAD_TOKEN × (n_head + 1) × ctx
                   (hybrid: blend local window and global ctx by layer share)
    baseline      = b0 + b_tensor × tensor_count + b_output × output_tensor_bytes
                  (+ optional digest_offset)
    vram_bytes    = weight_bytes + projector_bytes + kv_bytes + graph_bytes + baseline

Stdlib + numpy only when fitting; predict path is pure arithmetic.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

GRAPH_BYTES_PER_HEAD_TOKEN = 2048
FORMULA_NAME = "capped_kv_graph_v2"
CONSTANTS_VERSION = 2

# Defaults from the L4 fit (overwritten when constants.json is loaded).
DEFAULT_BASELINE = {
    "b0": -17_431_749.4,
    "b_tensor": 448_138.557,
    "b_output": -1.07380337,
}


def effective_ctx(ctx: int, sliding_window: int | None) -> int:
    sw = int(sliding_window or 0)
    if 0 < sw < ctx:
        return sw
    return int(ctx)


def resolve_kv_slopes(
    kv_per_token_f16: float = 0,
    sliding_window: int | None = 0,
    kv_local_bytes_per_token_f16: float | None = None,
    kv_global_bytes_per_token_f16: float | None = None,
) -> tuple[float, float]:
    """Return (local_per_tok, global_per_tok). Legacy rows have only one slope."""
    if kv_local_bytes_per_token_f16 is not None or kv_global_bytes_per_token_f16 is not None:
        return float(kv_local_bytes_per_token_f16 or 0), float(kv_global_bytes_per_token_f16 or 0)
    kv = float(kv_per_token_f16 or 0)
    if int(sliding_window or 0) > 0:
        return kv, 0.0
    return 0.0, kv


def kv_bytes(ctx: int, kv_per_token_f16: float, sliding_window: int | None = 0,
             kv_elem_factor: float = 1.0,
             kv_local_bytes_per_token_f16: float | None = None,
             kv_global_bytes_per_token_f16: float | None = None) -> float:
    local, glob = resolve_kv_slopes(
        kv_per_token_f16, sliding_window,
        kv_local_bytes_per_token_f16, kv_global_bytes_per_token_f16,
    )
    fac = float(kv_elem_factor)
    return (local * effective_ctx(ctx, sliding_window) + glob * int(ctx)) * fac


def graph_bytes(
    ctx: int,
    n_head: int,
    n_layer: int = 0,
    n_swa_layers: int = 0,
    n_global_layers: int = 0,
    sliding_window: int | None = 0,
) -> float:
    n_layer = int(n_layer or 0)
    n_swa = int(n_swa_layers or 0)
    n_glo = int(n_global_layers or 0)
    if n_layer > 0 and n_swa > 0 and n_glo > 0:
        ctx_eff = (
            n_swa * effective_ctx(ctx, sliding_window) + n_glo * int(ctx)
        ) / n_layer
        return float(GRAPH_BYTES_PER_HEAD_TOKEN) * (int(n_head) + 1) * ctx_eff
    return float(GRAPH_BYTES_PER_HEAD_TOKEN) * (int(n_head) + 1) * int(ctx)


def baseline_bytes(
    tensor_count: int,
    output_tensor_bytes: float,
    coeffs: dict | None = None,
    digest_offset: float | None = None,
) -> float:
    c = coeffs or DEFAULT_BASELINE
    base = (
        float(c.get("b0", 0.0))
        + float(c.get("b_tensor", 0.0)) * int(tensor_count or 0)
        + float(c.get("b_output", 0.0)) * float(output_tensor_bytes or 0.0)
    )
    if digest_offset is not None:
        return float(digest_offset)
    return base


def hybrid_from(fields: dict) -> dict:
    """Hybrid KV/graph fields accepted on crawl rows and fit points."""
    local = fields.get("kv_local_bytes_per_token_f16")
    glob = fields.get("kv_global_bytes_per_token_f16")
    if local is None and glob is None:
        # fit points use kv_per_token
        local = fields.get("kv_local_per_token")
        glob = fields.get("kv_global_per_token")
    return {
        "kv_local_bytes_per_token_f16": local,
        "kv_global_bytes_per_token_f16": glob,
        "n_layer": int(fields.get("n_layer") or 0),
        "n_swa_layers": int(fields.get("n_swa_layers") or 0),
        "n_global_layers": int(fields.get("n_global_layers") or 0),
    }


def predict_vram(
    *,
    weight_bytes: float,
    projector_bytes: float = 0.0,
    kv_bytes_per_token_f16: float,
    n_head: int,
    ctx: int,
    sliding_window: int | None = 0,
    tensor_count: int = 0,
    output_tensor_bytes: float = 0.0,
    kv_elem_factor: float = 1.0,
    coeffs: dict | None = None,
    digest_offset: float | None = None,
    kv_local_bytes_per_token_f16: float | None = None,
    kv_global_bytes_per_token_f16: float | None = None,
    n_layer: int = 0,
    n_swa_layers: int = 0,
    n_global_layers: int = 0,
) -> int:
    """Full-residency VRAM demand in bytes (Ollama ps_size target)."""
    kv = kv_bytes(
        ctx, kv_bytes_per_token_f16, sliding_window, kv_elem_factor,
        kv_local_bytes_per_token_f16, kv_global_bytes_per_token_f16,
    )
    gr = graph_bytes(ctx, n_head, n_layer, n_swa_layers, n_global_layers, sliding_window)
    base = baseline_bytes(tensor_count, output_tensor_bytes, coeffs, digest_offset)
    return int(round(float(weight_bytes) + float(projector_bytes or 0) + kv + gr + base))


def residual_after_kv_graph(
    resident: float,
    weight_bytes: float,
    projector_bytes: float,
    kv_bytes_per_token_f16: float,
    n_head: int,
    ctx: int,
    sliding_window: int | None = 0,
    kv_elem_factor: float = 1.0,
    kv_local_bytes_per_token_f16: float | None = None,
    kv_global_bytes_per_token_f16: float | None = None,
    n_layer: int = 0,
    n_swa_layers: int = 0,
    n_global_layers: int = 0,
) -> float:
    """ps_size − W − proj − KV − graph; what the baseline must explain."""
    return (
        float(resident)
        - float(weight_bytes)
        - float(projector_bytes or 0)
        - kv_bytes(
            ctx, kv_bytes_per_token_f16, sliding_window, kv_elem_factor,
            kv_local_bytes_per_token_f16, kv_global_bytes_per_token_f16,
        )
        - graph_bytes(ctx, n_head, n_layer, n_swa_layers, n_global_layers, sliding_window)
    )


def structural_signature(fields: dict) -> str:
    """Group quantizations of the same architecture/dimensions together."""
    return "|".join([
        str(fields.get("arch") or ""),
        str(fields.get("n_layer") or 0),
        str(fields.get("n_head") or 0),
        str(fields.get("n_head_kv") or 0),
        str(fields.get("key_length") or 0),
        str(fields.get("value_length") or 0),
        str(fields.get("sliding_window") or 0),
        str(fields.get("shared_kv_layers") or 0),
        str(fields.get("n_swa_layers") or 0),
        str(fields.get("kv_local_bytes_per_token_f16") or fields.get("kv_local_per_token") or 0),
        str(fields.get("expert_count") or 0),
        str(fields.get("param_count") or fields.get("active_params") or 0),
    ])


def err_stats(errs: list[float]) -> dict:
    if not errs:
        return {"max_rel_err": None, "p95_rel_err": None, "mean_rel_err": None, "n_points": 0}
    s = sorted(errs)
    n = len(s)
    p95 = s[min(n - 1, int(round(0.95 * (n - 1))))]
    return {
        "max_rel_err": float(max(s)),
        "p95_rel_err": float(p95),
        "mean_rel_err": float(sum(s) / n),
        "n_points": n,
    }


def fit_baseline(points: list[dict]) -> dict:
    """Least-squares fit of b0, b_tensor, b_output on residual-after-KV-graph points.

    Each point needs: residual, tensor_count, output_tensor_bytes.
    """
    import numpy as np

    if not points:
        return {"coeffs": dict(DEFAULT_BASELINE), "n": 0, "mode": "default"}
    y = np.array([p["residual"] for p in points], float)
    T = np.array([p.get("tensor_count") or 0 for p in points], float)
    O = np.array([p.get("output_tensor_bytes") or 0 for p in points], float)
    X = np.column_stack([np.ones(len(points)), T, O])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {
        "coeffs": {"b0": float(coef[0]), "b_tensor": float(coef[1]), "b_output": float(coef[2])},
        "n": len(points),
        "mode": "lstsq",
    }


def digest_offsets(points: list[dict], coeffs: dict | None = None) -> dict[str, float]:
    """Per-digest baseline that replaces the global formula for known digests.

    digest_offset = median(ps_size − W − proj − KV − graph) over measured ctx points.
    """
    by: dict[str, list[float]] = defaultdict(list)
    for p in points:
        dig = p.get("digest")
        if dig:
            by[dig].append(p["residual"])
    return {dig: statistics_median(vals) for dig, vals in by.items()}


def statistics_median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return float(s[n // 2])
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def predict_point(p: dict, coeffs: dict, offsets: dict[str, float] | None = None) -> float:
    dig = p.get("digest")
    off = offsets.get(dig) if offsets and dig else None
    hy = hybrid_from(p)
    return float(predict_vram(
        weight_bytes=p["W"],
        projector_bytes=p.get("proj") or 0,
        kv_bytes_per_token_f16=p["kv_per_token"],
        n_head=p["n_head"],
        ctx=p["ctx"],
        sliding_window=p.get("sliding_window") or 0,
        tensor_count=p.get("tensor_count") or 0,
        output_tensor_bytes=p.get("output_tensor_bytes") or 0,
        kv_elem_factor=p.get("kv_elem_factor") or 1.0,
        coeffs=coeffs,
        digest_offset=off,
        **hy,
    ))


def rel_errs(points: list[dict], coeffs: dict, offsets: dict[str, float] | None = None) -> list[float]:
    errs = []
    for p in points:
        pred = predict_point(p, coeffs, offsets)
        if p["resident"] > 0:
            errs.append(abs(pred - p["resident"]) / p["resident"])
    return errs


def validate_forward_context(points: list[dict], coeffs: dict) -> dict:
    """Fit digest offset from the smallest ctx only; score later contexts."""
    by_dig: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        if p.get("digest"):
            by_dig[p["digest"]].append(p)
    errs = []
    for dig, ps in by_dig.items():
        ps = sorted(ps, key=lambda x: x["ctx"])
        if len(ps) < 2:
            continue
        first = ps[0]
        # digest_offset = residual of first ctx (absolute baseline replacement)
        offset = first["residual"]
        for p in ps[1:]:
            pred = predict_point(p, coeffs, {dig: offset})
            errs.append(abs(pred - p["resident"]) / p["resident"])
    return err_stats(errs)


def validate_grouped_holdout(points: list[dict]) -> dict:
    """Leave-one-structural-signature-out: refit baseline without that group."""
    by_sig: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        by_sig[p.get("signature") or structural_signature(p)].append(p)
    if len(by_sig) < 2:
        return err_stats([])
    errs = []
    for hold_sig, held in by_sig.items():
        train = [p for sig, ps in by_sig.items() if sig != hold_sig for p in ps]
        if len(train) < 3:
            continue
        fit = fit_baseline(train)
        errs.extend(rel_errs(held, fit["coeffs"], offsets=None))
    return err_stats(errs)


def fit_vram_v2(points: list[dict]) -> dict:
    """Fit VRAM v2 on full-residency points.

    Required keys per point: model, digest, arch, ctx, resident, W, proj,
    kv_per_token (f16, already scaled by kv_elem_factor if needed — store raw f16
    and kv_elem_factor separately), n_head, sliding_window, tensor_count,
    output_tensor_bytes, signature (optional).
    """
    if not points:
        return {
            "formula": FORMULA_NAME,
            "graph_bytes_per_head_token": GRAPH_BYTES_PER_HEAD_TOKEN,
            "coeffs": dict(DEFAULT_BASELINE),
            "digest_offsets": {},
            "train": err_stats([]),
            "forward_context": err_stats([]),
            "grouped_holdout": err_stats([]),
            "max_rel_err": None,
            "n_points": 0,
        }

    enriched = []
    for p in points:
        q = dict(p)
        hy = hybrid_from(p)
        q["kv_elem_factor"] = p.get("kv_elem_factor") or 1.0
        q["residual"] = residual_after_kv_graph(
            p["resident"], p["W"], p.get("proj") or 0, p["kv_per_token"],
            p["n_head"], p["ctx"], p.get("sliding_window") or 0, q["kv_elem_factor"],
            hy.get("kv_local_bytes_per_token_f16"), hy.get("kv_global_bytes_per_token_f16"),
            hy.get("n_layer") or 0, hy.get("n_swa_layers") or 0, hy.get("n_global_layers") or 0,
        )
        if not q.get("signature"):
            q["signature"] = structural_signature(q)
        enriched.append(q)

    fit = fit_baseline(enriched)
    coeffs = fit["coeffs"]
    offsets = digest_offsets(enriched, coeffs)
    train = err_stats(rel_errs(enriched, coeffs, offsets=None))
    train_digest = err_stats(rel_errs(enriched, coeffs, offsets))
    forward = validate_forward_context(enriched, coeffs)
    holdout = validate_grouped_holdout(enriched)

    return {
        "formula": FORMULA_NAME,
        "graph_bytes_per_head_token": GRAPH_BYTES_PER_HEAD_TOKEN,
        "coeffs": coeffs,
        "digest_offsets": offsets,
        "fit_mode": fit["mode"],
        "train_global": train,
        "train_digest": train_digest,
        "forward_context": forward,
        "grouped_holdout": holdout,
        # Gate metric: global formula max (must be ≤ 5%).
        "max_rel_err": train["max_rel_err"],
        "n_points": train["n_points"],
        "n_digests": len(offsets),
    }


def load_vram_model(constants: dict) -> dict[str, Any]:
    """Extract vram_model from constants.json; raise if version is wrong and no fallback."""
    ver = constants.get("constants_version")
    vm = constants.get("vram_model") or {}
    if ver == CONSTANTS_VERSION and vm.get("formula") == FORMULA_NAME:
        return vm
    if ver in (None, 1) and not vm:
        # Legacy v1 — caller may fall back or reject.
        return {"formula": "legacy_g0_kv_ratio", "legacy": True}
    return vm


def predict_from_model_row(m: dict, ctx: int, vram_model: dict, kv_elem_factor: float = 1.0) -> int:
    """Predict using a crawl models.jsonl row + vram_model block from constants."""
    if vram_model.get("legacy"):
        # Should not be used for new index builds; keep a safe structural lower bound.
        kv = kv_bytes(
            ctx, float(m.get("kv_bytes_per_token_f16") or 0), m.get("sliding_window") or 0,
            kv_elem_factor, m.get("kv_local_bytes_per_token_f16"),
            m.get("kv_global_bytes_per_token_f16"),
        )
        return int(m["weight_bytes"] + (m.get("projector_bytes") or 0) + kv)

    coeffs = vram_model.get("coeffs") or DEFAULT_BASELINE
    dig = m.get("weight_digest") or m.get("digest")
    offsets = vram_model.get("digest_offsets") or {}
    groups = m.get("tensor_groups_bytes") or {}
    hy = hybrid_from(m)
    weight = m.get("gpu_weight_bytes")
    if weight is None and m.get("ple_bytes"):
        weight = float(m["weight_bytes"]) - float(m["ple_bytes"])
    if weight is None:
        weight = m["weight_bytes"]
    tensors = m.get("gpu_tensor_count")
    if tensors is None:
        tensors = int(m.get("tensor_count") or 0)
    return predict_vram(
        weight_bytes=weight,
        projector_bytes=m.get("projector_bytes") or 0,
        kv_bytes_per_token_f16=m["kv_bytes_per_token_f16"],
        n_head=int(m.get("n_head") or 0),
        ctx=ctx,
        sliding_window=m.get("sliding_window") or 0,
        tensor_count=int(tensors or 0),
        output_tensor_bytes=float(groups.get("output") or 0),
        kv_elem_factor=kv_elem_factor,
        coeffs=coeffs,
        digest_offset=offsets.get(dig) if dig else None,
        **hy,
    )
