#!/usr/bin/env python3
"""Build compact, browser-ready production data from the expanded model index.

The expanded index repeats fit and speed values for every model/GPU pair. The web
application can derive those values from model metadata and per-GPU coefficients,
so this exporter publishes each unique weight digest once.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from build_index import gpu_constants
from gguf_header import normalize_quant

SCHEMA_VERSION = 2
QUANT_SUFFIX = re.compile(
    r"(?:^|[-_:])(?:f16|fp16|q\d(?:_[a-z0-9]+)?|iq\d(?:_[a-z0-9]+)?)$",
    re.IGNORECASE,
)
CODE_HINTS = (
    "code", "coder", "codellama", "starcoder", "codegemma", "deepseek-coder",
    "devstral", "magicoder", "phind", "sqlcoder",
)
EMBED_HINTS = (
    "embed", "embedding", "bge-", "e5-", "gte-", "nomic-embed", "snowflake-arctic-embed",
    "all-minilm", "paraphrase-", "sentence-transform",
)
REASON_HINTS = (
    "reason", "deepseek-r1", "qwq", "r1-", "r1:", "thinking", "marco-o1",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def canonical_key(model: dict) -> tuple:
    """Prefer a concise default tag, then a concise non-quantized alias."""
    tag = str(model.get("tag") or "")
    ref = str(model.get("ref") or "")
    return (
        tag not in ("latest", "default"),
        bool(QUANT_SUFFIX.search(tag)),
        len(ref),
        ref.casefold(),
    )


def capability_signals(model: dict) -> list[str]:
    text = f"{model.get('model', '')} {model.get('ref', '')}".casefold()
    signals = []
    if model.get("projector_bytes") or any(x in text for x in ("vision", "-vl", "llava", "moondream")):
        signals.append("vision")
    if any(x in text for x in CODE_HINTS):
        signals.append("code")
    if model.get("arch") == "bert" or any(x in text for x in EMBED_HINTS):
        signals.append("embedding")
    if any(x in text for x in REASON_HINTS):
        signals.append("reasoning")
    return signals


def resolve_hf_source(digest: str, family: str, hf_sources: dict | None) -> dict | None:
    """Verified (hash-matched) beats likely (readme/homepage-matched); either can be absent."""
    if not hf_sources:
        return None
    hit = (hf_sources.get("by_digest") or {}).get(digest)
    if hit:
        return {"repo": hit["repo"], "url": hit["url"], "confidence": "verified"}
    hit = (hf_sources.get("by_family") or {}).get(family)
    if hit:
        return {"repo": hit["repo"], "url": hit["url"], "confidence": "likely", "method": hit.get("method", "readme")}
    return None


def compact_model(
    primary: dict,
    aliases: list[str],
    pushed_at: str | None = None,
    hf_sources: dict | None = None,
) -> dict:
    return {
        "ref": primary["ref"],
        "aliases": sorted(a for a in aliases if a != primary["ref"]),
        "model": primary["model"],
        "tag": primary["tag"],
        "digest": primary["digest"],
        "arch": primary["arch"],
        "quant": normalize_quant(primary.get("quant"), primary.get("tag")),
        "params": primary["param_count"],
        "active_params": primary["active_params"],
        "weight_bytes": primary["weight_bytes"],
        "active_weight_bytes": primary["active_weight_bytes"],
        "gpu_weight_bytes": primary.get("gpu_weight_bytes") or primary["weight_bytes"],
        "ple_bytes": primary.get("ple_bytes") or 0,
        "projector_bytes": primary.get("projector_bytes") or 0,
        "layers": primary["n_layer"],
        "context_length": primary["context_length"],
        "experts": primary.get("expert_count") or 0,
        "expert_used": primary.get("expert_used_count") or 0,
        "sliding_window": primary.get("sliding_window") or 0,
        "shared_kv_layers": primary.get("shared_kv_layers") or 0,
        "n_swa_layers": primary.get("n_swa_layers") or 0,
        "n_global_layers": primary.get("n_global_layers") or 0,
        "kv_local_bytes_per_token_f16": primary.get("kv_local_bytes_per_token_f16"),
        "kv_global_bytes_per_token_f16": primary.get("kv_global_bytes_per_token_f16"),
        "vram_digest_calibrated": bool(primary.get("vram_digest_calibrated")),
        "vram_bytes_at_ctx": primary["vram_bytes_at_ctx"],
        "pushed_at": pushed_at or primary.get("pushed_at"),
        "description": (primary.get("description") or "").strip(),
        "signals": capability_signals(primary),
        "hf_source": resolve_hf_source(primary["digest"], primary.get("model") or "", hf_sources),
    }


def deduplicate_models(models: list[dict], library: dict | None = None, hf_sources: dict | None = None) -> list[dict]:
    families = (library or {}).get("families") or library or {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for model in models:
        grouped[model["digest"]].append(model)
    result = []
    for digest_models in grouped.values():
        primary = min(digest_models, key=canonical_key)
        times = [m["pushed_at"] for m in digest_models if m.get("pushed_at")]
        compact = compact_model(primary, [m["ref"] for m in digest_models], max(times) if times else None, hf_sources)
        fam = families.get(primary.get("model") or "") if isinstance(families, dict) else None
        if isinstance(fam, dict) and fam.get("description") and not compact["description"]:
            compact["description"] = str(fam["description"]).strip()
        result.append(compact)
    return sorted(result, key=lambda m: m["ref"].casefold())


def compact_library(library: dict | None, model_names: set[str]) -> dict:
    families = (library or {}).get("families") or {}
    out = {}
    for name in sorted(model_names):
        fam = families.get(name) or {}
        if not isinstance(fam, dict):
            continue
        description = str(fam.get("description") or "").strip()
        readme = str(fam.get("readme") or "").strip()
        if description or readme:
            out[name] = {"description": description, "readme": readme}
    return out


def compact_gpu(gpu: dict, constants: dict) -> dict:
    estimate = gpu_constants(gpu, constants)
    return {
        "id": gpu["id"],
        "name": gpu["name"],
        "vendor": gpu["vendor"],
        "type": gpu["type"],
        "generation": gpu["generation"],
        "memory_gb": gpu["memory_gb"],
        "usable_fraction": gpu["usable_fraction"],
        "bandwidth_gbs": gpu["bandwidth_gbs"],
        "fp16_tflops": gpu["fp16_tflops"],
        "notes": gpu.get("notes") or "",
        "estimate": {
            "bw_eff_gbs": estimate["bw_eff_gbs"],
            "c_s_per_layer": estimate["c_s_per_layer"],
            "d_s": estimate["d_s"],
            "quant_factors": estimate.get("quant_factors") or {},
            "prefill_derate": estimate["prefill_derate"],
            "calibrated": bool(estimate["calibrated"]),
            "calibration_source": estimate.get("calibration_source"),
        },
    }


def _source_by_eval_id(q_base: dict | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in (q_base or {}).get("rows") or []:
        eid = row.get("eval_id")
        src = row.get("source")
        if eid and src:
            out[str(eid)] = str(src)
    return out


def lookup_score(model: dict, by_ref: dict[str, dict]) -> dict | None:
    """Find a sidecar score for a canonical prod model via ref or aliases."""
    candidates = [model["ref"], *(model.get("aliases") or [])]
    for ref in candidates:
        hit = by_ref.get(ref)
        if hit:
            return hit
    return None


def compact_quality(models: list[dict], scores: dict | None, q_base: dict | None = None) -> dict:
    """GPU-independent Q_file map keyed by canonical prod refs. Omits unmatched models."""
    by_ref = (scores or {}).get("by_ref") or {}
    sources = _source_by_eval_id(q_base)
    out: dict[str, dict] = {}
    families: set[str] = set()
    for model in models:
        hit = lookup_score(model, by_ref)
        if not hit:
            continue
        eval_id = hit.get("eval_id")
        out[model["ref"]] = {
            "eval_id": eval_id,
            "q_base": hit.get("q_base"),
            "q_file": hit.get("q_file"),
            "q_tasks": hit.get("q_tasks") or {},
            "quant_factor": hit.get("quant_factor"),
            "quality_source": sources.get(str(eval_id or ""), "measured"),
        }
        families.add(model["model"])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated": (scores or {}).get("generated"),
        "quality_refs": len(out),
        "quality_families": sorted(families),
        "by_ref": out,
    }


def build(
    index: dict,
    constants: dict,
    library: dict | None = None,
    scores: dict | None = None,
    q_base: dict | None = None,
    hf_sources: dict | None = None,
) -> tuple[dict, dict, dict, dict, dict]:
    models = deduplicate_models(index["models"], library, hf_sources)
    gpus = [compact_gpu(g, constants) for g in index["gpus"]]
    quality = compact_quality(models, scores, q_base)
    calibrated_digest_count = sum(m["vram_digest_calibrated"] for m in models)
    gate_summary = {}
    for name, gpu in (constants.get("gpus") or {}).items():
        gate_summary[name] = gpu.get("gates") or {}
    caveats = [
        "L4 speed uses a calibrated formula; it is not a measurement of every catalog tag.",
        "Speed on GPUs other than L4 is extrapolated from hardware specifications and L4-derived coefficients.",
        "MoE speed is experimental because the G4.3 validation gate does not pass.",
        "Partial-offload speed is not modeled; shown speed assumes full GPU residency.",
        "VRAM estimates assume the default f16 KV cache.",
        "Use-case recommendations use task-specific benchmark mixes (q_tasks), not name heuristics.",
        "Quality (q_file) is a vendor-benchmark mix times a quant fidelity factor; it is not Artificial Analysis proprietary evals.",
        "Models without a quality row fall back to an active-parameter size prior with exponential age decay in the finder.",
        "Task scores (q_tasks) use the same min–max + weighted-average shape as q_file, with per-usecase suite weights.",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated": index.get("generated") or constants.get("generated"),
        "source_generated": index.get("generated"),
        "constants_version": constants.get("constants_version"),
        "vram_formula": index.get("vram_formula"),
        "source_model_rows": len(index["models"]),
        "unique_models": len(models),
        "gpu_count": len(gpus),
        "calibrated_gpus": index.get("calibrated_gpus") or [],
        "vram_digest_calibrated_models": calibrated_digest_count,
        "quality_refs": quality["quality_refs"],
        "quality_families": len(quality["quality_families"]),
        "gates": gate_summary,
        "caveats": caveats,
    }
    hardware = {
        "schema_version": SCHEMA_VERSION,
        "ctx_points": index["ctx_points"],
        "gpus": gpus,
    }
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "models": models,
    }
    library_out = {
        "schema_version": SCHEMA_VERSION,
        "families": compact_library(library, {m["model"] for m in models}),
    }
    return manifest, hardware, catalog, library_out, quality


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/out/index.json")
    parser.add_argument("--constants", default="data/out/constants.json")
    parser.add_argument("--library", default="data/out/library.json")
    parser.add_argument("--scores", default="data/quality/scores.json")
    parser.add_argument("--q-base", default="data/quality/q_base.json")
    parser.add_argument("--hf-sources", default="data/out/hf_sources.json")
    parser.add_argument("--out-dir", default="prod/data")
    args = parser.parse_args(argv)

    index_path = Path(args.index)
    constants_path = Path(args.constants)
    out_dir = Path(args.out_dir)
    index = load_json(index_path)
    constants = load_json(constants_path)
    library_path = Path(args.library)
    library = load_json(library_path) if library_path.exists() else {}
    scores_path = Path(args.scores)
    scores = load_json(scores_path) if scores_path.exists() else None
    q_base_path = Path(args.q_base)
    q_base = load_json(q_base_path) if q_base_path.exists() else None
    hf_sources_path = Path(args.hf_sources)
    hf_sources = load_json(hf_sources_path) if hf_sources_path.exists() else None
    manifest, hardware, catalog, library_out, quality = build(
        index, constants, library, scores, q_base, hf_sources,
    )
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "gpus.json", hardware)
    write_json(out_dir / "models.json", catalog)
    write_json(out_dir / "library.json", library_out)
    write_json(out_dir / "quality.json", quality)
    total = sum(
        (out_dir / name).stat().st_size
        for name in ("manifest.json", "gpus.json", "models.json", "library.json", "quality.json")
    )
    hf_hits = sum(1 for m in catalog["models"] if m.get("hf_source"))
    print(
        f"[prod] {manifest['unique_models']} unique models, {manifest['gpu_count']} GPUs, "
        f"{len(library_out['families'])} library pages, {manifest['quality_refs']} quality refs, "
        f"{hf_hits} with an hf_source -> {out_dir} ({total / 1e6:.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
