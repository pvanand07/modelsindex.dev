#!/usr/bin/env python3
"""
Enumerate the Ollama library and build data/out/models.jsonl without downloading weights.

Per tag: manifest (exact layer sizes) -> config blob (param size, quant, family) ->
GGUF header via Range request (architecture fields, tensor-table byte sums).
Everything is cached under data/cache/ keyed by digest, so re-runs only fetch new tags.

    python scripts/crawl.py --models llama3.2,qwen3,gemma3     # smoke crawl
    python scripts/crawl.py                                     # full library
    python scripts/crawl.py --limit-tags 3 --workers 2

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_header import USER_AGENT, header_summary  # noqa: E402

REGISTRY = "https://registry.ollama.ai/v2/library"
SITE = "https://ollama.com"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "data" / "out"

MT_MODEL = "application/vnd.ollama.image.model"
MT_PROJECTOR = "application/vnd.ollama.image.projector"

HEADER_FIELDS = (
    "arch", "name", "size_label", "n_layer", "n_embd", "n_head", "n_head_kv", "kv_heads_sum",
    "key_length", "value_length", "context_length", "vocab_size", "expert_count",
    "expert_used_count", "sliding_window", "param_count", "active_params", "tensor_count",
    "expert_bytes", "active_weight_bytes", "kv_bytes_per_token_f16", "tensor_groups_bytes",
    "header_bytes",
)


# --------------------------------------------------------------------------- http + cache
def http_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 " + USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(1.5 * (i + 1))
    assert last is not None
    raise last


def cached_json(path: Path, ttl_days: float | None, producer):
    """Return JSON from `path` if fresh, else call producer(), store, return."""
    if path.exists():
        if ttl_days is None or (time.time() - path.stat().st_mtime) < ttl_days * 86400:
            with open(path, encoding="utf-8") as f:
                return json.load(f), True
    data = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data, False


# --------------------------------------------------------------------------- enumeration
def list_models() -> list[str]:
    html = http_get(f"{SITE}/library").decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="/library/([a-z0-9][a-z0-9._-]*)"', html)))


def list_tags(model: str) -> list[str]:
    html = http_get(f"{SITE}/library/{model}/tags").decode("utf-8", "replace")
    pat = rf'href="/library/{re.escape(model)}:([A-Za-z0-9._-]+)"'
    return sorted(set(re.findall(pat, html)))


# --------------------------------------------------------------------------- per-tag work
def get_manifest(model: str, tag: str, ttl_days: float) -> tuple[dict, bool]:
    def produce():
        return json.loads(http_get(f"{REGISTRY}/{model}/manifests/{tag}"))
    return cached_json(CACHE / "manifests" / model / f"{tag}.json", ttl_days, produce)


def get_config(model: str, digest: str) -> tuple[dict, bool]:
    def produce():
        return json.loads(http_get(f"{REGISTRY}/{model}/blobs/{digest}"))
    return cached_json(CACHE / "config" / f"{digest.replace(':', '_')}.json", None, produce)


def get_header(model: str, digest: str) -> tuple[dict, bool]:
    def produce():
        return header_summary(f"{REGISTRY}/{model}/blobs/{digest}")
    return cached_json(CACHE / "headers" / f"{digest.replace(':', '_')}.json", None, produce)


def build_row(model: str, tag: str, ttl_days: float, stats: dict) -> dict:
    row: dict = {"model": model, "tag": tag, "ref": f"{model}:{tag}"}
    try:
        manifest, hit = get_manifest(model, tag, ttl_days)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"manifest: {e}"
        return row
    stats["manifest_hits" if hit else "manifest_fetches"] += 1

    weight = next((l for l in manifest.get("layers", []) if l["mediaType"] == MT_MODEL), None)
    proj = next((l for l in manifest.get("layers", []) if l["mediaType"] == MT_PROJECTOR), None)
    if not weight:
        row["error"] = "manifest has no model layer"
        return row
    row["weight_digest"] = weight["digest"]
    row["weight_bytes"] = weight["size"]
    row["projector_digest"] = proj["digest"] if proj else None
    row["projector_bytes"] = proj["size"] if proj else 0
    row["config_digest"] = manifest.get("config", {}).get("digest")

    try:
        cfg, _ = get_config(model, row["config_digest"])
        row["config"] = {k: cfg.get(k) for k in ("model_format", "model_family", "model_families", "model_type", "file_type")}
    except Exception as e:  # noqa: BLE001
        row["config"] = None
        row["config_error"] = str(e)
    return row


def attach_header(row: dict, stats: dict) -> dict:
    if "weight_digest" not in row:
        row["header_ok"] = False
        return row
    try:
        h, hit = get_header(row["model"], row["weight_digest"])
        stats["header_hits" if hit else "header_fetches"] += 1
        for k in HEADER_FIELDS:
            row[k] = h.get(k)
        row["header_ok"] = True
    except Exception as e:  # noqa: BLE001
        row["header_ok"] = False
        row["header_error"] = str(e)
        stats["header_failures"] += 1
    return row


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated subset of library model names")
    ap.add_argument("--limit-tags", type=int, default=0, help="max tags per model (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ttl-days", type=float, default=7, help="manifest cache TTL (tags can be re-pushed)")
    ap.add_argument("--refresh", action="store_true", help="ignore manifest cache")
    ap.add_argument("--out", default=str(OUT / "models.jsonl"))
    a = ap.parse_args(argv)

    t0 = time.time()
    stats = {k: 0 for k in ("manifest_hits", "manifest_fetches", "header_hits", "header_fetches", "header_failures")}
    ttl = 0 if a.refresh else a.ttl_days

    models = a.models.split(",") if a.models else list_models()
    print(f"[crawl] {len(models)} models")

    # 1. tags (parallel over models)
    tags_by_model: dict[str, list[str]] = {}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(list_tags, m): m for m in models}
        for f in as_completed(futs):
            m = futs[f]
            try:
                tags = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[crawl] tags failed for {m}: {e}", file=sys.stderr)
                tags = []
            if a.limit_tags:
                tags = tags[: a.limit_tags]
            tags_by_model[m] = tags
    n_tags = sum(len(v) for v in tags_by_model.values())
    print(f"[crawl] {n_tags} tags")

    # 2. manifests + config (parallel over tags)
    rows: list[dict] = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(build_row, m, t, ttl, stats) for m, ts in tags_by_model.items() for t in ts]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 200 == 0:
                print(f"[crawl] manifests {i}/{n_tags}")
    rows.sort(key=lambda r: (r["model"], r["tag"]))

    # 3. headers, one fetch per unique digest
    digests = sorted({r["weight_digest"] for r in rows if "weight_digest" in r})
    print(f"[crawl] {len(digests)} unique weight digests for {n_tags} tags "
          f"(dedupe ratio {n_tags / max(1, len(digests)):.2f}x)")
    first_row_for: dict[str, dict] = {}
    for r in rows:
        d = r.get("weight_digest")
        if d and d not in first_row_for:
            first_row_for[d] = r
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(attach_header, r, stats): d for d, r in first_row_for.items()}
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 50 == 0:
                print(f"[crawl] headers {i}/{len(digests)}")
    # copy header fields to sibling rows sharing the digest
    for r in rows:
        d = r.get("weight_digest")
        if d and r is not first_row_for.get(d):
            src = first_row_for[d]
            for k in HEADER_FIELDS + ("header_ok", "header_error"):
                if k in src:
                    r[k] = src[k]
        r.setdefault("header_ok", False)

    # 4. write
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = sum(1 for r in rows if r.get("header_ok"))
    summary = {
        "models": len(models),
        "tags": n_tags,
        "unique_digests": len(digests),
        "rows_header_ok": ok,
        "rows_header_ok_pct": round(100 * ok / max(1, len(rows)), 1),
        "seconds": round(time.time() - t0, 1),
        **stats,
    }
    with open(out.parent / "crawl_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[crawl] " + json.dumps(summary))
    print(f"[crawl] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
