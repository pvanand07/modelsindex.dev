#!/usr/bin/env python3
"""Resolve each Ollama GGUF digest to a Hugging Face source repo, via modelindex.dev.

modelindex.dev keeps a public sha256 -> {source, org, name, path} reverse index built
from crawling hf/modelscope/civitai/ollama. Since Ollama's `digest` is already the
sha256 of the raw GGUF blob, a hit on that same hash from source="hf" means the file
is byte-identical to a real Hugging Face upload -- a zero-ambiguity dedupe key, not a
name guess. Coverage is bounded by what modelindex.dev has crawled (pilot: ~3-5% of
digests resolve, two independent 300-model samples). Hits skew toward two cases: (a)
non-calibrated formats (f16/f32/q8_0/q4_0/q5_0) that any tool reproduces identically,
and (b) publishers who upload their own GGUF to HF (e.g. ibm-granite, Qwen) and whose
release Ollama mirrors verbatim -- those match at every quant level, K-quants included.

    python scripts/hf_source.py --refresh          # crawl every digest, cache-first
    python scripts/hf_source.py --refresh --limit 300   # pilot a sample
    python scripts/hf_source.py                     # just rebuild data/out/hf_sources.json from cache
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
from crawl import cached_json  # noqa: E402
from gguf_header import USER_AGENT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MODELINDEX_DATA", str(ROOT / "data")))
CACHE = DATA / "cache" / "hf_source"
OUT = DATA / "out" / "hf_sources.json"
API = "https://modelindex.dev/api/v1/hash/sha256:{}"
MISS_TTL_DAYS = 14  # modelindex.dev's own crawl grows over time; retry misses periodically
HIT_TTL_DAYS = None  # a content-hash hit never goes stale


def fetch_hash(digest_hex: str) -> dict | None:
    req = urllib.request.Request(API.format(digest_hex), headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # HASH_UNKNOWN: valid "no match", cache it
            if e.code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1.0 * (attempt + 1))
    return None


def lookup(digest: str) -> dict | None:
    """digest is `sha256:<hex>`. Returns the cached API response, or None on a miss."""
    hexd = digest.split(":", 1)[-1]
    cache_path = CACHE / f"{hexd}.json"
    # Cache hits and misses at different TTLs without two cache files: a cached miss is
    # `null`; re-fetch it once MISS_TTL_DAYS has passed, but never expire a real hit.
    if cache_path.exists():
        stale = (time.time() - cache_path.stat().st_mtime) >= MISS_TTL_DAYS * 86400
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached is not None or not stale:
            return cached
    data, _ = cached_json(cache_path, 0, lambda: fetch_hash(hexd))
    return data


def best_hf_match(result: dict | None) -> dict | None:
    if not result:
        return None
    hf = [m for m in result.get("matches", []) if m.get("source") == "hf"]
    if not hf:
        return None
    # Prefer the shortest org/name (canonical repo over a random re-upload/fork).
    hf.sort(key=lambda m: (len(m["org"]) + len(m["name"]), m["org"], m["name"]))
    m = hf[0]
    return {
        "repo": f"{m['org']}/{m['name']}",
        "commit_sha": m.get("commit_sha"),
        "path": m.get("path"),
        "url": f"https://huggingface.co/{m['org']}/{m['name']}",
        "confidence": "verified",
    }


# --------------------------------------------------------------------------- readme text fallback
# For families with no exact-hash hit, mine the ollama.com readme (already crawled into
# library.json) for a linked Hugging Face repo. This is a human-authored pointer, not a
# content match -- it names the right model far more often than it doesn't, but it is
# unverified: no guarantee the specific GGUF tag is byte-identical to anything at that URL.
_LABELED_LINK = re.compile(r"\[([^\]]*)\]\((https?://huggingface\.co/[^\s)]+)\)", re.I)
_ANY_LINK = re.compile(r"https?://huggingface\.co/[^\s)\]\"'<>]+", re.I)
_BAD_PREFIX = ("datasets/", "spaces/", "papers/", "blog/", "learn/", "collections/")


def _clean_repo(url: str) -> str | None:
    path = url.split("huggingface.co/", 1)[-1].split("?")[0].split("#")[0]
    if any(path.startswith(p) for p in _BAD_PREFIX):
        return None
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return None  # org page only, no repo
    for stop in ("blob", "resolve", "tree", "commit", "discussions"):
        if stop in parts:
            parts = parts[: parts.index(stop)]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def extract_readme_repo(readme: str) -> str | None:
    """First repo linked as "Hugging Face"/"HF", else the first repo-shaped link, else None."""
    labeled, any_repo = [], []
    for label, url in _LABELED_LINK.findall(readme or ""):
        repo = _clean_repo(url)
        if not repo:
            continue
        (labeled if ("hugg" in label.lower() or label.strip().lower() == "hf") else any_repo).append(repo)
    if labeled:
        return labeled[0]
    if any_repo:
        return any_repo[0]
    for url in _ANY_LINK.findall(readme or ""):
        repo = _clean_repo(url)
        if repo:
            return repo
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="query modelindex.dev for uncached/stale digests")
    ap.add_argument("--limit", type=int, help="cap how many digests to query this run (for a pilot)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests when --refresh (be polite)")
    a = ap.parse_args(argv)

    models = json.loads((ROOT / "prod/data/models.json").read_text(encoding="utf-8"))["models"]
    library = json.loads((ROOT / "prod/data/library.json").read_text(encoding="utf-8")).get("families", {})
    digests = sorted({m["digest"] for m in models})

    if a.refresh:
        todo = [d for d in digests if not (CACHE / f"{d.split(':', 1)[-1]}.json").exists()]
        if a.limit:
            todo = todo[: a.limit]
        print(f"[hf_source] {len(digests)} digests, {len(todo)} to query, {a.workers} workers", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = {ex.submit(lookup, d): d for d in todo}
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 100 == 0:
                    print(f"[hf_source] {done}/{len(todo)}", flush=True)

    verified: dict[str, dict] = {}
    for d in digests:
        hexd = d.split(":", 1)[-1]
        cache_path = CACHE / f"{hexd}.json"
        if not cache_path.exists():
            continue
        with open(cache_path, encoding="utf-8") as f:
            result = json.load(f)
        match = best_hf_match(result)
        if match:
            verified[d] = match

    verified_families = {m["model"] for m in models if m["digest"] in verified}
    likely: dict[str, dict] = {}
    for fam, meta in library.items():
        if fam in verified_families:
            continue  # already have a real hash match for this family; don't downgrade it
        repo = extract_readme_repo((meta or {}).get("readme") or "")
        if repo:
            likely[fam] = {"repo": repo, "url": f"https://huggingface.co/{repo}", "confidence": "likely"}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "digest_count": len(digests),
                "verified_count": len(verified),
                "likely_family_count": len(likely),
                "by_digest": verified,
                "by_family": likely,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[hf_source] {len(verified)}/{len(digests)} digests verified by hash, "
        f"{len(likely)} more families have a likely (unverified) repo from readme text -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
