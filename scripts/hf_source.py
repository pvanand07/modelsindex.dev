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

A second tier fetches the family's readme-linked Hugging Face page and asks an LLM
(OpenRouter) whether it actually describes this model:

--verify-tier2  fetches the README of the Hugging Face repo a readme-derived candidate points
                at (scripts/link_common.py:fetch_hf_readme -- a plain GET of the repo's raw
                README.md, not the client-rendered SPA page) and asks the LLM whether that
                content actually describes this model. A confirmed candidate is promoted to
                method "readme_verified"; a rejected one falls through to scripts/links.py's
                homepage/GitHub/paper-based resolution in the same overall pipeline run.

For families with nothing after tier 1/tier 2, scripts/links.py takes over: it resolves the
family's homepage/GitHub/paper links from the same readme and, as a side effect of fetching
their content, can also turn up a Hugging Face repo -- see scripts/links.py and
docs/spec/links.md for that tier.

Needs OPENROUTER_API_KEY + OPENROUTER_BASE_URL (--verify-tier2), read from the environment or
from a .env file at the repo root (see load_dotenv()).

    python scripts/hf_source.py --refresh          # crawl every digest, cache-first
    python scripts/hf_source.py --refresh --limit 300   # pilot a sample
    python scripts/hf_source.py --verify-tier2      # LLM-verify readme-derived candidates
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
from link_common import DATA, ROOT, atomic_write_json, load_dotenv, openrouter_chat  # noqa: E402,F401

CACHE = DATA / "cache" / "hf_source"
OUT = DATA / "out" / "hf_sources.json"
HF_README_CACHE = DATA / "cache" / "hf_readme"
API = "https://modelindex.dev/api/v1/hash/sha256:{}"
MISS_TTL_DAYS = 14  # modelindex.dev's own crawl grows over time; retry misses periodically
HIT_TTL_DAYS = None  # a content-hash hit never goes stale
DEFAULT_LLM_MODEL = "~z-ai/glm-flash-latest"


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


# --------------------------------------------------------------------------- tier-2 verification
def verify_readme_candidate(family: str, description: str, repo: str, llm_model: str) -> dict:
    """Fetch the candidate repo's real README (link_common.fetch_hf_readme -- a plain GET of
    the raw file, not the client-rendered SPA page) and ask the LLM if it matches this family.

    `confirmed` is a tri-state: True/False is a real judgement, None means "not attempted"
    (missing OPENROUTER_* or the README fetch turned up nothing) -- callers should treat
    None as "leave the existing unverified readme match alone", not as a rejection.
    """
    from link_common import fetch_hf_readme  # local import: keeps the module import graph acyclic

    cache_path = DATA / "cache" / "hf_verify" / f"verify-{family}.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    or_key = os.environ.get("OPENROUTER_API_KEY")
    or_base = os.environ.get("OPENROUTER_BASE_URL")
    hf_url = f"https://huggingface.co/{repo}"

    confirmed: bool | None = None
    readme_text = fetch_hf_readme(repo, HF_README_CACHE)
    if readme_text and or_key and or_base:
        prompt = (
            f'Ollama model family: "{family}"\n'
            f"Description: {description or '(none)'}\n\n"
            f'Hugging Face README found for the candidate repo "{repo}":\n'
            f"{readme_text[:1500]}\n\n"
            "Does this README describe the same model as the Ollama family above? "
            "Reply with ONLY YES or NO."
        )
        try:
            reply = openrouter_chat(prompt, llm_model, or_key, or_base)
            confirmed = reply.strip().upper().startswith("YES")
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError, json.JSONDecodeError):
            confirmed = None

    result = {"repo": repo, "url": hf_url, "confirmed": confirmed}
    atomic_write_json(cache_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="query modelindex.dev for uncached/stale digests")
    ap.add_argument("--limit", type=int, help="cap how many digests to query this run (for a pilot)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests when --refresh (be polite)")
    ap.add_argument(
        "--verify-tier2", action="store_true",
        help="fetch the HF README a readme-derived candidate points at, ask the LLM to confirm it",
    )
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="OpenRouter model id for --verify-tier2")
    ap.add_argument("--llm-workers", type=int, default=3, help="parallel README fetches/LLM calls (be polite)")
    ap.add_argument("--llm-limit", type=int, help="cap how many families --verify-tier2 processes")
    a = ap.parse_args(argv)

    load_dotenv()
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
    unresolved: list[str] = []
    readme_candidates: dict[str, str] = {}
    for fam, meta in library.items():
        if fam in verified_families:
            continue  # already have a real hash match for this family; don't downgrade it
        repo = extract_readme_repo((meta or {}).get("readme") or "")
        if repo:
            readme_candidates[fam] = repo
        else:
            unresolved.append(fam)

    verify_hits = verify_misses = 0
    if a.verify_tier2:
        todo = list(readme_candidates.items())
        if a.llm_limit:
            todo = todo[: a.llm_limit]
        print(f"[hf_source] --verify-tier2: {len(todo)} readme-derived candidates, model={a.llm_model}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, a.llm_workers)) as ex:
            futs = {
                ex.submit(
                    verify_readme_candidate, fam, (library.get(fam) or {}).get("description") or "", repo, a.llm_model
                ): fam
                for fam, repo in todo
            }
            for fut in as_completed(futs):
                fam = futs[fut]
                result = fut.result()
                if result["confirmed"] is True:
                    likely[fam] = {"repo": result["repo"], "url": result["url"], "confidence": "likely", "method": "readme_verified"}
                    verify_hits += 1
                elif result["confirmed"] is False:
                    unresolved.append(fam)  # rejected -- scripts/links.py gets a shot at this family
                    verify_misses += 1
                # confirmed is None (not attempted): leave it for the plain "readme" fallback below
        print(f"[hf_source] --verify-tier2: {verify_hits} confirmed, {verify_misses} rejected", flush=True)

    # readme candidates that were never verified (flag off) or couldn't be (missing creds/fetch
    # miss) keep the original unverified method, unless verification already placed or rejected them
    for fam, repo in readme_candidates.items():
        if fam not in likely and fam not in unresolved:
            likely[fam] = {"repo": repo, "url": f"https://huggingface.co/{repo}", "confidence": "likely", "method": "readme"}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "digest_count": len(digests),
                "verified_count": len(verified),
                "likely_family_count": len(likely),
                "unresolved_families": sorted(unresolved),
                "by_digest": verified,
                "by_family": likely,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[hf_source] {len(verified)}/{len(digests)} digests verified by hash, "
        f"{len(likely)} families have a likely (unverified) repo, {len(unresolved)} unresolved"
        + (f" ({verify_hits} readme_verified)" if a.verify_tier2 else "")
        + f" -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
