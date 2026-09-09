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

A second tier mines the family's readme-linked Hugging Face repo(s) and asks an LLM
(OpenRouter) whether the readme-derived pick actually describes this model:

--verify-tier2  fetches the README of a candidate repo (scripts/link_common.py:fetch_hf_readme
                -- a plain GET of the repo's raw README.md, not the client-rendered SPA page)
                and asks the LLM whether that content actually describes this model. Verified
                once per distinct (family, repo) pair, not once per release, even though the
                pick itself is now release-scoped (see below) -- many releases in a family
                share the same picked repo. A confirmed candidate is promoted to method
                "readme_verified"; a rejected one falls through to scripts/links.py's
                homepage/GitHub/paper-based resolution in the same overall pipeline run.

A single Ollama "family" (e.g. "llava") can bundle genuinely different upstream releases at
different sizes -- llava:7b, llava:13b and llava:34b are different Vicuna checkpoints with
different Hugging Face repos, not the same file at different quants. The family readme (one
per family, not one per tag) often only names the top-of-family checkpoint, so a release's
siblings are mined two ways: `extract_readme_candidates` returns *every* HF link the readme
names (not just the first), and `sibling_candidates` scans whatever github/homepage/hf content
scripts/links.py has already fetched for the family (data/link_content/<family>.json) for bare
`org/repo` tokens (e.g. a `--model-path org/model-13b` in a shell snippet, no URL prefix) whose
repo name contains the family slug. For each release lacking a digest-verified hit, a size
token pulled from its tag (`13b`, `8x7b`, ...) is matched deterministically against candidate
repo names when unambiguous (no LLM spent); otherwise -- when `--verify-tier2` is set -- an LLM
call disambiguates, one per distinct (family, size-token) group so releases sharing a size don't
repeat the call.

--brave-search  for families where readme mining AND sibling-content scanning both come up
                completely empty, query Brave's web search (site:huggingface.co <family>) as a
                last resort. Every candidate this turns up flows through the exact same
                deterministic-size-match / LLM-disambiguate / LLM-verify pipeline as readme/
                sibling candidates -- the query only needs to maximize recall, not precision.
                Method vocabulary "brave_search" / "brave_search_verified" (as opposed to
                "readme" / "readme_verified") marks a pick as having come from a search API, not
                the family's own readme -- opt-in since it's a paid/rate-limited external API,
                needs BRAVE_SEARCH_API_KEY.

For families/releases with nothing after tiers 1-3, scripts/links.py takes over: it resolves
the family's homepage/GitHub/paper links from the same readme and, as a side effect of fetching
their content, can also turn up a release-specific Hugging Face repo -- see scripts/links.py and
docs/spec/links.md for that tier.

Needs OPENROUTER_API_KEY + OPENROUTER_BASE_URL (--verify-tier2, and release-level disambiguation
when a family's readme names more than one ambiguous candidate), read from the environment or
from a .env file at the repo root (see load_dotenv()). --brave-search additionally needs
BRAVE_SEARCH_API_KEY.

    python scripts/hf_source.py --refresh          # crawl every digest, cache-first
    python scripts/hf_source.py --refresh --limit 300   # pilot a sample
    python scripts/hf_source.py --verify-tier2      # LLM-verify readme-derived candidates
    python scripts/hf_source.py --verify-tier2 --families llava,wizardlm    # pilot specific families
    python scripts/hf_source.py --brave-search --verify-tier2   # + web-search fallback for the rest
    python scripts/hf_source.py                     # just rebuild data/out/hf_sources.json from cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crawl import cached_json  # noqa: E402
from gguf_header import USER_AGENT  # noqa: E402
from link_common import (  # noqa: E402,F401
    DATA, ROOT, atomic_write_json, clean_hf_repo, http_get, llm_pick_repo, load_dotenv,
    openrouter_chat, pick_candidate_for_release, read_json, release_key, size_token,
)

CACHE = DATA / "cache" / "hf_source"
OUT = DATA / "out" / "hf_sources.json"
HF_README_CACHE = DATA / "cache" / "hf_readme"
PICK_CACHE = DATA / "cache" / "hf_pick"
BRAVE_CACHE = DATA / "cache" / "brave_search"
BRAVE_RESULTS = DATA / "hf_links" / "brave_search_results.json"  # durable audit, NOT gitignored
BRAVE_API_BASE = "https://api.search.brave.com/res/v1"
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


# --------------------------------------------------------------------------- readme text mining
# For families with no exact-hash hit, mine the ollama.com readme (already crawled into
# library.json) for linked Hugging Face repos. This is a human-authored pointer, not a content
# match -- it names the right model(s) far more often than it doesn't, but it is unverified: no
# guarantee any specific GGUF tag is byte-identical to anything at that URL.
_LABELED_LINK = re.compile(r"\[([^\]]*)\]\((https?://huggingface\.co/[^\s)]+)\)", re.I)
_ANY_LINK = re.compile(r"https?://huggingface\.co/[^\s)\]\"'<>]+", re.I)


def extract_readme_candidates(readme: str) -> list[str]:
    """Every Hugging Face repo-shaped link in the readme, in priority order (labeled "Hugging
    Face"/"HF" first, then any repo-shaped link, then a bare URL scan), deduplicated.
    """
    labeled, any_repo = [], []
    for label, url in _LABELED_LINK.findall(readme or ""):
        repo = clean_hf_repo(url)
        if not repo:
            continue
        (labeled if ("hugg" in label.lower() or label.strip().lower() == "hf") else any_repo).append(repo)
    bare = [r for r in (clean_hf_repo(u) for u in _ANY_LINK.findall(readme or "")) if r]
    seen: set[str] = set()
    out: list[str] = []
    for repo in labeled + any_repo + bare:
        if repo not in seen:
            seen.add(repo)
            out.append(repo)
    return out


def extract_readme_repo(readme: str) -> str | None:
    """First repo linked as "Hugging Face"/"HF", else the first repo-shaped link, else None."""
    candidates = extract_readme_candidates(readme)
    return candidates[0] if candidates else None


# ------------------------------------------------------------------------- sibling-repo mining
# A family's readme often names only the top-of-family checkpoint (llava's readme links only
# the 7B repo); sibling sizes still turn up as bare `org/repo` tokens with no URL prefix inside
# already-fetched github/homepage/hf content (e.g. a `--model-path org/model-13b` in a shell
# snippet) -- read-only against scripts/links.py's prior output, no new fetch here.
_BARE_REPO = re.compile(r"(?<![\w/.\-])([\w][\w.\-]{1,39}/[\w][\w.\-]{1,80})(?![\w/.\-])")
_NON_REPO_EXT = re.compile(r"\.(png|jpe?g|gif|svg|webp|mp4|mov|md|py|sh|ya?ml|json|txt|css|js|ts|ico|pdf|zip|tar|gz)$", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def sibling_candidates(family: str, texts: list[str]) -> list[str]:
    """Bare-token repo candidates whose repo *name* (not org) contains the family slug --
    matching by name, not org, because a picked repo's own README often links siblings under a
    different org than the one already picked (confirmed live: wizardlm's TheBloke-org pick
    links WizardLM-org siblings).
    """
    seen: set[str] = set()
    out: list[str] = []
    fam_norm = _norm(family)
    if not fam_norm:
        return out
    for text in texts:
        for token in _BARE_REPO.findall(text or ""):
            if _NON_REPO_EXT.search(token):
                continue
            repo = clean_hf_repo(f"https://huggingface.co/{token}")
            if not repo or repo in seen:
                continue
            name = repo.split("/", 1)[-1]
            if fam_norm in _norm(name):
                seen.add(repo)
                out.append(repo)
    return out


def _load_family_content_texts(family: str) -> list[str]:
    """Text scripts/links.py already fetched for this family (github/homepage/hf content), used
    read-only as an extra sibling-candidate source. Empty until links.py has run for this family.
    """
    path = DATA / "link_content" / f"{family}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for kind in ("github", "homepage"):
        entry = data.get(kind)
        if isinstance(entry, dict) and entry.get("content"):
            texts.append(entry["content"])
    hf_entry = data.get("hf")
    if isinstance(hf_entry, dict):
        for repo_entry in hf_entry.values():
            if isinstance(repo_entry, dict) and repo_entry.get("content"):
                texts.append(repo_entry["content"])
    return texts


# --------------------------------------------------------------------------- Brave search (tier 3)
# For families where neither the readme nor already-fetched sibling content name any HF repo at
# all -- readme mining and sibling scanning both come up completely empty -- a site-restricted
# web search is the last resort. Opt-in (--brave-search) since it's a paid/rate-limited external
# API, same treatment as scripts/links.py gives Exa for homepage fetching.
def brave_search(query: str, api_key: str) -> dict | None:
    """One synchronous GET against Brave's Web Search API. Single attempt, no retry -- matches
    scripts/links.py's exa_fetch convention for a paid/rate-limited external API.
    """
    url = f"{BRAVE_API_BASE}/web/search?{urlencode({'q': query, 'count': 10})}"
    result = http_get(url, headers={"Accept": "application/json", "X-Subscription-Token": api_key})
    if not result or result[0] != 200:
        return None
    return json.loads(result[1])


def _brave_cache_path(query: str) -> Path:
    return BRAVE_CACHE / f"{hashlib.sha1(query.encode('utf-8')).hexdigest()}.json"


def _append_brave_audit(family: str, query: str, results: list[dict]) -> None:
    """Durable audit trail, family-keyed -- Brave is called exactly once per family here (unlike
    Exa in links.py, called per-URL), so family is the natural key. Own path/schema, not reusing
    the retired data/hf_links/exa_results.json shape.
    """
    existing = read_json(BRAVE_RESULTS) if BRAVE_RESULTS.exists() else {}
    existing[family] = {"query": query, "results": results}
    atomic_write_json(BRAVE_RESULTS, existing)


def brave_search_cached(family: str, query: str, api_key: str) -> list[dict]:
    """Cached by content-hash of the query; negative-cached (an empty/failed result is cached
    too) so a dead-end family costs one call ever, not one per run -- same discipline as
    scripts/links.py's exa_fetch_cached.
    """
    cache_path = _brave_cache_path(query)
    if cache_path.exists():
        return read_json(cache_path) or []
    try:
        data = brave_search(query, api_key)
        results = ((data or {}).get("web") or {}).get("results") or []
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
        results = []
    atomic_write_json(cache_path, results)
    if results:
        _append_brave_audit(family, query, results)
    return results


def brave_candidates(family: str, api_key: str) -> list[str]:
    """A site:huggingface.co web search, turned into clean HF repo candidates in Brave's own
    relevance-rank order, deduplicated. Precision is enforced downstream (clean_hf_repo rejects
    non-repo pages; verify_readme_candidate LLM-checks the eventual pick against its real
    README) -- the query itself only needs to maximize recall, so no query enrichment from the
    family's description (tried inconsistently across families: some carry the org name, most
    don't, and there's no reliable way to tell which in advance).
    """
    query = f"site:huggingface.co {family}"
    results = brave_search_cached(family, query, api_key)
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        repo = clean_hf_repo(r.get("url") or "")
        if repo and repo not in seen:
            seen.add(repo)
            out.append(repo)
    return out


def pick_repo_for_release(
    family: str, description: str, release_tag_suffix: str, candidates: list[str], llm_model: str, allow_llm: bool,
) -> str | None:
    """Deterministic size-token match first (free); LLM disambiguation only when ambiguous and
    `allow_llm` -- cached and deduped by (family, size-token) so releases sharing a size don't
    repeat the same call (e.g. `qwen3:4b-instruct-2507` and `qwen3:4b-thinking-2507` both key on
    "4b").
    """
    repo, ambiguous = pick_candidate_for_release(release_tag_suffix, candidates)
    if not ambiguous:
        return repo
    if not allow_llm:
        return None

    dedup_key = size_token(release_tag_suffix) or release_tag_suffix
    cache_path = PICK_CACHE / f"{family}__{dedup_key}.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f).get("repo")

    picked = llm_pick_repo(
        family, description,
        [{"repo": c, "source_url": "readme/sibling scan"} for c in candidates],
        llm_model, release_hint=release_tag_suffix,
    )
    atomic_write_json(cache_path, {"repo": picked})
    return picked


# --------------------------------------------------------------------------- tier-2 verification
def verify_readme_candidate(family: str, description: str, repo: str, llm_model: str) -> dict:
    """Fetch the candidate repo's real README (link_common.fetch_hf_readme -- a plain GET of
    the raw file, not the client-rendered SPA page) and ask the LLM if it matches this family.
    Cached per (family, repo) -- called once per distinct pick, not once per release, even
    though picks are release-scoped; many releases in a family share one picked repo.

    `confirmed` is a tri-state: True/False is a real judgement, None means "not attempted"
    (missing OPENROUTER_* or the README fetch turned up nothing) -- callers should treat
    None as "leave the existing unverified readme match alone", not as a rejection.
    """
    from link_common import fetch_hf_readme  # local import: keeps the module import graph acyclic

    cache_path = DATA / "cache" / "hf_verify" / f"verify-{family}__{repo.replace('/', '__')}.json"
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


def _distinct_releases_by_family(models: list[dict]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for m in models:
        out[m["model"]].add(release_key(m["model"], m["tag"]))
    return {fam: sorted(releases) for fam, releases in out.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="query modelindex.dev for uncached/stale digests")
    ap.add_argument("--limit", type=int, help="cap how many digests to query this run (for a pilot)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests when --refresh (be polite)")
    ap.add_argument(
        "--verify-tier2", action="store_true",
        help="fetch each picked repo's README, ask the LLM to confirm it; also enables "
             "LLM disambiguation for releases with an ambiguous readme/sibling candidate set",
    )
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="OpenRouter model id for --verify-tier2")
    ap.add_argument("--llm-workers", type=int, default=3, help="parallel README fetches/LLM calls (be polite)")
    ap.add_argument("--llm-limit", type=int, help="cap how many families get LLM disambiguation/verification this run")
    ap.add_argument("--families", help="comma-separated family names to restrict this run to (pilot targeting)")
    ap.add_argument(
        "--brave-search", action="store_true",
        help="for families with no readme/sibling HF candidate, query Brave's web search "
             "(site:huggingface.co) as a last-resort tier 3; needs BRAVE_SEARCH_API_KEY",
    )
    a = ap.parse_args(argv)

    load_dotenv()
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if a.brave_search and not brave_api_key:
        print("[hf_source] --brave-search set but BRAVE_SEARCH_API_KEY is missing, skipping", flush=True)
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

    verified_releases = {release_key(m["model"], m["tag"]) for m in models if m["digest"] in verified}
    verified_families = {m["model"] for m in models if m["digest"] in verified}
    releases_by_family = _distinct_releases_by_family(models)

    families_filter = set(a.families.split(",")) if a.families else None
    family_names = sorted(f for f in library if not families_filter or f in families_filter)

    # Phase A: candidate gathering + release-level picking (deterministic, free; LLM-backed
    # disambiguation only when a.verify_tier2 and within the per-run llm_limit budget).
    family_repo: dict[str, str] = {}                          # today's by_family: top candidate
    release_repo: dict[str, str] = {}                         # new: by_release
    unresolved_families: list[str] = []
    unresolved_releases: list[str] = []
    verify_targets: dict[str, set[str]] = defaultdict(set)    # fam -> {repo, ...} needing verification
    brave_families: set[str] = set()                          # families whose candidates came from Brave
    llm_budget_families = 0

    for fam in family_names:
        meta = library.get(fam) or {}
        readme = meta.get("readme") or ""
        description = meta.get("description") or ""
        # Union, not fallback: a family's readme very often names only the top-of-family
        # checkpoint (llava's readme has exactly one HF link), so a release needing a *sibling*
        # size only ever gets one if the sibling scan runs even when the readme already found
        # something -- an earlier version of this gated sibling_candidates behind "readme found
        # nothing," which meant llava:13b/34b never saw their own repos at all.
        candidates = extract_readme_candidates(readme)
        siblings = sibling_candidates(fam, [readme] + _load_family_content_texts(fam))
        for repo in siblings:
            if repo not in candidates:
                candidates.append(repo)

        # Tier 3, last resort: readme + sibling mining found nothing at all for this family.
        # Every candidate this family gets this run is therefore Brave-sourced -- one set is
        # enough to track that, no per-repo/per-release bookkeeping needed, since `candidates`
        # (and everything downstream keyed off it) is computed once per family and shared by
        # both the family-wide pick and every release in it.
        if not candidates and a.brave_search and brave_api_key:
            brave_repos = brave_candidates(fam, brave_api_key)
            if brave_repos:
                candidates = brave_repos
                brave_families.add(fam)

        allow_llm = bool(a.verify_tier2) and (a.llm_limit is None or llm_budget_families < a.llm_limit)
        if allow_llm:
            llm_budget_families += 1

        if fam not in verified_families:
            if candidates:
                family_repo[fam] = candidates[0]
                verify_targets[fam].add(candidates[0])
            else:
                unresolved_families.append(fam)

        for release in releases_by_family.get(fam, []):
            if release in verified_releases:
                continue
            if not candidates:
                unresolved_releases.append(release)
                continue
            tag_suffix = release.split(":", 1)[-1]
            picked = pick_repo_for_release(fam, description, tag_suffix, candidates, a.llm_model, allow_llm)
            if picked:
                release_repo[release] = picked
                verify_targets[fam].add(picked)
            else:
                unresolved_releases.append(release)

    # Phase B: verify each distinct (family, repo) pair once, regardless of how many releases
    # (or the family-wide pick) share it.
    verify_hits = verify_misses = 0
    verified_repo: dict[tuple[str, str], bool | None] = {}
    if a.verify_tier2:
        todo = [(fam, repo) for fam, repos in verify_targets.items() for repo in sorted(repos)]
        print(f"[hf_source] --verify-tier2: {len(todo)} distinct (family, repo) candidates, model={a.llm_model}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, a.llm_workers)) as ex:
            futs = {
                ex.submit(verify_readme_candidate, fam, (library.get(fam) or {}).get("description") or "", repo, a.llm_model): (fam, repo)
                for fam, repo in todo
            }
            for fut in as_completed(futs):
                fam, repo = futs[fut]
                result = fut.result()
                verified_repo[(fam, repo)] = result["confirmed"]
                if result["confirmed"] is True:
                    verify_hits += 1
                elif result["confirmed"] is False:
                    verify_misses += 1
        print(f"[hf_source] --verify-tier2: {verify_hits} confirmed, {verify_misses} rejected", flush=True)

    # Phase C: assemble by_family / by_release, applying verification results. A rejected pick
    # (confirmed is False) drops to unresolved rather than shipping a match already known wrong;
    # confirmed is True/None keep "*_verified"/plain respectively, same as before. A family whose
    # candidates came from Brave (tier 3) gets its own method vocabulary -- "brave_search" was
    # never named in a readme, so labeling it "readme" would misrepresent where it came from
    # (web/app.js's tooltip text depends on this distinction being accurate).
    likely: dict[str, dict] = {}
    for fam, repo in family_repo.items():
        confirmed = verified_repo.get((fam, repo))
        if confirmed is False:
            unresolved_families.append(fam)
            continue
        if fam in brave_families:
            method = "brave_search_verified" if confirmed is True else "brave_search"
        else:
            method = "readme_verified" if confirmed is True else "readme"
        likely[fam] = {"repo": repo, "url": f"https://huggingface.co/{repo}", "confidence": "likely", "method": method}

    likely_release: dict[str, dict] = {}
    for release, repo in release_repo.items():
        fam = release.split(":", 1)[0]
        confirmed = verified_repo.get((fam, repo))
        if confirmed is False:
            unresolved_releases.append(release)
            continue
        if fam in brave_families:
            method = "brave_search_verified" if confirmed is True else "brave_search"
        else:
            method = "readme_verified" if confirmed is True else "readme"
        likely_release[release] = {"repo": repo, "url": f"https://huggingface.co/{repo}", "confidence": "likely", "method": method}

    # A --families run only touches the named families -- merge its results into whatever's
    # already on disk for every other family, rather than replacing the whole file. Without
    # this, a pilot run would silently wipe out every previously-resolved family/release.
    processed = set(family_names)
    prior = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    merged_likely = {**{f: v for f, v in (prior.get("by_family") or {}).items() if f not in processed}, **likely}
    merged_likely_release = {
        **{r: v for r, v in (prior.get("by_release") or {}).items() if r.split(":", 1)[0] not in processed},
        **likely_release,
    }
    merged_unresolved_families = sorted(
        {f for f in (prior.get("unresolved_families") or []) if f not in processed} | set(unresolved_families)
    )
    merged_unresolved_releases = sorted(
        {r for r in (prior.get("unresolved_releases") or []) if r.split(":", 1)[0] not in processed} | set(unresolved_releases)
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "digest_count": len(digests),
                "verified_count": len(verified),
                "likely_family_count": len(merged_likely),
                "likely_release_count": len(merged_likely_release),
                "unresolved_families": merged_unresolved_families,
                "unresolved_releases": merged_unresolved_releases,
                "by_digest": verified,
                "by_family": merged_likely,
                "by_release": merged_likely_release,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[hf_source] {len(verified)}/{len(digests)} digests verified by hash, "
        f"{len(merged_likely)} families / {len(merged_likely_release)} releases have a likely (unverified) repo, "
        f"{len(merged_unresolved_families)} families and {len(merged_unresolved_releases)} releases unresolved"
        + (f" ({verify_hits} readme_verified)" if a.verify_tier2 else "")
        + f" -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
