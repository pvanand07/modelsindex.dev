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

Two more tiers fetch a page and ask an LLM (OpenRouter) to judge what it finds:

--verify-tier2  Exa-fetches the Hugging Face page a readme-derived candidate points at, and
                asks the LLM whether that page actually describes this model. Exa specifically
                (not hrequests) because huggingface.co is a client-rendered SPA -- a plain GET
                sees an empty shell, not the model card. A confirmed candidate is promoted to
                method "readme_verified"; a rejected one falls through to --llm-extract in the
                same run.
--llm-extract   for families with nothing yet, fetches the family's homepage, paper and/or
                GitHub repo (whichever the readme links) and asks the LLM to pick the correct
                Hugging Face repo out of whatever those pages link to. Fetches via hrequests
                first (a real browser-fingerprint GET with proper DOM link parsing -- Exa's own
                `extras.links` picker turned out to drop real anchors on link-heavy pages, e.g.
                it missed the huggingface.co link on mistral.ai/news/devstral even at
                `links: 100`); Exa is only consulted when hrequests can't fetch a page at all
                (network block, non-2xx, or the package isn't installed). Still "likely", not
                "verified" -- these are unverified content matches, not hash matches.

Needs OPENROUTER_API_KEY + OPENROUTER_BASE_URL (both tiers) and EXA_API_KEY (fallback fetch +
tier-2 verification), read from the environment or from a .env file at the repo root (see
load_dotenv()). `pip install hrequests` for the primary fetch path in --llm-extract; without it,
that tier falls back to Exa for every URL. Every page fetched this way is saved to
data/hf_links/exa_results.json (not gitignored -- it's the audit trail of what was actually
found, not a rebuildable cache).

    python scripts/hf_source.py --refresh          # crawl every digest, cache-first
    python scripts/hf_source.py --refresh --limit 300   # pilot a sample
    python scripts/hf_source.py --verify-tier2      # Exa+LLM verify readme-derived candidates
    python scripts/hf_source.py --llm-extract       # homepage/paper/github fallback for the rest
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
import uuid
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


# --------------------------------------------------------------------------- Exa-backed fallback
# When the readme has no usable Hugging Face link, it often still links the model's project
# homepage, its paper, or its GitHub repo. All three commonly link the Hugging Face repo
# themselves (a "Model" badge, a "Resources" section, a README badge) even when the ollama.com
# readme doesn't. Exa's /contents API returns a crawled/rendered copy of the page plus its
# outbound links directly -- a plain `urllib` GET only sees pre-render HTML, which is why an
# earlier raw-fetch version of this tier missed most client-rendered marketing pages. Since a
# homepage/paper/repo page can carry many unrelated links (socials, other products, other orgs'
# models), an LLM call picks the one that actually matches this model family.
_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)", re.I)
_LABEL_PAPER = re.compile(r"paper|arxiv|preprint|technical report", re.I)
_LABEL_HOMEPAGE = re.compile(r"website|homepage|home page|project page|\bblog\b|project site", re.I)
_ARXIV_LINK = re.compile(r"https?://arxiv\.org/abs/[0-9.]+v?\d*", re.I)
_GITHUB_BAD_PATH = ("issues", "pull", "blob", "tree", "wiki", "actions", "releases", "discussions", "commit", "compare")
_GITHUB_BAD_ORG = ("orgs", "sponsors", "marketplace", "topics", "features", "about", "pricing")
DEFAULT_LLM_MODEL = "~z-ai/glm-flash-latest"
LLM_CACHE = DATA / "cache" / "hf_llm_extract"
EXA_CACHE = DATA / "cache" / "hf_exa"
EXA_RESULTS = DATA / "hf_links" / "exa_results.json"  # durable audit trail -- not gitignored
EXA_CONTENTS_API = "https://api.exa.ai/contents"


def _clean_github(url: str) -> str | None:
    if "github.com" not in url.lower():
        return None
    path = url.split("github.com/", 1)[-1].split("?")[0].split("#")[0]
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2 or parts[0].lower() in _GITHUB_BAD_ORG:
        return None
    for stop in _GITHUB_BAD_PATH:
        if stop in parts:
            parts = parts[: parts.index(stop)]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def classify_readme_links(readme: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (huggingface repos, paper urls, homepage urls, github repos) found in the readme."""
    hf, paper, homepage, github = [], [], [], []
    for label, url in _MD_LINK.findall(readme or ""):
        low = label.lower()
        if "huggingface.co" in url.lower():
            repo = _clean_repo(url)
            if repo:
                hf.append(repo)
        elif "github.com" in url.lower():
            repo = _clean_github(url)
            if repo:
                github.append(repo)
        elif _LABEL_PAPER.search(low) or "arxiv.org" in url.lower():
            paper.append(url)
        elif _LABEL_HOMEPAGE.search(low):
            homepage.append(url)
    return hf, paper, homepage, github


def load_dotenv(path: Path | None = None) -> None:
    """Minimal `.env` loader (stdlib only). Real environment variables always win."""
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically, safe under concurrent writers hitting the same path.

    Two ThreadPoolExecutor workers can legitimately fetch the identical URL (two families
    linking the same paper/GitHub repo) and race to cache it -- a shared `<hash>.tmp` name
    caused a WinError 32 (file in use) when both threads' os.replace landed at once. A
    per-call unique suffix avoids the collision; whichever writer finishes last wins, which is
    fine since both are writing the same content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def exa_fetch(urls: list[str], api_key: str, livecrawl: str | None = None) -> dict:
    body = {"ids": urls, "extras": {"links": 25}, "text": {"maxCharacters": 20000}}
    if livecrawl:
        body["livecrawl"] = livecrawl
    req = urllib.request.Request(
        EXA_CONTENTS_API,
        data=json.dumps(body).encode("utf-8"),
        headers={"x-api-key": api_key, "content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read())


def _exa_cache_path(url: str) -> Path:
    return EXA_CACHE / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"


def exa_fetch_cached(urls: list[str], api_key: str) -> dict[str, dict | None]:
    """Exa-fetch each url, cached per-URL forever. One live-crawl retry for a non-success status."""
    out: dict[str, dict | None] = {}
    to_fetch = []
    for url in urls:
        cache_path = _exa_cache_path(url)
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                out[url] = json.load(f)
        else:
            to_fetch.append(url)
    if not to_fetch:
        return out

    def _call(ids: list[str], livecrawl: str | None = None) -> tuple[dict[str, dict], dict[str, str]]:
        try:
            data = exa_fetch(ids, api_key, livecrawl=livecrawl)
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            return {}, {}
        results = {r["id"]: r for r in data.get("results", [])}
        statuses = {s["id"]: s.get("status") for s in data.get("statuses", [])}
        return results, statuses

    results, statuses = _call(to_fetch)
    retry = [u for u in to_fetch if statuses.get(u) not in ("success", "cached")]
    if retry:
        retry_results, retry_statuses = _call(retry, livecrawl="always")
        results.update(retry_results)
        statuses.update(retry_statuses)

    for url in to_fetch:
        result = results.get(url)  # None cached as a genuine "Exa has nothing for this URL"
        _atomic_write_json(_exa_cache_path(url), result)
        out[url] = result
    return out


# Content extraction: hrequests first, Exa as fallback. Confirmed on mistral.ai/news/devstral
# that Exa's `extras.links` is a curated subset, not a full anchor dump -- on a link-heavy page
# (product nav, footer, etc.) it filled up on boilerplate before ever reaching the one
# in-content link that mattered, even at `links: 100`. hrequests does a real browser-fingerprint
# HTTP GET (bypasses the TLS/JA3 bot-detection that can silently blank a plain `urllib` GET) and
# parses the actual DOM, so `resp.html.absolute_links` is every anchor on the page, not a picked
# subset. Exa only gets called for a URL when hrequests couldn't fetch it at all (network block,
# non-2xx, or the package isn't installed) -- Exa's own crawl/cache can reach pages a direct GET
# can't (heavier bot walls, or truly JS-only content hrequests' non-browser mode can't execute).
HREQUESTS_CACHE = DATA / "cache" / "hf_hrequests"


def fetch_links_hrequests(url: str) -> list[str] | None:
    """GET url via hrequests and return its absolute outbound links, or None if the fetch failed."""
    try:
        import hrequests
    except ImportError:
        return None
    try:
        resp = hrequests.get(url, timeout=20)
        if resp.status_code >= 400:
            return None
        return list(resp.html.absolute_links)
    except Exception:
        return None


def fetch_links_cached(url: str, exa_api_key: str | None) -> tuple[list[str], str]:
    """Return (links, source): hrequests first, Exa fallback when hrequests can't fetch it
    *or* comes back with zero anchors -- a real page essentially never has none, so an empty
    result from a non-erroring GET means the page is a client-rendered shell (confirmed on
    notion.site: 200 OK, 0 anchors, real content only materializes after JS runs), not that
    the page genuinely has no links.
    """
    cache_path = HREQUESTS_CACHE / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return cached["links"], cached["source"]

    links = fetch_links_hrequests(url)
    source = "hrequests"
    if not links:
        source = "exa"
        links = []
        if exa_api_key:
            raw = exa_fetch_cached([url], exa_api_key).get(url)
            if raw:
                links = list((raw.get("extras") or {}).get("links") or [])

    _atomic_write_json(cache_path, {"links": links, "source": source})
    return links, source


def find_hf_repos_in_links(links: list[str]) -> list[str]:
    seen, out = set(), []
    for url in links:
        repo = _clean_repo(url) if "huggingface.co" in url.lower() else None
        if repo and repo not in seen:
            seen.add(repo)
            out.append(repo)
    return out


def find_github_repos_in_links(links: list[str]) -> list[str]:
    seen, out = set(), []
    for url in links:
        repo = _clean_github(url) if "github.com" in url.lower() else None
        if repo and repo not in seen:
            seen.add(repo)
            out.append(repo)
    return out


def find_arxiv_links(links: list[str]) -> list[str]:
    return list(dict.fromkeys(u for u in links if _ARXIV_LINK.match(u)))


def openrouter_chat(prompt: str, model: str, api_key: str, base_url: str) -> str:
    # max_tokens covers reasoning tokens too on thinking models (e.g. glm-flash) -- a low
    # budget leaves nothing for the actual answer after the model "thinks" through it.
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 400}
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.loads(r.read())
    return (data["choices"][0]["message"].get("content") or "").strip()


def llm_pick_repo(family: str, description: str, candidates: list[dict], model: str) -> str | None:
    """Ask the LLM which candidate repo (if any) is the correct match for this model family.

    candidates: [{"repo": "org/name", "source_url": "...", "snippet": "..."}, ...]
    """
    if not candidates:
        return None
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL")
    if not api_key or not base_url:
        return None
    repos = [c["repo"] for c in candidates]
    listing = "\n".join(
        f"- {c['repo']} (found on {c.get('source_url') or 'an unknown page'}"
        + (f': "{c["snippet"][:200]}"' if c.get("snippet") else "")
        + ")"
        for c in candidates
    )
    prompt = (
        f'Ollama model family: "{family}"\n'
        f"Description: {description or '(none)'}\n\n"
        f"Candidate Hugging Face repositories found while researching this model:\n{listing}\n\n"
        "Which one, if any, is the correct upstream Hugging Face repository for this exact model? "
        "Reply with ONLY the repo in org/repo format, or reply NONE if none of them match."
    )
    try:
        reply = openrouter_chat(prompt, model, api_key, base_url)
    except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError, json.JSONDecodeError):
        return None
    if reply.strip().upper().startswith("NONE"):
        return None
    m = re.search(r"[\w.\-]+/[\w.\-]+", reply)
    picked = m.group(0) if m else None
    return picked if picked in repos else None


def resolve_via_pages(family: str, meta: dict, llm_model: str) -> dict:
    """Fetch the family's homepage/paper/GitHub links (hrequests, Exa fallback) and ask the LLM
    to pick the HF repo, if any.

    Cached per family regardless of outcome -- a `repo: None` result means "checked, nothing
    found", not "not checked yet", so reruns don't repeat network/LLM calls for a family that
    genuinely has no discoverable Hugging Face link. Every fetch made along the way is kept in
    `raw` so the caller can persist it to data/hf_links/exa_results.json.
    """
    cache_path = LLM_CACHE / f"{family}.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    exa_api_key = os.environ.get("EXA_API_KEY")
    readme = (meta or {}).get("readme") or ""
    description = (meta or {}).get("description") or ""
    _hf_links, paper_links, homepage_links, github_links = classify_readme_links(readme)

    raw: dict[str, dict] = {}
    candidates: list[dict] = []
    seen_repos: set[tuple[str, str]] = set()

    def _fetch_and_scan(url: str) -> list[str]:
        links, source = fetch_links_cached(url, exa_api_key)
        raw[url] = {"source": source, "links": links}
        for repo in find_hf_repos_in_links(links):
            if (url, repo) in seen_repos:
                continue
            seen_repos.add((url, repo))
            candidates.append({"repo": repo, "source_url": url, "snippet": ""})
        return links

    github_urls = [f"https://github.com/{r}" for r in github_links[:1]]  # classify_readme_links returns org/repo, not a URL
    first_round = list(dict.fromkeys(homepage_links[:1] + paper_links[:1] + github_urls))
    all_links: list[str] = []
    for url in first_round:
        all_links += _fetch_and_scan(url)

    if not candidates and first_round:
        # nothing on the first round -- try one more hop via a paper/GitHub link that showed
        # up *on* those pages, if we hadn't already tried it directly
        second_round = list(dict.fromkeys(find_arxiv_links(all_links) + [f"https://github.com/{r}" for r in find_github_repos_in_links(all_links)]))
        second_round = [u for u in second_round if u not in raw][:2]
        for url in second_round:
            _fetch_and_scan(url)

    picked = llm_pick_repo(family, description, candidates, llm_model) if candidates else None
    result = {"repo": picked, "urls_fetched": list(raw.keys()), "candidates": candidates, "raw": raw}
    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result), encoding="utf-8")
    return result


def verify_readme_candidate(family: str, description: str, repo: str, llm_model: str) -> dict:
    """Exa-fetch the HF page a readme link named, and ask the LLM if it matches this family.

    `confirmed` is a tri-state: True/False is a real judgement, None means "not attempted"
    (missing EXA_API_KEY/OPENROUTER_* or the fetch turned up nothing) -- callers should treat
    None as "leave the existing unverified readme match alone", not as a rejection.
    """
    cache_path = LLM_CACHE / f"verify-{family}.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    exa_key = os.environ.get("EXA_API_KEY")
    or_key = os.environ.get("OPENROUTER_API_KEY")
    or_base = os.environ.get("OPENROUTER_BASE_URL")
    hf_url = f"https://huggingface.co/{repo}"

    confirmed: bool | None = None
    raw: dict | None = None
    if exa_key and or_key and or_base:
        raw = exa_fetch_cached([hf_url], exa_key).get(hf_url)
        if raw:
            prompt = (
                f'Ollama model family: "{family}"\n'
                f"Description: {description or '(none)'}\n\n"
                f'Hugging Face page found for the candidate repo "{repo}":\n'
                f"Title: {raw.get('title') or ''}\n"
                f"Text: {(raw.get('text') or '')[:1500]}\n\n"
                "Does this Hugging Face page describe the same model as the Ollama family above? "
                "Reply with ONLY YES or NO."
            )
            try:
                reply = openrouter_chat(prompt, llm_model, or_key, or_base)
                confirmed = reply.strip().upper().startswith("YES")
            except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError, json.JSONDecodeError):
                confirmed = None

    result = {"repo": repo, "url": hf_url, "confirmed": confirmed, "raw": {hf_url: raw} if raw else {}}
    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="query modelindex.dev for uncached/stale digests")
    ap.add_argument("--limit", type=int, help="cap how many digests to query this run (for a pilot)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests when --refresh (be polite)")
    ap.add_argument(
        "--verify-tier2", action="store_true",
        help="Exa-fetch the HF page a readme-derived candidate points at, ask the LLM to confirm it",
    )
    ap.add_argument(
        "--llm-extract", action="store_true",
        help="for families with nothing yet, Exa-fetch their homepage/paper/GitHub and ask an LLM to pick the repo",
    )
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="OpenRouter model id for --verify-tier2/--llm-extract")
    ap.add_argument("--llm-workers", type=int, default=3, help="parallel Exa fetches/LLM calls (be polite)")
    ap.add_argument("--llm-limit", type=int, help="cap how many families each of --verify-tier2/--llm-extract processes")
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

    exa_dump: dict[str, dict] = {}

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
                exa_dump[f"verify:{fam}"] = result
                if result["confirmed"] is True:
                    likely[fam] = {"repo": result["repo"], "url": result["url"], "confidence": "likely", "method": "readme_verified"}
                    verify_hits += 1
                elif result["confirmed"] is False:
                    unresolved.append(fam)  # rejected -- give --llm-extract a shot in this same run
                    verify_misses += 1
                # confirmed is None (not attempted): leave it for the plain "readme" fallback below
        print(f"[hf_source] --verify-tier2: {verify_hits} confirmed, {verify_misses} rejected", flush=True)

    # readme candidates that were never verified (flag off) or couldn't be (missing creds/fetch
    # miss) keep the original unverified method, unless verification already placed or rejected them
    for fam, repo in readme_candidates.items():
        if fam not in likely and fam not in unresolved:
            likely[fam] = {"repo": repo, "url": f"https://huggingface.co/{repo}", "confidence": "likely", "method": "readme"}

    llm_hits = 0
    if a.llm_extract:
        todo = unresolved[: a.llm_limit] if a.llm_limit else unresolved
        print(f"[hf_source] --llm-extract: {len(todo)} families with nothing yet, model={a.llm_model}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, a.llm_workers)) as ex:
            futs = {ex.submit(resolve_via_pages, fam, library.get(fam), a.llm_model): fam for fam in todo}
            done = 0
            for fut in as_completed(futs):
                fam = futs[fut]
                result = fut.result()
                exa_dump[fam] = result
                done += 1
                if result.get("repo"):
                    likely[fam] = {
                        "repo": result["repo"],
                        "url": f"https://huggingface.co/{result['repo']}",
                        "confidence": "likely",
                        "method": "homepage_llm",
                        "source": (result.get("urls_fetched") or [None])[0],
                    }
                    llm_hits += 1
                if done % 20 == 0:
                    print(f"[hf_source] --llm-extract: {done}/{len(todo)} ({llm_hits} hits)", flush=True)

    if exa_dump:
        EXA_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if EXA_RESULTS.exists():
            with open(EXA_RESULTS, encoding="utf-8") as f:
                existing = json.load(f)
        existing.update(exa_dump)
        EXA_RESULTS.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"[hf_source] wrote {len(exa_dump)} Exa result records -> {EXA_RESULTS}", flush=True)

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
        f"{len(likely)} families have a likely (unverified) repo"
        + (f" ({verify_hits} readme_verified)" if a.verify_tier2 else "")
        + (f" ({llm_hits} via --llm-extract)" if a.llm_extract else "")
        + f" -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
