#!/usr/bin/env python3
"""Resolve homepage / GitHub / paper links per Ollama model family, fetch page content for all
but paper, and (for families scripts/hf_source.py didn't already resolve) reuse that same
content as one more chance at finding the Hugging Face repo.

Every link type comes from the same place: scripts/link_common.py:classify_readme_links()
splits the family's ollama.com/library readme (already crawled by scripts/crawl.py into
library.json) into huggingface / paper / homepage / github buckets. This script handles the
three hf_source.py doesn't:

- github / homepage: one readme-linked candidate is accepted directly (method "readme"); more
  than one goes to an LLM to pick between them (link_common.llm_pick_url).
- paper: a readme-linked arxiv.org link wins if there is one, else the first candidate. Never
  fetched for content -- link only.
- github's README (via the GitHub API) and the homepage (via link_common.http_get, Exa as a
  last resort for client-rendered pages) get their content fetched and saved. That content is
  also scanned for an embedded huggingface.co link: for a family scripts/hf_source.py's tier
  1/2 left unresolved, any Hugging Face repos turned up this way go to the LLM for a final pick
  (link_common.llm_pick_repo), same as before -- but now as a side effect of content already
  being fetched for its own sake, not a separate fetch pass that throws its findings away.

Content is saved once per family at data/link_content/<family>.json (not gitignored -- like
data/hf_links/ before it, this is a durable record of what was actually fetched, not a
rebuildable cache) and folded into prod/data/link_content.json by build_prod_data.py.

Needs OPENROUTER_API_KEY + OPENROUTER_BASE_URL (github/homepage disambiguation and the HF pick),
and optionally EXA_API_KEY (homepage fetch fallback for client-rendered pages) and GITHUB_TOKEN
(lifts the GitHub API's 60/hr anonymous rate limit to 5000/hr) -- env or a .env file at the repo
root (link_common.load_dotenv()).

    python scripts/links.py --resolve                    # fill in every family, cache-first
    python scripts/links.py --resolve --limit 20          # pilot a sample
    python scripts/links.py                                # rebuild data/out/links.json from cache only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from link_common import (  # noqa: E402
    CONTENT_MAX,
    DATA,
    ROOT,
    DEFAULT_LLM_MODEL,
    atomic_write_json,
    classify_readme_links,
    extract_links,
    fetch_github_readme,
    fetch_hf_readme,
    find_hf_repos_in_links,
    html_to_text,
    http_get,
    llm_pick_repo,
    llm_pick_url,
    load_dotenv,
    read_json,
)

LINKS_OUT = DATA / "out" / "links.json"
CONTENT_DIR = DATA / "link_content"
RESOLVE_CACHE = DATA / "cache" / "links" / "resolve"
GITHUB_README_CACHE = DATA / "cache" / "github_readme"
HF_README_CACHE = DATA / "cache" / "hf_readme"
HOMEPAGE_CACHE = DATA / "cache" / "links" / "homepage"
EXA_CACHE = DATA / "cache" / "links" / "exa"
# durable audit trail, not gitignored -- keyed by URL, unlike the pre-existing (now-retired)
# tier-2b's family-keyed data/hf_links/exa_results.json, so this intentionally doesn't reuse
# that path/schema.
EXA_RESULTS = DATA / "hf_links" / "homepage_exa_results.json"
EXA_CONTENTS_API = "https://api.exa.ai/contents"


# ----------------------------------------------------------------------------------- Exa fallback
# Homepage pages vary too much for a fixed API to help (unlike Hugging Face/GitHub's raw-content
# endpoints); a real GET (via link_common.http_get, hrequests-first) covers most of them, but a
# client-rendered page (React/Vue marketing sites, notion.site) comes back as an empty shell.
# Exa's /contents API returns a crawled/rendered copy for those, at the cost of an API call.
def exa_fetch(urls: list[str], api_key: str) -> dict:
    body = {"ids": urls, "extras": {"links": 25}, "text": {"maxCharacters": CONTENT_MAX}, "livecrawl": "always"}
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


def exa_fetch_cached(url: str, api_key: str) -> dict | None:
    cache_path = _exa_cache_path(url)
    if cache_path.exists():
        return read_json(cache_path)
    try:
        data = exa_fetch([url], api_key)
        result = next(iter(data.get("results") or []), None)
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
        result = None
    atomic_write_json(cache_path, result)
    if result:
        _append_exa_audit(url, result)
    return result


def _append_exa_audit(url: str, result: dict) -> None:
    existing = read_json(EXA_RESULTS) if EXA_RESULTS.exists() else {}
    existing[url] = result
    atomic_write_json(EXA_RESULTS, existing)


# ------------------------------------------------------------------------------------- fetch+save
def fetch_homepage(url: str, exa_api_key: str | None, allow_network: bool = True) -> dict:
    """Returns {"text": str|None, "links": [str], "source": "direct"|"exa"}, cached forever."""
    cache_path = HOMEPAGE_CACHE / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"
    if cache_path.exists():
        return read_json(cache_path)
    if not allow_network:
        return {"text": None, "links": [], "source": None}

    text, links, source = None, [], "direct"
    result = http_get(url)
    if result and result[0] < 400 and result[1].strip():
        links = extract_links(result[1], url)
        text = html_to_text(result[1])
    if not links and exa_api_key:
        source = "exa"
        raw = exa_fetch_cached(url, exa_api_key)
        if raw:
            links = list((raw.get("extras") or {}).get("links") or [])
            text = raw.get("text") or text

    out = {"text": (text or "")[:CONTENT_MAX] or None, "links": links, "source": source}
    atomic_write_json(cache_path, out)
    return out


def _content_path(family: str) -> Path:
    return CONTENT_DIR / f"{family}.json"


def write_content(family: str, kind: str, url: str, text: str) -> None:
    """Merge one link type's content into the family's durable content file. Families are each
    processed by a single worker thread, so different kinds for the same family never race.
    """
    path = _content_path(family)
    existing = read_json(path) if path.exists() else {}
    existing[kind] = {"url": url, "content": text[:CONTENT_MAX]}
    atomic_write_json(path, existing)


def _pick_paper(candidates: list[str]) -> str | None:
    for c in candidates:
        if "arxiv.org" in c.lower():
            return c
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------------------- resolver
_HF_URL_IN_TEXT = re.compile(r"https?://huggingface\.co/\S+", re.I)


def _scan_text_for_hf(text: str) -> list[str]:
    """A GitHub README is markdown, so a huggingface.co link in it is plain text, not an <a
    href> -- reuse find_hf_repos_in_links on whatever URL-shaped substrings turn up.
    """
    return find_hf_repos_in_links(_HF_URL_IN_TEXT.findall(text))


def resolve_family(
    family: str,
    meta: dict | None,
    existing_hf: dict | None,
    exa_api_key: str | None,
    llm_model: str,
    allow_network: bool = True,
) -> dict:
    """existing_hf is whatever scripts/hf_source.py's tier 1/2 already resolved for this family
    (verified hash match or a likely readme match), or None if it found nothing. Either way this
    always (re)fetches and saves that repo's README content -- hf_source.py never persisted page
    content, only the link. When existing_hf is None, github/homepage content already being
    fetched for their own sake doubles as one more chance at finding an HF repo (cached per
    family in RESOLVE_CACHE/<family>-hf.json so the LLM pick isn't repeated on every rerun).
    """
    readme = (meta or {}).get("readme") or ""
    description = (meta or {}).get("description") or ""
    _hf_readme_links, paper_links, homepage_links, github_repos = classify_readme_links(readme)

    result: dict = {"github": None, "homepage": None, "paper": None, "hf": None}
    hf_candidates: list[dict] = []  # [{"repo", "source_url"}] gathered as a side effect below

    gh_pick_cache = RESOLVE_CACHE / f"{family}-github.json"
    if gh_pick_cache.exists():
        result["github"] = read_json(gh_pick_cache)
    elif allow_network:
        github_repo = llm_pick_url(family, description, "GitHub repository", github_repos, llm_model)
        result["github"] = (
            {"repo": github_repo, "url": f"https://github.com/{github_repo}", "confidence": "likely", "method": "readme"}
            if github_repo else None
        )
        atomic_write_json(gh_pick_cache, result["github"])
    if result["github"]:
        content = fetch_github_readme(result["github"]["repo"], GITHUB_README_CACHE, allow_network)
        if content:
            write_content(family, "github", result["github"]["url"], content)
            if existing_hf is None:
                for repo in _scan_text_for_hf(content):
                    hf_candidates.append({"repo": repo, "source_url": result["github"]["url"]})

    hp_pick_cache = RESOLVE_CACHE / f"{family}-homepage.json"
    if hp_pick_cache.exists():
        result["homepage"] = read_json(hp_pick_cache)
    elif allow_network:
        homepage_url = llm_pick_url(family, description, "project homepage", homepage_links, llm_model)
        result["homepage"] = {"url": homepage_url, "confidence": "likely", "method": "readme"} if homepage_url else None
        atomic_write_json(hp_pick_cache, result["homepage"])
    if result["homepage"]:
        fetched = fetch_homepage(result["homepage"]["url"], exa_api_key, allow_network)
        if fetched.get("text"):
            write_content(family, "homepage", result["homepage"]["url"], fetched["text"])
        if existing_hf is None:
            for repo in find_hf_repos_in_links(fetched.get("links") or []):
                hf_candidates.append({"repo": repo, "source_url": result["homepage"]["url"]})

    paper_url = _pick_paper(paper_links)
    result["paper"] = {"url": paper_url, "confidence": "likely", "method": "readme"} if paper_url else None

    if existing_hf is not None:
        result["hf"] = existing_hf
    elif hf_candidates:
        hf_pick_cache = RESOLVE_CACHE / f"{family}-hf.json"
        if hf_pick_cache.exists():
            result["hf"] = read_json(hf_pick_cache)
        elif allow_network:
            seen, deduped = set(), []
            for c in hf_candidates:
                if c["repo"] not in seen:
                    seen.add(c["repo"])
                    deduped.append(c)
            picked = llm_pick_repo(family, description, deduped, llm_model)
            if picked:
                method = "github_llm" if any(c["repo"] == picked and "github.com" in c["source_url"] for c in deduped) else "homepage_llm"
                result["hf"] = {"repo": picked, "url": f"https://huggingface.co/{picked}", "confidence": "likely", "method": method}
            atomic_write_json(hf_pick_cache, result["hf"])

    if result["hf"]:
        content = fetch_hf_readme(result["hf"]["repo"], HF_README_CACHE, allow_network)
        if content:
            write_content(family, "hf", result["hf"]["url"], content)

    return result


# ------------------------------------------------------------------------------------------- main
def _existing_hf_by_family(models: list[dict], hf_sources: dict) -> dict[str, dict]:
    """What scripts/hf_source.py already resolved, keyed by family: a digest-level verified hit
    (preferred) or a family-level likely hit, matching build_prod_data.resolve_hf_source's
    precedence (verified always wins).
    """
    by_digest = hf_sources.get("by_digest") or {}
    by_family = hf_sources.get("by_family") or {}
    out: dict[str, dict] = {}
    for m in models:
        hit = by_digest.get(m["digest"])
        if hit and m.get("model") not in out:
            out[m["model"]] = {"repo": hit["repo"], "url": hit["url"], "confidence": "verified"}
    for fam, hit in by_family.items():
        if fam not in out:
            out[fam] = {"repo": hit["repo"], "url": hit["url"], "confidence": "likely", "method": hit.get("method", "readme")}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolve", action="store_true", help="fetch/compute anything not already cached")
    ap.add_argument("--limit", type=int, help="cap how many families to process this run (for a pilot)")
    ap.add_argument("--workers", type=int, default=3, help="parallel fetches/LLM calls (be polite)")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    a = ap.parse_args(argv)

    load_dotenv()
    library_path = ROOT / "data/out/library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")).get("families", {}) if library_path.exists() else {}
    models_path = ROOT / "prod/data/models.json"
    models = json.loads(models_path.read_text(encoding="utf-8"))["models"] if models_path.exists() else []
    hf_sources_path = ROOT / "data/out/hf_sources.json"
    hf_sources = json.loads(hf_sources_path.read_text(encoding="utf-8")) if hf_sources_path.exists() else {}
    existing_hf = _existing_hf_by_family(models, hf_sources)

    families = sorted(library.keys())
    if a.limit:
        families = families[: a.limit]

    exa_api_key = os.environ.get("EXA_API_KEY")
    by_family: dict[str, dict] = {}

    print(f"[links] {'resolving' if a.resolve else 'rebuilding from cache'} {len(families)} families, {a.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {
            ex.submit(resolve_family, fam, library.get(fam), existing_hf.get(fam), exa_api_key, a.llm_model, a.resolve): fam
            for fam in families
        }
        done = 0
        for fut in as_completed(futs):
            fam = futs[fut]
            result = fut.result()
            if any(result.values()):
                by_family[fam] = result
            done += 1
            if done % 20 == 0:
                print(f"[links] {done}/{len(families)}", flush=True)

    LINKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    LINKS_OUT.write_text(
        json.dumps({"schema_version": 1, "family_count": len(by_family), "by_family": by_family}, indent=2),
        encoding="utf-8",
    )
    counts = {k: sum(1 for v in by_family.values() if v.get(k)) for k in ("hf", "github", "homepage", "paper")}
    print(f"[links] families resolved={len(by_family)} {counts} -> {LINKS_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
