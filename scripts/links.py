#!/usr/bin/env python3
"""Resolve homepage / GitHub / paper links per Ollama model family, fetch page content for all
but paper, and (for releases scripts/hf_source.py didn't already resolve) reuse that same
content as one more chance at finding a release-specific Hugging Face repo.

Every link type comes from the same place: scripts/link_common.py:classify_readme_links()
splits the family's ollama.com/library readme (already crawled by scripts/crawl.py into
library.json) into huggingface / paper / homepage / github buckets. This script handles the
three hf_source.py doesn't:

- github / homepage: family-scoped -- one readme-linked candidate is accepted directly (method
  "readme"); more than one goes to an LLM to pick between them (link_common.llm_pick_url). A
  project has exactly one repo/site regardless of which size someone's asking about.
- paper: family-scoped too. A readme-linked arxiv.org link wins if there is one, else the first
  candidate. Never fetched for content -- link only.
- github's README (via the GitHub API) and the homepage (via link_common.http_get, Exa as a
  last resort for client-rendered pages) get their content fetched and saved once per family.
  That content is also scanned for embedded huggingface.co links / bare org/repo tokens: for
  each *release* (family + tag with the quant suffix stripped, link_common.release_key) that
  scripts/hf_source.py's tiers left unresolved, the candidates found this way get a
  deterministic size-token match when unambiguous, or an LLM pick
  (link_common.llm_pick_repo, cached and deduped by size so releases sharing a size don't repeat
  the call) when not -- a release-specific hf repo, not a single family-wide guess, as a side
  effect of content already being fetched for its own sake.

Content is saved once per family at data/link_content/<family>.json (not gitignored -- like
data/hf_links/ before it, this is a durable record of what was actually fetched, not a
rebuildable cache) and folded into prod/data/link_content.json by build_prod_data.py. hf content
specifically is keyed by repo (not release) within that file, since multiple releases in a
family can legitimately share one repo.

Needs OPENROUTER_API_KEY + OPENROUTER_BASE_URL (github/homepage disambiguation and the HF pick),
and optionally EXA_API_KEY (homepage fetch fallback for client-rendered pages) and GITHUB_TOKEN
(lifts the GitHub API's 60/hr anonymous rate limit to 5000/hr) -- env or a .env file at the repo
root (link_common.load_dotenv()).

    python scripts/links.py --resolve                    # fill in every family, cache-first
    python scripts/links.py --resolve --limit 20          # pilot a sample
    python scripts/links.py --resolve --families llava,wizardlm  # pilot specific families
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
    pick_candidate_for_release,
    read_json,
    release_key,
    size_token,
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
    `kind` is "github" or "homepage" here -- flat, one value per family, since a project has
    exactly one repo/site regardless of which size someone's asking about. `hf` content is
    keyed by repo instead (write_hf_content) since different releases in a family can resolve
    to different Hugging Face repos.
    """
    path = _content_path(family)
    existing = read_json(path) if path.exists() else {}
    existing[kind] = {"url": url, "content": text[:CONTENT_MAX]}
    atomic_write_json(path, existing)


def write_hf_content(family: str, repo: str, url: str, text: str) -> None:
    """Merge one Hugging Face repo's content into the family's content file, keyed by repo (not
    release) -- multiple releases in a family legitimately share one repo, and content-fetching
    is already cached per-repo (link_common.fetch_hf_readme), so this only ever writes a repo's
    content once even if several releases pick it.
    """
    path = _content_path(family)
    existing = read_json(path) if path.exists() else {}
    hf_entry = existing.get("hf")
    if not isinstance(hf_entry, dict) or any(not isinstance(v, dict) for v in hf_entry.values()):
        hf_entry = {}  # migrate an old flat {"url","content"} shape (or absent) to repo-keyed
    hf_entry[repo] = {"url": url, "content": text[:CONTENT_MAX]}
    existing["hf"] = hf_entry
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
    releases: list[str],
    existing_hf_by_release: dict[str, dict],
    exa_api_key: str | None,
    llm_model: str,
    allow_network: bool = True,
) -> dict:
    """github/homepage/paper stay family-scoped (one project repo/site/paper regardless of
    size). hf is release-scoped: `existing_hf_by_release` is whatever scripts/hf_source.py's
    tiers already resolved per release (verified hash match or a likely readme/sibling match);
    for any release with nothing there, the github-README/homepage content already being
    fetched for its own sake doubles as one more chance -- picked deterministically by size
    token when unambiguous, LLM-disambiguated (and cached, deduped by size so releases sharing
    a size don't repeat the call) otherwise. Every resolved hf repo -- from hf_source.py or from
    this fallback -- gets its own README content fetched and saved, keyed by repo.
    """
    readme = (meta or {}).get("readme") or ""
    description = (meta or {}).get("description") or ""
    _hf_readme_links, paper_links, homepage_links, github_repos = classify_readme_links(readme)

    result: dict = {"github": None, "homepage": None, "paper": None, "hf_by_release": {}}
    hf_candidates: list[dict] = []  # [{"repo", "source_url"}] gathered as a side effect below
    need_fallback = any(release not in existing_hf_by_release for release in releases)

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
            if need_fallback:
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
        if need_fallback:
            for repo in find_hf_repos_in_links(fetched.get("links") or []):
                hf_candidates.append({"repo": repo, "source_url": result["homepage"]["url"]})

    paper_url = _pick_paper(paper_links)
    result["paper"] = {"url": paper_url, "confidence": "likely", "method": "readme"} if paper_url else None

    if need_fallback and hf_candidates:
        seen, fallback_candidates = set(), []
        for c in hf_candidates:
            if c["repo"] not in seen:
                seen.add(c["repo"])
                fallback_candidates.append(c)
        candidate_repos = [c["repo"] for c in fallback_candidates]
        for release in releases:
            if release in existing_hf_by_release:
                continue
            tag_suffix = release.split(":", 1)[-1]
            picked = _pick_fallback_repo(family, description, tag_suffix, fallback_candidates, candidate_repos, llm_model, allow_network)
            if picked:
                method = "github_llm" if any(c["repo"] == picked and "github.com" in c["source_url"] for c in fallback_candidates) else "homepage_llm"
                result["hf_by_release"][release] = {"repo": picked, "url": f"https://huggingface.co/{picked}", "confidence": "likely", "method": method}

    # Content: fetch+save every distinct hf repo this family ended up with, from either source.
    distinct_repos = {hit["repo"] for hit in existing_hf_by_release.values()} | {hit["repo"] for hit in result["hf_by_release"].values()}
    for repo in distinct_repos:
        content = fetch_hf_readme(repo, HF_README_CACHE, allow_network)
        if content:
            write_hf_content(family, repo, f"https://huggingface.co/{repo}", content)

    return result


def _pick_fallback_repo(
    family: str, description: str, tag_suffix: str, candidates_with_source: list[dict],
    candidate_repos: list[str], llm_model: str, allow_network: bool,
) -> str | None:
    """Deterministic size-token match first (free, no cache needed); LLM disambiguation only
    when ambiguous, cached and deduped by (family, size-token) same as hf_source.py's release
    picking, so releases sharing a size don't repeat the LLM call.
    """
    repo, ambiguous = pick_candidate_for_release(tag_suffix, candidate_repos)
    if not ambiguous:
        return repo
    if not allow_network:
        return None

    dedup_key = size_token(tag_suffix) or tag_suffix
    cache_path = RESOLVE_CACHE / f"{family}-hf-{dedup_key}.json"
    if cache_path.exists():
        return read_json(cache_path)

    picked = llm_pick_repo(family, description, candidates_with_source, llm_model, release_hint=tag_suffix)
    atomic_write_json(cache_path, picked)
    return picked


# ------------------------------------------------------------------------------------------- main
def _releases_by_family(models: list[dict]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for m in models:
        out.setdefault(m["model"], set()).add(release_key(m["model"], m["tag"]))
    return {fam: sorted(releases) for fam, releases in out.items()}


def _existing_hf_by_release(models: list[dict], hf_sources: dict) -> dict[str, dict[str, dict]]:
    """What scripts/hf_source.py already resolved, keyed by family -> {release: hit}. A
    digest-level verified hit (release-exact by construction) beats a release-level likely hit,
    matching build_prod_data.resolve_links's precedence (verified always wins). Families/
    releases with nothing here are exactly the ones this module's fallback should attempt.
    """
    by_digest = hf_sources.get("by_digest") or {}
    by_release = hf_sources.get("by_release") or {}
    out: dict[str, dict[str, dict]] = {}
    for m in models:
        fam, release = m["model"], release_key(m["model"], m["tag"])
        hit = by_digest.get(m["digest"])
        if hit:
            out.setdefault(fam, {})[release] = {"repo": hit["repo"], "url": hit["url"], "confidence": "verified"}
    for release, hit in by_release.items():
        fam = release.split(":", 1)[0]
        if release not in out.get(fam, {}):
            out.setdefault(fam, {})[release] = {"repo": hit["repo"], "url": hit["url"], "confidence": "likely", "method": hit.get("method", "readme")}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolve", action="store_true", help="fetch/compute anything not already cached")
    ap.add_argument("--limit", type=int, help="cap how many families to process this run (for a pilot)")
    ap.add_argument("--workers", type=int, default=3, help="parallel fetches/LLM calls (be polite)")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--families", help="comma-separated family names to restrict this run to (pilot targeting)")
    a = ap.parse_args(argv)

    load_dotenv()
    library_path = ROOT / "data/out/library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")).get("families", {}) if library_path.exists() else {}
    models_path = ROOT / "prod/data/models.json"
    models = json.loads(models_path.read_text(encoding="utf-8"))["models"] if models_path.exists() else []
    hf_sources_path = ROOT / "data/out/hf_sources.json"
    hf_sources = json.loads(hf_sources_path.read_text(encoding="utf-8")) if hf_sources_path.exists() else {}
    existing_hf = _existing_hf_by_release(models, hf_sources)
    releases_by_family = _releases_by_family(models)

    families_filter = set(a.families.split(",")) if a.families else None
    families = sorted(f for f in library if not families_filter or f in families_filter)
    if a.limit:
        families = families[: a.limit]

    exa_api_key = os.environ.get("EXA_API_KEY")
    by_family: dict[str, dict] = {}

    print(f"[links] {'resolving' if a.resolve else 'rebuilding from cache'} {len(families)} families, {a.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {
            ex.submit(
                resolve_family, fam, library.get(fam), releases_by_family.get(fam, []),
                existing_hf.get(fam, {}), exa_api_key, a.llm_model, a.resolve,
            ): fam
            for fam in families
        }
        done = 0
        for fut in as_completed(futs):
            fam = futs[fut]
            result = fut.result()
            if result["github"] or result["homepage"] or result["paper"] or result["hf_by_release"]:
                by_family[fam] = result
            done += 1
            if done % 20 == 0:
                print(f"[links] {done}/{len(families)}", flush=True)

    # --families or --limit only touch a subset of families -- merge into whatever's already on
    # disk for every other family rather than replacing the whole file, or a pilot/partial run
    # would silently wipe out every previously-resolved family.
    processed = set(families)
    prior_by_family = json.loads(LINKS_OUT.read_text(encoding="utf-8")).get("by_family", {}) if LINKS_OUT.exists() else {}
    merged_by_family = {**{f: v for f, v in prior_by_family.items() if f not in processed}, **by_family}

    LINKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    LINKS_OUT.write_text(
        json.dumps({"schema_version": 2, "family_count": len(merged_by_family), "by_family": merged_by_family}, indent=2),
        encoding="utf-8",
    )
    counts = {k: sum(1 for v in merged_by_family.values() if v.get(k)) for k in ("github", "homepage", "paper")}
    counts["hf_releases"] = sum(len(v.get("hf_by_release") or {}) for v in merged_by_family.values())
    print(f"[links] families resolved={len(merged_by_family)} {counts} -> {LINKS_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
