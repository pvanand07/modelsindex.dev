#!/usr/bin/env python3
"""Shared helpers for scripts/hf_source.py and scripts/links.py.

Both scripts resolve provenance links for an Ollama model family and need the same three
things: a readme-link classifier, an LLM disambiguation call, and an HTTP fetch that prefers
`hrequests` (real browser fingerprint, bypasses the TLS/JA3 bot-detection a plain socket GET can
trip) and falls back to stdlib `urllib` when `hrequests` isn't installed. Living here instead of
in one script or the other avoids a circular import between the two.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_header import USER_AGENT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MODELINDEX_DATA", str(ROOT / "data")))
DEFAULT_LLM_MODEL = "~z-ai/glm-flash-latest"
CONTENT_MAX = 100_000  # matches crawl.py's README_MAX cap on the Ollama library readme

# A trailing quant token on an Ollama tag -- e.g. "-q4_K_M", "-fp16". `{0,2}` (not `?`) because
# K-quants need two trailing segments ("_K" then "_M"/"_L"/"_S"): "q4_K_M" is q4 + "_K" + "_M".
# An earlier single-segment version of this regex silently failed to strip any K-quant at all.
QUANT_SUFFIX = re.compile(
    r"(?:^|[-_:])(?:f16|fp16|bf16|q\d(?:_[a-z0-9]+){0,2}|iq\d(?:_[a-z0-9]+)?|mxfp\d)$",
    re.IGNORECASE,
)


_SIZE_TOKEN = re.compile(r"\d+(?:\.\d+)?x?\d*[bB]\b")


def size_token(tag_suffix: str) -> str | None:
    """A size-shaped token from a tag/release suffix (`13b`, `8x7b`, `1.5b`), lowercased -- used
    to disambiguate between candidate repos, not to compute a real parameter count.
    """
    m = _SIZE_TOKEN.search(tag_suffix or "")
    return m.group(0).lower() if m else None


def pick_candidate_for_release(release_tag_suffix: str, candidates: list[str]) -> tuple[str | None, bool]:
    """Deterministic size-token match against candidate repo *names*. Returns (repo, ambiguous)
    -- ambiguous=True means the caller should fall back to an LLM pick (or leave unresolved).
    Shared by scripts/hf_source.py (readme/sibling candidates) and scripts/links.py (homepage/
    GitHub-scan candidates) -- same matching problem, two different candidate sources.
    """
    if not candidates:
        return None, False
    size = size_token(release_tag_suffix)
    if len(candidates) == 1:
        candidate = candidates[0]
        # A lone candidate is only a free pass when there's no size to check, or it matches --
        # a single candidate scraped from arbitrary readme/changelog text (an old preview
        # release mentioned in passing, say) is not automatically "the" answer for every size in
        # the family just because nothing else turned up. Confirmed live: llava's github readme
        # mentions exactly one HF link in its changelog section (LLaVA-Lightning-MPT-7B-preview)
        # that isn't the actual repo for any of llava's real 7b/13b/34b releases -- accepting a
        # lone mismatched-size candidate stamped it onto llava:13b as a false match.
        if not size or size in candidate.split("/", 1)[-1].lower():
            return candidate, False
        return None, True
    if size:
        matches = [c for c in candidates if size in c.split("/", 1)[-1].lower()]
        if len(matches) == 1:
            return matches[0], False
    return None, True


def release_key(family: str, tag: str) -> str:
    """`family:tag` with any trailing quant token stripped.

    Two tags that differ only by quant (`13b-v1.5-fp16` vs `13b-v1.5-q5_K_M`) share one release;
    two tags that differ by base model or size (`13b-llama2-q4_0` vs `13b-q4_0`, or `7b` vs
    `13b`) do not -- this is deliberately a pure string operation (no numeric size parsing, no
    dependency on scripts/quality.py's separate and looser `_SIZE_B` regex) so it never has to
    guess at a MoE tag's total parameter count.
    """
    stripped = QUANT_SUFFIX.sub("", tag or "") or tag
    return f"{family}:{stripped}"

try:
    import hrequests
    HAVE_HREQUESTS = True
except ImportError:
    HAVE_HREQUESTS = False


# --------------------------------------------------------------------------------- basic utils
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


def atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically, safe under concurrent writers hitting the same path.

    A per-call unique suffix (not a fixed `.tmp` name) avoids two ThreadPoolExecutor workers
    racing to write the same cache entry (e.g. two families citing the same GitHub repo).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------- readme link classifier
_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)", re.I)
_LABEL_PAPER = re.compile(r"paper|arxiv|preprint|technical report", re.I)
_LABEL_HOMEPAGE = re.compile(r"website|homepage|home page|project page|\bblog\b|project site", re.I)
_ARXIV_LINK = re.compile(r"https?://arxiv\.org/abs/[0-9.]+v?\d*", re.I)
_HF_BAD_PREFIX = ("datasets/", "spaces/", "papers/", "blog/", "learn/", "collections/")
_GITHUB_BAD_PATH = ("issues", "pull", "blob", "tree", "wiki", "actions", "releases", "discussions", "commit", "compare")
_GITHUB_BAD_ORG = ("orgs", "sponsors", "marketplace", "topics", "features", "about", "pricing")


_REPO_SEGMENT = re.compile(r"[A-Za-z0-9_.\-]+")


def _sanitize_repo_segment(part: str) -> str:
    """Truncate at the first character that can't appear in an org/repo name -- a loosely
    written source regex (`\\S+`, embedded-HTML scans) can pull in a trailing quote/angle
    bracket from surrounding markup (`<a href="...naveraven-v2-13b">`); without this, that
    stray character rides along into a cache filename and breaks on Windows (illegal char).
    """
    m = _REPO_SEGMENT.match(part)
    return m.group(0) if m else ""


def clean_hf_repo(url: str) -> str | None:
    path = url.split("huggingface.co/", 1)[-1].split("?")[0].split("#")[0]
    if any(path.startswith(p) for p in _HF_BAD_PREFIX):
        return None
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return None  # org page only, no repo
    for stop in ("blob", "resolve", "tree", "commit", "discussions"):
        if stop in parts:
            parts = parts[: parts.index(stop)]
    if len(parts) < 2:
        return None
    org, name = _sanitize_repo_segment(parts[0]), _sanitize_repo_segment(parts[1])
    if not org or not name:
        return None
    return f"{org}/{name}"


def clean_github_repo(url: str) -> str | None:
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
    org, name = _sanitize_repo_segment(parts[0]), _sanitize_repo_segment(parts[1])
    if not org or not name:
        return None
    return f"{org}/{name}"


def classify_readme_links(readme: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (huggingface repos, paper urls, homepage urls, github repos) found in the readme."""
    hf, paper, homepage, github = [], [], [], []
    for label, url in _MD_LINK.findall(readme or ""):
        low = label.lower()
        if "huggingface.co" in url.lower():
            repo = clean_hf_repo(url)
            if repo:
                hf.append(repo)
        elif "github.com" in url.lower():
            repo = clean_github_repo(url)
            if repo:
                github.append(repo)
        elif _LABEL_PAPER.search(low) or "arxiv.org" in url.lower():
            paper.append(url)
        elif _LABEL_HOMEPAGE.search(low):
            homepage.append(url)
    return hf, paper, homepage, github


def find_hf_repos_in_links(links: list[str]) -> list[str]:
    seen, out = set(), []
    for url in links:
        repo = clean_hf_repo(url) if "huggingface.co" in url.lower() else None
        if repo and repo not in seen:
            seen.add(repo)
            out.append(repo)
    return out


def find_github_repos_in_links(links: list[str]) -> list[str]:
    seen, out = set(), []
    for url in links:
        repo = clean_github_repo(url) if "github.com" in url.lower() else None
        if repo and repo not in seen:
            seen.add(repo)
            out.append(repo)
    return out


# -------------------------------------------------------------------------------------- LLM pick
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


def llm_pick_repo(
    family: str, description: str, candidates: list[dict], model: str, release_hint: str | None = None,
) -> str | None:
    """Ask the LLM which candidate repo (if any) is the correct match for this model family.

    candidates: [{"repo": "org/name", "source_url": "...", "snippet": "..."}, ...]
    `release_hint` is the specific release's tag suffix (e.g. "13b-v1.5") when the pick is for
    one release within a family that has several (not every candidate necessarily matches --
    without this, the LLM has no way to know it should decline a same-family-wrong-size
    candidate rather than pick one anyway).
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
    release_line = f'Specific release/size to match: "{release_hint}"\n' if release_hint else ""
    prompt = (
        f'Ollama model family: "{family}"\n'
        f"{release_line}"
        f"Description: {description or '(none)'}\n\n"
        f"Candidate Hugging Face repositories found while researching this model:\n{listing}\n\n"
        "Which one, if any, is the correct upstream Hugging Face repository for this exact "
        + (f"release/size ({release_hint})? " if release_hint else "model? ")
        + "Reply with ONLY the repo in org/repo format, or reply NONE if none of them match "
        + ("this specific release/size." if release_hint else "the model.")
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


def llm_pick_url(family: str, description: str, kind: str, candidates: list[str], model: str) -> str | None:
    """Same idea as llm_pick_repo but for a flat list of candidate URLs/repo-strings (github
    repos, homepage URLs). A single candidate is accepted without spending an LLM call --
    disambiguation only matters once there's more than one to choose between.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL")
    if not api_key or not base_url:
        return candidates[0]  # best-effort: first readme mention wins without an LLM available
    listing = "\n".join(f"- {c}" for c in candidates)
    prompt = (
        f'Ollama model family: "{family}"\n'
        f"Description: {description or '(none)'}\n\n"
        f"Candidate {kind} links found in this family's readme:\n{listing}\n\n"
        f"Which one, if any, is the correct {kind} for this exact model? "
        "Reply with ONLY that link (verbatim), or reply NONE if none of them match."
    )
    try:
        reply = openrouter_chat(prompt, model, api_key, base_url)
    except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError, json.JSONDecodeError):
        return candidates[0]
    reply = reply.strip()
    if reply.upper().startswith("NONE"):
        return None
    return reply if reply in candidates else candidates[0]


# --------------------------------------------------------------------------------------- fetching
def http_get(url: str, headers: dict | None = None, timeout: int = 20) -> tuple[int, str] | None:
    """GET url, return (status_code, text), or None if the request could not be made at all.

    hrequests first -- a real browser-fingerprint client, needed for pages behind TLS/JA3 bot
    walls. Falls back to stdlib urllib only when hrequests isn't installed; a URL hrequests
    genuinely fails to reach (network error, non-2xx from a real attempt) is not retried via
    urllib, since urllib would hit the same wall for the same reason.
    """
    headers = dict(headers or {})
    if HAVE_HREQUESTS:
        try:
            resp = hrequests.get(url, headers=headers, timeout=timeout)
            return resp.status_code, (resp.text or "")
        except Exception:
            return None
    headers.setdefault("User-Agent", USER_AGENT)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return e.code, body
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def fetch_hf_readme(repo: str, cache_dir: Path, allow_network: bool = True) -> str | None:
    """The model card text at huggingface.co/<repo>/raw/<branch>/README.md -- a plain CDN file,
    not the rendered (client-side-only) model page, so no SPA problem and no Exa needed.
    """
    cache_path = cache_dir / f"{repo.replace('/', '__')}.json"
    if cache_path.exists():
        return read_json(cache_path).get("text")
    if not allow_network:
        return None
    text = None
    for branch in ("main", "master"):
        result = http_get(f"https://huggingface.co/{repo}/raw/{branch}/README.md")
        if result and result[0] == 200 and result[1].strip():
            text = result[1][:CONTENT_MAX]
            break
    atomic_write_json(cache_path, {"text": text})
    return text


def fetch_github_readme(repo: str, cache_dir: Path, allow_network: bool = True) -> str | None:
    """The repo's README via the GitHub API, which resolves filename/casing automatically
    (README.md, Readme.rst, no extension, ...) -- a raw `.../HEAD/README.md` guess would not.
    An optional GITHUB_TOKEN (read via load_dotenv(), like OPENROUTER_API_KEY) lifts the rate
    limit from 60/hr to 5000/hr.
    """
    cache_path = cache_dir / f"{repo.replace('/', '__')}.json"
    if cache_path.exists():
        return read_json(cache_path).get("text")
    if not allow_network:
        return None
    headers = {"Accept": "application/vnd.github.raw+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    result = http_get(f"https://api.github.com/repos/{repo}/readme", headers=headers)
    text = result[1][:CONTENT_MAX] if result and result[0] == 200 and result[1].strip() else None
    atomic_write_json(cache_path, {"text": text})
    return text


# --------------------------------------------------------------------------- HTML text/link mining
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
_HREF_RE = re.compile(r'href=[\'"]([^\'" >]+)', re.I)


def html_to_text(page_html: str) -> str:
    """Crude but dependency-free HTML->text: strip script/style blocks and tags, unescape
    entities, collapse whitespace. Good enough for readme-style content extraction; this is not
    a rendering engine, so client-rendered (JS-only) pages still come back mostly empty.
    """
    text = _SCRIPT_STYLE_RE.sub(" ", page_html or "")
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def extract_links(page_html: str, base_url: str) -> list[str]:
    """Every `href` on the page, resolved to an absolute URL, de-duplicated, order preserved."""
    return list(dict.fromkeys(urljoin(base_url, href) for href in _HREF_RE.findall(page_html or "")))
