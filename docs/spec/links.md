# Provenance links

Companion to [implementation-plan.md](implementation-plan.md). Ollama's registry carries no
provenance field back to a model's upstream project — not the Hugging Face repo it was
converted from, not its GitHub repo, not its paper, not even its own homepage. Not in the
manifest, not in the GGUF header's `general.*` metadata, not on the `ollama.com/library` page's
markup. This doc covers how MODELINDEX recovers all four of those anyway (and, for three of
them, saves the page content too), and how well it works.

This supersedes the earlier `hf-linking.md`, which covered only the Hugging Face link. Homepage,
GitHub and paper links were always *discoverable* along the way (`scripts/hf_source.py`'s old
tier 2b fetched them purely to hunt for more Hugging Face links, then discarded them) — this doc
describes making all four first-class, plus saving page content for everything but paper.

## Data model

Each row in `prod/data/models.json` (one per unique weight `digest`, see
`build_prod_data.deduplicate_models`) carries:

```json
"links": {
  "hf": {
    "repo": "ibm-granite/granite-4.1-8b-GGUF",
    "url": "https://huggingface.co/ibm-granite/granite-4.1-8b-GGUF",
    "confidence": "verified"
  },
  "github": {
    "repo": "ibm-granite/granite",
    "url": "https://github.com/ibm-granite/granite",
    "confidence": "likely",
    "method": "readme"
  },
  "homepage": null,
  "paper": {
    "url": "https://arxiv.org/abs/2408.12894",
    "confidence": "likely",
    "method": "readme"
  }
}
```

Any of the four can be `null`. `hf` is the only one that can reach `confidence: "verified"` —
byte-identity is inherently per-file (quant-specific); the other three have no finer grain than
"this family's readme links here," so they're always `"likely"`.

`method` names how a `likely` link was found:

| method | meaning |
|---|---|
| `readme` | a single readme-linked candidate, accepted directly |
| `readme_verified` | (hf only) an LLM confirmed the readme-linked HF repo's own README describes this model |
| `homepage_llm` | (hf only) found scanning the family's homepage's outbound links, LLM-picked |
| `github_llm` | (hf only) found scanning the family's GitHub README text, LLM-picked |

Page **content** (not just the link) is saved for `hf`, `github` and `homepage` — never `paper`
— at `prod/data/link_content.json`, family-keyed (every quant/tag of a family shares one
project, so there's nothing to gain fetching per-row):

```json
"families": {
  "granite": {
    "hf": {"url": "https://huggingface.co/ibm-granite/granite-4.1-8b-GGUF", "content": "---\nlicense: apache-2.0\n---\n# Granite..."},
    "github": {"url": "https://github.com/ibm-granite/granite", "content": "# Granite\n\n..."}
  }
}
```

## Where each link comes from

Every link type starts from the same source: `scripts/link_common.py:classify_readme_links()`
splits every markdown link in the family's `ollama.com/library/<family>` readme (already crawled
into `library.json` by `scripts/crawl.py`) into four buckets — `huggingface` (labeled "Hugging
Face"/"HF", or an `org/repo`-shaped link), `github` (any `github.com` URL), `paper` (labels
matching `paper|arxiv|preprint|technical report`, or an `arxiv.org` URL), `homepage` (labels
matching `website|homepage|project page|blog`).

### `hf` — `scripts/hf_source.py`, tiers 1–2 (unchanged from before)

1. **Tier 1, verified**: Ollama's `digest` on every manifest layer is `sha256:<hex>` of the raw
   GGUF blob. [modelindex.dev](https://modelindex.dev) is a public reverse hash index (`GET
   /api/v1/hash/sha256:<hex>`) built from crawling HF/ModelScope/CivitAI/Ollama; a hit with
   `source == "hf"` means the file is byte-identical to a real upload there — not a guess.
   Coverage is bounded by what modelindex.dev has crawled (~3–5% of digests, two independent
   300-sample pilots) and by GGUF quantization reproducibility (K-quants depend on a
   calibration step that differs by tool, so only non-calibrated formats or publisher-uploaded
   GGUFs match reliably).
2. **Tier 2, likely**: for families with no hash hit, `extract_readme_repo()` mines the readme
   for a linked `huggingface.co/<org>/<repo>` URL (prefer a "Hugging Face"-labeled link, else the
   first repo-shaped link).
3. **Tier 2 verification, `--verify-tier2`**: fetches the candidate repo's real README —
   `link_common.fetch_hf_readme()`, a plain GET of `huggingface.co/<repo>/raw/<branch>/README.md`
   — and asks an LLM (OpenRouter) whether it actually describes this Ollama family. Confirmed →
   `readme_verified`; rejected → the family falls through to `scripts/links.py` in the same
   pipeline run instead of shipping a match already known to be wrong.

   This step previously fetched the HF page via Exa (huggingface.co's rendered page is a
   client-only SPA, so a plain GET saw an empty shell). Fetching `/raw/<branch>/README.md`
   instead sidesteps the SPA entirely — it's a static file, not the rendered app — which also
   means this step no longer needs `EXA_API_KEY`. The tradeoff: a small number of real
   repos are access-gated (401 without a login) and return no README this way; those keep
   whatever tier 2 already found, unverified.

### `github` / `homepage` / `paper` — `scripts/links.py` (new)

For every family (not just ones missing an `hf` link — a project can have a GitHub repo
regardless of whether its HF link resolved):

- **github / homepage**: the readme's candidates for that bucket go to
  `link_common.llm_pick_url()` — one candidate is accepted directly (`method: "readme"`, no LLM
  spent); more than one goes to an LLM to disambiguate.
- **paper**: an `arxiv.org` candidate wins if there is one, else the first candidate. Never an
  LLM call — low enough stakes not to spend one, and rarely ambiguous.

### `hf` fallback — also `scripts/links.py`, when tiers 1–2 found nothing

This replaces the old tier 2b (`resolve_via_pages`), which fetched a family's homepage/paper/
GitHub purely to scan for an HF link and threw the fetch away afterward. Now that content is
being fetched for its own sake (next section), the same fetch doubles as one more chance at
finding `hf`: `scripts/links.py` scans the GitHub README text and the homepage's outbound links
for a `huggingface.co/<org>/<repo>` link, and if it finds any, sends the candidates to the LLM
for a final pick (`link_common.llm_pick_repo()`, same prompt shape as tier 2 verification) —
`method: "github_llm"` or `"homepage_llm"` depending on where it came from. This is strictly
weaker evidence than tiers 1–2 (an LLM guess off a secondary page, not the model's own readme or
a hash match), so it never overrides an existing `hf` link — `scripts/build_prod_data.py`'s
`resolve_links()` only falls back to it when `hf_sources.json` has nothing for that family.

## Content fetching

Content is fetched via `hrequests` (a real browser-fingerprint HTTP client — bypasses the
TLS/JA3 bot-detection a plain socket GET can trip), with a bare `urllib` fallback when
`hrequests` isn't installed:

- **`hf`**: `huggingface.co/<repo>/raw/<branch>/README.md` (tries `main` then `master`) — a
  plain CDN file, not the client-rendered page. No LLM, no Exa.
- **`github`**: `api.github.com/repos/<repo>/readme` with `Accept:
  application/vnd.github.raw+json`, which resolves filename/casing automatically
  (`README.md`, `Readme.rst`, no extension, ...) — a raw `.../HEAD/README.md` guess wouldn't. An
  optional `GITHUB_TOKEN` (read via `link_common.load_dotenv()`, same as `OPENROUTER_API_KEY`)
  lifts the API's rate limit from 60/hr to 5000/hr.
- **`homepage`**: arbitrary sites vary too much for a fixed endpoint. A direct GET (hrequests, or
  a small regex-based `<a href>` + text extractor in `link_common.py` — no `bs4`/`readability`
  dependency) covers most of them; when that comes back with genuinely zero outbound links (not
  an error — a *successful* fetch with zero anchors reliably means a client-rendered shell, not a
  page that truly has none), Exa's `/contents` API (`EXA_API_KEY`) is a last-resort fallback for
  the JS-only remainder. Exa is the only paid/API-metered step left in the whole pipeline.
- **`paper`**: link only, by design — never fetched.

Every fetch is cached forever, keyed by repo (`hf`/`github`) or URL (`homepage`), under
`data/cache/{hf_readme,github_readme,links/homepage}/` (gitignored, rebuildable). Content itself
is capped at 100,000 characters (`CONTENT_MAX`, matching `crawl.py`'s `README_MAX` cap on the
Ollama readme) and saved once per family at **`data/link_content/<family>.json`** — not
gitignored, same rationale as the pre-existing `data/hf_links/` audit trail: this took real
network/API time to produce and isn't cheaply rebuildable, so it's a durable record, not a
cache.

## Pipeline

```
prod/data/models.json (digest, model), data/out/library.json (readme per family)
  -> scripts/hf_source.py --refresh
       tier 1: GET modelindex.dev/api/v1/hash/sha256:<hex> per digest
       cache -> data/cache/hf_source/<hex>.json
     -> "verified" hits, per digest

  -> scripts/hf_source.py --verify-tier2
       tier 2: extract_readme_repo(library readme) -> a candidate repo
       verify: link_common.fetch_hf_readme(candidate) -> real README text -> ask the LLM
         confirmed=True  -> "likely" / method=readme_verified
         confirmed=False -> dropped; family added to the unresolved list for scripts/links.py
         confirmed=None  -> falls through to plain "likely" / method=readme
     -> data/out/hf_sources.json   {by_digest, by_family, unresolved_families}   (gitignored)

  -> scripts/links.py --resolve
       classify_readme_links() -> github / homepage / paper candidates (llm_pick_url if >1)
       fetch content: github (GitHub API), homepage (hrequests/Exa), and hf (whatever
         hf_source.py already resolved, OR a fallback pick from github/homepage content
         when hf_sources.json has nothing for this family)
       cache -> data/cache/{hf_readme,github_readme,links/homepage,links/resolve}/*.json (gitignored)
     -> data/link_content/<family>.json          (NOT gitignored -- durable)
     -> data/hf_links/homepage_exa_results.json   (NOT gitignored -- Exa fallback audit trail)
     -> data/out/links.json   {by_family: {hf, github, homepage, paper}}   (gitignored)

  -> scripts/build_prod_data.py
       resolve_links(digest, family, hf_sources, links) -> "links" field (verified digest beats
         likely family beats links.py's fallback, per link type)
       compact_link_content(data/link_content/*.json) -> "families" map, no "paper" key
     -> prod/data/models.json    (schema_version 2, "links" field)
     -> prod/data/link_content.json

  -> web/app.js
       sourceIconsHtml(row) renders an Ollama icon (always) + hf/github/homepage/paper icons
         for whichever links.<kind> is set
       technicalHtml(row) adds a collapsible <details> panel per available content type
         (Hugging Face model card, GitHub readme, Homepage) alongside the existing Ollama readme
```

Caching follows the shape established by `hf_source.py`: a real hit is cached forever; a `null`
result is cached too (a family that's genuinely checked-and-empty shouldn't be re-fetched and
re-asked on every rerun) except tier 1's `404 HASH_UNKNOWN` miss, which is retried after 14 days
since modelindex.dev's own crawl keeps growing. `scripts/links.py --resolve` fills in whatever
isn't cached yet; a plain `python scripts/links.py` (no `--resolve`) only rebuilds
`data/out/links.json` from what's already cached, touching no network.

## Known limitations

- `hf` coverage is bounded by modelindex.dev's crawl breadth (tier 1) and by what a readme names
  or links to at all (tiers 2/fallback) — neither is auditable beyond spot checks.
- Some real, correct Hugging Face repos are access-gated and return 401 on the raw README
  fetch without a login; those keep whichever earlier tier already resolved them (unverified if
  tier 2 alone), with no content saved.
- `homepage` content mining uses a small regex-based HTML→text/link extractor, not a real DOM
  parser — good enough for readme-style content, but a client-rendered (JS-only) page still
  comes back mostly empty from a direct fetch; Exa's crawl doesn't cover everything either, so
  some homepages remain genuinely unresolved for content even when the link itself was found.
- `github`/`homepage`/`paper` never reach `confidence: "verified"` — there's no hash-match
  equivalent for them the way there is for `hf`.
- No cross-registry linking beyond Hugging Face (modelindex.dev also indexes ModelScope and
  CivitAI; tier 1 only keeps `source == "hf"` matches).

## How to run

```bash
# hf tier 1: refresh the hash cache for every digest not already cached (~6,280 requests, ~15 min)
python scripts/hf_source.py --refresh --workers 6

# hf tier 2 verification (needs OPENROUTER_API_KEY + OPENROUTER_BASE_URL)
python scripts/hf_source.py --verify-tier2 --llm-workers 4

# github/homepage/paper + content + hf fallback for everything else (needs OPENROUTER_*;
# optionally EXA_API_KEY, GITHUB_TOKEN)
python scripts/links.py --resolve --workers 4

# pilot either step on a sample first
python scripts/hf_source.py --verify-tier2 --llm-limit 20
python scripts/links.py --resolve --limit 20

# fold everything into prod/data/
python scripts/build_prod_data.py
```
