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

**Release-scoping (`hf` only).** A single Ollama "family" (`ollama.com/library/<family>`) can
bundle genuinely different upstream releases at different sizes — verified live: `llava:7b`,
`llava:13b` and `llava:34b` are different Vicuna checkpoints with different Hugging Face repos,
not the same file at different quants, but Ollama serves exactly one readme per family
(confirmed byte-identical between `llava:7b` and `llava:13b`'s pages). Stamping one family-wide
"likely" `hf` pick onto every size in the family was a real bug — `by_family["llava"]` pointed at
the 7B repo, applied to the 13B and 34B tags too. `github`/`homepage`/`paper` don't have this
problem (a project has one repo/site/paper regardless of size), so only `hf` is release-scoped;
see "Release scoping" below for the mechanism.

## Data model

Each row in `prod/data/models.json` (one per unique weight `digest`, see
`build_prod_data.deduplicate_models`) carries a `release` field (family + tag with the quant
suffix stripped, `link_common.release_key()`) and a `links` object:

```json
"release": "llava:13b",
"links": {
  "hf": {
    "release": {
      "repo": "liuhaotian/llava-v1.5-13b",
      "url": "https://huggingface.co/liuhaotian/llava-v1.5-13b",
      "confidence": "likely",
      "method": "readme_verified"
    },
    "family": {
      "repo": "liuhaotian/llava-v1.5-7b",
      "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b",
      "confidence": "likely",
      "method": "readme_verified"
    }
  },
  "github": {
    "repo": "haotian-liu/LLaVA",
    "url": "https://github.com/haotian-liu/LLaVA",
    "confidence": "likely",
    "method": "readme"
  },
  "homepage": null,
  "paper": null
}
```

`hf` always carries both an envelope: `release` (the precise pick for this exact release, when
one was found) and `family` (the family-wide guess, kept alongside it rather than dropped, even
when it differs — a caller prefers `release` and falls back to `family`, labeled as not
size-confirmed). `github`/`homepage`/`paper` stay flat objects, `null` when nothing was found. A
digest-level hash match (`confidence: "verified"`) is release-exact by construction (one digest
is one file), so it fills both the `release` and `family` slots identically.

`method` names how a `likely` link was found:

| method | meaning |
|---|---|
| `readme` | a single candidate, accepted directly (no LLM spent) |
| `readme_verified` | (hf only) an LLM confirmed the picked repo's own README describes this model/release |
| `homepage_llm` | (hf only) found scanning the family's homepage's outbound links, LLM-picked |
| `github_llm` | (hf only) found scanning the family's GitHub README text (or a bare `org/repo` token in it), LLM-picked |
| `brave_search` | (hf only) found via a Brave web search, not yet/couldn't be LLM-verified |
| `brave_search_verified` | (hf only) found via a Brave web search, LLM-confirmed against the repo's real README |

Page **content** (not just the link) is saved for `hf`, `github` and `homepage` — never `paper`
— at `prod/data/link_content.json`, family-keyed. `github`/`homepage` are flat (one project per
family); `hf` is keyed by **repo string**, not release — multiple releases in a family
legitimately share one repo (every quant of `llava:13b-v1.5-*` shares one), and content-fetching
is already cached per-repo:

```json
"families": {
  "llava": {
    "github": {"url": "https://github.com/haotian-liu/LLaVA", "content": "# LLaVA\n\n..."},
    "hf": {
      "liuhaotian/llava-v1.5-7b":  {"url": "https://huggingface.co/liuhaotian/llava-v1.5-7b",  "content": "---\n..."},
      "liuhaotian/llava-v1.5-13b": {"url": "https://huggingface.co/liuhaotian/llava-v1.5-13b", "content": "---\n..."}
    }
  }
}
```

### The `release` key

`link_common.release_key(family, tag)` = `f"{family}:{tag}"` with any trailing quant token
stripped (`link_common.QUANT_SUFFIX`) — a pure string operation, no numeric size parsing, so it
never has to guess a MoE tag's total parameter count the way `scripts/quality.py`'s separate
`_SIZE_B` regex does (that regex misreads `8x7b` as size `7`; `release_key` never computes a
number at all).

| tag | release |
|---|---|
| `13b-v1.5-fp16` and `13b-v1.5-q5_K_M` | both `llava:13b-v1.5` — same checkpoint, differ only by quant |
| `wizardlm:13b-llama2-q4_0` | `wizardlm:13b-llama2` |
| `wizardlm:13b-q4_0` | `wizardlm:13b` — kept apart from the row above: same size, different base model |
| `qwen3:235b-a22b-instruct-2507-q4_K_M` | `qwen3:235b-a22b-instruct-2507` |
| `codellama:latest` | `codellama:latest` — no quant token, unchanged |

## Where each link comes from

Every link type starts from the same source: `scripts/link_common.py:classify_readme_links()`
splits every markdown link in the family's `ollama.com/library/<family>` readme (already crawled
into `library.json` by `scripts/crawl.py`) into four buckets — `huggingface` (labeled "Hugging
Face"/"HF", or an `org/repo`-shaped link), `github` (any `github.com` URL), `paper` (labels
matching `paper|arxiv|preprint|technical report`, or an `arxiv.org` URL), `homepage` (labels
matching `website|homepage|project page|blog`).

### `hf` — `scripts/hf_source.py`, tiers 1–2, release-scoped

1. **Tier 1, verified**: Ollama's `digest` on every manifest layer is `sha256:<hex>` of the raw
   GGUF blob. [modelindex.dev](https://modelindex.dev) is a public reverse hash index (`GET
   /api/v1/hash/sha256:<hex>`) built from crawling HF/ModelScope/CivitAI/Ollama; a hit with
   `source == "hf"` means the file is byte-identical to a real upload there — not a guess.
   Already release-exact (one digest is one file), so it never needed the fix below. Coverage is
   bounded by what modelindex.dev has crawled (~3–5% of digests, two independent 300-sample
   pilots) and by GGUF quantization reproducibility (K-quants depend on a calibration step that
   differs by tool, so only non-calibrated formats or publisher-uploaded GGUFs match reliably).
2. **Tier 2, likely, per release**: for releases with no hash hit,
   `extract_readme_candidates()` mines the readme for **every** linked
   `huggingface.co/<org>/<repo>` URL (not just the first — a family's readme very often names
   only the top-of-family checkpoint, so the smaller/larger siblings, if named at all, are
   further down the same link list). `sibling_candidates()` adds a second source: bare
   `org/repo` tokens (no URL prefix — e.g. a `--model-path org/model-13b` inside a shell
   snippet) found in whatever github/homepage/hf content `scripts/links.py` has *already*
   fetched for the family, filtered to tokens whose repo *name* contains the family slug
   (matching by name, not org, because a picked repo's own README often links siblings under a
   different org than the one already picked — confirmed live for `wizardlm`: the picked
   `TheBloke`-org repo's README links `WizardLM`-org siblings). The two sources are unioned, not
   one-a-fallback-for-the-other — an earlier version of this only ran the sibling scan when the
   readme found *nothing*, which meant a family whose readme names one repo (like `llava`) never
   saw its siblings at all.

   For each release, a size token pulled from its tag (`13b`, `8x7b`, `1.5b`,
   `link_common.size_token()`) is matched against candidate repo *names*. Exactly one match →
   accepted deterministically (`method: "readme"`, no LLM spent). Zero or multiple matches → an
   LLM call (`link_common.llm_pick_repo()`), cached and **deduped by (family, size token)** so
   releases sharing a size (e.g. `qwen3:4b-instruct-2507` and `qwen3:4b-thinking-2507`) don't
   repeat the call.
3. **Tier 2 verification, `--verify-tier2`**: fetches a picked repo's real README —
   `link_common.fetch_hf_readme()`, a plain GET of `huggingface.co/<repo>/raw/<branch>/README.md`
   — and asks an LLM (OpenRouter) whether it actually describes this model. Verified **once per
   distinct (family, repo) pair**, not once per release — many releases in a family pick the same
   repo, so this stays cheap even though picking is now release-scoped. Confirmed →
   `readme_verified`; rejected → that release falls through to `scripts/links.py` in the same
   pipeline run instead of shipping a match already known to be wrong.

   This step previously fetched the HF page via Exa (huggingface.co's rendered page is a
   client-only SPA, so a plain GET saw an empty shell). Fetching `/raw/<branch>/README.md`
   instead sidesteps the SPA entirely — it's a static file, not the rendered app — which also
   means this step no longer needs `EXA_API_KEY`. The tradeoff: a small number of real
   repos are access-gated (401 without a login) and return no README this way; those keep
   whatever tier 2 already found, unverified.

`by_family` (the pre-release-scoping "top candidate" pick) is kept alongside `by_release`,
unchanged in meaning — it's exactly the fallback value `build_prod_data.resolve_links()` uses
for the `family` slot when a specific release has nothing release-specific.

4. **Tier 3, `--brave-search`**: last resort, for families where tiers 1–2 found **nothing at
   all** (`extract_readme_candidates()` and `sibling_candidates()` both come up empty for that
   family — not merely "the candidate got rejected," see below). Queries Brave's web search API
   (`site:huggingface.co <family>`, the bare family slug, no query enrichment — precision is
   enforced downstream by `clean_hf_repo()` filtering and tier 2's verification step, not by the
   query) and turns the results into repo candidates the exact same way readme/sibling candidates
   are: `clean_hf_repo()` on each result URL, in Brave's own relevance-rank order, deduplicated.
   From there every candidate flows through the identical size-token-match /
   LLM-disambiguate / LLM-verify pipeline as tiers 1–2 — no separate code path. Opt-in
   (`--brave-search`, needs `BRAVE_SEARCH_API_KEY`) since it's a paid/rate-limited external API,
   same treatment `links.py` gives Exa. Cached by content-hash of the query
   (`data/cache/brave_search/`), negative-cached (a dead-end family costs one call ever, not one
   per run), with a durable family-keyed audit trail at `data/hf_links/brave_search_results.json`
   (own path/schema, not reusing the retired family-keyed `exa_results.json` shape).

   Method vocabulary: `brave_search` / `brave_search_verified` — kept distinct from `readme` /
   `readme_verified` because the pick didn't come from the family's own readme, and
   `web/app.js`'s tooltip text depends on that distinction being accurate.

   Measured on a full run (2026-09-10): 76 of the then-95 unresolved families resolved (240 →
   307 distinct (family, repo) verification pairs, 232 confirmed / 24 rejected), leaving 19
   families genuinely unresolvable even via web search. Spot-checked correct against known
   publishers (`deepseek-r1` → `deepseek-ai/DeepSeek-R1`, `codellama` →
   `codellama/CodeLlama-7b-hf`, `bge-large` → `BAAI/bge-large-en`).

   **What tier 3 does *not* do**: retroactively help a family whose tier-2 candidate got
   *rejected* by verification within the same run. `yi`'s readme names `01-ai/Yi-34B` (the
   correct publisher) — a non-empty candidate list, so tier 3 never fires for it even though
   verification went on to reject that specific pick as a family-level match. This is a real,
   narrow gap (a rejected family stays without an `hf` entry until a *later* run happens to
   re-confirm it, since Phase A recomputes the same non-empty candidate list every time) — not
   something this feature was scoped to close; it targets "no candidate found," not "the one
   candidate found didn't verify."

### `github` / `homepage` / `paper` — `scripts/links.py`, family-scoped (unchanged)

For every family (not just ones missing an `hf` link — a project can have a GitHub repo
regardless of whether its HF link resolved):

- **github / homepage**: the readme's candidates for that bucket go to
  `link_common.llm_pick_url()` — one candidate is accepted directly (`method: "readme"`, no LLM
  spent); more than one goes to an LLM to disambiguate.
- **paper**: an `arxiv.org` candidate wins if there is one, else the first candidate. Never an
  LLM call — low enough stakes not to spend one, and rarely ambiguous.

These three stay family-scoped deliberately — a project has one repo, one site, one paper
regardless of which size someone's asking about, unlike `hf`.

### `hf` fallback — also `scripts/links.py`, per release, when tiers 1–2 found nothing

This replaces the old tier 2b (`resolve_via_pages`), which fetched a family's homepage/GitHub
purely to scan for an HF link and threw the fetch away afterward. Now that content is being
fetched for its own sake (next section), the same fetch doubles as one more chance at finding
`hf` — but per release, not once for the whole family: `scripts/links.py` scans the GitHub
README text and the homepage's outbound links for `huggingface.co/<org>/<repo>` links, then
applies the same deterministic-size-token-first / LLM-if-ambiguous pick (cached and deduped by
size, same as tier 2) for each release tier 1/2 left unresolved — `method: "github_llm"` or
`"homepage_llm"` depending on where the candidate came from. This is strictly weaker evidence
than tiers 1–2 (an LLM guess off a secondary page, not the model's own readme or a hash match),
so it never overrides an existing per-release pick — `scripts/build_prod_data.py`'s
`resolve_links()` only falls back to it when `hf_sources.json`'s `by_release`/`by_digest` have
nothing for that specific release. Because this fallback is inherently release-scoped, it never
produces a family-wide guess — the `links.hf.family` slot only ever comes from `hf_source.py`'s
own `by_family`.

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
prod/data/models.json (digest, model, tag), data/out/library.json (readme per family)
  -> scripts/hf_source.py --refresh
       tier 1: GET modelindex.dev/api/v1/hash/sha256:<hex> per digest
       cache -> data/cache/hf_source/<hex>.json
     -> "verified" hits, per digest

  -> scripts/hf_source.py --verify-tier2 [--families f1,f2 for a pilot]
       tier 2, per release: extract_readme_candidates() + sibling_candidates(already-fetched
         content) -> candidate repos; size-token match if unambiguous, else LLM pick (cached,
         deduped by size)
       verify: once per distinct (family, repo) pick -> link_common.fetch_hf_readme(repo) ->
         real README text -> ask the LLM
         confirmed=True  -> "likely" / method=readme_verified
         confirmed=False -> dropped; release added to the unresolved list for scripts/links.py
         confirmed=None  -> falls through to plain "likely" / method=readme
  -> scripts/hf_source.py --brave-search --verify-tier2 [optional, needs BRAVE_SEARCH_API_KEY]
       tier 3, only for families where tier 2 found ZERO candidates (not "rejected", "none at
         all"): site:huggingface.co <family> -> clean_hf_repo() per result -> same size-token
         match / LLM-disambiguate / LLM-verify pipeline as tier 2
       cache -> data/cache/brave_search/<sha1(query)>.json (gitignored, negative-cached)
     -> data/hf_links/brave_search_results.json   (NOT gitignored -- durable audit, family-keyed)
     -> data/out/hf_sources.json
          {by_digest, by_family, by_release, unresolved_families, unresolved_releases} (gitignored)
        (a --families run merges into the existing file rather than replacing it wholesale --
        every family not named this run keeps its prior by_family/by_release entries untouched)

  -> scripts/links.py --resolve [--families f1,f2 / --limit N for a pilot]
       classify_readme_links() -> github / homepage / paper candidates (llm_pick_url if >1,
         family-scoped, unchanged)
       fetch content: github (GitHub API), homepage (hrequests/Exa) -- family-scoped, once each
       per release still unresolved after hf_source.py: scan that same github/homepage content
         for hf candidates -> size-token match if unambiguous, else LLM pick (cached, deduped)
       fetch+save hf content for every distinct repo the family ended up with (from hf_source.py
         or this fallback), keyed by repo
       cache -> data/cache/{hf_readme,github_readme,links/homepage,links/resolve}/*.json (gitignored)
     -> data/link_content/<family>.json          (NOT gitignored -- durable; hf keyed by repo)
     -> data/hf_links/homepage_exa_results.json   (NOT gitignored -- Exa fallback audit trail)
     -> data/out/links.json   {by_family: {github, homepage, paper, hf_by_release}}   (gitignored)
        (--families/--limit merge the same way as hf_source.py's --families)

  -> scripts/build_prod_data.py
       release_key(model, tag) -> "release" field, computed once per compacted row
       resolve_links(digest, family, release, hf_sources, links) -> "links" field:
         hf.release = digest hit, else by_release hit, else links.py's hf_by_release fallback
         hf.family  = digest hit, else by_family hit  (links.py's fallback is release-only,
                                                         never contributes a family-wide guess)
         github/homepage/paper: unchanged, family-scoped straight from links.json
       compact_link_content(data/link_content/*.json) -> "families" map, no "paper" key,
         hf sub-map passed through unchanged (still repo-keyed)
     -> prod/data/models.json    (schema_version 2, "release" + "links" fields)
     -> prod/data/link_content.json

  -> web/app.js
       resolvedHf(row) = links.hf.release || links.hf.family (family-only flagged in the tooltip)
       sourceIconsHtml(row) renders an Ollama icon (always) + hf/github/homepage/paper icons
       technicalHtml(row) adds a collapsible <details> panel per available content type
         (Hugging Face model card -- looked up by the resolved repo, GitHub readme, Homepage)
```

Caching follows the shape established by `hf_source.py`: a real hit is cached forever; a `null`
result is cached too (a family/release that's genuinely checked-and-empty shouldn't be
re-fetched and re-asked on every rerun) except tier 1's `404 HASH_UNKNOWN` miss, which is
retried after 14 days since modelindex.dev's own crawl keeps growing. `scripts/links.py
--resolve` fills in whatever isn't cached yet; a plain `python scripts/links.py` (no
`--resolve`) only rebuilds `data/out/links.json` from what's already cached, touching no
network. Both scripts' `--families` (and `links.py`'s pre-existing `--limit`) restrict *what
gets processed*, not what gets written — a partial run always merges into the existing output
file rather than replacing it, so piloting a few families never discards every other family's
already-resolved data.

## Known limitations

- `hf` coverage is bounded by modelindex.dev's crawl breadth (tier 1) and by what a readme (plus
  already-fetched sibling content) names or links to at all (tiers 2/fallback) — neither is
  auditable beyond spot checks.
- The size-token match is a substring check against candidate repo *names* — it works well
  because HF repo names commonly embed size (`llava-v1.5-13b`), but it's a naming convention,
  not a guarantee. Two candidates naming the same size under different orgs (confirmed live:
  `liuhaotian/llava-v1.5-7b` vs `llava-hf/llava-1.5-7b-hf`) are correctly treated as ambiguous
  and sent to the LLM rather than guessed between.
- A release whose readme names only the top-of-family checkpoint and whose family has no
  already-fetched github/homepage content yet (a brand-new family, first pipeline pass) won't
  see its siblings until a *second* `scripts/links.py` run has populated that content —
  `sibling_candidates()` reads it read-only, it doesn't fetch anything new itself.
- Some real, correct Hugging Face repos are access-gated and return 401 on the raw README
  fetch without a login; those keep whichever earlier tier already resolved them (unverified if
  tier 2 alone), with no content saved.
- `homepage` content mining uses a small regex-based HTML→text/link extractor, not a real DOM
  parser — good enough for readme-style content, but a client-rendered (JS-only) page still
  comes back mostly empty from a direct fetch; Exa's crawl doesn't cover everything either, so
  some homepages remain genuinely unresolved for content even when the link itself was found.
- `github`/`homepage`/`paper` never reach `confidence: "verified"` — there's no hash-match
  equivalent for them the way there is for `hf`, and they're deliberately family-scoped, not
  release-scoped.
- No cross-registry linking beyond Hugging Face (modelindex.dev also indexes ModelScope and
  CivitAI; tier 1 only keeps `source == "hf"` matches).
- Benchmark matching (`scripts/quality.py`) is a separate, not-yet-integrated system — it has its
  own looser size/variant matcher (`tag_size_b`, `match_eval`) that doesn't consume `release`.
  Aligning the two is a deliberate follow-up, not part of this design.
- Tier 3 (Brave search) only fires when tier 2 found zero candidates for a family, not when it
  found one that verification later rejected — a family whose sole readme-derived candidate is
  wrong (confirmed `yi`: readme names the correct publisher `01-ai/Yi-34B`, but the LLM rejected
  it as a family-level match) stays without an `hf` entry rather than getting a second attempt in
  the same run. A later run can still pick it up if verification happens to confirm it instead
  (LLM judgement on borderline cases isn't perfectly stable run to run), but there's no code path
  that automatically escalates a rejection to tier 3.
- `clean_hf_repo()`'s prefix denylist for Hugging Face's own site sections (`docs/`, `tasks/`,
  `chat/`, ...) is necessarily incomplete — it's grown from concrete false positives (a Brave
  search for `gemma2` ranked `huggingface.co/docs/transformers/...` above the actual model repo,
  and slipped through because a docs page has no raw README.md for tier 2's verification step to
  reject either), not from an exhaustive list of HF's site map.

## How to run

```bash
# hf tier 1: refresh the hash cache for every digest not already cached (~6,280 requests, ~15 min)
python scripts/hf_source.py --refresh --workers 6

# hf tier 2, release-scoped: readme/sibling candidates, size-token match, LLM disambiguation +
# verification where ambiguous (needs OPENROUTER_API_KEY + OPENROUTER_BASE_URL)
python scripts/hf_source.py --verify-tier2 --llm-workers 4

# hf tier 3: Brave web search for families tier 2 found nothing at all for (needs
# BRAVE_SEARCH_API_KEY, plus OPENROUTER_* for the same disambiguation/verification as tier 2)
python scripts/hf_source.py --brave-search --verify-tier2 --llm-workers 4

# github/homepage/paper + content + per-release hf fallback for everything else (needs
# OPENROUTER_*; optionally EXA_API_KEY, GITHUB_TOKEN)
python scripts/links.py --resolve --workers 4

# pilot either step on specific families first (merges into the existing output, doesn't
# replace it -- safe to run repeatedly against a growing family list)
python scripts/hf_source.py --verify-tier2 --families llava,wizardlm,codellama,qwen
python scripts/links.py --resolve --families llava,wizardlm,codellama,qwen

# fold everything into prod/data/
python scripts/build_prod_data.py
```
