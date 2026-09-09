# Hugging Face source linking

Companion to [implementation-plan.md](implementation-plan.md). Ollama's registry carries no
provenance field back to the upstream Hugging Face repo a GGUF was converted from — not in
the manifest, not in the GGUF header's `general.*` metadata, not on the `ollama.com/library`
page's markup. This doc covers how MODELINDEX recovers that link anyway, and how well it works.

## Data model

Each row in `prod/data/models.json` (one per unique weight `digest`, see
`build_prod_data.deduplicate_models`) carries:

```json
"hf_source": {
  "repo": "ibm-granite/granite-4.1-8b-GGUF",
  "url": "https://huggingface.co/ibm-granite/granite-4.1-8b-GGUF",
  "confidence": "verified"
}
```

or, for a `likely` match, with a `method` naming which tier found it:

```json
"hf_source": {
  "repo": "mistralai/Devstral-Small-2505",
  "url": "https://huggingface.co/mistralai/Devstral-Small-2505",
  "confidence": "likely",
  "method": "homepage_llm"
}
```

or `null` when nothing was found. `confidence` is one of:

| value | meaning | derived from |
|---|---|---|
| `verified` | the exact same file (byte-for-byte, same sha256) exists at that HF repo | reverse hash lookup |
| `likely` | some page names that repo for this model, but no file hash confirms it | readme text (`method: readme`), an LLM-confirmed readme link (`readme_verified`), or a homepage/paper/GitHub page + LLM (`homepage_llm`) |

`verified` always wins when more than one tier finds something for the same digest/family —
`likely` is a fallback for what the hash crawl didn't confirm, never a downgrade of a real
match (`scripts/hf_source.py:main`, the `verified_families` exclusion set). A `readme_verified`
candidate that the LLM *rejects* isn't kept at all — the family falls through to tier 2b in the
same run instead of shipping a match we already know is wrong.

## Tier 1 — verified, via modelindex.dev's reverse hash index

Ollama's `digest` on every manifest layer is already `sha256:<hex>` of the raw GGUF blob
(`scripts/crawl.py`). [modelindex.dev](https://modelindex.dev) is a public, unauthenticated,
read-only provenance index (`GET /api/v1/hash/sha256:<hex>` — full contract at
[modelindex.dev/llms.txt](https://modelindex.dev/llms.txt)) that has independently crawled and
hashed ~3M files across HuggingFace, ModelScope, CivitAI and Ollama. Querying it with our own
digest either returns nothing, or returns every `{source, org, name, path}` elsewhere in its
index carrying that exact byte content. A hit with `source == "hf"` is a zero-ambiguity match —
not a guess, the file is identical.

```
scripts/hf_source.py: fetch_hash(digest_hex)
  -> GET https://modelindex.dev/api/v1/hash/sha256:<hex>
  -> best_hf_match(): keep source=="hf" matches, prefer the shortest org/name
```

**This only ever catches a fraction of the catalog**, for two independent reasons:

1. **Coverage** — a hit requires modelindex.dev to have already crawled and hashed that
   specific upstream repo. It hasn't crawled everything on Hugging Face.
2. **Reproducibility** — most GGUF quantization is not deterministic across tools. F16/F32/Q8_0
   and the legacy Q4_0/Q5_0 formats are simple rounding, so any converter reproduces identical
   bytes. K-quants (Q4_K_M, Q5_K_S, ...) depend on an importance-matrix calibration step that
   differs by tool and calibration data, so two independently-produced "Q4_K_M" files of the
   same model are almost never byte-identical — *unless* Ollama's copy is a verbatim mirror of
   a publisher's own GGUF upload (e.g. `ibm-granite`, `Qwen` publish GGUF directly; Ollama's
   library entry is that same file), in which case every quant level matches, K-quants included.

## Tier 2 — likely, via readme link extraction

For families with no hash hit, `scripts/hf_source.py:extract_readme_repo` mines the family's
`ollama.com/library/<family>` readme (already crawled into `library.json` by `crawl.py`) for a
linked `huggingface.co/<org>/<repo>` URL:

1. Prefer a markdown link whose label contains "Hugging Face" / "HF".
2. Otherwise, the first link that resolves to an `org/repo` shape.
3. Drop `datasets/`, `spaces/`, `papers/`, `blog/`, `learn/`, `collections/` paths, and any link
   that only names an **org** page (`huggingface.co/codellama`, no repo) — those aren't
   resolvable to one specific model.
4. Strip `/blob/`, `/resolve/`, `/tree/`, `/commit/`, `/discussions/` suffixes back to `org/repo`.

This is a *family-level* signal (the readme is shared across every tag/quant in that family), so
one text match covers every row in the family — the readme names the base model, not any
specific GGUF artifact, hence `likely` rather than `verified`.

## Tier 2 verification — `--verify-tier2`, Exa + LLM

Tier 2's `extract_readme_repo` is a blind regex pick — it trusts whatever the readme's first
labeled link says with no check. `scripts/hf_source.py:verify_readme_candidate` adds a check:
fetch the candidate's Hugging Face page via Exa (see below — huggingface.co is a client-rendered
SPA, so a plain GET sees an empty shell, not the model card; Exa's crawl/cache has the rendered
content) and ask the LLM "does this page's title/text actually describe this Ollama family?"

`confirmed` is a tri-state, not a bool: `True`/`False` is a real judgement; `None` means
"not attempted" (missing `EXA_API_KEY`/`OPENROUTER_*`, or the fetch turned up nothing) and is
treated the same as never having run `--verify-tier2` at all — the candidate keeps its plain
`method: "readme"`. A confirmed candidate is promoted to `method: "readme_verified"`. A
*rejected* one is dropped entirely and the family falls through to tier 2b in the same run —
run against the full catalog, this caught 6 wrong picks out of 85 (`deepseek-coder-v2`,
`llava`, `mistral`, `openchat`, `samantha-mistral`, `wizardlm` — mostly cases where the readme's
first HF link was a related-but-different model, e.g. a base model instead of the actual
upstream of this specific fine-tune).

## Tier 2b — likely, via homepage/paper/GitHub + LLM, `--llm-extract`

For families with nothing yet (no tier 2 match, or a tier 2 candidate tier-2-verification just
rejected), `scripts/hf_source.py:resolve_via_pages` goes further out:

1. `classify_readme_links` splits every markdown link in the readme into four buckets:
   `huggingface` (tier 2's job), `paper` (labels matching `paper|arxiv|preprint|technical
   report`, or an `arxiv.org` URL), `homepage` (labels matching `website|homepage|project
   page|blog`), and `github` (any `github.com` URL, classified by host, not label).
2. Fetch the homepage + paper + GitHub links (whichever exist) and scan each page's outbound
   links for a `huggingface.co/<org>/<repo>` link. If nothing turns up on that first round, take
   one more hop: fetch an `arxiv.org` or GitHub link found *on* those pages that wasn't tried
   directly, and scan that too.
3. Whatever Hugging Face repos turn up (zero, one, or several — these pages link plenty of
   things that aren't the model, including on a busy GitHub org page) go to the LLM for a final
   pick, same shape as tier 2 verification: "which one, if any, is correct?"

This is strictly weaker evidence than tier 2 (further from Ollama's own readme, an LLM guess
instead of a direct link), so it never overrides an existing `likely` — only families with
nothing at all reach it (the `unresolved` list in `main()`, which tier-2-verification rejections
get added to during the same run).

**Fetching: hrequests first, Exa as fallback.** The original version of this tier used only Exa's
`/contents` API (`extras.links`) to read a page's outbound links. That turned out to be
unreliable on link-heavy pages: confirmed on `mistral.ai/news/devstral` that Exa's link picker
dropped the one `huggingface.co` link the page actually has (twice, in fact) even at
`links: 100` — it filled up on product-nav boilerplate first. `hrequests` (a real
browser-fingerprint HTTP client with proper DOM parsing, `pip install hrequests`,
https://github.com/daijro/hrequests) reads *every* anchor on a page via
`resp.html.absolute_links`, so it's the primary fetch now; Exa is only consulted for a URL when
hrequests can't get it at all — non-2xx, a network exception, the package not installed, or
(this took a live test to find) a *successful* fetch that comes back with **zero** anchors,
which turned out to mean "this is a client-rendered shell, not a page with genuinely no links"
(confirmed on a `notion.site` page: 200 OK, 0 anchors via a plain GET, but real content once
rendered). `fetch_links_cached` records which of the two actually served each URL.

The LLM call (`openrouter_chat`) hits `OPENROUTER_BASE_URL/chat/completions` with
`OPENROUTER_API_KEY` (read from the real environment, or from a `.env` file at the repo root via
`load_dotenv()` — a stdlib-only loader, no `python-dotenv` dependency). The default model is
`~z-ai/glm-flash-latest` (yes, with the leading `~` — that's the real OpenRouter model id, found
by listing `/api/v1/models` and grepping for `glm`), a reasoning model: `max_tokens` is set
generously (400) because reasoning tokens are billed against the same budget as the answer, and
a low cap silently truncates the actual answer to nothing (`content: null`) after the model
finishes "thinking." Every fetch and LLM pick is cached per family at
`data/cache/hf_llm_extract/<family>.json` regardless of outcome, so a family that genuinely has
nothing to find isn't re-fetched and re-asked on every run.

**Coverage**: of 109 families reaching this tier (103 with no readme link + 6 rejected by
tier-2 verification), 46 had a homepage/paper/GitHub link to try at all, and 14 resolved. Most
of the remainder are pages hrequests *and* Exa both come back empty on (a real coverage limit,
not a bug), plus a few genuinely ambiguous cases the LLM correctly declines rather than guesses
at — `bespoke-minicheck`'s GitHub link led to a busy `open-thoughts/open-thoughts` README listing
17 unrelated model repos, and the LLM returned `NONE` rather than pick one at random.

## Pipeline

```
prod/data/models.json (digest, model)
  -> scripts/hf_source.py --refresh
       for each unique digest:
         GET modelindex.dev/api/v1/hash/sha256:<hex>
         cache -> data/cache/hf_source/<hex>.json   (gitignored, like data/cache/config, data/cache/headers)
     -> "verified" hits, per digest

  -> scripts/hf_source.py --verify-tier2   (readme-derived candidates from families with no verified hit)
       extract_readme_repo(library.json[family].readme) -> a candidate repo
       verify_readme_candidate(): Exa-fetch the HF page, ask the LLM to confirm
         confirmed=True  -> "likely" / method=readme_verified
         confirmed=False -> dropped; family added to the unresolved list for --llm-extract
         confirmed=None  -> falls through to plain "likely" / method=readme

  -> scripts/hf_source.py --llm-extract   (families with nothing yet)
       classify_readme_links() -> homepage / paper / github candidates
       resolve_via_pages(): fetch_links_cached() [hrequests, Exa fallback] -> scan for HF links -> LLM pick
         cache -> data/cache/hf_llm_extract/<family>.json            (gitignored)
                                                                      -> "likely" / method=homepage_llm
     -> data/hf_links/exa_results.json   (durable audit trail -- NOT gitignored)
     -> data/out/hf_sources.json   {by_digest: {...}, by_family: {...}}   (gitignored)

  -> scripts/build_prod_data.py
       compact_model() -> resolve_hf_source(digest, family, hf_sources) -> "hf_source" field
     -> prod/data/models.json   (schema_version 2)

  -> web/app.js
       sourceIconsHtml(row) renders an Ollama icon (always) + an HF icon (if hf_source)
```

Caching follows the same shape throughout: a real hit is cached forever (content and LLM
judgements don't go stale on their own); tier 1's miss (`404 HASH_UNKNOWN`) is retried after
`MISS_TTL_DAYS = 14` since modelindex.dev's crawl keeps growing; tiers 2/2b/verification cache
regardless of outcome (a `repo: None` result means "checked, found nothing," not "not checked
yet") since there's no equivalent reason to expect a page's content to change. All the
`data/cache/hf_*` per-URL/per-digest caches use `_atomic_write_json` (a uuid-suffixed temp file)
rather than a fixed `.tmp` name — two ThreadPoolExecutor workers can legitimately fetch the
identical URL (two families citing the same paper or GitHub repo) and race to write the same
cache entry; a shared temp name caused a real `WinError 32` under that race.

## Measured results (2026-09-09, full catalog)

| | rows | families |
|---|---|---|
| `verified` (tier 1) | 240 | 38 |
| `likely`, method `readme_verified` (tier 2, LLM-confirmed) | 2,091 | 79 |
| `likely`, method `homepage_llm` (tier 2b) | 236 | 14 |
| none | 3,713 | 95 |
| **total with an `hf_source`** | **2,567 / 6,280 (41%)** | 131 / 226 |

Tier 1 pilot methodology before the full run: two independent random 300-digest samples both
landed at 3–5% verified — the full 6,280-digest crawl landed at 240 (3.8%), inside that range.

Tier 2 verification ran against all 85 readme-derived candidates and rejected 6 of them
(`deepseek-coder-v2`, `llava`, `mistral`, `openchat`, `samantha-mistral`, `wizardlm`) — those
6 families' rows moved from "would have shipped a wrong likely match" to either tier 2b or
none, which is why the row/family counts here are *lower* than an earlier unverified-tier-2
measurement in this same investigation (2,866/6,280) despite tier 2b covering more ground than
it did then (14 families vs. 9) — the verification step is trading raw coverage for correctness,
which is the point of adding it.

Families with **no** HF link found by any tier (95/226) are mostly first-party Ollama/Meta
conversions (`llama3.2`'s own `ollama.com` page has zero mentions of `huggingface.co`) — for
these, no `hf_source` is the correct answer, not a coverage gap.

## UI

`web/app.js: sourceIconsHtml` renders, on every model card, next to the fit badge:

- an Ollama icon (brand mark, always present, links to `ollama.com/library/<family>`)
- an HF icon (🤗, present only when `hf_source` is set) — links to `hf_source.url`
  - `verified`: a small blue badge-check overlay (`VERIFIED_BADGE` in `app.js`), tooltip names
    the repo and says "byte-identical file match"
  - `likely`, method `readme`: no badge, tooltip says "named in the readme, not hash-verified"
  - `likely`, method `readme_verified`: no badge, tooltip says "named in the readme,
    LLM-confirmed but not hash-verified"
  - `likely`, method `homepage_llm`: no badge, tooltip says "found via the model's homepage,
    not hash-verified"

The expanded readme panel (`technicalHtml` / `sourceLinksHtml`) shows the same links as text,
for the case where the model has no readme markup at all to click through.

## Known limitations

- Coverage is bounded by modelindex.dev's own crawl breadth (tier 1) and by what hrequests/Exa
  can actually fetch (tier 2b) — neither is auditable beyond spot checks.
- Tier 2 verification and tier 2b's LLM pick are only as good as the prompt and the candidate
  list they're given. Confirmed both failure directions during development: a candidate list
  that's too large and noisy (18+ near-duplicate repos from a busy GitHub README) makes the LLM
  decline rather than guess (`tulu3` failed this way once, then succeeded once the candidate set
  was cleaner); a genuinely correct single candidate can still get value from verification (79/85
  confirmed outright).
- hrequests doesn't execute JavaScript, so client-rendered pages (huggingface.co itself, some
  `notion.site` pages) come back as an empty shell — `fetch_links_cached` treats a *successful*
  fetch with zero anchors as a signal to fall back to Exa, but Exa's own crawl doesn't cover
  everything either, so some of these are still genuinely unresolved.
- Neither tier attempts cross-registry linking beyond Hugging Face (modelindex.dev also indexes
  ModelScope and CivitAI; `best_hf_match` only keeps `source == "hf"`).

## How to run

```bash
# Tier 1: refresh the hash cache for every digest not already cached (~6,280 requests, ~15 min)
python scripts/hf_source.py --refresh --workers 6

# Pilot a sample instead of the full catalog
python scripts/hf_source.py --refresh --limit 300 --workers 6

# Tier 2 verification + tier 2b fallback, in one invocation (recommended -- lets a tier-2
# rejection fall through to tier 2b in the same run; running --llm-extract alone in a later,
# separate invocation would not retry a family --verify-tier2 already rejected earlier).
# Needs OPENROUTER_API_KEY + OPENROUTER_BASE_URL (both) and EXA_API_KEY (fetch fallback +
# tier-2 verification), env or .env at the repo root. `pip install hrequests` for the primary
# tier-2b fetch path.
python scripts/hf_source.py --verify-tier2 --llm-extract --llm-workers 4
python scripts/hf_source.py --verify-tier2 --llm-extract --llm-limit 20  # pilot a sample

# Rebuild data/out/hf_sources.json from whatever's cached, without querying anything new
python scripts/hf_source.py

# Fold the result into prod/data/models.json
python scripts/build_prod_data.py
```
