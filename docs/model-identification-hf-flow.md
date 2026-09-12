# Model identification and Hugging Face linking

This is the evidence flow represented by the currently fetched data. Counts are from the
production snapshot generated on **2026-09-09**.

```mermaid
flowchart TD
    A["Ollama library pages<br/>family, description, README links"]
    B["Ollama registry manifests<br/>tag, layer digest, exact layer sizes, push time"]
    C["Ollama config blobs<br/>model format, family, type, quant"]
    D["GGUF Range reads<br/>general.name, architecture, params, quant,<br/>context, tensor byte totals"]

    A --> E["Family identity<br/>model slug + description + README"]
    B --> F["Exact artifact identity<br/>family:tag + sha256 digest"]
    C --> G["Format/family corroboration"]
    D --> H["Checkpoint identity signals<br/>canonical-name hint + size/architecture"]

    F --> I{"Reverse SHA-256 lookup<br/>modelindex.dev"}
    I -->|"source = hf and same bytes"| J["Verified HF repo<br/>digest-scoped"]

    A --> K["Mine every HF URL in README"]
    H --> L["Build release key<br/>family:tag minus quant suffix"]
    L --> M["Match release size token<br/>against candidate repo names"]
    K --> M

    N["Previously fetched HF/GitHub/homepage text"] --> O["Mine sibling org/repo tokens"]
    O --> M

    M -->|"one size match"| P["Likely release candidate"]
    M -->|"zero or ambiguous"| Q["LLM selects candidate<br/>using family/release context"]
    Q --> P

    R["Brave search results<br/>only when local candidates are empty"] --> M

    P --> S["Fetch raw HF README<br/>main, then master"]
    S --> T{"README describes<br/>the same model?"}
    T -->|yes| U["Likely + verified-by-README<br/>release-scoped HF link"]
    T -->|no| V["Reject candidate / leave unresolved"]
    T -->|gated or unavailable| W["Keep likely candidate<br/>without README verification"]

    A --> X["Classify GitHub, homepage, paper links"]
    X --> Y["Fetch GitHub README and homepage content"]
    Y --> N
    Y --> Z["Secondary HF candidates<br/>github_llm / homepage_llm"]
    Z --> P

    J --> AA["Merge precedence"]
    U --> AA
    W --> AA
    AA --> AB["links.hf.release<br/>exact release when available"]
    AA --> AC["links.hf.family<br/>explicit fallback/context"]
    AB --> AD["prod/data/models.json<br/>6,280 unique digest rows"]
    AC --> AD
    X --> AD
```

## What was fetched and how it identifies a model

| Fetched evidence | Durable location | Identification value |
|---|---|---|
| Ollama family page and README | `data/out/library.json` | Family slug, description, upstream URLs, and textual model/release context |
| Registry manifest and config | `data/out/models.jsonl` | Exact tag, raw GGUF SHA-256 digest, byte sizes, format, family, type, and quantization |
| GGUF header via HTTP Range | `data/out/models.jsonl` | `general.name`, architecture, parameter count, context, quantization, and tensor sizes without downloading the full model |
| Reverse-hash responses | `data/cache/identity/imported/hash_lookup/` | Byte-identical HF upload; strongest evidence because the Ollama layer and HF file share a digest |
| HF candidates and decisions | `data/out/identity_v2.json` | Artifact- and release-scoped repo selections, method, confidence, and unresolved sets |
| Brave search audit | `data/identity/evidence/search_audits/brave_search_results.json` | Last-resort HF candidates for families with no README/content candidates |
| GitHub/homepage/paper decisions | `data/out/links.json` | Project-level provenance and secondary routes to an HF repo |
| Saved HF/GitHub/homepage content | `data/identity/evidence/link_content/<family>.json` | Model-card/README text used to verify identity and discover sibling checkpoints |
| Final joined catalog | `prod/data/models.json` | One record per unique digest with `release` and resolved provenance links |

## Current HF-linking coverage

| Measure | Count |
|---|---:|
| Unique production model artifacts | 6,280 |
| Digest lookups attempted | 6,280 |
| Byte-identical, digest-verified HF matches | 240 |
| Likely family mappings | 177 |
| Likely release mappings | 380 |
| Production rows with a release-specific HF link | 2,676 |
| Production rows using only the explicit family fallback | 1,974 |
| Production rows without an HF link | 1,630 |
| Distinct HF repos linked from production rows | 351 |
| Saved family content bundles | 194 |
| Unresolved families / releases in the HF audit | 11 / 549 |

The key safety rule is that an HF link is **release-scoped**. For example, different sizes in
one Ollama family may originate from different Hugging Face repositories. Consumers should use
`links.hf.release` first and only fall back to `links.hf.family`, visibly treating that fallback
as not size-confirmed.

## Example final link

For `alfred:40b-1023-q4_1`, the quant suffix is removed to form release
`alfred:40b-1023`. Its release-specific result is
[lightonai/alfred-40b-1023](https://huggingface.co/lightonai/alfred-40b-1023), with
`confidence: likely` and `method: readme_verified`.

Related implementation: [`scripts/crawl.py`](../scripts/crawl.py), the
[`identity/`](../identity/) package, and
[`scripts/build_prod_data.py`](../scripts/build_prod_data.py). Retired resolver code is retained
only in the gitignored `legacy/` workspace tree.
