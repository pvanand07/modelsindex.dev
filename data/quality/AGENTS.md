# Quality sidecar — agent instructions

This directory holds **benchmark-derived quality scores** for MODELINDEX. It is intentionally separate from crawl/prod artifacts.

## Do not touch

- `data/out/models.jsonl` — crawl output; read-only for quality scoring
- `prod/data/models.json` — browser export; never write eval scores here
- `data/out/constants.json`, measurements, index — fit/bench pipeline

## Layout

| Path | Purpose |
|------|---------|
| `seed/web_evals.json` | Curated HF/blog/arXiv benchmark rows |
| `seed/hf_map.csv` | Maps eval_id ↔ HF repo ↔ Ollama family/tag hints |
| `evals.jsonl` | Merged eval subjects (readme + web) |
| `references.jsonl` | Citation index: URL, accessed date, snippet path, benches |
| `refs/*.md` | On-disk verification snippets (copy table rows from sources) |
| `q_base.json` | Q_base from min–max intelligence suite |
| `quant_factors.json` | Seed F(quant) multipliers; replace from fit.py when calibrated |
| `scores.json` | Per expanded catalog ref: Q_base, Q_file (GPU-independent) |
| `ingest_summary.json` | Row counts, missing families, readme coverage |
| `../../prod/data/quality.json` | Browser export: compact `by_ref` map of Q_file |

## Scoring formulas

**Q_base** — fixed intelligence suite (see `INTELLIGENCE_SUITE` in `scripts/quality.py`), inspired by [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking/) v4.3 category weighting but mapped to vendor benches we store (not AA proprietary evals or scores):

| Category (AA v4.3) | MODELINDEX benches | Weight |
|--------------------|-------------------|--------|
| Scientific (20%) | `gpqa_diamond`, `aime25`, `aime24` | 15%, 10%, 5% |
| Coding (20%) | `livecodebench`, `humaneval` | 12%, 8% |
| General (30%) | `mmlu_pro`, `math500` | 20%, 10% |
| Agents (30%) | `swe_bench`, `bfcl` | 15%, 5% |

Per benchmark: min–max normalize to 0–100 within sidecar population, then weighted average (renormalized over available benches). No Arena Elo; saturated `ifeval` / plain `mmlu` dropped from mix.

`Q_base = Σ(w_i × norm_i) / Σ(w_i)` where `norm_i = 100 × (score − pop_min) / (pop_max − pop_min)`.

Embedding rows: `mteb` only. Vision rows: suite + `mmmu` (15%, renormalized).

**Q_file** — quant degradation (multiplicative):

`Q_file = Q_base * F(quant)`

Seed factors: F16=1.0, Q8_0=0.97, Q6_K=0.94, Q5_K_M=0.92, Q4_K_M=0.90, Q4_0=0.82, Q3_K_M=0.72, Q2_K=0.55

**q_tasks** — per use-case scores with the same min–max + weighted-average shape, then × F(quant). Suites in `TASK_SUITES` (`scripts/quality.py`):

| Use case | Suite emphasis | Required bench |
|----------|----------------|----------------|
| `chat` | Full intelligence suite | — |
| `code` | LiveCodeBench, HumanEval, SWE-Bench, BFCL | — |
| `long` | MMLU-Pro, MATH-500, GPQA, AIME25, SWE-Bench (long-doc proxies) | — |
| `vision` | MMMU-weighted + text mix | `mmmu` |
| `embedding` | MTEB only | `mteb` |

Vision/embedding tasks are omitted when the modality bench is absent (no text-only fake score).

**Q_rec is browser-side.** The sidecar publishes GPU-independent `Q_file` and `q_tasks`. The finder uses `q_file/100` as capability and `q_tasks[usecase]/100` as the match term (replacing name/signal heuristics). When either is missing: size prior × **exponential age decay** \(2^{-t/1.1}\) years since `pushed_at` (half-life 1.10y; see `refs/capability-age-decay.md`). Measured scores are **not** age-discounted. Architecture hard-filters for vision/embedding catalogs remain (eligibility, not ranking). Do not bake an L4/8k Q_rec into `scores.json`.

Ranking shape (matches `web/app.js` balanced priority):

`capability*50 + speed*14 + match*26 + fit*18` (+ recency / quant nudges)

- `capability` = `q_file/100` when present, else log(params)+quant proxy × age decay
- `match` = `q_tasks[usecase]/100` when present, else same size prior
- `speed_score = clamp(log10(tps+1)/2.4, 0, 1)` from the GPU estimate formula
- `fit_score` — full=1, partial=0.3 — from VRAM at selected context

## Benchmark keys

Canonical keys: `aime24`, `aime25`, `gpqa_diamond` (fallback `gpqa`), `math500`, `mmlu_pro` (fallback `mmlu`), `livecodebench`, `swe_bench`, `humaneval`, `bfcl`, `mmmu` (vision), `mteb` (embed-only). Legacy keys (`ifeval`, plain `mmlu`, `gpqa`) may appear in ingest but are not in `INTELLIGENCE_SUITE`.

## External methodology refs

| Ref | Purpose |
|-----|---------|
| `refs/artificial-analysis-intelligence.md` | AA Intelligence Index v4.3 suite/weighting/normalization — design inspiration for Q_base only |

## Ingest sources (priority)

1. **Ollama README tables** — `prod/data/library.json` or `data/cache/library/*.json`; parse markdown benchmark tables
2. **Web seed** — `seed/web_evals.json` for official cards when README lacks numbers
3. **WebSearch/WebFetch** — for missing families; always persist citations

## Adding missing scores (checklist)

1. Search official sources: HuggingFace model card, vendor blog, arXiv, GitHub eval_details
2. Add row to `seed/web_evals.json` with `eval_id`, `family`, `size_b`, `variant`, `benchmarks`
3. Add mapping line to `seed/hf_map.csv` if HF repo helps disambiguate
4. Create `refs/<slug>.md` with URL, accessed date, snippet table, supported benchmarks
5. Append entry to `REFERENCE_SEED` in `scripts/quality.py` (or extend references.jsonl writer)
6. Run `python scripts/quality.py all`
7. Verify `ingest_summary.json` — check `missing_eval_families` and `readme_table_families_missing`

## Citation rules

- Every non-readme score must have at least one `references.jsonl` row
- Snippet files under `refs/` must include: URL, accessed date, which benchmarks, eval_ids
- When vendors disagree (e.g. MMLU-Pro 66.4 HF vs 65.1 Meta/NIM), prefer HF unless README is authoritative; document conflict in snippet
- Record `accessed` as ISO date (UTC)

## Expected README table families

These Ollama families often ship benchmark tables: olmo-3, olmo-3.1, gemma3, gemma4, deepseek-v2.5, deepseek-v4-flash, nemotron-3-super, glm-4.7-flash, qwen3.8-flash-next, qwen3.5, llama4

If `ingest_summary.json` lists them under `readme_table_families_missing`, fetch vendor docs via WebSearch.

## Commands

```bash
python scripts/quality.py ingest   # rebuild evals + references
python scripts/quality.py score    # recompute q_base + scores.json
python scripts/quality.py all      # both
python scripts/build_prod_data.py  # writes prod/data/quality.json for the finder
python -m pytest tests/test_quality.py tests/test_prod_data.py -q
```

## Tests

Keep unit tests small: table parsing, bench alias normalization, Q_base monotonicity, quant factor application.
