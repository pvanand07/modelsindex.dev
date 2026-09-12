# Model identity rebuild: session handoff

Last updated: 2026-09-13 (Asia/Calcutta).
Spec: [model-identity-spec.md](model-identity-spec.md).

## User request

Rebuild model identification and HF linking as a new module; maintain a spec and session handoff
under `docs/`, recording decisions, changes, and next steps. The preceding review identified
substring-size matching, size-only cache collisions, ignored aliases, hash-hit suppression,
and inconsistent secondary-source verification in the legacy pipeline.

## Delivered files

| File | Purpose |
|---|---|
| `identity/resolver.py` | Independent, standard-library offline resolver |
| `tests/identity_tests/` | Resolver, fetch, search, verify, comparison, and export tests |
| `identity/fetch.py` | Public model-card fetcher, plan-only by default |
| `identity/search.py` | Targeted, cached Brave candidate discovery |
| `identity/verify.py` | Evidence-grounded semantic decision verifier |
| `identity/export.py` | Conservative production catalog preview adapter |
| `tests/identity_tests/fixtures/model_identity_review.json` | Reviewed alias and selection cases with evidence notes |
| `identity/compare.py` | Digest-aligned comparison CLI with input checksums |
| `identity/cli.py` | Unified `python -m identity <command>` dispatcher |
| `docs/model-identity-spec.md` | Implemented contract, limitations, validation and extension plan |
| `docs/model-identity-handoff.md` | This continuing session log |
| `data/out/identity_v2.json` | Generated offline audit report; gitignored/rebuildable |
| `data/out/identity_v2_comparison.json` | Generated comparison against legacy output; gitignored/rebuildable |
| `data/out/identity_v2_fetch_plan.json` | Latest bounded fetch plan |
| `data/out/identity_v2_fetch_pilot.json` | Three-repository live pilot result |
| `docs/model-identity-review.md` | Conflict review and nine changed repo-pair findings |

## Decision log

| Decision | Reason |
|---|---|
| Build independently; do not import legacy resolver helpers | Avoid carrying over matching and caching defects |
| Read raw crawl, not compact production models | Preserve all tags and same-digest alias evidence |
| Make the first module an offline rebuild | The user placed fetched data in scope; replay is reproducible without new API use |
| Keep every exact-file upload separately from upstream picks | Byte equality cannot establish publisher authority |
| Use deterministic constraints and explicit abstention | Current family-level LLM verdicts cannot prove exact release identity |
| Require a card but label picks `release_name_constraints` | Card availability alone is not semantic verification |
| Keep output separate from production | A precision benchmark and migration contract are not yet established |
| Recompute decisions from evidence hashes | Full-release/evidence changes cannot reuse a stale size-only decision |
| Fail closed on contradictory alias evidence | Conflicting mappings must remain visible rather than silently choosing one |

## Changes and findings this session

1. Inspected working tree before implementation. Existing user changes were present in
   `scripts/hf_source.py`, `scripts/link_common.py`, `scripts/links.py`, `tests/test_links.py`, and
   `data/hf_links/brave_search_results.json`. The earlier flow document was untracked.
   These were preserved; this implementation uses new files.
2. Added artifact/release inventory, source-preserving extraction, strict HF host validation,
   exact size and explicit mode/version checks, card-presence gate, candidate ambiguity,
   independent exact hash records, alias joins, and a deferred enrichment queue.
3. Added CLI output with stable ordering and atomic replacement. All data paths derive from
   `--data-dir`; no production or legacy decision-cache input is required.
4. First replay produced 220 likely release decisions and ten artifact conflicts. Inspection
   showed an unspecified tag such as `qwen2:7b` selecting a base repo while its explicit
   instruct alias selected another repo. Added a regression and a conservative rule: unspecified
   variant cannot silently mean base when named variants are in the candidate set.
5. Final replay below reflects that correction. Remaining conflicts are intentionally reported;
   no manual guesses were added.

## Initial version validation (1.0.0; historical)

Command:

```powershell
python -m unittest discover -s tests -p test_model_identity.py
python scripts/model_identity.py --data-dir data --out data/out/identity_v2.json
```

Ten new tests passed. The fixture integration test rebuilds twice and compares complete output.

| Measure | Result |
|---|---:|
| Raw crawl rows | 7,374 |
| Unique artifacts | 6,281 |
| Release keys | 1,258 |
| Artifacts with exact-file HF matches | 240 |
| Likely release decisions | 147 |
| Ambiguous release decisions | 7 |
| Unresolved release decisions | 1,104 |
| Artifacts with likely upstream picks | 1,028 |
| Artifact conflicts | 4 |
| Artifacts with unresolved upstream | 5,249 |
| Releases queued for enrichment/review | 1,111 |

These are status counts, not accuracy measurements. The older production snapshot contained
6,280 artifacts; this module inventories raw fetched rows, including one additional artifact.
Do not compare these counts to the prior production denominator without aligning snapshots.
The first implementation prioritizes abstention and has substantially lower upstream coverage.

## Initial next steps (historical; subsequently implemented)

1. Independently review new mappings and broaden the fixture set beyond the current agent-reviewed
   cases. Keep the remaining Qwen 72B/110B contradiction visible; no source data was corrected.
2. Address family renames, default version ambiguity, candidate ancestors/comparison mentions,
   and card semantic verification. Current name constraints cannot settle these reliably.
3. Extend the implemented bounded fetch adapter with candidate discovery/targeted search only
   after defining semantic verification. Continue controlled batches; do not bypass access denials.
4. Add an opt-in production adapter after precision review. The digest-aligned comparison is now
   implemented; it does not claim the old and new evidence were fetched on the same date. Keep the existing
   production catalog and UI on the legacy path until precision and field semantics are reviewed.

## Continuation: alias review and comparison (1.1.0)

User requested continuation of next steps. Reviewed all four conflicts, three initial positive
cases, and all nine repo pairs changed by the final comparison. Details and local source paths
are in [model-identity-review.md](model-identity-review.md).

Decisions and changes:

- Three apparent conflicts were notation/context parsing errors: Llama 4 Scout/Maverick expert
  aliases and InternLM's `1m` context alias. Added evidence-conditioned normalization; retained
  Qwen's genuine 72B-versus-110B metadata contradiction.
- Resolve combined (digest, family) alias constraints before picking the artifact upstream repo.
  Same-digest defaults benefit; other quantizations do not inherit those constraints or hash confidence.
- Scope unspecified-variant gating to otherwise compatible candidates, avoiding contamination by
  unrelated sizes or families.
- Added a source-noted fixture file: four alias cases and twelve selection cases, nine of which
  explicitly compare against the conflicting legacy repository. This is agent review, not human approval.
- Added evidence file checksums, snapshot hash, and schema version 2 with shared detailed artifact
  decisions. Deduplicating these decisions reduced the intermediate report from about 76 MB to 28 MB.
- Added comparison CLI, which keeps likely upstream claims separate from exact uploads and
  family fallbacks. It aligns on digest and hashes its captured input bytes.
- Added output/input collision guards and comparison duplicate-digest validation.

Validation: all sixteen identity tests pass, including deterministic fixture replay, context/MoE
normalization prerequisites, cross-digest isolation, and comparison semantics.

Latest replay: 7,374 rows; 6,281 artifacts; 1,258 release keys; 240 exact-file matches.
182 likely / 12 ambiguous / 1,064 unresolved release decisions. Artifact upstream results:
1,151 likely / 1 conflict / 5,129 unresolved. This is +123 likely artifacts and three fewer
false conflicts relative to 1.0.0, not a measured accuracy increase.

Comparison against 6,280 shared legacy artifacts: 658 agree, 75 differ, 418 new-only likely
upstream claims, 1,703 legacy-only likely claims, 3,426 neither. One raw artifact has no legacy
counterpart. All 75 disagreements fall into the nine documented repo pairs; tag/card evidence
supports the new choices, but no independent precision benchmark has been completed.

Reproduce comparison:

```powershell
python scripts/model_identity_compare.py --identity data/out/identity_v2.json --legacy prod/data/models.json --out data/out/identity_v2_comparison.json
```

No live enrichment, API calls, semantic model-card verifier or production integration was added
in this continuation. Current queues are still release-based; some generic aliases already
covered at artifact level can remain queued. Prioritize artifact gaps when implementing a worker.

## Continuation: bounded enrichment and incomplete evidence (1.2.0)

User requested another continuation. Implemented `model_identity_fetch.py` and integrated
successful card evidence into offline rebuilds. The spec describes CLI flags, states and TTLs.

Changes and decisions:

- Added an artifact-based queue, deduplicated by constraint decision; resolved aliases do not
  trigger redundant network work. Conflict groups remain review-only.
- The adapter deduplicates by repo, ranks by affected artifacts and enforces repository, HTTP
  request, elapsed-time and response-size budgets. Redirects count as requests and remain on HF.
- Main-to-master fallback resumes across budget-limited runs. Cache records distinguish missing,
  denied, transient, rate-limited, invalid, oversized and successful responses. Retain last-good
  content independently from refresh status. A persistent global backoff honors HTTP 429.
- Fresh independent cards override legacy content only after SHA-256 validation. No legacy cache
  was modified and no resolver decision is changed by a fetch until a new offline rebuild.
- The first sandbox pilot encountered transport errors. Repeated the bounded public read with
  approved network escalation and `--retry-transient`; this override cannot bypass rate limits.
- Live pilot: `lmsys/vicuna-13b-v1.5-16k` and `lmsys/vicuna-13b-v1.5` returned HTTP 200;
  `meta-llama/Llama-3.1-8B-Instruct` returned HTTP 401 and stays denied. The successful cards
  contained 2,071 and 1,965 characters. The cache retains both attempts for each repository.
- Pilot initially added fifteen likely artifact mappings (1,151 -> 1,166). Review then found a
  deeper bug: an equally compatible candidate missing its card was treated as rejected. Added
  an ambiguity gate and regression: missing evidence must never select a competing repo.
  This removed 271 provisional artifact picks, leaving 895 likely upstream mappings. Treat the
  reduction as corrected confidence handling, not a loss of fetched evidence.

Latest validation: **29 tests passed** (`test_model_identity*.py`), including simulated timeouts,
404 fallback, denied/empty/HTML/oversized responses, retry expiration, persisted 429 backoff,
request/deadline budgets, redirect restrictions, last-good retention, offline integration and
tampered-card detection. `git diff --check` passed (legacy CRLF warnings only).

Latest output: 7,374 rows; 6,281 artifacts; 1,258 release keys; 240 exact-file matches.
131 likely / 120 ambiguous / 1,007 unresolved release decisions. Artifact upstream:
895 likely / 1 conflict / 5,385 unresolved. There are 851 artifact work groups and 210 eligible
missing-card repositories; the latest three-repo plan records one cached denial and zero HTTP reads.

Latest legacy comparison: 6,280 shared artifacts; 559 agreeing likely claims, 60 different,
276 new-only, 1,817 legacy-only, 3,568 neither. One raw-only artifact. Historical 1.1.0 comparison
numbers and reviewed repo pairs above remain a record of that version, not current coverage.

Reproduce a plan:

```powershell
python scripts/model_identity_fetch.py --identity data/out/identity_v2.json --data-dir data --out data/out/identity_v2_fetch_plan.json --max-repos 3
```

The separate `--fetch` flag enables public HTTP; see the spec for bounded execution.

## Continuation: search, semantic verification, and production preview (1.3.0)

The user authorized live APIs, LLM calls, and Brave searches. Added cached, bounded modules for
all three remaining pipeline stages and ran pilots.

- Five targeted Brave searches completed for Aya Expanse 32B, three CodeGemma constraints, and
  CodeLlama 13B. Results include canonical, mirror, and conversion repos; all enter the shared
  constraint resolver and none receives trust from search rank alone.
- Fetched ten high-impact candidate cards: seven succeeded; gated Meta Llama 3.3 and two Cohere
  repos returned denied states. A rebuild moved likely artifacts from 895 to 986.
- First semantic pilot exposed empty-content responses from a reasoning model. Increased the
  completion budget, requested JSON output, and made empty response handling explicit.
- Fixed family derivation for prompts and grounded whitespace-normalized quotes to exact card
  substrings. Nine of the first ten decision groups cached valid verdicts.
- A second 20-item batch returned 15 valid `yes/yes` verdicts and five invalid/un-groundable
  responses, which were discarded. The semantic cache now contains 24 verified decision groups.
- Resolver consumes matching semantic verdicts by decision key and card hash. `yes/yes` upgrades
  method; an explicit `no` rejects the artifact choice; uncertainty does not upgrade it.
- Added the conservative export adapter. Current preview contains 6,280 rows, 240 rows with exact
  upload evidence, and 347 rows with semantic-verified upstream links. Existing links are intact.

Validation now totals **36 passing identity tests**. New tests cover verifier vocabulary and
quote grounding, cache reuse, strict Brave filtering, search caching, semantic-only export,
provisional opt-in, duplicate digest rejection, and legacy-link preservation. `git diff --check`
passes apart from pre-existing line-ending warnings in legacy modified files.

Current resolver replay after live enrichment: 7,374 rows, 6,281 artifacts, 1,258 release keys,
240 exact-file matches, 986 likely artifact upstream picks, one conflict, and 5,294 unresolved.
The artifact decisions comprise 24 semantic-verified and 142 provisional likely groups, plus
108 ambiguous, 732 unresolved, and one conflict group. Counts are evidence coverage, not accuracy.

Remaining steps: independently review semantic verdicts and a representative new-only sample;
use structured HF metadata/card sections to reduce LLM ambiguity; continue bounded enrichment;
then enable the opt-in path in the release build. No live production catalog, deployment, or
commit was changed.

## Continuation: opt-in build and UI integration

Added `--identity` and `--identity-include-provisional` to `build_prod_data.py`. The default build
path remains unchanged. An identity-enabled preview completed at
`data/out/prod_identity_preview/`: 6,280 model rows, 347 semantic-verified upstream rows, and
240 exact-upload rows. Its manifest records resolver 1.3.0 and evidence snapshot
`69b719cafa2967bc232533ec2ebe83687eb86d5e15cdc64bd06313e5373f0805`.

Updated `web/app.js` to prefer `identity_v2.upstream` when present. Semantic-verified links receive
the verified badge and a model-card/release-evidence tooltip; provisional links remain explicitly
unverified; rows without an exported upstream retain the legacy resolution behavior. Exact-file
uploads remain data-only so the UI does not confuse a matching conversion with upstream authorship.

Validation:

- 36 identity-specific tests pass.
- The identity-enabled production preview builds successfully.
- `node --check web/app.js` passes.
- `git diff --check` passes except pre-existing line-ending warnings in legacy modified files.
- Full Python discovery ran 165 tests: 163 passed; two VRAM-model tests could not import NumPy in
  this environment. Neither failure touches identity or link integration.

The release catalog under `prod/data` was not overwritten. The next integration gate is an
independent review of the 24 semantic decision groups and a representative sample of provisional,
ambiguous, and legacy-only cases. After that, enable `--identity` in the real build command and
add an exact-upload detail panel if it provides useful provenance to users.

## Continuation rules

Update this document after material decisions, code changes, or validation runs. Record both
implemented behavior and deferred work explicitly. Preserve unrelated worktree changes. The
generated JSON is an audit artifact, not a replacement `prod/data/models.json`. Record live
network activity and its bounds explicitly; do not describe a fetched card as semantic proof.

## Refactor: active package and legacy quarantine (2026-09-13)

The new resolver is now a top-level `identity/` package with a unified
`python -m identity <command>` entry point. Tests moved to `tests/identity_tests/`; durable evidence
moved to `data/identity/evidence/`; active caches use `data/cache/identity/`. The production
builder no longer imports the retired linking helper and legacy JSON inputs are explicit,
deprecated compatibility options instead of defaults.

Retired `hf_source.py`, `links.py`, `link_common.py`, their test, generated reports, search
audits not used by the new resolver, and obsolete caches moved under `legacy/`. `/legacy/` is
gitignored, so this is a local recovery area and will not be merged. Reusable Brave audit data
and saved source content were migrated into the active evidence namespace rather than discarded.

The planned refactor validation, offline replay, identity-enabled preview, and diff inspection
were completed below.

Validation completed:

- 36/36 identity tests pass.
- Offline replay: 7,374 rows, 6,281 artifacts, 240 byte-matched artifacts, 986 likely artifact
  upstreams, 5,294 unresolved, and one conflict.
- Identity-enabled preview: 6,280 rows, 347 semantic-verified upstreams, and 240 rows with exact
  upload provenance; evidence snapshot
  `851bfd7f8ad0bef252b31c69d35cb1c993e30d5707a0063284b4131f8c3226e3`.
- Browser JavaScript syntax and `git diff --check` pass (line-ending warnings only).
- Full discovery runs 93 tests; 91 pass and the same two unrelated VRAM tests remain blocked by
  missing NumPy in this environment.

Remaining merge gate: review/stage the large evidence-directory rename so Git records it as a
migration, and either run the two NumPy-dependent tests in the project environment or explicitly
accept that pre-existing environment limitation.

## Merge-gate execution and safety audit (2026-09-13)

- Verified all 194 tracked `data/link_content` files have matching filenames in
  `data/identity/evidence/link_content`; no evidence file is missing.
- Ran the complete suite with the project-declared NumPy dependency available: **96/96 tests pass**.
- Audited all 24 cached semantic groups plus representative `new_only` and `different` mappings.
  The audit caught two unsafe patterns: placeholder (`...`) verifier reasons and versioned HF
  checkpoints selected for unversioned aliases.
- Added deterministic `version_unspecified` rejection and require affirmative semantic verdicts
  to contain both a grounded quote and a substantive reason. Prompt version is now 2, so future
  verifier cache entries use the stricter contract. Added regressions for both gates.
- Corrected the verifier's remaining pre-refactor card paths to the active evidence/cache namespace.

Corrected validation: **37/37 identity tests pass**. Offline replay now reports 942 likely
artifact upstreams, 5,338 unresolved, one conflict, and 240 byte matches. Eighteen semantic groups
pass the stricter gate, exporting 259 semantic-verified production rows. The lower coverage is an
intentional precision correction.

Decision: keep identity integration opt-in for this merge. The module and UI support are ready,
but enabling it by default should wait for human review of the 18 surviving semantic groups and a
documented release-build invocation that first creates `data/out/identity_v2.json`. The ignored
audit report must not become an undeclared build prerequisite.

Remaining steps after staging: inspect Git's rename detection, review the staged diff, then commit
and merge. Default enablement and further bounded enrichment belong in a follow-up change.
