# Saved-evidence review, 2026-09-12

Scope: agent review of fetched data and explicit aliases, not independent human adjudication or
a catalog-wide precision measurement. Fixtures: `tests/identity_tests/fixtures/model_identity_review.json`.

## Four original artifact conflicts

| Case | Fetched evidence | Resolution |
|---|---|---|
| Qwen `72b-text` / `110b-text-v1.5` | Same digest; `name=Qwen1.5-110B`; 111,209,914,368 parameters | Keep conflict. Do not silently rewrite the 72B ref. |
| Llama 4 Scout | Same digest has `16x17b` and `17b-scout-16e-instruct`; GGUF expert count 16 | Normalize notation using explicit alias plus header. No total-parameter multiplication. |
| Llama 4 Maverick | Same digest has `128x17b` and `17b-maverick-128e-instruct`; expert count 128 | Same conditional notation rule. |
| InternLM `1m` | Same digest also has `7b-chat-1m-v2.5`; 7,737,708,544 parameters; name `Internlm2_5 7b Chat 1m` | Keep `1m` as a context qualifier, not parameter size. |

Source: `data/out/models.jsonl`. Exact digests, aliases, expected constraints and provenance are
recorded in the fixture. No source row was edited. The MoE rule requires the header and explicit
same-digest expert alias; isolated small-model M-size tokens remain parameter tokens.

## Changed mappings compared with legacy (1.1.0 snapshot)

The comparison aligns 6,280 shared digests. All 75 different likely-upstream picks fall into
these nine repo pairs. Saved card availability and explicit tag constraints support the new
choices; the comparison itself does not establish correctness.

| Rows | Explicit release evidence | Legacy repo | New repo |
|---:|---|---|---|
| 12 | `xwinlm:7b-v0.2` | `Xwin-LM/Xwin-LM-7B-V0.1` | `Xwin-LM/Xwin-LM-7B-V0.2` |
| 14 | `starcoder2:15b-instruct-v0.1` | `bigcode/starcoder2-15b` | `bigcode/starcoder2-15b-instruct-v0.1` |
| 14 | `yi-coder:9b-chat` | `01-ai/Yi-Coder-9B` | `01-ai/Yi-Coder-9B-Chat` |
| 15 | `llava:7b-v1.5` | `liuhaotian/LLaVA-Lightning-MPT-7B-preview` | `liuhaotian/llava-v1.5-7b` |
| 14 | `zephyr:7b-beta` | `HuggingFaceH4/zephyr-7b-alpha` | `HuggingFaceH4/zephyr-7b-beta` |
| 2 | `sailor2:8b-chat` | `sail/Sailor2-8B` | `sail/Sailor2-8B-Chat` |
| 1 | `orca2:latest` shares digest with `7b` | `microsoft/Orca-2-13b` | `microsoft/Orca-2-7b` |
| 1 | `granite3.2:latest` shares digest with `8b-instruct` | `ibm-granite/collections` | `ibm-granite/granite-3.2-8b-instruct` |
| 2 | `olmo-3.1:32b-think` | `allenai/Olmo-3.1-32B-Instruct` | `allenai/Olmo-3.1-32B-Think` |

Model cards are saved under `data/identity/evidence/link_content/<family>.json` and/or `data/cache/identity/imported/cards/`.
Inspected headings explicitly name StarCoder2-Instruct, Zephyr beta, Granite-3.2-8B-Instruct,
and Olmo 3.1 32B Think. The other cases were checked against the saved candidate card and
explicit tag/alias evidence. These are identity-constraint regressions; the fixtures do not
simulate a semantic verifier or establish publisher authority.

Additional positive fixtures: Alfred `40b-1023`, CodeBooga `34b-v0.1`, Qwen2 `7b-instruct`.
The saved Ollama references / model-card loading examples name the expected repositories.

## Comparison interpretation (1.1.0 snapshot)

658 shared artifacts have agreeing likely-upstream picks; 75 differ; 418 have only a new likely
upstream pick; 1,703 have only a legacy likely release pick; 3,426 have neither. A legacy exact
upload or family fallback is recorded separately and is not counted as a likely release claim.
There is one additional artifact in the raw inventory and no legacy-only artifact.

The generated report `data/out/identity_v2_comparison.json` contains per-digest rows and input
hashes. The underlying report records consumed evidence hashes. This captures input identity,
not a guarantee that the old and new pipelines used contemporaneous upstream fetches.

Remaining review work: the 418 new-only claims, unresolved defaults/renamed families, ancestor
mentions, and false negatives among legacy-only claims. Production cutover should follow an
independent representative accuracy evaluation rather than a coverage threshold.

## 1.2.0 update

The newer resolver no longer treats a missing competing card as rejection. Some earlier likely
picks therefore return to ambiguity while their evidence remains stored. Latest comparison:
559 agree, 60 differ, 276 new-only, 1,817 legacy-only, 3,568 neither, over the same 6,280 shared
digests. The historical repo-pair review above remains useful regression evidence, not a current
precision or coverage claim. Two Vicuna cards were fetched in a three-repository public pilot;
Llama 3.1 returned 401. See the handoff for attempt states, test results and current counts.

## 1.3.0 live-evidence update

Five targeted Brave searches were cached and ten additional card fetches were attempted. Seven
cards succeeded and three gated repos returned denied. Two bounded OpenRouter verification runs
produced 24 cached, evidence-grounded `yes/yes` decision groups; invalid or ungrounded responses
were discarded rather than interpreted. These verdicts support 347 production-preview rows due
to shared digest/alias decisions. This is not an independent accuracy result: the verifier is an
LLM judging stored evidence, and the next review must sample its positive and any future negative
verdicts independently.
