# Artificial Analysis Intelligence Index — methodology notes

Primary sources (accessed **2026-09-09**):

- https://artificialanalysis.ai/methodology/intelligence-benchmarking/
- https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3
- https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index

## What it measures

> "The Artificial Analysis Intelligence Index is a composite benchmark score that measures language model capabilities across reasoning, coding, knowledge, instruction following, scientific reasoning, and completing multi-step tasks."

> "Artificial Analysis Intelligence Index is a primarily text-based, English-language evaluation suite. We benchmark models for image inputs, speech inputs and multilingual performance separately to the Intelligence Index evaluation suite."

## Current suite (v4.3, September 2026)

Ten evaluations in four categories. Category weights:

| Category | Share of index |
|----------|----------------|
| Agents | 30% |
| Coding | 20% |
| General | 30% |
| Scientific Reasoning | 20% |

Per-evaluation weights (v4.3):

| Category | Evaluation | Private test set | Weight |
|----------|------------|------------------|--------|
| Agents (30%) | AA-Briefcase | Yes | 15% |
| | GDPval-AA v2 | No | 10% |
| | AutomationBench-AA | Yes | 5% |
| Coding (20%) | Terminal-Bench v4.0 | No | 10% |
| | SciCode | No | 10% |
| General (30%) | AA-Omniscience — Accuracy | Yes | 10% |
| | AA-Omniscience — Non-hallucination | Yes | 5% |
| | GDP.pdf | No | 10% |
| | AA-LCR v1.1 | No | 5% |
| Scientific Reasoning (20%) | Humanity's Last Exam (HLE) | No | 10% |
| | CritPt | Yes | 10% |

45% of v4.3 weighting uses private questions or answers (held-out test sets).

**Not in the index:** Chatbot Arena Elo, MMLU-Pro, LiveCodeBench, AIME (removed in v4.0 per AA changelog), IFBench (removed v4.1 for saturation). Reported separately.

## Index calculation (quoted)

> "Intelligence Index is calculated as a weighted average across four categories: Agents (30%), Coding (20%), Scientific Reasoning (20%) and General (30%). The weighting emphasizes agentic tasks."

> "The Artificial Analysis Intelligence Index is calculated as a weighted average of production benchmark scores, scaled from 0 to 100."

### pass@1 scoring

> "We generally use pass@1 scoring across our evaluations, where a model must produce the correct answer on its first attempt. For evaluations with multiple repeats, pass@1 is calculated by aggregating results across all repeats."

Formula (methodology page): average of per-attempt correctness across all repeats and instances.

### Elo-based agent eval normalization

GDPval-AA v2 and AA-Briefcase use pairwise Elo (Bradley–Terry), anchored to human expert deliverables at 1000. For index inclusion:

> "GDPval-AA v2 Elo scores are frozen at the time of a model's addition and normalized as **clamp((Elo - 500) / 2000)** for inclusion in the Intelligence Index."

Same clamp mapping applies to AA-Briefcase combined Elo.

### Separate axes

Intelligence, cost per Intelligence Index task, and time per Intelligence Index task are reported separately (not folded into the index score).

## MODELINDEX use

This file documents **external** methodology only. MODELINDEX does not scrape AA scores or reuse AA branding. Our `Q_base` takes **inspiration** from the fixed-suite, category-weighted, 0–100 weighted-average shape, mapped to vendor-reported benchmarks we already store (`mmlu_pro`, `gpqa_diamond`, `aime24/25`, `livecodebench`, `swe_bench`, etc.).
