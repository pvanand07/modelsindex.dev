#!/usr/bin/env python3
"""
MODELINDEX quality sidecar — ingest benchmark tables and compute Q scores.

Separate from crawl/prod data under data/quality/. Stdlib only.

    python scripts/quality.py ingest   # readme tables + seed web evals
    python scripts/quality.py score    # Q_base, Q_file, scores.json
    python scripts/quality.py all      # ingest then score
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "data" / "quality"
SEED = QUALITY / "seed"
REFS = QUALITY / "refs"
DEFAULT_LIBRARY = ROOT / "prod" / "data" / "library.json"
DEFAULT_LIBRARY_CACHE = ROOT / "data" / "cache" / "library"
DEFAULT_MODELS = ROOT / "data" / "out" / "models.jsonl"
FALLBACK_MODELS = ROOT / "prod" / "data" / "models.json"

SCHEMA_VERSION = 2

# Canonical benchmark keys used across eval rows and references.
CANONICAL_BENCHES = (
    "aime24",
    "aime25",
    "gpqa_diamond",
    "gpqa",
    "math500",
    "mmlu_pro",
    "mmlu",
    "livecodebench",
    "swe_bench",
    "humaneval",
    "ifeval",
    "mmmu",
    "bfcl",
    "mteb",
)

# Row header aliases -> canonical bench key (first match wins).
BENCH_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("aime24", ("aime 2024", "aime24", "aime 24", "aime2024")),
    ("aime25", ("aime 2025", "aime25", "aime 25", "aime2025", "aime25 (no tools)")),
    ("gpqa_diamond", ("gpqa-diamond", "gpqa diamond", "gpqa-d", "gpqa diamond (no tools)")),
    ("gpqa", ("gpqa", "gpqa (no tools)")),
    ("math500", ("math-500", "math 500", "math500")),
    ("mmlu_pro", ("mmlu-pro", "mmlu pro")),
    ("mmlu", ("mmlu",)),
    ("livecodebench", (
        "livecodebench", "live code bench", "lcb",
        "livecodebench (v5 2024-07 to 2024-12)",
    )),
    ("swe_bench", ("swe-bench", "swe bench", "swe-bench pro", "swe bench pro")),
    ("humaneval", ("humaneval", "humanevalplus", "humaneval+", "humaneval plus")),
    ("ifeval", ("ifeval", "if eval", "ifeval (strict)")),
    ("mmmu", ("mmmu",)),
    ("bfcl", ("bfcl",)),
    ("mteb", ("mteb",)),
]

# Fixed intelligence suite for Q_base (chat/default).
# Inspired by AA Intelligence Index v4.3 category shape (Agents/Coding/General/Scientific)
# but mapped to vendor-reported benches we store — not AA proprietary evals or scores.
# Weights sum to 1.0; renormalized over available benches per eval row at score time.
INTELLIGENCE_SUITE: dict[str, float] = {
    # Scientific reasoning proxies (30%) — AA: HLE + CritPt
    "gpqa_diamond": 0.15,
    "aime25": 0.10,
    "aime24": 0.05,
    # Coding (20%) — AA: Terminal-Bench + SciCode
    "livecodebench": 0.12,
    "humaneval": 0.08,
    # General knowledge (30%) — AA: Omniscience + long-doc + LCR
    "mmlu_pro": 0.20,
    "math500": 0.10,
    # Agentic / tool use (20%) — AA: Briefcase + GDPval + AutomationBench
    "swe_bench": 0.15,
    "bfcl": 0.05,
}
# Legacy alias for tests/docs that referenced TASK_MIX.
TASK_MIX = INTELLIGENCE_SUITE

# Per use-case suites (same min–max + weighted-average shape as Q_base).
# Keys match web finder usecase ids. vision/embedding require a modality bench.
TASK_SUITES: dict[str, dict[str, float]] = {
    "chat": dict(INTELLIGENCE_SUITE),
    "code": {
        "livecodebench": 0.40,
        "humaneval": 0.25,
        "swe_bench": 0.30,
        "bfcl": 0.05,
    },
    "long": {
        # Proxies for AA General long-doc / LCR (no GDP.pdf / AA-LCR in sidecar yet)
        "mmlu_pro": 0.40,
        "math500": 0.15,
        "gpqa_diamond": 0.20,
        "aime25": 0.10,
        "swe_bench": 0.15,
    },
    "vision": {
        "mmmu": 0.50,
        "mmlu_pro": 0.20,
        "gpqa_diamond": 0.15,
        "livecodebench": 0.15,
    },
    "embedding": {
        "mteb": 1.0,
    },
}
TASK_REQUIRED_BENCHES: dict[str, frozenset[str]] = {
    "vision": frozenset({"mmmu"}),
    "embedding": frozenset({"mteb"}),
}

SEED_QUANT_FACTORS: dict[str, float] = {
    "F16": 1.0,
    "Q8_0": 0.97,
    "Q6_K": 0.94,
    "Q5_K_M": 0.92,
    "Q4_K_M": 0.90,
    "Q4_0": 0.82,
    "Q3_K_M": 0.72,
    "Q2_K": 0.55,
}

REFERENCE_SEED: list[dict[str, Any]] = [
    {
        "ref_id": "ref-llama31-8b-hf",
        "url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        "title": "Llama 3.1 8B Instruct HF card",
        "snippet_path": "refs/llama31-8b-hf.md",
        "benchmarks": ["mmlu", "mmlu_pro", "gpqa", "ifeval", "humaneval", "math500"],
        "eval_ids": ["llama3.1:8b-instruct"],
    },
    {
        "ref_id": "ref-llama31-70b-hf",
        "url": "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
        "title": "Llama 3.1 70B Instruct HF card",
        "snippet_path": "refs/llama31-70b-hf.md",
        "benchmarks": ["mmlu", "mmlu_pro", "gpqa", "ifeval", "humaneval", "math500"],
        "eval_ids": ["llama3.1:70b-instruct"],
        "notes": "MMLU-Pro 66.4 on HF; NVIDIA NIM lists 65.1 — sidecar prefers HF.",
    },
    {
        "ref_id": "ref-llama31-eval-details",
        "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md",
        "title": "Meta Llama 3.1 eval_details",
        "snippet_path": "refs/llama31-eval-details.md",
        "benchmarks": ["mmlu", "mmlu_pro", "gpqa", "ifeval", "humaneval", "math500"],
        "eval_ids": ["llama3.1:8b-instruct", "llama3.1:70b-instruct"],
    },
    {
        "ref_id": "ref-gemma3-card",
        "url": "https://ai.google.dev/gemma/docs/core/model_card_3",
        "title": "Gemma 3 model card",
        "snippet_path": "refs/gemma3-card.md",
        "benchmarks": ["mmlu_pro", "gpqa_diamond", "livecodebench", "ifeval", "humaneval"],
        "eval_ids": ["gemma3:1b-it", "gemma3:4b-it", "gemma3:12b-it", "gemma3:27b-it"],
    },
    {
        "ref_id": "ref-gemma3-arxiv",
        "url": "https://arxiv.org/abs/2503.19786",
        "title": "Gemma 3 arXiv paper",
        "snippet_path": "refs/gemma3-arxiv.md",
        "benchmarks": ["mmlu_pro", "gpqa_diamond", "livecodebench", "ifeval", "humaneval"],
        "eval_ids": ["gemma3:27b-it"],
    },
    {
        "ref_id": "ref-qwen25-blog",
        "url": "https://qwenlm.github.io/blog/qwen2.5-llm/",
        "title": "Qwen2.5 blog benchmarks",
        "snippet_path": "refs/qwen25-blog.md",
        "benchmarks": ["mmlu_pro", "gpqa", "livecodebench", "ifeval", "humaneval"],
        "eval_ids": ["qwen2.5:7b-instruct", "qwen2.5:14b-instruct", "qwen2.5:32b-instruct"],
    },
    {
        "ref_id": "ref-deepseek-r1-distill-32b",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "title": "DeepSeek-R1 distill HF card",
        "snippet_path": "refs/deepseek-r1-distill-32b.md",
        "benchmarks": ["aime24", "math500", "gpqa_diamond", "livecodebench"],
        "eval_ids": [
            "deepseek-r1:1.5b-qwen-distill",
            "deepseek-r1:7b-qwen-distill",
            "deepseek-r1:14b-qwen-distill",
            "deepseek-r1:32b-qwen-distill",
            "deepseek-r1:8b-llama-distill",
            "deepseek-r1:70b-llama-distill",
        ],
    },
    {
        "ref_id": "ref-phi4-arxiv",
        "url": "https://arxiv.org/html/2412.08905v1",
        "title": "Phi-4 arXiv paper",
        "snippet_path": "refs/phi4-arxiv.md",
        "benchmarks": ["mmlu", "gpqa", "math500", "humaneval"],
        "eval_ids": ["phi4:14b"],
    },
    {
        "ref_id": "ref-llama32-model-card",
        "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md",
        "title": "Llama 3.2 model card",
        "snippet_path": "refs/llama32-model-card.md",
        "benchmarks": [
            "mmlu",
            "ifeval",
            "gpqa",
            "math500"
        ],
        "eval_ids": [
            "llama3.2:1b-instruct",
            "llama3.2:3b-instruct"
        ]
    },
    {
        "ref_id": "ref-llama33-model-card",
        "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md",
        "title": "Llama 3.3 model card",
        "snippet_path": "refs/llama33-model-card.md",
        "benchmarks": [
            "mmlu",
            "mmlu_pro",
            "gpqa_diamond",
            "ifeval",
            "humaneval",
            "math500"
        ],
        "eval_ids": [
            "llama3.3:70b-instruct"
        ]
    },
    {
        "ref_id": "ref-llama4-model-card",
        "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md",
        "title": "Llama 4 model card",
        "snippet_path": "refs/llama4-model-card.md",
        "benchmarks": [
            "mmlu_pro",
            "gpqa_diamond",
            "livecodebench",
            "mmmu"
        ],
        "eval_ids": [
            "llama4:scout-17b-instruct",
            "llama4:maverick-17b-instruct"
        ]
    },
    {
        "ref_id": "ref-qwen3-arxiv",
        "url": "https://arxiv.org/abs/2505.09388",
        "title": "Qwen3 technical report",
        "snippet_path": "refs/qwen3-arxiv.md",
        "benchmarks": [
            "mmlu_pro",
            "gpqa_diamond",
            "ifeval",
            "livecodebench",
            "math500",
            "aime24"
        ],
        "eval_ids": [
            "qwen3:8b-instruct",
            "qwen3:14b-instruct",
            "qwen3:32b-instruct",
            "qwen3:235b-a22b-instruct",
            "qwq:32b-reasoning"
        ]
    },
    {
        "ref_id": "ref-qwen35-hf-card",
        "url": "https://huggingface.co/Qwen/Qwen3.5-122B-A10B",
        "title": "Qwen3.5 HF model card",
        "snippet_path": "refs/qwen35-hf-card.md",
        "benchmarks": [
            "mmlu_pro",
            "gpqa_diamond",
            "ifeval",
            "livecodebench",
            "swe_bench"
        ],
        "eval_ids": [
            "qwen3.5:122b-a10b-instruct",
            "qwen3.5:27b-instruct"
        ]
    },
    {
        "ref_id": "ref-qwen25-coder-arxiv",
        "url": "https://arxiv.org/abs/2409.12186",
        "title": "Qwen2.5-Coder technical report",
        "snippet_path": "refs/qwen25-coder-arxiv.md",
        "benchmarks": [
            "humaneval",
            "livecodebench"
        ],
        "eval_ids": [
            "qwen2.5-coder:7b-instruct",
            "qwen2.5-coder:14b-instruct",
            "qwen2.5-coder:32b-instruct"
        ]
    },
    {
        "ref_id": "ref-gemma2-arxiv",
        "url": "https://arxiv.org/html/2408.00118",
        "title": "Gemma 2 paper",
        "snippet_path": "refs/gemma2-arxiv.md",
        "benchmarks": [
            "mmlu",
            "humaneval"
        ],
        "eval_ids": [
            "gemma2:2b-it",
            "gemma2:9b-it",
            "gemma2:27b-it"
        ]
    },
    {
        "ref_id": "ref-phi4-mini-hf",
        "url": "https://huggingface.co/microsoft/phi-4-mini-instruct",
        "title": "Phi-4-mini HF card",
        "snippet_path": "refs/phi4-mini-hf.md",
        "benchmarks": [
            "mmlu",
            "mmlu_pro",
            "gpqa",
            "math500"
        ],
        "eval_ids": [
            "phi4-mini:3.8b-instruct"
        ]
    },
    {
        "ref_id": "ref-phi4-reasoning-hf",
        "url": "https://huggingface.co/microsoft/Phi-4-reasoning",
        "title": "Phi-4-reasoning HF card",
        "snippet_path": "refs/phi4-reasoning-hf.md",
        "benchmarks": [
            "aime24",
            "gpqa_diamond",
            "livecodebench",
            "mmlu_pro",
            "humaneval",
            "ifeval"
        ],
        "eval_ids": [
            "phi4-reasoning:14b",
            "phi4-reasoning:14b-plus"
        ]
    },
    {
        "ref_id": "ref-mistral7b-arxiv",
        "url": "https://arxiv.org/pdf/2310.06825",
        "title": "Mistral 7B paper",
        "snippet_path": "refs/mistral7b-arxiv.md",
        "benchmarks": [
            "mmlu",
            "humaneval",
            "math500"
        ],
        "eval_ids": [
            "mistral:7b-instruct"
        ]
    },
    {
        "ref_id": "ref-mixtral-arxiv",
        "url": "https://arxiv.org/pdf/2401.04088",
        "title": "Mixtral 8x7B paper",
        "snippet_path": "refs/mixtral-arxiv.md",
        "benchmarks": [
            "mmlu",
            "humaneval",
            "math500"
        ],
        "eval_ids": [
            "mixtral:8x7b-instruct"
        ]
    },
    {
        "ref_id": "ref-mixtral-8x22b-blog",
        "url": "https://mistral.ai/news/mixtral-8x22b",
        "title": "Mixtral 8x22B blog",
        "snippet_path": "refs/mixtral-8x22b-blog.md",
        "benchmarks": [
            "math500"
        ],
        "eval_ids": [
            "mixtral:8x22b-instruct"
        ]
    },
    {
        "ref_id": "ref-devstral-blog",
        "url": "https://mistral.ai/news/devstral",
        "title": "Devstral blog",
        "snippet_path": "refs/devstral-blog.md",
        "benchmarks": [
            "swe_bench"
        ],
        "eval_ids": [
            "devstral:24b-instruct"
        ]
    },
    {
        "ref_id": "ref-olmo2-hf",
        "url": "https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct",
        "title": "OLMo-2 instruct HF card",
        "snippet_path": "refs/olmo2-hf.md",
        "benchmarks": [
            "mmlu",
            "ifeval",
            "math500"
        ],
        "eval_ids": [
            "olmo2:7b-instruct",
            "olmo2:13b-instruct"
        ]
    },
    {
        "ref_id": "ref-granite33-hf",
        "url": "https://huggingface.co/ibm-granite/granite-3.3-8b-instruct",
        "title": "Granite 3.3 HF card",
        "snippet_path": "refs/granite33-hf.md",
        "benchmarks": [
            "mmlu",
            "humaneval",
            "ifeval",
            "aime24",
            "math500"
        ],
        "eval_ids": [
            "granite3.3:2b-instruct",
            "granite3.3:8b-instruct"
        ]
    },
    {
        "ref_id": "ref-artificial-analysis-intelligence",
        "url": "https://artificialanalysis.ai/methodology/intelligence-benchmarking/",
        "title": "Artificial Analysis Intelligence Index methodology",
        "snippet_path": "refs/artificial-analysis-intelligence.md",
        "benchmarks": [],
        "eval_ids": [],
        "notes": "External index methodology reference for Q_base suite design; no AA scores ingested.",
    },
]

# Families where readme tables are expected (agent checklist).
README_TABLE_FAMILIES = (
    "olmo-3", "olmo-3.1", "gemma3", "gemma4", "deepseek-v2.5", "deepseek-v4-flash",
    "nemotron-3-super", "glm-4.7-flash", "qwen3.8-flash-next", "qwen3.5", "llama4",
)

_NUM = re.compile(r"[-+]?\d*\.?\d+")
_SIZE_B = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.I)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]+\)")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$", re.M)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_cell(text: str) -> str:
    text = _LINK.sub(r"\1", text.strip())
    text = _BOLD.sub(r"\1", text)
    return text.strip()


def parse_number(cell: str) -> float | None:
    cell = clean_cell(cell).replace(",", "").replace("—", "").replace("�", "").strip()
    if not cell or cell in {"-", "–", "N/A", "n/a", "--"}:
        return None
    m = _NUM.search(cell)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def normalize_bench(name: str) -> str | None:
    key = clean_cell(name).casefold()
    key = re.sub(r"\s+", " ", key)
    for canon, aliases in BENCH_ALIASES:
        if key == canon:
            return canon
        for alias in aliases:
            if key == alias or alias in key:
                # Avoid GPQA matching GPQA-Diamond
                if canon == "gpqa" and "diamond" in key:
                    continue
                if canon == "mmlu" and "pro" in key:
                    continue
                if canon == "math500" and key == "math" or key.endswith(" math"):
                    pass
                return canon
    if key == "math":
        return "math500"
    return None


def column_size_b(header: str) -> float | None:
    header_cf = clean_cell(header).casefold()
    m = _SIZE_B.search(header_cf)
    if m:
        return float(m.group(1))
    # Nemotron 3 Super style — no explicit B in header; skip
    return None


def column_variant(header: str) -> str:
    h = clean_cell(header).casefold()
    if any(x in h for x in ("distill", "r1")):
        return "distill"
    if any(x in h for x in ("instruct", " it", "-it", "chat")):
        return "instruct"
    if "vision" in h or "-vl" in h or " vl" in h:
        return "vision"
    if "embed" in h:
        return "embedding"
    if "base" in h or "pretrain" in h:
        return "base"
    return "unknown"


def slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return text or "unknown"


def eval_id_for(family: str, size_b: float | None, variant: str, col_header: str) -> str:
    size = f"{size_b:g}b" if size_b is not None else slugify(col_header)[:24]
    var = variant if variant not in ("unknown", "base") else "it"
    return f"{family}:{size}-{var}"


def iter_markdown_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|") or line.count("|") < 2:
            i += 1
            continue
        block = [line]
        i += 1
        while i < len(lines) and lines[i].strip().startswith("|"):
            block.append(lines[i].strip())
            i += 1
        if len(block) < 2:
            continue
        rows = [[clean_cell(c) for c in ln.strip("|").split("|")] for ln in block]
        # skip separator row(s)
        body = [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in r)]
        if len(body) >= 2:
            tables.append(body)
    return tables


def parse_readme_tables(family: str, readme: str, source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ref_id = f"ref-readme-{slugify(family)}"
    for table in iter_markdown_tables(readme):
        header = table[0]
        if len(header) < 2:
            continue
        bench_col = 0
        # Some tables put benchmark in col0, models in rest
        model_cols: list[tuple[int, str, float | None, str]] = []
        for idx, col in enumerate(header[1:], start=1):
            size = column_size_b(col)
            variant = column_variant(col)
            if size is None and variant == "unknown" and idx == 1 and len(header) == 2:
                # two-column score table — handled separately
                continue
            model_cols.append((idx, col, size, variant))

        if not model_cols:
            # Two-column benchmark/score layout for a single implicit subject
            benches: dict[str, float] = {}
            for row in table[1:]:
                if len(row) < 2:
                    continue
                bench = normalize_bench(row[0])
                val = parse_number(row[1])
                if bench and val is not None:
                    benches[bench] = val
            if benches:
                out.append({
                    "eval_id": f"{family}:readme-single",
                    "family": family,
                    "size_b": None,
                    "variant": "unknown",
                    "benchmarks": benches,
                    "source": source,
                    "reference_ids": [ref_id],
                    "column_header": "single-column",
                })
            continue

        by_col: dict[int, dict[str, float]] = defaultdict(dict)
        for row in table[1:]:
            if len(row) <= bench_col:
                continue
            bench = normalize_bench(row[bench_col])
            if not bench:
                continue
            for idx, col_name, size, variant in model_cols:
                if idx >= len(row):
                    continue
                val = parse_number(row[idx])
                if val is None:
                    continue
                by_col[idx][bench] = val

        for idx, col_name, size, variant in model_cols:
            benches = by_col.get(idx)
            if not benches:
                continue
            eid = eval_id_for(family, size, variant, col_name)
            out.append({
                "eval_id": eid,
                "family": family,
                "size_b": size,
                "variant": variant,
                "benchmarks": dict(benches),
                "source": source,
                "reference_ids": [ref_id],
                "column_header": col_name,
            })
    return out


def load_library_families(library_json: Path, cache_dir: Path) -> dict[str, dict[str, str]]:
    families: dict[str, dict[str, str]] = {}
    if cache_dir.is_dir():
        for path in sorted(cache_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            families[path.stem] = {
                "description": str(data.get("description") or ""),
                "readme": str(data.get("readme") or ""),
            }
    if library_json.is_file():
        blob = json.loads(library_json.read_text(encoding="utf-8"))
        for name, copy in blob.get("families", {}).items():
            families.setdefault(name, {
                "description": str(copy.get("description") or ""),
                "readme": str(copy.get("readme") or ""),
            })
    return families


def load_web_seed(path: Path) -> list[dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for subj in blob.get("subjects", []):
        rows.append({
            "eval_id": subj["eval_id"],
            "family": subj["family"],
            "size_b": subj.get("size_b"),
            "variant": subj.get("variant", "unknown"),
            "benchmarks": dict(subj.get("benchmarks") or {}),
            "source": "web_seed",
            "reference_ids": list(subj.get("reference_ids") or []),
            "notes": subj.get("notes"),
        })
    return rows


def merge_eval_rows(readme_rows: list[dict], web_rows: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in readme_rows:
        merged[row["eval_id"]] = row
    for row in web_rows:
        prev = merged.get(row["eval_id"])
        if prev:
            benches = dict(prev.get("benchmarks") or {})
            benches.update(row.get("benchmarks") or {})
            row = {**prev, **row, "benchmarks": benches, "source": "web_seed+readme"}
        merged[row["eval_id"]] = row
    return sorted(merged.values(), key=lambda r: (r["family"], r.get("size_b") or 0, r["eval_id"]))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme_reference(family: str, accessed: str, eval_ids: list[str]) -> dict[str, Any]:
    ref_id = f"ref-readme-{slugify(family)}"
    url = f"https://ollama.com/library/{family}"
    snippet_name = f"readme-{slugify(family)}.md"
    snippet_path = REFS / snippet_name
    snippet_rel = f"refs/{snippet_name}"
    snippet_path.parent.mkdir(parents=True, exist_ok=True)
    if not snippet_path.exists():
        snippet_path.write_text(
            f"# Ollama library README — {family}\n\n"
            f"- **URL:** {url}\n"
            f"- **Accessed:** {accessed}\n"
            f"- **Supports:** parsed markdown benchmark tables\n"
            f"- **Eval IDs:** {', '.join(sorted(set(eval_ids))) or 'see evals.jsonl'}\n\n"
            f"## Note\n\n"
            f"Benchmark numbers extracted automatically from the Ollama library README table.\n",
            encoding="utf-8",
        )
    return {
        "ref_id": ref_id,
        "url": url,
        "title": f"Ollama library README — {family}",
        "snippet_path": snippet_rel,
        "benchmarks": list(CANONICAL_BENCHES),
        "eval_ids": sorted(set(eval_ids)),
        "accessed": accessed,
        "schema_version": SCHEMA_VERSION,
        "source": "readme",
    }


def write_references(accessed: str, readme_refs: list[dict] | None = None) -> list[dict]:
    rows = []
    for ref in REFERENCE_SEED:
        rows.append({**ref, "accessed": accessed, "schema_version": SCHEMA_VERSION})
    rows.extend(readme_refs or [])
    write_jsonl(QUALITY / "references.jsonl", rows)
    return rows


def bench_population_stats(eval_rows: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in eval_rows:
        for bench, val in (row.get("benchmarks") or {}).items():
            if isinstance(val, (int, float)):
                buckets[bench].append(float(val))
    stats: dict[str, dict[str, float]] = {}
    for bench, vals in buckets.items():
        if len(vals) < 2:
            v = vals[0]
            stats[bench] = {"mean": v, "std": 1.0, "min": v, "max": v, "n": len(vals)}
        else:
            stats[bench] = {
                "mean": statistics.mean(vals),
                "std": max(statistics.pstdev(vals), 1e-6),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    return stats


def normalize_bench_to_index(val: float, st: dict[str, float]) -> float:
    """Map a raw benchmark score to 0–100 using population min–max (AA uses raw pass@1)."""
    lo = st["min"]
    hi = st["max"]
    if hi <= lo:
        return max(0.0, min(100.0, val))
    return max(0.0, min(100.0, 100.0 * (val - lo) / (hi - lo)))


def resolve_bench_value(benches: dict[str, float], canon: str) -> float | None:
    if canon in benches:
        return benches[canon]
    if canon == "gpqa_diamond" and "gpqa" in benches:
        return benches["gpqa"]
    if canon == "gpqa" and "gpqa_diamond" in benches:
        return benches["gpqa_diamond"]
    if canon == "mmlu_pro" and "mmlu" in benches:
        return benches["mmlu"]
    if canon == "mmlu" and "mmlu_pro" in benches:
        return benches["mmlu_pro"]
    if canon == "math500" and "math" in benches:
        return benches["math"]
    return None


def task_weights_for(row: dict) -> dict[str, float]:
    """Weights for overall Q_base (intelligence). Variant may add modality benches."""
    variant = str(row.get("variant") or "")
    if variant == "embedding" or "embed" in row.get("eval_id", ""):
        return {"mteb": 1.0}
    weights = dict(INTELLIGENCE_SUITE)
    if variant == "vision" or "vl" in row.get("eval_id", ""):
        # Multimodal reported separately in AA; add vision bench without kitchen-sink dilution.
        weights = dict(weights)
        weights["mmmu"] = 0.15
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    return weights


def weighted_suite_score(
    benches: dict,
    weights: dict[str, float],
    stats: dict[str, dict[str, float]],
    required: frozenset[str] | None = None,
) -> tuple[float | None, dict[str, float], dict[str, float]]:
    """Min–max normalize each bench, then weighted average (renormalized over available).

    Returns (score_or_None, used_raw, norm_by_bench). None when no usable benches,
    or when a required modality bench is missing.
    """
    if required:
        if not any(resolve_bench_value(benches, key) is not None for key in required):
            return None, {}, {}
    mix_sum = 0.0
    w_sum = 0.0
    norm_by_bench: dict[str, float] = {}
    used: dict[str, float] = {}
    for canon, weight in weights.items():
        val = resolve_bench_value(benches, canon)
        if val is None:
            continue
        st = stats.get(canon)
        if not st:
            continue
        norm = normalize_bench_to_index(val, st)
        norm_by_bench[canon] = round(norm, 4)
        used[canon] = val
        mix_sum += weight * norm
        w_sum += weight
    if w_sum <= 0:
        return None, {}, {}
    return mix_sum / w_sum, used, norm_by_bench


def compute_task_scores(row: dict, stats: dict[str, dict[str, float]]) -> dict[str, float]:
    """Task-specific 0–100 scores (same shape as Q_base). Omits tasks with no usable benches."""
    benches = row.get("benchmarks") or {}
    out: dict[str, float] = {}
    for task, weights in TASK_SUITES.items():
        required = TASK_REQUIRED_BENCHES.get(task)
        score, _, _ = weighted_suite_score(benches, weights, stats, required)
        if score is None:
            continue
        out[task] = round(score, 3)
    return out


def compute_q_base(row: dict, stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    benches = row.get("benchmarks") or {}
    weights = task_weights_for(row)
    mix_avg, used, norm_by_bench = weighted_suite_score(benches, weights, stats)
    if mix_avg is None:
        q_base = 50.0
        mix_avg = 0.0
    else:
        q_base = mix_avg
    q_tasks = compute_task_scores(row, stats)
    return {
        "eval_id": row["eval_id"],
        "family": row["family"],
        "size_b": row.get("size_b"),
        "variant": row.get("variant"),
        "q_base": round(q_base, 3),
        "mix_avg": round(mix_avg, 4),
        "benchmarks_used": used,
        "norm_by_bench": norm_by_bench,
        "q_tasks": q_tasks,
        "suite": "intelligence_v2",
        "source": row.get("source"),
    }


def quant_factor(quant: str, factors: dict[str, float]) -> float:
    q = quant.upper() if quant else "UNKNOWN"
    if q in factors:
        return factors[q]
    # Prefix fallback Q4_K_S -> Q4_K_M family
    for key, val in sorted(factors.items(), key=lambda kv: len(kv[0]), reverse=True):
        if q.startswith(key.split("_")[0]) and q[0] == key[0]:
            return val
    return factors.get("Q4_K_M", 0.90)


def context_score(context_length: int | None) -> float:
    if not context_length or context_length <= 0:
        return 0.0
    ctx = max(context_length, 2048)
    return max(0.0, min(1.0, math.log2(ctx / 2048) / 6.0))


def tag_size_b(tag: str, params: int | None) -> float | None:
    m = _SIZE_B.search(tag)
    if m:
        return float(m.group(1))
    if params:
        return round(params / 1e9, 2)
    return None


def tag_variant(tag: str, model: str) -> str:
    t = tag.casefold()
    if "distill" in t or "r1" in model.casefold():
        return "distill"
    if any(x in t for x in ("it", "instruct", "chat")):
        return "instruct"
    if "embed" in t:
        return "embedding"
    if "vl" in t or "vision" in t:
        return "vision"
    if "base" in t:
        return "base"
    return "unknown"


def load_catalog_models(models_path: Path) -> list[dict]:
    if models_path.suffix == ".jsonl":
        rows = []
        with models_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    blob = json.loads(models_path.read_text(encoding="utf-8"))
    return blob.get("models") or []


def index_eval_rows(rows: list[dict]) -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    return by_family


def match_eval(row: dict, eval_rows: list[dict]) -> dict | None:
    family = row["model"]
    tag = row.get("tag") or ""
    params = row.get("active_params") or row.get("param_count") or row.get("params")
    size = tag_size_b(tag, params)
    variant = tag_variant(tag, family)
    candidates = [e for e in eval_rows if e.get("size_b") is not None]
    if not candidates:
        candidates = eval_rows
    best = None
    best_key = (999.0, 999, 999)
    for ev in candidates:
        ev_size = ev.get("size_b")
        dist = abs((ev_size or 0) - (size or 0)) if size is not None and ev_size is not None else 5.0
        ev_var = ev.get("variant") or "unknown"
        var_pen = 0 if ev_var == variant else (0.5 if ev_var == "unknown" or variant == "unknown" else 2.0)
        key = (dist + var_pen, dist, 0 if ev.get("source", "").startswith("web") else 1)
        if key < best_key:
            best_key = key
            best = ev
    if best and best_key[0] > 8.0:
        return None
    return best


def ingest(
    library_json: Path,
    library_cache: Path,
    web_seed: Path,
    models_path: Path,
) -> dict[str, Any]:
    accessed = utc_now()[:10]
    families = load_library_families(library_json, library_cache)
    readme_rows: list[dict] = []
    readme_families_with_tables: set[str] = set()
    readme_eval_ids_by_family: dict[str, list[str]] = defaultdict(list)
    for name, copy in sorted(families.items()):
        readme = copy.get("readme") or ""
        if not readme:
            continue
        parsed = parse_readme_tables(name, readme, "readme")
        if parsed:
            readme_families_with_tables.add(name)
            readme_rows.extend(parsed)
            for row in parsed:
                readme_eval_ids_by_family[name].append(row["eval_id"])

    readme_refs = [
        write_readme_reference(fam, accessed, ids)
        for fam, ids in sorted(readme_eval_ids_by_family.items())
    ]
    refs = write_references(accessed, readme_refs)

    web_rows = load_web_seed(web_seed)
    eval_rows = merge_eval_rows(readme_rows, web_rows)
    write_jsonl(QUALITY / "evals.jsonl", eval_rows)

    catalog = load_catalog_models(models_path)
    catalog_families = sorted({m["model"] for m in catalog})
    eval_families = sorted({r["family"] for r in eval_rows})
    missing_families = [f for f in catalog_families if f not in eval_families]

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated": utc_now(),
        "sources": {
            "readme_rows": len(readme_rows),
            "web_seed_rows": len(web_rows),
            "merged_eval_rows": len(eval_rows),
            "readme_families_with_tables": sorted(readme_families_with_tables),
            "references": len(refs),
            "readme_references": len(readme_refs),
        },
        "catalog": {
            "models_path": str(models_path.relative_to(ROOT) if models_path.is_relative_to(ROOT) else models_path),
            "family_count": len(catalog_families),
            "eval_family_count": len(eval_families),
            "missing_eval_families": missing_families,
            "missing_eval_family_count": len(missing_families),
        },
        "expected_readme_table_families": list(README_TABLE_FAMILIES),
        "readme_table_families_missing": [
            f for f in README_TABLE_FAMILIES if f not in readme_families_with_tables
        ],
    }
    write_json(QUALITY / "ingest_summary.json", summary)
    print(
        f"[quality] ingest: readme={len(readme_rows)} web={len(web_rows)} "
        f"merged={len(eval_rows)} missing_families={len(missing_families)}",
        flush=True,
    )
    return summary


def score(models_path: Path) -> dict[str, Any]:
    eval_rows = [json.loads(ln) for ln in (QUALITY / "evals.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    stats = bench_population_stats(eval_rows)
    q_rows = [compute_q_base(row, stats) for row in eval_rows]
    write_json(QUALITY / "q_base.json", {
        "schema_version": SCHEMA_VERSION,
        "generated": utc_now(),
        "population_stats": stats,
        "rows": q_rows,
    })

    quant_doc = {
        "schema_version": SCHEMA_VERSION,
        "generated": utc_now(),
        "source": "seed",
        "factors": dict(SEED_QUANT_FACTORS),
        "default": 0.67,
        "notes": "Q_file = Q_base * F(quant). Replace with fit.py quant factors when calibrated.",
    }
    write_json(QUALITY / "quant_factors.json", quant_doc)
    factors = quant_doc["factors"]

    by_family = index_eval_rows(eval_rows)
    q_by_eval = {r["eval_id"]: r for r in q_rows}
    catalog = load_catalog_models(models_path)

    scores_by_ref: dict[str, dict] = {}
    matched = 0
    for model in catalog:
        ref = model.get("ref") or f"{model.get('model')}:{model.get('tag')}"
        ev = match_eval(model, by_family.get(model["model"], []))
        if not ev:
            continue
        q_meta = q_by_eval.get(ev["eval_id"])
        if not q_meta:
            continue
        quant = model.get("quant") or model.get("config", {}).get("file_type") or "UNKNOWN"
        if isinstance(model.get("config"), dict):
            ft = model["config"].get("file_type")
            if ft and quant in ("", "unknown", None):
                quant = str(ft)
        q_base = q_meta["q_base"]
        f_quant = quant_factor(str(quant), factors)
        q_file = round(q_base * f_quant, 3)
        q_tasks_base = q_meta.get("q_tasks") or {}
        q_tasks = {task: round(val * f_quant, 3) for task, val in q_tasks_base.items()}
        ctx = context_score(model.get("context_length"))
        # Q_rec template — speed/fit need GPU measurements; leave null placeholders.
        components = {
            "q_file_term": round(q_file * 0.50, 3),
            "speed_term": None,
            "context_term": round(ctx * 26.0, 3),
            "fit_term": None,
        }
        q_rec_partial = round(components["q_file_term"] + components["context_term"], 3)
        scores_by_ref[ref] = {
            "eval_id": ev["eval_id"],
            "family": model["model"],
            "tag": model.get("tag"),
            "quant": quant,
            "q_base": q_base,
            "quant_factor": f_quant,
            "q_file": q_file,
            "q_tasks": q_tasks,
            "context_score": round(ctx, 4),
            "speed_score": None,
            "fit_score": None,
            "q_rec_partial": q_rec_partial,
            "q_rec_formula": "q_file*50 + speed*14 + q_task*26 + fit*18 (speed/fit live in browser)",
            "components": components,
        }
        matched += 1

    scores_doc = {
        "schema_version": SCHEMA_VERSION,
        "generated": utc_now(),
        "matched_refs": matched,
        "catalog_refs": len(catalog),
        "by_ref": scores_by_ref,
    }
    write_json(QUALITY / "scores.json", scores_doc)
    print(f"[quality] score: q_base={len(q_rows)} matched_refs={matched}/{len(catalog)}", flush=True)
    return scores_doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MODELINDEX quality sidecar")
    ap.add_argument("command", choices=("ingest", "score", "all"), nargs="?", default="all")
    ap.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    ap.add_argument("--library-cache", type=Path, default=DEFAULT_LIBRARY_CACHE)
    ap.add_argument("--web-seed", type=Path, default=SEED / "web_evals.json")
    ap.add_argument("--models", type=Path, default=None, help="read-only catalog (default models.jsonl or prod/models.json)")
    args = ap.parse_args(argv)

    models_path = args.models
    if models_path is None:
        models_path = DEFAULT_MODELS if DEFAULT_MODELS.is_file() else FALLBACK_MODELS

    QUALITY.mkdir(parents=True, exist_ok=True)
    REFS.mkdir(parents=True, exist_ok=True)

    if args.command in ("ingest", "all"):
        ingest(args.library, args.library_cache, args.web_seed, models_path)
    if args.command in ("score", "all"):
        score(models_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
