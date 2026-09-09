#!/usr/bin/env python3
"""Unit tests for scripts/quality.py."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import quality  # noqa: E402


class TestBenchNormalization(unittest.TestCase):
    def test_gpqa_diamond_not_plain_gpqa(self):
        self.assertEqual(quality.normalize_bench("GPQA-Diamond"), "gpqa_diamond")
        self.assertEqual(quality.normalize_bench("GPQA"), "gpqa")

    def test_mmlu_pro_not_plain_mmlu(self):
        self.assertEqual(quality.normalize_bench("MMLU-Pro"), "mmlu_pro")
        self.assertEqual(quality.normalize_bench("MMLU"), "mmlu")

    def test_aime_and_math_aliases(self):
        self.assertEqual(quality.normalize_bench("AIME 2024"), "aime24")
        self.assertEqual(quality.normalize_bench("MATH-500"), "math500")
        self.assertEqual(quality.normalize_bench("MATH"), "math500")


class TestTableParsing(unittest.TestCase):
    def test_two_column_single_subject(self):
        md = """
| Benchmark | Score |
|-----------|------:|
| MMLU-Pro | 67.5 |
| GPQA-Diamond | 42.4 |
"""
        rows = quality.parse_readme_tables("demo", md, "readme")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["benchmarks"]["mmlu_pro"], 67.5)
        self.assertAlmostEqual(rows[0]["benchmarks"]["gpqa_diamond"], 42.4)

    def test_multi_column_olmo_style(self):
        md = """
| **Benchmark** | **Olmo3 Instruct 7B** | **Qwen 2.5 7B** |
|:---|---:|---:|
| MATH | 87.3 | 71 |
| AIME 2024 | 44.3 | 11.3 |
"""
        rows = quality.parse_readme_tables("olmo-3", md, "readme")
        by_header = {r["column_header"]: r for r in rows}
        self.assertAlmostEqual(by_header["Olmo3 Instruct 7B"]["benchmarks"]["math500"], 87.3)
        self.assertAlmostEqual(by_header["Qwen 2.5 7B"]["benchmarks"]["aime24"], 11.3)


class TestScoring(unittest.TestCase):
    def test_q_base_minmax_ordering(self):
        stats = {
            "mmlu_pro": {"mean": 50.0, "std": 10.0, "min": 30.0, "max": 70.0, "n": 3},
            "gpqa_diamond": {"mean": 40.0, "std": 10.0, "min": 20.0, "max": 50.0, "n": 3},
        }
        low = quality.compute_q_base(
            {"eval_id": "x", "family": "f", "variant": "instruct", "benchmarks": {"mmlu_pro": 30.0, "gpqa_diamond": 20.0}},
            stats,
        )
        high = quality.compute_q_base(
            {"eval_id": "y", "family": "f", "variant": "instruct", "benchmarks": {"mmlu_pro": 70.0, "gpqa_diamond": 50.0}},
            stats,
        )
        self.assertLess(low["q_base"], high["q_base"])
        self.assertGreaterEqual(low["q_base"], 0.0)
        self.assertLessEqual(high["q_base"], 100.0)
        self.assertEqual(low["suite"], "intelligence_v2")

    def test_normalize_bench_to_index(self):
        st = {"min": 0.0, "max": 100.0, "mean": 50.0, "std": 10.0, "n": 5}
        self.assertAlmostEqual(quality.normalize_bench_to_index(0.0, st), 0.0)
        self.assertAlmostEqual(quality.normalize_bench_to_index(100.0, st), 100.0)
        self.assertAlmostEqual(quality.normalize_bench_to_index(50.0, st), 50.0)

    def test_quant_factor_degrades_q_file(self):
        q_base = 60.0
        f16 = quality.quant_factor("F16", quality.SEED_QUANT_FACTORS)
        q4 = quality.quant_factor("Q4_K_M", quality.SEED_QUANT_FACTORS)
        self.assertAlmostEqual(f16, 1.0)
        self.assertLess(q4, f16)
        self.assertAlmostEqual(q_base * q4, q_base * 0.90)

    def test_context_score_monotonic(self):
        self.assertLess(quality.context_score(2048), quality.context_score(8192))
        self.assertLessEqual(quality.context_score(10_000_000), 1.0)

    def test_task_scores_code_vs_chat(self):
        stats = {
            "mmlu_pro": {"mean": 50.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "math500": {"mean": 50.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "gpqa_diamond": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "aime25": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "aime24": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "livecodebench": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "humaneval": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "swe_bench": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "bfcl": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "mmmu": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
            "mteb": {"mean": 50.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 3},
        }
        coder = quality.compute_q_base(
            {
                "eval_id": "coder",
                "family": "f",
                "variant": "instruct",
                "benchmarks": {
                    "livecodebench": 90.0,
                    "humaneval": 90.0,
                    "swe_bench": 80.0,
                    "mmlu_pro": 40.0,
                },
            },
            stats,
        )
        chatty = quality.compute_q_base(
            {
                "eval_id": "chatty",
                "family": "f",
                "variant": "instruct",
                "benchmarks": {
                    "livecodebench": 20.0,
                    "humaneval": 20.0,
                    "swe_bench": 20.0,
                    "mmlu_pro": 90.0,
                    "math500": 90.0,
                    "gpqa_diamond": 80.0,
                },
            },
            stats,
        )
        self.assertIn("code", coder["q_tasks"])
        self.assertIn("chat", coder["q_tasks"])
        self.assertGreater(coder["q_tasks"]["code"], chatty["q_tasks"]["code"])
        # Strong coding benches also lift chat (coding is part of intelligence);
        # a knowledge-heavy model should still beat a code specialist on long.
        self.assertIn("long", chatty["q_tasks"])
        self.assertGreater(chatty["q_tasks"]["long"], coder["q_tasks"]["long"])

    def test_vision_task_requires_mmmu(self):
        stats = {
            "mmlu_pro": {"mean": 50.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 2},
            "mmmu": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0, "n": 2},
        }
        no_vision = quality.compute_task_scores(
            {"eval_id": "t", "benchmarks": {"mmlu_pro": 80.0}},
            stats,
        )
        with_vision = quality.compute_task_scores(
            {"eval_id": "t", "benchmarks": {"mmlu_pro": 80.0, "mmmu": 70.0}},
            stats,
        )
        self.assertNotIn("vision", no_vision)
        self.assertIn("vision", with_vision)

    def test_capability_age_decay_half_life(self):
        # Mirrors web/app.js QUALITY_AGE_HALF_LIFE_YEARS (arXiv:2603.28576).
        half_life = 1.1
        decay = lambda t: math.exp(-math.log(2) * t / half_life)
        self.assertAlmostEqual(decay(0.0), 1.0)
        self.assertAlmostEqual(decay(half_life), 0.5)
        self.assertAlmostEqual(decay(2 * half_life), 0.25)
        self.assertLess(decay(2.0), decay(1.0))


class TestPipelineSmoke(unittest.TestCase):
    def test_ingest_and_score_on_minimal_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lib = tmp_path / "library.json"
            lib.write_text(json.dumps({
                "families": {
                    "gemma3": {
                        "description": "Gemma 3",
                        "readme": "| Benchmark | Gemma 3 4B IT |\n|---|---:|\n| MMLU-Pro | 43.6 |\n| IFEval | 80.2 |\n",
                    }
                }
            }), encoding="utf-8")
            seed = tmp_path / "web_evals.json"
            seed.write_text(json.dumps({"subjects": []}), encoding="utf-8")
            models = tmp_path / "models.json"
            models.write_text(json.dumps({"models": [{
                "ref": "gemma3:4b-it-q4_K_M",
                "model": "gemma3",
                "tag": "4b-it-q4_K_M",
                "quant": "Q4_K_M",
                "context_length": 8192,
                "param_count": 4_000_000_000,
                "active_params": 4_000_000_000,
            }]}), encoding="utf-8")

            out = tmp_path / "quality"
            out.mkdir()
            quality.QUALITY = out
            quality.REFS = out / "refs"
            quality.REFS.mkdir()

            quality.ingest(lib, tmp_path / "cache", seed, models)
            doc = quality.score(models)

            self.assertTrue((out / "evals.jsonl").is_file())
            self.assertTrue((out / "references.jsonl").is_file())
            self.assertGreaterEqual(doc["matched_refs"], 1)
            sample = next(iter(doc["by_ref"].values()))
            self.assertIn("q_tasks", sample)
            self.assertIn("chat", sample["q_tasks"])


if __name__ == "__main__":
    unittest.main()
