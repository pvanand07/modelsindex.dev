#!/usr/bin/env python3
"""Tests for the compact production-data export."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_prod_data import (  # noqa: E402  # pylint: disable=import-error
    build,
    canonical_key,
    compact_link_content,
    compact_quality,
    deduplicate_models,
    lookup_score,
    main,
    resolve_links,
)
from gguf_header import normalize_quant  # noqa: E402  # pylint: disable=import-error


def model(ref: str, digest: str, *, tag: str | None = None, params: int = 3_000_000_000) -> dict:
    name, inferred_tag = ref.split(":", 1)
    return {
        "ref": ref,
        "model": name,
        "tag": tag or inferred_tag,
        "digest": digest,
        "arch": "llama",
        "quant": "Q4_K_M",
        "param_count": params,
        "active_params": params,
        "weight_bytes": 2_000_000_000,
        "active_weight_bytes": 2_000_000_000,
        "projector_bytes": 0,
        "n_layer": 28,
        "context_length": 8192,
        "expert_count": 0,
        "expert_used_count": 0,
        "sliding_window": 0,
        "vram_digest_calibrated": False,
        "vram_bytes_at_ctx": {"2048": 2_200_000_000, "4096": 2_300_000_000},
    }


class TestDeduplication(unittest.TestCase):
    def test_default_tag_is_stable_canonical(self):
        rows = [
            model("demo:3b-q4_K_M", "sha256:a"),
            model("demo:latest", "sha256:a"),
            model("demo:3b", "sha256:a"),
        ]
        self.assertEqual(min(rows, key=canonical_key)["ref"], "demo:latest")
        compact = deduplicate_models(list(reversed(rows)))
        self.assertEqual(compact[0]["ref"], "demo:latest")
        self.assertEqual(compact[0]["aliases"], ["demo:3b", "demo:3b-q4_K_M"])

    def test_description_is_family_copy(self):
        older = model("demo:latest", "sha256:a")
        older["description"] = "A short family blurb."
        compact = deduplicate_models([older])
        self.assertEqual(compact[0]["description"], "A short family blurb.")

    def test_pushed_at_uses_latest_alias(self):
        older = model("demo:latest", "sha256:a")
        older["pushed_at"] = "2024-01-01T00:00:00Z"
        newer = model("demo:3b", "sha256:a")
        newer["pushed_at"] = "2025-06-01T00:00:00Z"
        compact = deduplicate_models([older, newer])
        self.assertEqual(compact[0]["pushed_at"], "2025-06-01T00:00:00Z")

    def test_capability_signals_are_conservative(self):
        coding = model("starcoder2:latest", "sha256:code")
        embedding = model("nomic-embed-text:latest", "sha256:embed")
        rows = deduplicate_models([coding, embedding])
        by_ref = {row["ref"]: row for row in rows}
        self.assertIn("code", by_ref["starcoder2:latest"]["signals"])
        self.assertIn("embedding", by_ref["nomic-embed-text:latest"]["signals"])

    def test_release_field_present_and_quant_variants_share_it(self):
        """Two K-quant tags of the same checkpoint must produce the same release key -- this is
        the regression guard for the QUANT_SUFFIX two-segment (q5_K_M-style) bugfix."""
        fp16 = model("llava:13b-v1.5-fp16", "sha256:a", tag="13b-v1.5-fp16")
        kquant = model("llava:13b-v1.5-q5_K_M", "sha256:a", tag="13b-v1.5-q5_K_M")
        compact = deduplicate_models([fp16, kquant])
        self.assertEqual(compact[0]["release"], "llava:13b-v1.5")

    def test_release_differs_for_different_sizes(self):
        seven_b = model("llava:7b", "sha256:x", tag="7b")
        thirteen_b = model("llava:13b", "sha256:y", tag="13b")
        rows = deduplicate_models([seven_b, thirteen_b])
        by_ref = {row["ref"]: row for row in rows}
        self.assertEqual(by_ref["llava:7b"]["release"], "llava:7b")
        self.assertEqual(by_ref["llava:13b"]["release"], "llava:13b")
        self.assertNotEqual(by_ref["llava:7b"]["release"], by_ref["llava:13b"]["release"])


class TestLinksExport(unittest.TestCase):
    def test_digest_verified_beats_release_and_family_likely(self):
        hf_sources = {
            "by_digest": {"sha256:a": {"repo": "org/exact", "url": "https://huggingface.co/org/exact"}},
            "by_release": {"demo:13b": {"repo": "org/release-guess", "url": "https://huggingface.co/org/release-guess", "method": "readme"}},
            "by_family": {"demo": {"repo": "org/guess", "url": "https://huggingface.co/org/guess", "method": "readme"}},
        }
        links = resolve_links("sha256:a", "demo", "demo:13b", hf_sources, None)
        self.assertEqual(links["hf"]["release"]["confidence"], "verified")
        self.assertEqual(links["hf"]["release"]["repo"], "org/exact")
        self.assertEqual(links["hf"]["family"]["confidence"], "verified")
        self.assertEqual(links["hf"]["family"]["repo"], "org/exact")

    def test_release_likely_used_when_digest_has_no_hit(self):
        hf_sources = {
            "by_digest": {},
            "by_release": {"demo:13b": {"repo": "org/release-guess", "url": "https://huggingface.co/org/release-guess", "method": "readme_verified"}},
            "by_family": {"demo": {"repo": "org/family-guess", "url": "https://huggingface.co/org/family-guess", "method": "readme_verified"}},
        }
        links = resolve_links("sha256:b", "demo", "demo:13b", hf_sources, None)
        self.assertEqual(links["hf"]["release"]["repo"], "org/release-guess")
        self.assertEqual(links["hf"]["release"]["confidence"], "likely")

    def test_release_and_family_hf_can_differ_and_both_survive(self):
        """The core requirement: a release-specific pick never silently discards a differing
        family-wide guess -- both are kept so a consumer can choose which to trust."""
        hf_sources = {
            "by_digest": {},
            "by_release": {"llava:13b": {"repo": "liuhaotian/llava-v1.5-13b", "url": "https://huggingface.co/liuhaotian/llava-v1.5-13b", "method": "readme_verified"}},
            "by_family": {"llava": {"repo": "liuhaotian/llava-v1.5-7b", "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b", "method": "readme_verified"}},
        }
        links = resolve_links("sha256:c", "llava", "llava:13b", hf_sources, None)
        self.assertEqual(links["hf"]["release"]["repo"], "liuhaotian/llava-v1.5-13b")
        self.assertEqual(links["hf"]["family"]["repo"], "liuhaotian/llava-v1.5-7b")
        self.assertNotEqual(links["hf"]["release"]["repo"], links["hf"]["family"]["repo"])

    def test_family_likely_used_when_release_has_no_hit(self):
        hf_sources = {
            "by_digest": {},
            "by_release": {},
            "by_family": {"demo": {"repo": "org/guess", "url": "https://huggingface.co/org/guess", "method": "readme_verified"}},
        }
        links = resolve_links("sha256:b", "demo", "demo:13b", hf_sources, None)
        self.assertIsNone(links["hf"]["release"])
        self.assertEqual(links["hf"]["family"]["repo"], "org/guess")
        self.assertEqual(links["hf"]["family"]["method"], "readme_verified")

    def test_links_fallback_hf_used_only_when_hf_source_has_nothing(self):
        """scripts/links.py's fallback (github/homepage-content scan) is inherently
        release-scoped -- it never produces a family-wide guess, so `family` stays null here
        even though `release` resolves via the fallback."""
        links_data = {
            "by_family": {
                "demo": {
                    "hf_by_release": {"demo:13b": {"repo": "org/fallback", "url": "https://huggingface.co/org/fallback", "confidence": "likely", "method": "homepage_llm"}},
                    "github": {"repo": "org/demo", "url": "https://github.com/org/demo", "confidence": "likely", "method": "readme"},
                    "homepage": None,
                    "paper": None,
                }
            }
        }
        resolved = resolve_links("sha256:c", "demo", "demo:13b", None, links_data)
        self.assertEqual(resolved["hf"]["release"]["method"], "homepage_llm")
        self.assertEqual(resolved["hf"]["release"]["repo"], "org/fallback")
        self.assertIsNone(resolved["hf"]["family"])
        self.assertEqual(resolved["github"]["repo"], "org/demo")
        self.assertIsNone(resolved["homepage"])
        self.assertIsNone(resolved["paper"])

    def test_no_links_at_all_is_all_none(self):
        resolved = resolve_links("sha256:d", "demo", "demo:13b", None, None)
        self.assertEqual(resolved, {"hf": {"release": None, "family": None}, "github": None, "homepage": None, "paper": None})

    def test_compact_link_content_drops_paper_and_unknown_families(self):
        link_content = {
            # hf content is keyed by repo (not release) -- multiple releases can share one repo.
            "demo": {"hf": {"org/demo-7b": {"url": "https://huggingface.co/org/demo-7b", "content": "card text"}}},
            "other": {"github": {"url": "https://github.com/org/other", "content": "readme text"}},
        }
        out = compact_link_content(link_content, {"demo"})
        self.assertIn("demo", out)
        self.assertNotIn("other", out)
        self.assertNotIn("paper", out["demo"])
        self.assertIn("org/demo-7b", out["demo"]["hf"])

    def test_compact_link_content_skips_empty_families(self):
        out = compact_link_content({"demo": {}}, {"demo"})
        self.assertEqual(out, {})


class TestQuantNormalization(unittest.TestCase):
    def test_unknown_fp16_tag_becomes_f16(self):
        self.assertEqual(normalize_quant("unknown", "7b-instruct-fp16"), "F16")
        self.assertEqual(normalize_quant("unknown", "qwen:7b-fp16"), "F16")

    def test_known_names_are_canonical(self):
        self.assertEqual(normalize_quant("Q4_K_M", "latest"), "Q4_K_M")
        self.assertEqual(normalize_quant("fp16", "latest"), "F16")
        self.assertEqual(normalize_quant(1, "latest"), "F16")

    def test_placeholder_without_hint_is_none(self):
        self.assertIsNone(normalize_quant("unknown", "cloud"))
        self.assertIsNone(normalize_quant(None, "latest"))


class TestQualityExport(unittest.TestCase):
    def test_lookup_via_alias(self):
        compact = {
            "ref": "llama3.1:8b",
            "aliases": ["llama3.1:8b-instruct-q4_K_M"],
            "model": "llama3.1",
        }
        by_ref = {
            "llama3.1:8b-instruct-q4_K_M": {
                "eval_id": "llama3.1:8b-instruct",
                "q_base": 55.0,
                "q_file": 49.5,
                "quant_factor": 0.9,
            },
        }
        hit = lookup_score(compact, by_ref)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["q_file"], 49.5)

    def test_compact_quality_keys_canonical_refs(self):
        models = deduplicate_models([
            model("llama3.1:8b-instruct-q4_K_M", "sha256:a", params=8_000_000_000),
            model("llama3.1:8b", "sha256:a", params=8_000_000_000),
            model("alfred:40b-1023-q4_1", "sha256:b", params=40_000_000_000),
        ])
        scores = {
            "generated": "2026-01-01T00:00:00Z",
            "by_ref": {
                "llama3.1:8b": {
                    "eval_id": "llama3.1:8b-instruct",
                    "q_base": 55.0,
                    "q_file": 49.5,
                    "q_tasks": {"chat": 49.5, "code": 42.0},
                    "quant_factor": 0.9,
                },
            },
        }
        q_base = {
            "rows": [{"eval_id": "llama3.1:8b-instruct", "source": "web_seed"}],
        }
        quality = compact_quality(models, scores, q_base)
        self.assertEqual(quality["quality_refs"], 1)
        self.assertIn("llama3.1", quality["quality_families"])
        self.assertNotIn("alfred:40b-1023-q4_1", quality["by_ref"])
        row = quality["by_ref"]["llama3.1:8b"]
        self.assertEqual(row["q_file"], 49.5)
        self.assertEqual(row["q_tasks"]["code"], 42.0)
        self.assertEqual(row["quality_source"], "web_seed")

    def test_empty_scores_exports_empty_map(self):
        models = deduplicate_models([model("demo:latest", "sha256:a")])
        quality = compact_quality(models, None)
        self.assertEqual(quality["quality_refs"], 0)
        self.assertEqual(quality["by_ref"], {})


@unittest.skipUnless(
    (ROOT / "data/out/index.json").exists() and (ROOT / "data/out/constants.json").exists(),
    "generated index artifacts are not available",
)
class TestProductionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads((ROOT / "data/out/index.json").read_text(encoding="utf-8"))
        cls.constants = json.loads((ROOT / "data/out/constants.json").read_text(encoding="utf-8"))
        scores_path = ROOT / "data/quality/scores.json"
        q_base_path = ROOT / "data/quality/q_base.json"
        cls.scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else None
        cls.q_base = json.loads(q_base_path.read_text(encoding="utf-8")) if q_base_path.exists() else None
        cls.manifest, cls.hardware, cls.catalog, cls.library, cls.quality, cls.link_content = build(
            cls.index, cls.constants, scores=cls.scores, q_base=cls.q_base,
        )

    def test_schema_counts_and_calibration(self):
        unique_digests = {row["digest"] for row in self.index["models"]}
        self.assertEqual(self.manifest["unique_models"], len(unique_digests))
        self.assertEqual(len(self.catalog["models"]), len(unique_digests))
        self.assertEqual(self.manifest["gpu_count"], len(self.index["gpus"]))
        self.assertEqual(self.manifest["calibrated_gpus"], ["l4"])
        l4 = next(gpu for gpu in self.hardware["gpus"] if gpu["id"] == "l4")
        self.assertTrue(l4["estimate"]["calibrated"])
        self.assertEqual(self.manifest["quality_refs"], self.quality["quality_refs"])
        self.assertEqual(self.manifest["quality_families"], len(self.quality["quality_families"]))

    def test_unknown_quant_is_cleaned(self):
        quants = {row.get("quant") for row in self.catalog["models"]}
        self.assertNotIn("unknown", quants)
        self.assertNotIn("UNKNOWN", quants)
        gemma = next(row for row in self.catalog["models"] if row["ref"] == "gemma:7b-instruct-fp16")
        self.assertEqual(gemma["quant"], "F16")

    def test_vram_curve_matches_source(self):
        source_by_digest = {row["digest"]: row for row in self.index["models"]}
        for compact in self.catalog["models"][::500]:
            self.assertEqual(
                compact["vram_bytes_at_ctx"],
                source_by_digest[compact["digest"]]["vram_bytes_at_ctx"],
            )

    def test_quality_beats_size_prior_for_known_pair(self):
        if not self.scores:
            self.skipTest("quality scores unavailable")
        by_ref = self.quality["by_ref"]
        llama = next((m for m in self.catalog["models"] if m["ref"].startswith("llama3.1:8b")), None)
        alfred = next((m for m in self.catalog["models"] if m["ref"].startswith("alfred:")), None)
        if not llama or llama["ref"] not in by_ref:
            self.skipTest("llama3.1:8b quality row missing")
        if alfred and alfred["ref"] in by_ref:
            self.assertGreater(by_ref[llama["ref"]]["q_file"], by_ref[alfred["ref"]]["q_file"])
        else:
            # alfred should remain unmatched (size prior only)
            self.assertNotIn(alfred["ref"] if alfred else "", by_ref)
            self.assertGreater(by_ref[llama["ref"]]["q_file"], 0)

    def test_export_is_compact_and_repeatable(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            args = [
                "--index", str(ROOT / "data/out/index.json"),
                "--constants", str(ROOT / "data/out/constants.json"),
            ]
            self.assertEqual(main([*args, "--out-dir", first]), 0)
            self.assertEqual(main([*args, "--out-dir", second]), 0)
            for name in ("manifest.json", "gpus.json", "models.json", "library.json", "quality.json", "link_content.json"):
                a = (Path(first) / name).read_bytes()
                b = (Path(second) / name).read_bytes()
                self.assertEqual(a, b)
            # Ceiling raised from 7MB: each row's links.hf now carries a {release, family}
            # envelope (two link objects) instead of one flat object, plus a new "release"
            # field -- a real, intentional size increase from the release-scoping change, not
            # unbounded growth. Re-check this ceiling if it starts getting tight again.
            self.assertLess((Path(first) / "models.json").stat().st_size, 9_000_000)
            quality = json.loads((Path(first) / "quality.json").read_text(encoding="utf-8"))
            self.assertIn("by_ref", quality)
            self.assertGreater(quality["quality_refs"], 0)


if __name__ == "__main__":
    unittest.main()