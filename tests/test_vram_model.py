#!/usr/bin/env python3
"""Unit + L4 regression tests for VRAM v2. Run: python -m unittest tests.test_vram_model -v"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gguf_header import kv_slopes_from_layers  # noqa: E402
from vram_model import (  # noqa: E402
    CONSTANTS_VERSION,
    DEFAULT_BASELINE,
    FORMULA_NAME,
    GRAPH_BYTES_PER_HEAD_TOKEN,
    effective_ctx,
    fit_vram_v2,
    graph_bytes,
    kv_bytes,
    load_vram_model,
    predict_from_model_row,
    predict_vram,
    residual_after_kv_graph,
)


class TestFormulaUnits(unittest.TestCase):
    def test_effective_ctx_dense(self):
        self.assertEqual(effective_ctx(32768, 0), 32768)
        self.assertEqual(effective_ctx(4096, None), 4096)

    def test_effective_ctx_swa(self):
        self.assertEqual(effective_ctx(32768, 1024), 1024)
        self.assertEqual(effective_ctx(512, 1024), 512)  # window larger than ctx

    def test_graph_heads(self):
        # 24 heads -> 51200 B/tok; 32 -> 67584; 40 -> 83968; 64 -> 133120
        self.assertEqual(graph_bytes(1, 24), 2048 * 25)
        self.assertEqual(graph_bytes(1, 32), 2048 * 33)
        self.assertEqual(graph_bytes(1, 40), 2048 * 41)
        self.assertEqual(graph_bytes(1, 64), 2048 * 65)
        self.assertEqual(GRAPH_BYTES_PER_HEAD_TOKEN, 2048)

    def test_kv_q8_halves_only_kv(self):
        kv_f16 = kv_bytes(4096, 1000, 0, 1.0)
        kv_q8 = kv_bytes(4096, 1000, 0, 0.5)
        self.assertEqual(kv_q8, kv_f16 * 0.5)
        g = graph_bytes(4096, 32)
        # graph unchanged by kv_elem_factor
        a = predict_vram(
            weight_bytes=1e9, kv_bytes_per_token_f16=1000, n_head=32, ctx=4096,
            tensor_count=100, output_tensor_bytes=1e6, kv_elem_factor=1.0, coeffs=DEFAULT_BASELINE,
        )
        b = predict_vram(
            weight_bytes=1e9, kv_bytes_per_token_f16=1000, n_head=32, ctx=4096,
            tensor_count=100, output_tensor_bytes=1e6, kv_elem_factor=0.5, coeffs=DEFAULT_BASELINE,
        )
        self.assertAlmostEqual(a - b, kv_f16 - kv_q8, places=0)
        self.assertGreater(g, 0)

    def test_digest_offset_replaces_baseline(self):
        with_off = predict_vram(
            weight_bytes=1e9, kv_bytes_per_token_f16=100, n_head=8, ctx=2048,
            tensor_count=50, output_tensor_bytes=1e6, coeffs=DEFAULT_BASELINE, digest_offset=123456789,
        )
        # Same call but without digest: different unless by chance
        without = predict_vram(
            weight_bytes=1e9, kv_bytes_per_token_f16=100, n_head=8, ctx=2048,
            tensor_count=50, output_tensor_bytes=1e6, coeffs=DEFAULT_BASELINE,
        )
        self.assertNotEqual(with_off, without)
        # Reconstruct: W + 0 + kv + graph + digest_offset
        expected = int(round(1e9 + kv_bytes(2048, 100, 0, 1.0) + graph_bytes(2048, 8) + 123456789))
        self.assertEqual(with_off, expected)

    def test_context_cap_swa_saturates_kv(self):
        # KV stops growing after window; graph keeps growing with full ctx
        r1 = residual_after_kv_graph(3e9, 2e9, 0, 50_000, 8, 2048, 1024)
        r2 = residual_after_kv_graph(3e9, 2e9, 0, 50_000, 8, 16384, 1024)
        # residual = resident - W - kv - graph; different graphs => different residuals
        self.assertNotEqual(r1, r2)
        self.assertEqual(kv_bytes(2048, 50_000, 1024), kv_bytes(16384, 50_000, 1024))


class TestGemma4Hybrid(unittest.TestCase):
    def test_12b_slopes(self):
        n = 48
        pattern = [False if (i % 6 == 5 or i == n - 1) else True for i in range(n)]
        n_kv = [1 if not pattern[i] else 8 for i in range(n)]
        s = kv_slopes_from_layers(
            n_layer=n, n_kv=n_kv, key_length=512, value_length=512,
            key_length_swa=256, value_length_swa=256,
            sliding_window_pattern=pattern, shared_kv_layers=0,
        )
        self.assertEqual(s["n_swa_layers"], 40)
        self.assertEqual(s["n_global_layers"], 8)
        self.assertEqual(s["n_kv_alloc_layers"], 48)
        self.assertEqual(s["kv_local_bytes_per_token_f16"], 40 * 8 * (256 + 256) * 2)
        self.assertEqual(s["kv_global_bytes_per_token_f16"], 8 * 1 * (512 + 512) * 2)
        # Legacy uniform cap overstated short-ctx KV (~1.86x at 2k).
        legacy = 328 * (512 + 512) * 2 * 1024
        hybrid = kv_bytes(
            2048, s["kv_bytes_per_token_f16"], 1024, 1.0,
            s["kv_local_bytes_per_token_f16"], s["kv_global_bytes_per_token_f16"],
        )
        self.assertLess(hybrid, legacy)
        self.assertAlmostEqual(legacy / hybrid, 1.86, places=1)

    def test_e2b_shared_kv_only_prefix_allocates(self):
        n = 35
        pattern = [False if (i % 5 == 4 or i == n - 1) else True for i in range(n)]
        s = kv_slopes_from_layers(
            n_layer=n, n_kv=[1] * n, key_length=512, value_length=512,
            key_length_swa=256, value_length_swa=256,
            sliding_window_pattern=pattern, shared_kv_layers=20,
        )
        self.assertEqual(s["n_kv_alloc_layers"], 15)
        self.assertEqual(s["n_swa_layers"], 28)
        self.assertEqual(s["n_global_layers"], 7)
        none_shared = kv_slopes_from_layers(
            n_layer=n, n_kv=[1] * n, key_length=512, value_length=512,
            key_length_swa=256, value_length_swa=256,
            sliding_window_pattern=pattern, shared_kv_layers=0,
        )
        self.assertLess(s["kv_bytes_per_token_f16"], none_shared["kv_bytes_per_token_f16"])

    def test_hybrid_graph_blends_window_and_full_ctx(self):
        full = graph_bytes(131072, 16)
        hyb = graph_bytes(131072, 16, n_layer=48, n_swa_layers=40, n_global_layers=8, sliding_window=1024)
        self.assertLess(hyb, full)
        # Uniform SWA (gemma3): n_global_layers=0 keeps full-ctx graph.
        self.assertEqual(graph_bytes(131072, 16, n_layer=34, n_swa_layers=34, n_global_layers=0, sliding_window=1024), full)

    def test_ple_table_is_not_gpu_decode_weight(self):
        from gguf_header import tensor_group
        self.assertEqual(tensor_group("per_layer_token_embd.weight"), "ple")
        self.assertEqual(tensor_group("token_embd.weight"), "token_embd")
        self.assertEqual(tensor_group("blk.0.attn_q.weight"), "attn")

    def test_e2b_vram_excludes_ple_table(self):
        row = {
            "weight_bytes": 7_162_394_016,
            "gpu_weight_bytes": 7_162_394_016 - 4_697_600_000,
            "ple_bytes": 4_697_600_000,
            "projector_bytes": 0,
            "kv_bytes_per_token_f16": 18432,
            "kv_local_bytes_per_token_f16": 12288,
            "kv_global_bytes_per_token_f16": 6144,
            "n_head": 8,
            "n_layer": 35,
            "n_swa_layers": 28,
            "n_global_layers": 7,
            "sliding_window": 512,
            "gpu_tensor_count": 400,
            "tensor_count": 2012,
            "tensor_groups_bytes": {"output": 0},
        }
        vm = {"formula": FORMULA_NAME, "coeffs": DEFAULT_BASELINE, "digest_offsets": {}}
        pred = predict_from_model_row(row, 8192, vm)
        self.assertLess(pred / 1e9, 4.0)
        self.assertGreater(pred / 1e9, 2.0)
        row = {
            "weight_bytes": 7_381_000_000,
            "projector_bytes": 0,
            "kv_bytes_per_token_f16": 344064,
            "kv_local_bytes_per_token_f16": 327680,
            "kv_global_bytes_per_token_f16": 16384,
            "n_head": 16,
            "n_layer": 48,
            "n_swa_layers": 40,
            "n_global_layers": 8,
            "sliding_window": 1024,
            "tensor_count": 667,
            "tensor_groups_bytes": {"output": 0},
        }
        vm = {"formula": FORMULA_NAME, "coeffs": DEFAULT_BASELINE, "digest_offsets": {}}
        pred = predict_from_model_row(row, 2048, vm)
        legacy_row = dict(row)
        legacy_row["kv_local_bytes_per_token_f16"] = None
        legacy_row["kv_global_bytes_per_token_f16"] = None
        legacy_row["n_swa_layers"] = 0
        legacy_row["n_global_layers"] = 0
        legacy_row["kv_bytes_per_token_f16"] = 671744
        legacy = predict_from_model_row(legacy_row, 2048, vm)
        self.assertLess(pred, legacy)


class TestConstantsVersion(unittest.TestCase):
    def test_reject_legacy(self):
        vm = load_vram_model({"constants_version": 1, "arch": {"llama": {"g0_bytes": 1, "kv_ratio": 1.5}}})
        self.assertTrue(vm.get("legacy"))

    def test_accept_v2(self):
        vm = load_vram_model({
            "constants_version": CONSTANTS_VERSION,
            "vram_model": {"formula": FORMULA_NAME, "coeffs": DEFAULT_BASELINE, "digest_offsets": {}},
        })
        self.assertEqual(vm["formula"], FORMULA_NAME)
        self.assertFalse(vm.get("legacy"))

    def test_build_index_rejects_v1(self):
        import build_index
        with self.assertRaises(SystemExit):
            build_index.resolve_vram_model({"constants_version": 1})


class TestFitSynthetic(unittest.TestCase):
    def test_fit_recovers_and_excludes_nothing_partial(self):
        # Two structural models, residuals explained by tensor/output coeffs
        pts = []
        for dig, n_head, tc, outb, W in [
            ("sha256:aaa", 8, 100, 1_000_000, 2_000_000_000),
            ("sha256:bbb", 32, 200, 5_000_000, 8_000_000_000),
        ]:
            for ctx in (2048, 4096, 8192):
                base = DEFAULT_BASELINE["b0"] + DEFAULT_BASELINE["b_tensor"] * tc + DEFAULT_BASELINE["b_output"] * outb
                resident = W + kv_bytes(ctx, 10_000, 0) + graph_bytes(ctx, n_head) + base
                pts.append({
                    "model": dig, "digest": dig, "arch": "llama", "ctx": ctx, "resident": resident,
                    "W": W, "proj": 0, "kv_per_token": 10_000, "n_head": n_head, "n_head_kv": 8,
                    "n_layer": 32, "key_length": 128, "value_length": 128, "sliding_window": 0,
                    "tensor_count": tc, "output_tensor_bytes": outb, "expert_count": 0,
                    "param_count": W // 2,
                })
        vfit = fit_vram_v2(pts)
        self.assertLessEqual(vfit["max_rel_err"], 0.01)
        self.assertEqual(vfit["n_points"], 6)
        self.assertEqual(vfit["n_digests"], 2)


@unittest.skipUnless(
    (ROOT / "data/out/measurements/l4-20260908T150419Z-main.jsonl").exists()
    and (ROOT / "data/out/models.jsonl").exists(),
    "L4 measurements / models.jsonl not present",
)
class TestL4Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fit as fit_mod
        cls.fit_mod = fit_mod
        meas = str(ROOT / "data/out/measurements/l4-20260908T150419Z-main.jsonl")
        kv = ROOT / "data/out/measurements/l4-20260908T150419Z-kvq8.jsonl"
        paths = [meas]
        if kv.exists():
            paths.append(str(kv))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "constants.json"
            rc = fit_mod.main([
                "--measurements", *paths,
                "--models", str(ROOT / "data/out/models.jsonl"),
                "--gpus", str(ROOT / "data/gpus.csv"),
                "--out", str(out),
            ])
            cls.assertIs = None
            assert rc == 0
            cls.constants = json.loads(out.read_text(encoding="utf-8"))

    def test_version_and_gates(self):
        self.assertEqual(self.constants["constants_version"], CONSTANTS_VERSION)
        vm = self.constants["vram_model"]
        self.assertEqual(vm["formula"], FORMULA_NAME)
        self.assertLessEqual(vm["train_global"]["max_rel_err"], 0.05)
        hold = vm.get("grouped_holdout") or {}
        if hold.get("max_rel_err") is not None:
            self.assertLessEqual(hold["max_rel_err"], 0.05)
        fwd = vm.get("forward_context") or {}
        if fwd.get("max_rel_err") is not None:
            self.assertLessEqual(fwd["max_rel_err"], 0.02)
        dig = vm.get("train_digest") or {}
        if dig.get("max_rel_err") is not None:
            self.assertLessEqual(dig["max_rel_err"], 0.02)

        gpu = next(iter(self.constants["gpus"].values()))
        self.assertTrue(gpu["gates"]["G4.2_vram_max_rel_err_le_0.05"])
        # Partial points must not be in the full-demand fit count relative to main vram rows
        self.assertGreaterEqual(vm["n_points"], 50)

    def test_partial_excluded(self):
        # qwen3:30b-a3b @ 32768 is partial on L4; must not appear as a training digest-only issue
        # Verify via re-building points: fully_on_gpu False excluded
        recs = self.fit_mod.load_jsonl([str(ROOT / "data/out/measurements/l4-20260908T150419Z-main.jsonl")])
        partial = [r for r in recs if r.get("kind") == "vram" and r.get("model") == "qwen3:30b-a3b"
                   and r.get("ctx") == 32768 and r.get("fully_on_gpu") is False]
        self.assertEqual(len(partial), 1)
        # n_points should be total full vram rows only
        full = [r for r in recs if r.get("kind") == "vram" and r.get("fully_on_gpu") is True
                and (r.get("label") or "main") in ("main", "local")]
        self.assertEqual(self.constants["vram_model"]["n_points"], len(full))

    def test_predict_from_row(self):
        models = self.fit_mod.load_models(str(ROOT / "data/out/models.jsonl"))
        row = models["llama3.2:3b"]
        vm = self.constants["vram_model"]
        pred = predict_from_model_row(row, 4096, vm)
        self.assertGreater(pred, row["weight_bytes"])


if __name__ == "__main__":
    unittest.main()
