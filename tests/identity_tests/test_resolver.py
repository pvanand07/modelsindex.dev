"""Offline contract and regression tests for the independent identity resolver."""
import json
from pathlib import Path
import tempfile
import unittest

from identity.resolver import alias_constraints, build, clean_repo, evaluate, inventory, release_key, resolve
from identity.compare import compare

D1 = "sha256:" + "a" * 64
D2 = "sha256:" + "b" * 64


class IdentityTests(unittest.TestCase):
    def test_quant_release_and_variant_preservation(self):
        self.assertEqual(release_key("demo", "7b-thinking-2507-q4_K_M"), "demo:7b-thinking-2507")
        self.assertEqual(release_key("demo", "q4_K_M"), "demo:default")
        self.assertEqual(release_key("demo", "8x7b-iq4_xs"), "demo:8x7b")

    def test_exact_sizes(self):
        for tag, repo in [("7b", "org/demo-27B"), ("3b", "org/demo-13B"),
                          ("7b", "org/demo-8x7B"), ("1.5b", "org/demo-11.5B")]:
            with self.subTest(tag=tag, repo=repo):
                self.assertIn("size_mismatch_or_missing", evaluate("demo", tag, repo, "card"))
        self.assertEqual(evaluate("demo", "8x7b", "org/demo-8x7B", "card"), [])

    def test_variant_and_version_constraints(self):
        self.assertIn("variant_mismatch_or_missing", evaluate("demo", "4b-thinking-2507", "org/demo-4B-Instruct-2507", "card"))
        self.assertIn("release_qualifier_missing", evaluate("demo", "4b-instruct-2507", "org/demo-4B-Instruct-2508", "card"))
        self.assertIn("variant_unspecified", evaluate("demo", "4b", "org/demo-4B-Instruct", "card"))

    def test_unsupported_and_generic_stay_unresolved(self):
        self.assertIn("unspecified_release", evaluate("demo", "latest", "org/demo", "card"))
        self.assertIn("model_card_unavailable", evaluate("demo", "7b", "org/demo-7B", None))
        self.assertIn("converted_repository_not_canonical", evaluate("demo", "7b", "org/demo-7B-GGUF", "card"))

    def test_url_host_validation(self):
        for url in ["https://evil.test/huggingface.co/org/demo", "https://huggingface.co.evil.test/org/demo", "https://huggingface.co/datasets/org/demo"]:
            self.assertIsNone(clean_repo(url))
        self.assertEqual(clean_repo("https://huggingface.co/org/demo/blob/main/README.md"), "org/demo")

    def test_inventory_keeps_aliases(self):
        artifacts, releases = inventory([{"digest": D1, "model": "demo", "tag": "latest", "aliases": ["demo:7b-q4_K_M"]}])
        self.assertEqual(set(releases), {"demo:latest", "demo:7b"})
        self.assertEqual(len(artifacts[D1]["refs"]), 2)

    def test_ambiguous_candidates_and_cache_invalidation(self):
        candidates = [{"repo": "org/demo-7B", "source": "readme", "kind": "readme", "excerpt": ""}]
        cards = {"org/demo-7B": "card"}
        first = resolve("demo", "7b", candidates, cards)
        self.assertEqual(first["status"], "likely")
        second = resolve("demo", "7b", candidates, {"org/demo-7B": "new card"})
        self.assertNotEqual(first["decision_key"], second["decision_key"])
        self.assertNotEqual(first["decision_key"], resolve("demo", "7b-base", candidates, cards)["decision_key"])
        candidates.append({"repo": "other/demo-7B", "source": "search", "kind": "search", "excerpt": ""})
        cards["other/demo-7B"] = "card"
        self.assertEqual(resolve("demo", "7b", candidates, cards)["status"], "ambiguous")

    def test_unspecified_variant_does_not_silently_mean_base(self):
        candidates = [{"repo": repo} for repo in ["org/demo-7B", "org/demo-7B-Instruct"]]
        cards = {c["repo"]: "card" for c in candidates}
        self.assertEqual(resolve("demo", "7b", candidates, cards)["status"], "unresolved")
        self.assertEqual(resolve("demo", "7b-instruct", candidates, cards)["repo"], "org/demo-7B-Instruct")

    def test_missing_competing_card_is_not_negative_identity_evidence(self):
        candidates = [{"repo": "org/demo-7B"}, {"repo": "other/demo-7B"}]
        result = resolve("demo", "7b", candidates, {"org/demo-7B": "card"})
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["repo"])
        self.assertEqual(result["checks"]["other/demo-7B"], ["model_card_unavailable"])

    def fixture(self, root, conflicting=False):
        def write(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        rows = [{"model": "demo", "tag": tag, "weight_digest": digest}
                for tag, digest in [("latest", D1), ("7b-q4_0", D1), ("7b-q8_0", D2)]]
        if conflicting:
            rows.append({"model": "demo", "tag": "13b", "weight_digest": D1})
        (root / "out").mkdir()
        (root / "out/models.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        write("out/library.json", {"families": {"demo": {"readme": "[HF](https://huggingface.co/org/demo-7B)"}}})
        write("cache/identity/imported/cards/org__demo-7B.json", {"text": "Demo 7B model card"})
        write("cache/identity/imported/hash_lookup/" + "a" * 64 + ".json", {"matches": [
            {"source": "hf", "org": "quantizer", "name": "demo-7B-GGUF", "commit_sha": "rev", "path": "q4.gguf"},
            {"source": "hf", "org": "mirror", "name": "demo-7B-GGUF", "commit_sha": "rev2", "path": "q4.gguf"}]})

    def test_offline_build_alias_join_and_no_quant_suppression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            result = build(root)
            self.assertEqual(result, build(root))
            self.assertEqual(result["by_digest"][D1]["upstream"]["repo"], "org/demo-7B")
            self.assertEqual(result["by_digest"][D2]["upstream"]["repo"], "org/demo-7B")
            self.assertEqual(len(result["by_digest"][D1]["file_matches"]), 2)
            self.assertEqual(result["by_digest"][D2]["file_matches"], [])
            self.assertEqual(result["by_digest"][D1]["file_matches"][0]["commit_sha"], "rev2")
            self.assertEqual(result["by_release"]["demo:latest"]["status"], "unresolved")

    def test_conflicting_alias_sizes_block_artifact_pick(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, conflicting=True)
            artifact = build(root)["by_digest"][D1]
            self.assertEqual(artifact["upstream"]["status"], "conflict")
            self.assertIsNone(artifact["upstream"]["repo"])

    def test_reviewed_saved_evidence_fixtures(self):
        path = Path(__file__).parent / "fixtures/model_identity_review.json"
        reviewed = json.loads(path.read_text(encoding="utf-8"))
        for case in reviewed["alias_cases"]:
            with self.subTest(case=case["id"]):
                result = alias_constraints(case["tags"], case["expert_counts"])
                self.assertEqual(result["conflict"], case["conflict"])
                self.assertEqual(result["features"]["sizes"], case["expected_sizes"])
        for case in reviewed["selection_cases"]:
            with self.subTest(repo=case["repo"]):
                candidates = [{"repo": case["repo"], "source": case["source"]}]
                cards = {case["repo"]: case["evidence"]}
                if case.get("rejected_repo"):
                    candidates.append({"repo": case["rejected_repo"], "source": "legacy_decision"})
                    cards[case["rejected_repo"]] = "Competing model card available"
                result = resolve(case["family"], case["tag"], candidates, cards)
                self.assertEqual(result["repo"], case["repo"])

    def test_moe_equivalence_needs_header_and_explicit_alias(self):
        tags = ["16x17b", "17b-scout-16e-instruct"]
        self.assertTrue(alias_constraints(tags)["conflict"])
        self.assertTrue(alias_constraints(tags, [128])["conflict"])
        self.assertFalse(alias_constraints(tags, [16])["conflict"])
        self.assertEqual(alias_constraints(["137m"])["features"]["sizes"], ["137m"])

    def test_unrelated_variant_does_not_contaminate_other_size(self):
        candidates = [{"repo": "org/demo-7B"}, {"repo": "org/demo-13B-Instruct"}]
        self.assertEqual(resolve("demo", "7b", candidates, {c["repo"]: "card" for c in candidates})["repo"], "org/demo-7B")

    def test_unversioned_alias_does_not_select_versioned_checkpoint(self):
        candidate = {"repo": "org/demo-7B-Instruct-v2.1"}
        result = resolve("demo", "7b-instruct", [candidate], {candidate["repo"]: "card"})
        self.assertIsNone(result["repo"])
        self.assertIn("version_unspecified", result["checks"][candidate["repo"]])

    def test_alias_combination_resolves_default_without_cross_digest_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            rows = [{"model": "demo", "tag": t, "weight_digest": d} for t, d in
                    [("latest", D1), ("7b-instruct-q4_0", D1), ("7b-q8_0", D2)]]
            (root / "out/models.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            (root / "out/library.json").write_text(json.dumps({"families": {"demo": {"readme":
                "https://huggingface.co/org/demo-7B https://huggingface.co/org/demo-7B-Instruct"}}}), encoding="utf-8")
            write_path = root / "cache/identity/imported/cards/org__demo-7B-Instruct.json"
            write_path.parent.mkdir(parents=True, exist_ok=True)
            write_path.write_text(json.dumps({"text": "instruct card"}), encoding="utf-8")
            report = build(root)
            self.assertEqual(report["by_digest"][D1]["upstream"]["repo"], "org/demo-7B-Instruct")
            self.assertIsNone(report["by_digest"][D2]["upstream"]["repo"])
            self.assertFalse(any(D1 in group["digests"] for group in report["artifact_enrichment_queue"]))
            key = report["by_digest"][D1]["family_decisions"]["demo"]["decision_key"]
            self.assertIn(key, report["artifact_decisions"])
            self.assertEqual(report["artifact_decisions"][key]["constraints"]["features"]["modes"], ["instruct"])

    def test_snapshot_fingerprint_changes_with_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            before = build(root)
            (root / "cache/identity/imported/cards/org__demo-7B.json").write_text(json.dumps({"text": "changed card"}), encoding="utf-8")
            after = build(root)
            self.assertEqual(before["input_sha256"], after["input_sha256"])
            self.assertNotEqual(before["evidence_snapshot_sha256"], after["evidence_snapshot_sha256"])

    def test_comparison_aligns_digests_and_separates_verified_upload(self):
        identity = {"by_digest": {D1: {"upstream": {"repo": "org/upstream", "status": "likely"}},
                                  D2: {"upstream": {"repo": None, "status": "unresolved"}}}}
        legacy = {"models": [{"digest": D1, "ref": "demo:7b", "links": {"hf": {"release":
            {"repo": "quantizer/demo-GGUF", "confidence": "verified"}}}}]}
        result = compare(identity, legacy)
        self.assertEqual(result["summary"]["shared_artifacts"], 1)
        self.assertEqual(result["summary"]["identity_only_artifacts"], 1)
        self.assertEqual(result["rows"][0]["category"], "new_only")
        self.assertEqual(result["rows"][0]["legacy_exact_upload"], "quantizer/demo-GGUF")
        legacy["models"].append(legacy["models"][0])
        with self.assertRaises(ValueError):
            compare(identity, legacy)


if __name__ == "__main__":
    unittest.main()
