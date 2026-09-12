"""Conservative production-preview export tests."""
import unittest
from identity.export import export


class ExportTests(unittest.TestCase):
    def fixture(self, method="semantic_verified"):
        key, digest = "k", "sha256:a"
        identity = {"resolver_version": "x", "evidence_snapshot_sha256": "snap",
            "artifact_decisions": {key: {"repo": "org/model", "method": method,
                "semantic": {"verdict": {"exact_release": "yes", "upstream": "yes"}}}},
            "by_digest": {digest: {"upstream": {"status": "likely", "repo": "org/model",
                "supporting_releases": ["demo:7b"]}, "family_decisions": {"demo": {"decision_key": key}},
                "file_matches": [{"repo": "quant/model-GGUF", "confidence": "byte_verified"}]}}}
        legacy = {"schema_version": 2, "models": [{"digest": digest, "links": {"hf": {"release": None}}}]}
        return identity, legacy

    def test_default_exports_only_semantic_and_preserves_links(self):
        identity, legacy = self.fixture()
        result = export(identity, legacy)
        row = result["models"][0]
        self.assertIsNone(row["links"]["hf"]["release"])
        self.assertEqual(row["identity_v2"]["upstream"]["confidence"], "semantic_verified")
        self.assertEqual(row["identity_v2"]["exact_file_uploads"][0]["repo"], "quant/model-GGUF")

    def test_provisional_requires_opt_in(self):
        identity, legacy = self.fixture("release_name_constraints")
        self.assertIsNone(export(identity, legacy)["models"][0]["identity_v2"]["upstream"])
        value = export(identity, legacy, True)["models"][0]["identity_v2"]["upstream"]
        self.assertEqual(value["confidence"], "provisional")

    def test_duplicate_digest_fails(self):
        identity, legacy = self.fixture()
        legacy["models"].append(dict(legacy["models"][0]))
        with self.assertRaises(ValueError): export(identity, legacy)


if __name__ == "__main__": unittest.main()
