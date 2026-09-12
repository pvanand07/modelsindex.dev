"""Semantic verifier tests without live calls."""
import json
from pathlib import Path
import tempfile
import unittest

from identity.verify import parse_verdict, run


class VerifyTests(unittest.TestCase):
    def test_quote_must_exist_and_vocabulary_is_closed(self):
        self.assertEqual(parse_verdict('{"exact_release":"yes","upstream":"yes","quote":"Model 7B","reason":"The card identifies this exact upstream release."}', "Model 7B card")["quote"], "Model 7B")
        with self.assertRaises(ValueError):
            parse_verdict('{"exact_release":"yes","upstream":"yes","quote":"invented"}', "card")
        with self.assertRaises(ValueError):
            parse_verdict('{"exact_release":"probably","upstream":"yes"}', "card")
        with self.assertRaises(ValueError):
            parse_verdict(None, "card")
        with self.assertRaises(ValueError):
            parse_verdict('{"exact_release":"yes","upstream":"yes","quote":"card","reason":"..."}', "card")
        with self.assertRaises(ValueError):
            parse_verdict('{"exact_release":"yes","upstream":"yes","quote":"","reason":"A sufficiently long but ungrounded explanation."}', "card")
        grounded = parse_verdict('{"exact_release":"yes","upstream":"yes","quote":"Model  7B","reason":"The card identifies this exact upstream release."}', "Model\n7B card")
        self.assertEqual(grounded["quote"], "Model\n7B")

    def test_execution_writes_a_reusable_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            repo, key = "org/demo-7B", "decision"
            card_path = data / "cache/identity/imported/cards/org__demo-7B.json"
            card_path.parent.mkdir(parents=True)
            card_path.write_text(json.dumps({"text": "Official Demo 7B model"}), encoding="utf-8")
            report = {"artifact_decisions": {key: {"status": "likely", "repo": repo,
                "constraints": {"tags": ["demo:7b"], "features": {"sizes": ["7b"]}},
                "evidence": {repo: [{"source": "https://ollama.com/library/demo", "excerpt": "Demo"}]}}},
                "by_digest": {"sha256:a": {"family_decisions": {"demo": {"decision_key": key}}}}}
            response = '{"exact_release":"yes","upstream":"yes","quote":"Demo 7B","reason":"The card identifies this exact upstream release."}'
            caller = lambda *args: response
            old_key, old_base = __import__('os').environ.get("OPENROUTER_API_KEY"), __import__('os').environ.get("OPENROUTER_BASE_URL")
            __import__('os').environ.update(OPENROUTER_API_KEY="x", OPENROUTER_BASE_URL="https://example.test")
            try:
                first = run(report, data, execute=True, caller=caller)
                second = run(report, data, execute=True, caller=lambda *args: self.fail("cache missed"))
            finally:
                if old_key is None: __import__('os').environ.pop("OPENROUTER_API_KEY", None)
                else: __import__('os').environ["OPENROUTER_API_KEY"] = old_key
                if old_base is None: __import__('os').environ.pop("OPENROUTER_BASE_URL", None)
                else: __import__('os').environ["OPENROUTER_BASE_URL"] = old_base
            self.assertEqual(first["summary"], {"verified": 1})
            self.assertEqual(second["summary"], {"cached": 1})
            self.assertEqual(first["items"][0]["family"], "demo")


if __name__ == "__main__":
    unittest.main()
