"""Targeted search tests without network access."""
import json
from pathlib import Path
import tempfile
import unittest
from identity.search import plan, run


class SearchTests(unittest.TestCase):
    def report(self):
        return {"artifact_decisions": {"d": {"status": "unresolved", "constraints": {"features":
            {"sizes": ["7b"], "modes": ["instruct"], "qualifiers": ["v2"]}}}},
            "artifact_enrichment_queue": [{"action": "review_or_search", "family": "demo", "decision_key": "d",
                                            "digests": ["a", "b"], "repos": []}]}

    def test_plan_is_targeted(self):
        item = plan(self.report())[0]
        self.assertEqual(item["query"], "site:huggingface.co demo 7b instruct v2")

    def test_results_are_strictly_filtered_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            old = __import__('os').environ.get("BRAVE_SEARCH_API_KEY")
            __import__('os').environ["BRAVE_SEARCH_API_KEY"] = "x"
            calls = []
            def caller(*args):
                calls.append(1)
                return [{"url": "https://huggingface.co/org/demo-7B-Instruct-v2", "title": "ok"},
                        {"url": "https://huggingface.co/datasets/org/data"},
                        {"url": "https://evil.test/huggingface.co/org/bad"}]
            try:
                first = run(self.report(), data, execute=True, caller=caller)
                second = run(self.report(), data, execute=True, caller=lambda *a: self.fail("cache miss"))
            finally:
                if old is None: __import__('os').environ.pop("BRAVE_SEARCH_API_KEY", None)
                else: __import__('os').environ["BRAVE_SEARCH_API_KEY"] = old
            self.assertEqual(first["items"][0]["repos"], ["org/demo-7B-Instruct-v2"])
            self.assertEqual(second["summary"], {"cached": 1})


if __name__ == "__main__": unittest.main()
