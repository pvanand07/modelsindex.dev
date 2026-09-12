"""Bounded enrichment contract tests; no network access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from identity.resolver import build, write_report
from identity.fetch import cache_path, plan, run


def report(*repos):
    return {"artifact_enrichment_queue": [{"action": "fetch_cards", "repos": [r], "digests": ["sha256:a"],
                                           "decision_key": r} for r in repos]}


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 1000
        self.calls = []

    def transport(self, *responses):
        responses = iter(responses)
        def fetch(url, timeout):
            self.calls.append(url)
            value = next(responses)
            if isinstance(value, Exception):
                raise value
            return value
        return fetch

    def execute(self, responses, **kwargs):
        return run(report("org/model"), self.root, fetch=True, clock=lambda: self.now,
                   transport=self.transport(*responses), **kwargs)

    def cached(self):
        return json.loads(cache_path(self.root, "org/model").read_text(encoding="utf-8"))

    def test_plan_is_read_only_and_deduplicates(self):
        r = report("org/model", "org/model")
        r["artifact_enrichment_queue"].append({"action": "review_conflict", "repos": ["bad/repo"]})
        self.assertEqual(len(plan(r)), 1)
        result = run(r, self.root, transport=self.transport(), clock=lambda: self.now)
        self.assertEqual(result["summary"]["requests"], 0)
        self.assertFalse(cache_path(self.root, "org/model").exists())

    def test_main_missing_then_master_success_and_revision(self):
        self.execute([(404, {}, ""), (200, {"x-repo-commit": "revision"}, "# Model")])
        entry = self.cached()
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["document"]["branch"], "master")
        self.assertEqual(entry["document"]["revision"], "revision")
        self.assertEqual(entry["document"]["sha256"], hashlib.sha256(b"# Model").hexdigest())
        self.assertEqual(len(entry["attempts"]), 2)

    def test_budget_resume_does_not_repeat_main(self):
        self.execute([(404, {}, "")], max_requests=1)
        self.assertEqual(self.cached()["status"], "deferred")
        self.execute([(200, {}, "# Model")], max_requests=1)
        self.assertIn("/master/", self.calls[-1])

    def test_valid_miss_differs_from_transient_and_retries_expire(self):
        self.execute([(404, {}, ""), (404, {}, "")])
        self.assertEqual(self.cached()["status"], "not_found")
        cached = self.execute([])
        self.assertEqual(cached["summary"]["requests"], 0)
        self.now += 8 * 86400
        self.execute([TimeoutError()])
        self.assertEqual(self.cached()["status"], "transient_error")
        self.assertEqual(self.cached()["retry_at"], self.now + 300)

    def test_rate_limit_stops_run_and_honors_retry_after(self):
        result = run(report("org/model", "org/next"), self.root, fetch=True, clock=lambda: self.now,
                     transport=self.transport((429, {"retry-after": "600"}, "")))
        self.assertEqual(result["summary"]["requests"], 1)
        self.assertEqual(self.cached()["retry_at"], self.now + 600)
        retry = self.execute([], retry_transient=True)
        self.assertEqual(retry["summary"]["requests"], 0)
        other_repo = run(report("org/different"), self.root, fetch=True, retry_transient=True,
                         clock=lambda: self.now, transport=self.transport())
        self.assertEqual(other_repo["summary"]["requests"], 0)
        self.assertEqual(other_repo["items"][0]["action"], "rate_limit_deferred")

    def test_transport_retry_override_does_not_require_waiting(self):
        self.execute([TimeoutError()])
        self.execute([(200, {}, "# Recovered")], retry_transient=True)
        self.assertEqual(self.cached()["status"], "ok")

    def test_failed_refresh_retains_last_good_document(self):
        self.execute([(200, {}, "# Good")])
        self.now += 31 * 86400
        self.execute([(503, {}, "")])
        entry = self.cached()
        self.assertEqual(entry["status"], "transient_error")
        self.assertEqual(entry["document"]["text"], "# Good")

    def test_denied_empty_and_html_are_not_successes(self):
        for response, expected in [((401, {}, ""), "denied"), ((200, {}, ""), "empty"),
                                   ((200, {}, "<!DOCTYPE html><html>Login</html>"), "invalid_content"),
                                   ((200, {"x-identity-too-large": "true"}, ""), "too_large")]:
            with self.subTest(state=expected):
                self.now += 2 * 86400
                self.execute([response])
                self.assertEqual(self.cached()["status"], expected)
                self.assertIsNone(self.cached().get("document"))

    def test_redirects_count_against_budget_and_cannot_leave_hf(self):
        self.execute([(307, {"location": "https://example.test/steal"}, "")])
        self.assertEqual(self.cached()["status"], "redirect_error")
        self.assertEqual(len(self.calls), 1)
        self.now += 2 * 86400
        self.execute([(307, {"location": "/org/model/raw/master/README.md"}, "")], max_requests=1)
        self.assertEqual(self.cached()["status"], "deferred")

    def test_repo_limit_and_unsupported_report(self):
        result = run(report("org/a", "org/b"), self.root, fetch=True, max_repos=1, clock=lambda: self.now,
                     transport=self.transport((200, {}, "card")))
        self.assertEqual(result["summary"]["requests"], 1)
        self.assertEqual(result["items"][1]["action"], "budget_deferred")
        with self.assertRaises(ValueError):
            plan({})
        with self.assertRaises(ValueError):
            plan(report("../escape"))

    def test_clock_budget_stops_before_next_request(self):
        def transport(url, timeout):
            self.now += 61
            return 404, {}, ""
        result = run(report("org/model"), self.root, fetch=True, transport=transport, clock=lambda: self.now)
        self.assertEqual(result["summary"]["requests"], 1)
        self.assertEqual(self.cached()["status"], "deferred")

    def test_enrichment_changes_offline_rebuild_and_checks_integrity(self):
        (self.root / "out").mkdir()
        write_report(self.root / "out/library.json", {"families": {"model": {"readme": "https://huggingface.co/org/model-7B"}}})
        (self.root / "out/models.jsonl").write_text(json.dumps({"model": "model", "tag": "7b", "digest": "sha256:" + "a" * 64}), encoding="utf-8")
        before = build(self.root)
        self.assertEqual(before["summary"]["artifact_upstream_statuses"], {"unresolved": 1})
        run(before, self.root, fetch=True, clock=lambda: self.now, transport=self.transport((200, {}, "# Model 7B")))
        after = build(self.root)
        self.assertEqual(after["summary"]["artifact_upstream_statuses"], {"likely": 1})
        self.assertNotEqual(before["evidence_snapshot_sha256"], after["evidence_snapshot_sha256"])
        entry_path = cache_path(self.root, "org/model-7B")
        entry = json.loads(entry_path.read_bytes())
        entry["document"]["text"] = "tampered"
        write_report(entry_path, entry)
        with self.assertRaises(ValueError):
            build(self.root)


if __name__ == "__main__":
    unittest.main()
