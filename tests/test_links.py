#!/usr/bin/env python3
"""Tests for the pure/deterministic pieces of scripts/link_common.py and scripts/links.py --
classification, URL cleaning, and content mining. Network-touching functions (http_get,
fetch_*_readme, fetch_homepage, the LLM calls) aren't covered here; they're thin enough to be
exercised by an actual --resolve pilot run instead of mocked out.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from link_common import (  # noqa: E402  # pylint: disable=import-error
    classify_readme_links,
    clean_github_repo,
    clean_hf_repo,
    extract_links,
    find_github_repos_in_links,
    find_hf_repos_in_links,
    html_to_text,
)
from links import _existing_hf_by_family, _pick_paper, _scan_text_for_hf  # noqa: E402  # pylint: disable=import-error


class TestCleanRepo(unittest.TestCase):
    def test_clean_hf_repo_strips_blob_suffix(self):
        self.assertEqual(clean_hf_repo("https://huggingface.co/org/name/blob/main/README.md"), "org/name")

    def test_clean_hf_repo_rejects_org_only(self):
        self.assertIsNone(clean_hf_repo("https://huggingface.co/org"))

    def test_clean_hf_repo_rejects_datasets_and_spaces(self):
        self.assertIsNone(clean_hf_repo("https://huggingface.co/datasets/org/name"))
        self.assertIsNone(clean_hf_repo("https://huggingface.co/spaces/org/name"))

    def test_clean_github_repo_strips_issues_suffix(self):
        self.assertEqual(clean_github_repo("https://github.com/org/name/issues/12"), "org/name")

    def test_clean_github_repo_rejects_org_pages(self):
        self.assertIsNone(clean_github_repo("https://github.com/orgs/org/repositories"))
        self.assertIsNone(clean_github_repo("https://github.com/pricing"))


class TestClassifyReadmeLinks(unittest.TestCase):
    def test_splits_four_buckets(self):
        readme = (
            "[Hugging Face](https://huggingface.co/org/model) "
            "[GitHub](https://github.com/org/repo) "
            "[Paper](https://arxiv.org/abs/2401.00001) "
            "[Project website](https://example.com/project)"
        )
        hf, paper, homepage, github = classify_readme_links(readme)
        self.assertEqual(hf, ["org/model"])
        self.assertEqual(github, ["org/repo"])
        self.assertEqual(paper, ["https://arxiv.org/abs/2401.00001"])
        self.assertEqual(homepage, ["https://example.com/project"])

    def test_arxiv_url_classified_as_paper_without_label(self):
        readme = "[Read more](https://arxiv.org/abs/2401.00001)"
        _hf, paper, _homepage, _github = classify_readme_links(readme)
        self.assertEqual(paper, ["https://arxiv.org/abs/2401.00001"])

    def test_no_links_returns_empty_buckets(self):
        self.assertEqual(classify_readme_links(""), ([], [], [], []))
        self.assertEqual(classify_readme_links(None), ([], [], [], []))


class TestFindReposInLinks(unittest.TestCase):
    def test_find_hf_repos_dedupes_and_orders(self):
        links = [
            "https://huggingface.co/org/a",
            "https://example.com/unrelated",
            "https://huggingface.co/org/a/blob/main/config.json",
            "https://huggingface.co/org/b",
        ]
        self.assertEqual(find_hf_repos_in_links(links), ["org/a", "org/b"])

    def test_find_github_repos_ignores_non_github(self):
        links = ["https://huggingface.co/org/a", "https://github.com/org/repo"]
        self.assertEqual(find_github_repos_in_links(links), ["org/repo"])


class TestHtmlToText(unittest.TestCase):
    def test_strips_script_and_style(self):
        html_doc = "<html><head><style>.a{color:red}</style></head><body><script>evil()</script><p>Hello&nbsp;world</p></body></html>"
        text = html_to_text(html_doc)
        self.assertIn("Hello", text)
        self.assertNotIn("evil()", text)
        self.assertNotIn("color:red", text)

    def test_collapses_blank_lines(self):
        text = html_to_text("<p>one</p>\n\n\n\n<p>two</p>")
        self.assertNotIn("\n\n\n", text)


class TestExtractLinks(unittest.TestCase):
    def test_resolves_relative_links(self):
        html_doc = '<a href="/docs">docs</a><a href="https://other.example/x">x</a>'
        links = extract_links(html_doc, "https://example.com/page")
        self.assertIn("https://example.com/docs", links)
        self.assertIn("https://other.example/x", links)

    def test_dedupes_preserving_order(self):
        html_doc = '<a href="/a">a</a><a href="/b">b</a><a href="/a">a again</a>'
        self.assertEqual(extract_links(html_doc, "https://example.com"), ["https://example.com/a", "https://example.com/b"])


class TestPickPaper(unittest.TestCase):
    def test_prefers_arxiv(self):
        candidates = ["https://example.com/paper.pdf", "https://arxiv.org/abs/2401.00001"]
        self.assertEqual(_pick_paper(candidates), "https://arxiv.org/abs/2401.00001")

    def test_falls_back_to_first_candidate(self):
        candidates = ["https://example.com/paper.pdf"]
        self.assertEqual(_pick_paper(candidates), "https://example.com/paper.pdf")

    def test_empty_is_none(self):
        self.assertIsNone(_pick_paper([]))


class TestScanTextForHf(unittest.TestCase):
    def test_finds_repo_in_plain_markdown_text(self):
        text = "See the model card at https://huggingface.co/org/model for details."
        self.assertEqual(_scan_text_for_hf(text), ["org/model"])

    def test_no_match_returns_empty(self):
        self.assertEqual(_scan_text_for_hf("nothing here"), [])


class TestExistingHfByFamily(unittest.TestCase):
    def test_verified_digest_wins_over_family_likely(self):
        models = [{"digest": "sha256:a", "model": "demo"}]
        hf_sources = {
            "by_digest": {"sha256:a": {"repo": "org/exact", "url": "https://huggingface.co/org/exact"}},
            "by_family": {"demo": {"repo": "org/guess", "url": "https://huggingface.co/org/guess", "method": "readme"}},
        }
        out = _existing_hf_by_family(models, hf_sources)
        self.assertEqual(out["demo"]["confidence"], "verified")
        self.assertEqual(out["demo"]["repo"], "org/exact")

    def test_family_only_hit_is_likely(self):
        models = [{"digest": "sha256:a", "model": "demo"}]
        hf_sources = {"by_digest": {}, "by_family": {"demo": {"repo": "org/guess", "url": "https://huggingface.co/org/guess", "method": "readme"}}}
        out = _existing_hf_by_family(models, hf_sources)
        self.assertEqual(out["demo"]["confidence"], "likely")

    def test_unresolved_family_is_absent(self):
        models = [{"digest": "sha256:a", "model": "demo"}]
        out = _existing_hf_by_family(models, {})
        self.assertNotIn("demo", out)


if __name__ == "__main__":
    unittest.main()
