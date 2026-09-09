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
from unittest.mock import patch

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
    pick_candidate_for_release,
    release_key,
    size_token,
)
from links import _existing_hf_by_release, _pick_paper, _scan_text_for_hf  # noqa: E402  # pylint: disable=import-error
from hf_source import brave_candidates, extract_readme_candidates, sibling_candidates  # noqa: E402  # pylint: disable=import-error


class TestCleanRepo(unittest.TestCase):
    def test_clean_hf_repo_strips_blob_suffix(self):
        self.assertEqual(clean_hf_repo("https://huggingface.co/org/name/blob/main/README.md"), "org/name")

    def test_clean_hf_repo_rejects_org_only(self):
        self.assertIsNone(clean_hf_repo("https://huggingface.co/org"))

    def test_clean_hf_repo_rejects_datasets_and_spaces(self):
        self.assertIsNone(clean_hf_repo("https://huggingface.co/datasets/org/name"))
        self.assertIsNone(clean_hf_repo("https://huggingface.co/spaces/org/name"))

    def test_clean_hf_repo_rejects_docs_and_other_site_sections(self):
        # Regression: a Brave search for "gemma2" ranked huggingface.co's own docs page above
        # the actual model repo (huggingface.co/google/gemma-2-2b-it); "docs/transformers" was
        # wrongly accepted as a repo candidate since it wasn't in the bad-prefix denylist, and
        # verification couldn't catch it either (a docs page has no raw README.md to reject).
        self.assertIsNone(clean_hf_repo("https://huggingface.co/docs/transformers/model_doc/gemma2"))
        self.assertIsNone(clean_hf_repo("https://huggingface.co/tasks/text-generation"))
        self.assertIsNone(clean_hf_repo("https://huggingface.co/pricing/enterprise"))

    def test_clean_hf_repo_rejects_url_with_no_huggingface_domain(self):
        # Regression: without an explicit domain check, a URL missing "huggingface.co/"
        # entirely falls through split()'s no-op case (returns the whole original string
        # unchanged) and gets parsed as if it were a path, fabricating a bogus "repo" out of
        # the URL's own scheme/host -- caught while wiring an arbitrary (non-readme, non-HF-only)
        # source, a Brave search result, into clean_hf_repo.
        self.assertIsNone(clean_hf_repo("https://example.com/unrelated"))

    def test_clean_github_repo_strips_issues_suffix(self):
        self.assertEqual(clean_github_repo("https://github.com/org/name/issues/12"), "org/name")

    def test_clean_github_repo_rejects_org_pages(self):
        self.assertIsNone(clean_github_repo("https://github.com/orgs/org/repositories"))
        self.assertIsNone(clean_github_repo("https://github.com/pricing"))

    def test_clean_hf_repo_strips_trailing_quote_from_embedded_html(self):
        # A loose \S+-style URL scan over raw HTML (e.g. <a href="...url">) can pull in the
        # closing quote as part of the match -- confirmed live in nexusraven's GitHub readme,
        # which crashed scripts/links.py by writing an illegal Windows filename.
        self.assertEqual(
            clean_hf_repo('https://huggingface.co/Nexusflow/NexusRaven-V2-13B"'), "Nexusflow/NexusRaven-V2-13B"
        )

    def test_clean_github_repo_strips_trailing_html_junk(self):
        self.assertEqual(clean_github_repo('https://github.com/org/repo">'), "org/repo")


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


class TestExistingHfByRelease(unittest.TestCase):
    def test_verified_digest_wins_over_release_likely(self):
        models = [{"digest": "sha256:a", "model": "demo", "tag": "13b"}]
        hf_sources = {
            "by_digest": {"sha256:a": {"repo": "org/exact", "url": "https://huggingface.co/org/exact"}},
            "by_release": {"demo:13b": {"repo": "org/guess", "url": "https://huggingface.co/org/guess", "method": "readme"}},
        }
        out = _existing_hf_by_release(models, hf_sources)
        self.assertEqual(out["demo"]["demo:13b"]["confidence"], "verified")
        self.assertEqual(out["demo"]["demo:13b"]["repo"], "org/exact")

    def test_release_only_hit_is_likely(self):
        models = [{"digest": "sha256:a", "model": "demo", "tag": "13b"}]
        hf_sources = {"by_digest": {}, "by_release": {"demo:13b": {"repo": "org/guess", "url": "https://huggingface.co/org/guess", "method": "readme"}}}
        out = _existing_hf_by_release(models, hf_sources)
        self.assertEqual(out["demo"]["demo:13b"]["confidence"], "likely")

    def test_unresolved_release_is_absent(self):
        models = [{"digest": "sha256:a", "model": "demo", "tag": "13b"}]
        out = _existing_hf_by_release(models, {})
        self.assertNotIn("demo", out)

    def test_different_releases_in_one_family_kept_separate(self):
        models = [
            {"digest": "sha256:a", "model": "llava", "tag": "7b"},
            {"digest": "sha256:b", "model": "llava", "tag": "13b"},
        ]
        hf_sources = {
            "by_digest": {},
            "by_release": {
                "llava:7b": {"repo": "liuhaotian/llava-v1.5-7b", "url": "https://huggingface.co/liuhaotian/llava-v1.5-7b", "method": "readme"},
                "llava:13b": {"repo": "liuhaotian/llava-v1.5-13b", "url": "https://huggingface.co/liuhaotian/llava-v1.5-13b", "method": "readme"},
            },
        }
        out = _existing_hf_by_release(models, hf_sources)
        self.assertEqual(out["llava"]["llava:7b"]["repo"], "liuhaotian/llava-v1.5-7b")
        self.assertEqual(out["llava"]["llava:13b"]["repo"], "liuhaotian/llava-v1.5-13b")


class TestReleaseKey(unittest.TestCase):
    def test_quant_only_variants_merge(self):
        self.assertEqual(release_key("llava", "13b-v1.5-fp16"), release_key("llava", "13b-v1.5-q5_K_M"))

    def test_base_model_qualifier_kept_apart_despite_same_size(self):
        self.assertNotEqual(release_key("wizardlm", "13b-llama2-q4_0"), release_key("wizardlm", "13b-q4_0"))
        self.assertEqual(release_key("wizardlm", "13b-llama2-q4_0"), "wizardlm:13b-llama2")
        self.assertEqual(release_key("wizardlm", "13b-q4_0"), "wizardlm:13b")

    def test_moe_active_param_suffix_survives(self):
        self.assertEqual(release_key("qwen3", "235b-a22b-instruct-2507-q4_K_M"), "qwen3:235b-a22b-instruct-2507")

    def test_no_quant_token_unchanged(self):
        self.assertEqual(release_key("codellama", "latest"), "codellama:latest")

    def test_different_sizes_differ(self):
        self.assertNotEqual(release_key("llava", "7b"), release_key("llava", "13b"))


class TestExtractReadmeCandidates(unittest.TestCase):
    def test_returns_all_links_not_just_first(self):
        readme = (
            "[Hugging Face](https://huggingface.co/org/model-7b) "
            "and also [here](https://huggingface.co/org/model-13b)"
        )
        candidates = extract_readme_candidates(readme)
        self.assertEqual(candidates, ["org/model-7b", "org/model-13b"])

    def test_labeled_link_sorts_first(self):
        readme = (
            "[some other link](https://huggingface.co/org/unrelated) "
            "[Hugging Face](https://huggingface.co/org/the-real-one)"
        )
        self.assertEqual(extract_readme_candidates(readme)[0], "org/the-real-one")

    def test_no_links_is_empty(self):
        self.assertEqual(extract_readme_candidates(""), [])


class TestSiblingCandidates(unittest.TestCase):
    def test_finds_family_slug_matching_bare_tokens(self):
        text = (
            "Run it with:\n"
            "```bash\n"
            "python -m llava.serve.cli --model-path liuhaotian/llava-v1.5-13b\n"
            "```\n"
            "See also the unrelated other-project/other-model repo and images/llava_logo.png."
        )
        candidates = sibling_candidates("llava", [text])
        self.assertIn("liuhaotian/llava-v1.5-13b", candidates)
        self.assertNotIn("other-project/other-model", candidates)

    def test_rejects_file_extensions(self):
        text = "See docs/llava_architecture.png for the diagram."
        self.assertEqual(sibling_candidates("llava", [text]), [])

    def test_empty_family_yields_nothing(self):
        self.assertEqual(sibling_candidates("", ["org/repo-13b"]), [])


class TestPickCandidateForRelease(unittest.TestCase):
    def test_single_candidate_is_free_pass_when_no_size_to_check(self):
        repo, ambiguous = pick_candidate_for_release("latest", ["org/only-one"])
        self.assertEqual(repo, "org/only-one")
        self.assertFalse(ambiguous)

    def test_single_candidate_matching_size_is_free_pass(self):
        repo, ambiguous = pick_candidate_for_release("13b", ["org/model-13b"])
        self.assertEqual(repo, "org/model-13b")
        self.assertFalse(ambiguous)

    def test_single_candidate_with_mismatched_size_is_ambiguous(self):
        # Regression: a lone candidate scraped from arbitrary readme/changelog text (e.g. an old
        # preview release mentioned in passing) must not be blindly stamped onto every release
        # just because it's the only thing found -- confirmed live: llava's github readme
        # mentions exactly one bare HF link (a 7B preview build) that isn't the repo for
        # llava:13b, and an earlier version of this free-passed it anyway.
        repo, ambiguous = pick_candidate_for_release("13b", ["liuhaotian/LLaVA-Lightning-MPT-7B-preview"])
        self.assertIsNone(repo)
        self.assertTrue(ambiguous)

    def test_unique_size_match_is_deterministic(self):
        candidates = ["liuhaotian/llava-v1.5-7b", "liuhaotian/llava-v1.5-13b", "liuhaotian/llava-v1.6-34b"]
        repo, ambiguous = pick_candidate_for_release("13b", candidates)
        self.assertEqual(repo, "liuhaotian/llava-v1.5-13b")
        self.assertFalse(ambiguous)

    def test_multiple_size_matches_are_ambiguous(self):
        candidates = ["liuhaotian/llava-v1.5-7b", "llava-hf/llava-1.5-7b-hf"]
        repo, ambiguous = pick_candidate_for_release("7b", candidates)
        self.assertIsNone(repo)
        self.assertTrue(ambiguous)

    def test_no_size_token_with_multiple_candidates_is_ambiguous(self):
        candidates = ["org/a", "org/b"]
        repo, ambiguous = pick_candidate_for_release("latest", candidates)
        self.assertIsNone(repo)
        self.assertTrue(ambiguous)

    def test_no_candidates_is_not_ambiguous(self):
        repo, ambiguous = pick_candidate_for_release("13b", [])
        self.assertIsNone(repo)
        self.assertFalse(ambiguous)


class TestSizeToken(unittest.TestCase):
    def test_extracts_leading_size(self):
        self.assertEqual(size_token("13b-v1.5-fp16"), "13b")

    def test_extracts_moe_notation(self):
        self.assertEqual(size_token("8x22b-fp16"), "8x22b")

    def test_no_size_token_is_none(self):
        self.assertIsNone(size_token("latest"))


class TestBraveCandidates(unittest.TestCase):
    """brave_candidates() itself makes no network call -- it delegates to brave_search_cached()
    (mocked here) and only owns the results-&gt;repo-candidates extraction, which is what's
    tested. The Brave API call/cache/audit-trail plumbing is exercised by an actual
    --brave-search pilot run instead, same as this file's docstring already notes for the other
    network-touching functions.
    """

    def test_extracts_and_dedupes_repos_from_results(self):
        fake_results = [
            {"url": "https://huggingface.co/org/model-7b", "title": "model-7b"},
            {"url": "https://huggingface.co/org/model-7b/tree/main", "title": "duplicate, same repo"},
            {"url": "https://huggingface.co/datasets/org/data", "title": "not a model repo"},
            {"url": "https://example.com/unrelated", "title": "not huggingface at all"},
            {"url": "https://huggingface.co/org2/model-13b", "title": "model-13b"},
        ]
        with patch("hf_source.brave_search_cached", return_value=fake_results):
            candidates = brave_candidates("model", "fake-key")
        self.assertEqual(candidates, ["org/model-7b", "org2/model-13b"])

    def test_no_results_is_empty(self):
        with patch("hf_source.brave_search_cached", return_value=[]):
            self.assertEqual(brave_candidates("model", "fake-key"), [])

    def test_missing_url_field_is_skipped_not_crashed(self):
        with patch("hf_source.brave_search_cached", return_value=[{"title": "no url key"}]):
            self.assertEqual(brave_candidates("model", "fake-key"), [])


if __name__ == "__main__":
    unittest.main()
