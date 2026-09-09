#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from crawl import parse_library_html, rewrite_site_urls  # noqa: E402  # pylint: disable=import-error


class TestLibraryCopy(unittest.TestCase):
    def test_parse_description_and_readme(self):
        html = (ROOT / "tests/fixtures/ollama_library_snippet.html").read_text(encoding="utf-8")
        copy = parse_library_html(html)
        self.assertEqual(copy["description"], "Meta's Llama 3.2 goes small with 1B and 3B models.")
        self.assertIn("The Meta Llama 3.2 collection", copy["readme"])
        self.assertIn("https://ollama.com/assets/library/llama3.2/be01fadf.png", copy["readme"])
        self.assertIn("ollama run llama3.2", copy["readme"])

    def test_rewrite_markdown_asset_urls(self):
        self.assertEqual(
            rewrite_site_urls("![x](/assets/library/foo.png)"),
            "![x](https://ollama.com/assets/library/foo.png)",
        )


if __name__ == "__main__":
    unittest.main()
