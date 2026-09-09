#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from crawl import iso_from_push_unix  # noqa: E402  # pylint: disable=import-error


class TestPushTime(unittest.TestCase):
    def test_unix_to_iso(self):
        self.assertEqual(iso_from_push_unix(1727284764), "2024-09-25T17:19:24Z")

    def test_invalid(self):
        self.assertIsNone(iso_from_push_unix(None))
        self.assertIsNone(iso_from_push_unix("nope"))
        self.assertIsNone(iso_from_push_unix(0))
        self.assertIsNone(iso_from_push_unix(-1))


if __name__ == "__main__":
    unittest.main()
