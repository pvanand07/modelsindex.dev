#!/usr/bin/env python3
from __future__ import annotations

import filecmp
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = [
    "gpus.json",
    "library.json",
    "link_content.json",
    "manifest.json",
    "models.json",
    "quality.json",
]


class TestBuildDist(unittest.TestCase):
    def test_build_dist_populates_public_data(self):
        subprocess.run(["bash", str(ROOT / "scripts/build_dist.sh")], check=True, cwd=ROOT)
        public_data = ROOT / "public/data"
        prod_data = ROOT / "prod/data"
        self.assertEqual(
            sorted(p.name for p in public_data.glob("*.json")),
            sorted(DATA_FILES),
        )
        for name in DATA_FILES:
            self.assertTrue(
                filecmp.cmp(public_data / name, prod_data / name, shallow=False),
                f"{name} differs between prod/data and public/data",
            )


if __name__ == "__main__":
    unittest.main()
