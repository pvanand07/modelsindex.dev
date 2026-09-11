#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf public/data
mkdir -p public/data
cp prod/data/gpus.json prod/data/library.json prod/data/link_content.json \
   prod/data/manifest.json prod/data/models.json prod/data/quality.json public/data/
