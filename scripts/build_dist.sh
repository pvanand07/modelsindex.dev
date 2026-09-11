#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist
mkdir -p dist/data
cp web/index.html web/app.js web/styles.css dist/
cp prod/data/gpus.json prod/data/library.json prod/data/link_content.json \
   prod/data/manifest.json prod/data/models.json prod/data/quality.json dist/data/
sed -i 's#"../prod/data"#"./data"#' dist/index.html
