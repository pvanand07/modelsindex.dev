#!/usr/bin/env bash
# RunPod bootstrap for Phase 1 (crawl.py) — CPU-only, no GPU needed.
# crawl.py is stdlib Python: it only does HTTP GET/Range requests against
# registry.ollama.ai and ollama.com and writes small JSON files. Running it
# on a GPU pod would waste GPU-hour billing for zero benefit.
#
# Recommended: attach a RunPod Network Volume (e.g. 100 GB) at /workspace,
# so the header/manifest cache this pod builds is still there when a later
# GPU pod (run_pod.sh) mounts the same volume for Phase 3/5 — no re-fetch.
#
#   bash scripts/run_pod_crawl.sh                 # full library crawl
#   bash scripts/run_pod_crawl.sh --models llama3.2,qwen3,gemma3   # smoke
#
# Pod: any "CPU-Only" template (e.g. "CPU3" / Secure Cloud, 4 vCPU is plenty
# since this is I/O-bound, not compute-bound). No CUDA image, no GPU count.
set -euo pipefail

WORK=/workspace
mkdir -p "$WORK"
REPO_LINK="$WORK/MODELINDEX_APP"
cd "$(dirname "$0")/.."

# Keep the cache and output on the (persistent) network volume, not the
# ephemeral container disk, so a pod restart or a later GPU pod reuses it.
export MODELINDEX_DATA="${MODELINDEX_DATA:-$WORK/data}"
mkdir -p "$MODELINDEX_DATA/cache" "$MODELINDEX_DATA/out"
[ -e data ] || true
rm -rf data/cache data/out 2>/dev/null || true
mkdir -p data
ln -sfn "$MODELINDEX_DATA/cache" data/cache
ln -sfn "$MODELINDEX_DATA/out" data/out

python3 --version
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

echo "== crawl: $* =="
# --workers higher than the local default: this is network-bound and a
# datacenter VPS has far more concurrent-connection headroom than a laptop.
python3 scripts/crawl.py --workers 16 "$@" 2>&1 | tee "data/out/crawl-$STAMP.log"

echo
echo "Summary:"
cat data/out/crawl_summary.json
echo
echo "models.jsonl and data/cache/ are on the network volume at $MODELINDEX_DATA."
echo "Either keep that volume attached for the GPU pods (run_pod.sh), or copy off:"
echo "  scp -P <port> -r root@<pod-ip>:$MODELINDEX_DATA/out/models.jsonl ./data/out/"
