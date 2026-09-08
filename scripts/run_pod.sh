#!/usr/bin/env bash
# RunPod bootstrap for the calibration protocol (spec section 5.3 / 5.4).
#
#   bash scripts/run_pod.sh 24gb                 # 24 GB tier
#   bash scripts/run_pod.sh 48gb                 # 48 / 80 GB tier (adds 64k/128k ctx points)
#   bash scripts/run_pod.sh 24gb --kv-q8         # plus the flash-attention + q8_0 KV pass (4090 only)
#   bash scripts/run_pod.sh 24gb --include-partial   # plus llama3.3:70b forced partial offload
#
# Template: runpod/pytorch (python3, nvidia-smi, torch present). Container disk: 80 GB (24gb) / 150 GB (48gb).
# Results land in /workspace/results — scp them off before terminating the pod.
set -euo pipefail

TIER="${1:?tier required: 24gb | 48gb}"; shift || true
KVQ8=0; EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --kv-q8) KVQ8=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

WORK=/workspace
RESULTS=$WORK/results
mkdir -p "$RESULTS"
export OLLAMA_MODELS=$WORK/ollama-models      # keep pulls on the large disk
cd "$(dirname "$0")/.."

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr -cd '[:alnum:]-')
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CTX_BIG=()
[ "$TIER" = "48gb" ] && CTX_BIG=(--ctx-big)

start_server() {  # $1 = log suffix, uses $FA and $KVT
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 2
  OLLAMA_HOST=127.0.0.1:11434 \
  OLLAMA_NUM_PARALLEL=1 \
  OLLAMA_FLASH_ATTENTION="$FA" \
  OLLAMA_KV_CACHE_TYPE="$KVT" \
  nohup ollama serve > "$RESULTS/server-$1.log" 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf localhost:11434/api/version >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "ollama did not start; see $RESULTS/server-$1.log" >&2
  exit 1
}

echo "== main pass: $GPU tier=$TIER =="
FA=0 KVT=f16 start_server main
python3 scripts/bench.py --tier "$TIER" "${CTX_BIG[@]}" \
  --out "$RESULTS/$GPU-$STAMP-main.jsonl" \
  --server-log "$RESULTS/server-main.log" \
  --label main "${EXTRA[@]}"

if [ "$KVQ8" = 1 ]; then
  echo "== kv-q8 pass =="
  FA=1 KVT=q8_0 start_server kvq8
  python3 scripts/bench.py --models llama3.1:8b,qwen3:30b-a3b --skip-pull --skip-speed --no-bandwidth \
    --out "$RESULTS/$GPU-$STAMP-kvq8.jsonl" \
    --server-log "$RESULTS/server-kvq8.log" \
    --label kv-q8
fi

echo
echo "Done. Copy results before terminating:"
echo "  scp -P <port> -r root@<pod-ip>:$RESULTS ./data/out/measurements/"
