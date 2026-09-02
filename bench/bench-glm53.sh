#!/usr/bin/env bash
# Benchmark a running GLM-5.3-Flash server and print the numbers that decide
# which moat to fund: single-stream latency (comm-bound) and batched throughput.
#
#   ./bench-glm53.sh <mode-label> [port]
#
# Single-stream TPOT is the communication-sensitive number on this box: with
# P2P unavailable, every decode step pays 2 allreduces per layer through host
# RAM. If TP=8 vs TP8+EP=8 moves TPOT materially, comms (moat 2) dominate.
set -euo pipefail

LABEL="${1:-run}"
PORT="${2:-30000}"
VENV="${VENV:-/root/sgl-env}"
OUT="/root/logs/bench_${LABEL}.txt"

source "$VENV/bin/activate"
mkdir -p /root/logs

echo "=== waiting for server on :$PORT ==="
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "server never became healthy" >&2; exit 1; }

{
  echo "### $LABEL"
  echo "--- concurrency=1 (latency / comm-bound) ---"
  python -m sglang.bench_serving \
    --backend sglang --port "$PORT" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts 8 --max-concurrency 1

  echo "--- concurrency=16 (throughput) ---"
  python -m sglang.bench_serving \
    --backend sglang --port "$PORT" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts 64 --max-concurrency 16
} 2>&1 | tee "$OUT"

echo
echo "=== key numbers ($LABEL) ==="
grep -E "Output token throughput|Mean TPOT|Mean TTFT|Request throughput" "$OUT" || true
