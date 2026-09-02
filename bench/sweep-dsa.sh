#!/usr/bin/env bash
# Sweep DSA (sparse attention) backends on sm120 and record how each one fails,
# or which one serves. TileLang needs 148 KB of dynamic shared memory and the
# RTX 5090 caps out at 99 KB/block, so the question is whether any other backend
# is tiled to fit. trtllm is excluded: its TllmGenFmhaRunner is sm100-only.
set -uo pipefail

VENV="${VENV:-/root/sgl-env}"
MODEL="${MODEL:-/root/models/GLM-5.3-Flash-NVFP4}"
CU="$VENV/lib/python3.12/site-packages/nvidia/cu13"
OUT=/root/dsa_sweep_results.txt

export CUDA_HOME="$CU" PATH="$CU/bin:$PATH"
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export LIBRARY_PATH="$CU/lib" LD_LIBRARY_PATH="$CU/lib"
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_ENABLE_JIT_DEEPGEMM=0
export NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=32
export SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1

: > "$OUT"
for BK in flashinfer_sparse_mla flashmla_auto flashmla_kv fa3; do
  # flashmla/flashinfer sparse MLA are fp8-KV designs; fa3 wants bf16.
  case "$BK" in
    fa3) KV=bfloat16 ;;
    *)   KV=fp8_e4m3 ;;
  esac
  LOG="/root/dsa_${BK}.log"
  echo "=== $BK (kv=$KV) ===" | tee -a "$OUT"
  pkill -9 -f "launch_serve[r]" 2>/dev/null; sleep 5

  setsid bash -c "source $VENV/bin/activate && python -m sglang.launch_server \
      --model-path $MODEL --tp-size 8 --trust-remote-code \
      --kv-cache-dtype $KV \
      --dsa-prefill-backend $BK --dsa-decode-backend $BK \
      --moe-runner-backend triton \
      --max-running-requests 16 --host 0.0.0.0 --port 30000 \
      > $LOG 2>&1" < /dev/null > /dev/null 2>&1 &

  for i in $(seq 1 42); do
    if grep -qE "The server is fired up|Uvicorn running" "$LOG" 2>/dev/null; then
      echo "  RESULT: SERVED" | tee -a "$OUT"; echo "WINNER=$BK" >> "$OUT"; exit 0
    fi
    if grep -qiE "Scheduler hit an exception|Killed|ValueError" "$LOG" 2>/dev/null; then
      REASON=$(grep -ihoE "shared memory size to [0-9]+|not supported[^,.]*|ValueError: [^,]{0,90}|InternalError[^,]{0,70}|NameError: [^,]{0,50}" "$LOG" | head -1)
      echo "  RESULT: FAILED - ${REASON:-see $LOG}" | tee -a "$OUT"
      break
    fi
    sleep 10
  done
done
pkill -9 -f "launch_serve[r]" 2>/dev/null
echo "SWEEP DONE" | tee -a "$OUT"
