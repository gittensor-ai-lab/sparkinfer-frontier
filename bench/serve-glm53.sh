#!/usr/bin/env bash
# Serve GLM-5.3-Flash-NVFP4 on 8x RTX 5090 (sm120) under one named config.
#
#   ./serve-glm53.sh tp8          TP=8, MoE backend auto        (baseline)
#   ./serve-glm53.sh tp8_ep8      TP=8 + EP=8                   (moat 2: cut TP allreduce traffic)
#   ./serve-glm53.sh tp8_triton   TP=8, MoE backend triton      (sm120-safe fallback)
#   ./serve-glm53.sh tp8_marlin   TP=8, MoE backend marlin      (moat 1: warp-level FP4, SM90/SM120)
#
# Why these: the compressed-tensors W4A4 NVFP4 MoE scheme prefers the FlashInfer
# TRT-LLM FP4 path, which targets sm100 (tcgen05 tensor-memory MMA). sm120 has no
# tcgen05, so that kernel is expected to be unavailable; marlin/triton are the
# warp-level paths that do exist on sm120.
set -euo pipefail

MODE="${1:-tp8}"
MODEL="${MODEL:-/root/models/GLM-5.3-Flash-NVFP4}"
PORT="${PORT:-30000}"
VENV="${VENV:-/root/sgl-env}"

# The image ships nvcc 12.8, but SM 12.x needs CUDA >= 12.9 — without this,
# flashinfer's JIT cannot emit sm120 code and device-capability probes fail.
# nvidia-cuda-nvcc-cu13 (pip) supplies CUDA 13.3.
CU="${CU:-$VENV/lib/python3.12/site-packages/nvidia/cu13}"
if [ -x "$CU/bin/nvcc" ]; then
  export CUDA_HOME="$CU"
  export PATH="$CU/bin:$PATH"
  # That nvcc is 13.3 while the runtime headers are 13.0 (CUDART_VERSION 13000).
  # CCCL hard-errors on any mismatch; the header documents this opt-out for
  # exactly this case. Without it every JIT kernel fails to compile on sm120.
  export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
  # The wheel ships libcudart.so.13 only; ld needs the unversioned soname.
  [ -e "$CU/lib/libcudart.so" ] || ln -sf "$CU/lib/libcudart.so.13" "$CU/lib/libcudart.so"
  export LIBRARY_PATH="$CU/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$CU/lib:${LD_LIBRARY_PATH:-}"
fi

# PCIe-only, dual-NUMA box: P2P is CNS on all pairs, so every collective stages
# through host RAM. More NCCL channels helps; custom all-reduce is disabled by
# SGLang anyway (needs full NVLink above world_size 2).
# SGLang already disables DeepGEMM on sm120 ("DeepGEMM requires TMEM/tcgen05
# (SM100+datacenter), not available on SM120" -- deep_gemm_wrapper/configurer.py),
# which leaves the `deep_gemm` module unimported. But the GLM-5.3 mHC prenorm path
# is gated on this separate flag, which defaults True and never consults that
# decision, so it calls deep_gemm.tf32_hc_prenorm_gemm and dies with
# `NameError: name 'deep_gemm' is not defined`. Force the non-DeepGEMM branch.
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0

# Deliberately NOT setting NCCL_P2P_DISABLE / NCCL_MIN_NCHANNELS. Measured on this
# box (bench_allreduce.py, 8 GPUs): forcing them made every case worse than NCCL's
# own defaults -- 8 KiB all-reduce 125 vs 106 us p50 and 1357 vs 190 us p99, and a
# 64 MiB all-reduce 154 ms vs 35.5 ms (0.8 vs 3.3 GB/s). Those flags come from
# 4x RTX PRO 6000 recipes and do not transfer to this dual-NUMA PCIe topology.
# Let NCCL detect the topology itself.
export SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1  # it self-disables anyway above world_size 2

COMMON=(
  --model-path "$MODEL"
  --tp-size 8
  --trust-remote-code
  --host 0.0.0.0 --port "$PORT"
  # 189 GiB of weights over 8x32 GiB leaves ~7 GiB/GPU for KV + activations +
  # graphs. Keep concurrency modest until the real headroom is measured.
  --max-running-requests 16
)

case "$MODE" in
  tp8)         EXTRA=() ;;
  tp8_ep8)     EXTRA=(--ep-size 8) ;;
  tp8_triton)  EXTRA=(--moe-runner-backend triton) ;;
  tp8_marlin)  EXTRA=(--moe-runner-backend marlin) ;;
  # DeepGEMM JIT targets SM90/SM100 and fails to compile on sm120 ("NVCC
  # compilation failed"), and it is reached from more than the MoE runner, so it
  # has to be switched off globally rather than routed around. SGLang documents
  # the cutlass FP8 GEMM backend as "optimal for SM120 GPUs".
  tp8_sm120)   export SGLANG_ENABLE_JIT_DEEPGEMM=0
               EXTRA=(--fp8-gemm-backend cutlass --moe-runner-backend triton) ;;
  tp8_sm120_ep8)
               export SGLANG_ENABLE_JIT_DEEPGEMM=0
               EXTRA=(--fp8-gemm-backend cutlass --moe-runner-backend triton --ep-size 8) ;;
  # DeepGEMM is still reached during CUDA-graph capture of the decode path
  # (capture_decode_graph -> decode_cuda_graph_runner) even with the JIT env off.
  # Dropping graph capture trades decode speed for finding out whether sm120 can
  # run this model correctly at all -- correctness first, then performance.
  tp8_sm120_nocg)
               export SGLANG_ENABLE_JIT_DEEPGEMM=0
               EXTRA=(--fp8-gemm-backend cutlass --moe-runner-backend triton --disable-cuda-graph) ;;
  # sm120 identifies as Blackwell, so auto-selection pairs an FP8 KV cache with the
  # TRT-LLM DSA backend -- whose TllmGenFmhaRunner is built for sm100 and throws
  # inside CUDA-graph capture. The cookbook's non-Blackwell pairing (BF16 KV +
  # TileLang DSA) is the combination that has kernels here. Keep dtype and both
  # DSA backends switched together: TileLang DSA with FP8 KV is not a valid combo.
  # SGLang ships a purpose-built sm120 sparse-MLA kernel (flash_mla_sm120.py) for
  # GLM DSA models with an FP8 KV cache. Both prefill and decode must use it --
  # the validator rejects any mix. This is the only DSA path with sm120 kernels.
  # The FlashInfer default inside flash_mla_sm120 routes into TRTLLM-GEN MLA, which
  # refuses GLM-5.3's qk_rope_head_dim=0 ("requires sparse_mla_top_k_lens"). The
  # torch fallback in the same file is layout-independent after the gather, so with
  # a flat-latent gather it serves GLM-5.3 -- unfused and slow, but correct first.
  tp8_sm120mla)
               export SGLANG_ENABLE_JIT_DEEPGEMM=0
               # Leave SGLANG_SM120_FLASHMLA_BACKEND at flashinfer: with
               # sparse_mla_top_k_lens supplied for NoPE, the real CUTLASS sm120
               # kernel is usable, and the torch fallback is orders slower.
               # The NoPE reference path does a boolean-mask assignment
               # (gathered_kv[invalid_mask] = 0), which forces a device-to-host
               # sync and deadlocks inside CUDA-graph capture -- the server hangs
               # at 0% GPU rather than erroring. Eager decode is required until a
               # capturable kernel replaces it.
               # The Triton MoE runner has no NVFP4 path at all -- only
               # flashinfer_cutlass and flashinfer_trtllm implement W4A4, and
               # trtllm targets sm100. Routing NVFP4 experts through triton
               # collapses the hidden state and the logits go uniform.
               EXTRA=(--kv-cache-dtype fp8_e4m3
                      --dsa-prefill-backend flashinfer_sparse_mla
                      --dsa-decode-backend flashinfer_sparse_mla
                      --moe-runner-backend marlin
                      # The routed experts are NVFP4 but the shared expert stays
                      # BF16 (the quant config's ignore list), so fusing it into
                      # expert slot n_routed_experts leaves that slot without
                      # weight_packed/weight_scale/weight_global_scale. The zero
                      # global scale then becomes inf via 1/x and NaNs the layer.
                      --disable-shared-experts-fusion
                      # Without these the chat template's raw </think> and tool
                      # syntax leak into message.content.
                      --reasoning-parser glm45
                      --tool-call-parser glm47
                      --disable-cuda-graph) ;;
  tp8_tilelang)
               export SGLANG_ENABLE_JIT_DEEPGEMM=0
               # bfloat16 must be explicit: `auto` resolves to fp8_e4m3 on Blackwell,
               # and tilelang DSA only accepts fp8 KV on ROCm/HIP, bf16 on CUDA.
               EXTRA=(--kv-cache-dtype bfloat16
                      --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
                      --moe-runner-backend triton) ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

source "$VENV/bin/activate"
mkdir -p /root/logs
echo "=== serving mode=$MODE ==="
exec python -m sglang.launch_server "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "/root/logs/serve_${MODE}.log"
