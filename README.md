# Frontier models on 8x RTX 5090

Running **GLM-5.3-Flash (NVFP4)** on eight consumer RTX 5090s, from a fork of
[SGLang](https://github.com/sgl-project/sglang).

```
"Q: What is 15 times 4?"                    ->  "15 times 4 is 60."
"Q: Name the largest planet..."             ->  "Jupiter is the largest planet in our solar system."
"Q: Who wrote Romeo and Juliet?"            ->  "Romeo and Juliet was written by William Shakespeare."
"The sky appears blue because sunlight..."  ->  "...is scattered by the atmosphere."
```

## The node

| | |
|---|---|
| GPUs | 8x RTX 5090 (sm120), 32 GB each, 256 GB total |
| Interconnect | PCIe Gen4 x16, **no P2P** (CNS on all 28 pairs), dual NUMA |
| Host | 2x EPYC 7K62, 192 threads, 503 GiB RAM |
| Model | `RedHatAI/GLM-5.3-Flash-NVFP4` -- 320B total / 18B active, 185 GiB |
| Residency | 24.8 GB per GPU, ~62 s to load |

## Run it

```bash
bench/serve-glm53.sh tp8_sm120mla     # TP=8, sm120 sparse MLA, NVFP4 MoE
bench/bench-glm53.sh tp8_sm120mla     # latency + throughput
bench/bench_allreduce.py              # torchrun --nproc_per_node=8
```

`serve-glm53.sh` carries every workaround below, each commented with the failure
it prevents.

## What it took

Five things had to be fixed before this model produced a correct token on sm120.

**Toolchain.** The image ships nvcc 12.8 but SM 12.x needs >= 12.9, so every JIT
kernel failed. `nvidia-cuda-nvcc-cu13` supplies 13.3, which then mismatches the
13.0 runtime headers (CCCL hard-errors), needing
`-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`, plus a `libcudart.so` soname symlink.

**Shared-expert fusion.** The routed experts are NVFP4 but the shared expert
stays BF16, and the loader fuses it into expert slot `n_routed_experts` anyway.
That slot receives no scale, so `1/weight_global_scale` is `inf` and `inf * 0`
NaNs the whole layer. Every MoE backend read that poisoned scale, which is why
all of them failed on healthy weights. `--disable-shared-experts-fusion` is the
fix, and this is **not** sm120-specific.

**NoPE sparse MLA.** GLM-5.3 is `qk_rope_head_dim=0`; the compiled sm120
sparse-MLA kernel is shaped for `512 + rope 64` and misses on geometry, not on
dispatch. The NoPE route falls back to a layout-independent reference, fed by a
flat-latent gather for `DSATokenToKVPool`'s 528 B rows (512 FP8 + 4 FP32 block
scales).

**NVFP4 MoE.** TRT-LLM FP4 is sm100-only and FlashInfer CUTLASS reads the
ModelOpt layout, so `sm120_nvfp4_moe_ref.py` dequantises and runs a grouped GEMM.

**Tensor-parallel deadlock.** `x[mask] = 0` sizes its write on the host, which
desynchronises ranks and hangs NCCL at 0% GPU with no error. `masked_fill` is
the sync-free form. The same sync deadlocks CUDA-graph capture, so this path
runs eager.

## Kernels

`sm120_nvfp4_moe_triton.py` is a fused NVFP4 GEMM: it unpacks the e2m1 nibbles
and applies the per-group FP8 scale inside the matmul, so expert weights are read
once at 4 bits and never materialised in bfloat16. Tiles are 64x64x64, sized for
sm120's 99 KB of shared memory rather than sm100's 228 KB.

`sm120_nvfp4_moe_fused.py` then removes the per-expert launch: `moe_align_block_size`
sorts the routed (token, expert) pairs so a block owns 64 rows sharing one expert,
and the whole MoE becomes **two grouped GEMMs** instead of two launches per expert.

Against the per-expert reference at GLM-5.3 shapes, 288 experts / top-8
(`bench/test_moe_fused.py`), matching it exactly:

| tokens | experts hit | reference | fused | |
|---|---|---|---|---|
| 1 | 8 | 3.84 ms | 0.65 ms | **5.9x** |
| 8 | 62 | 28.23 ms | 0.87 ms | **32.5x** |
| 32 | 166 | 75.57 ms | 1.66 ms | **45.4x** |
| 128 | 280 | 129.16 ms | 2.50 ms | **51.7x** |

End to end that is **2.5 -> 6.0 tok/s** (16 tokens in 2.65 s, warm; the first
request pays Triton JIT).

The single GEMM underneath (`bench/test_nvfp4_gemm.py`), against
`dequantize_nvfp4` + `torch.matmul` on identical packed weights:

| M | N | K | rel err | dequant+matmul | fused | |
|---|---|---|---|---|---|---|
| 64 | 512 | 4096 | 0.0024 | 0.298 ms | 0.087 ms | **3.42x** |
| 256 | 512 | 4096 | 0.0043 | 0.297 ms | 0.087 ms | **3.41x** |
| 1024 | 4096 | 2048 | 0.0030 | 0.382 ms | 0.303 ms | 1.26x |

Small M is the case that matters: each expert only sees the tokens routed to it.
The global scale is read from a device pointer inside the kernel -- passing it as
a Python float syncs once per expert, roughly 90 times per layer.

## Known limitations

- End to end is **6.0 tok/s** (16 tokens in 2.65 s, warm). CUDA graphs are still
  off because the NoPE attention reference does a host sync, and attention is
  unfused. **A capturable attention path is the next win** now that the MoE is
  grouped.
- Long greedy decodes still show phrase-level repetition. Short answers, facts,
  arithmetic and chain-of-thought are correct.
- Communication costs ~106 us for an 8 KiB all-reduce and 3.3 GB/s at 64 MiB,
  which bounds decode near 105 tok/s before any compute. NCCL is used as-is:
  `bench/pcie_allreduce.py` measures a NUMA-hierarchical variant (1.5-3x
  **slower** -- NCCL already reduces topology-aware) and reduce_scatter + FP8
  all_gather (1.07x faster, 4.5e-2 relative error, a poor trade). The 3.3 GB/s
  figure is much closer to the practical ceiling than PCIe's 25 GB/s per-GPU
  number suggests, since an 8-way all-reduce moves 2(N-1)/N of the data through
  shared host memory.

## Upstream bugs found

Worth reporting to sgl-project/sglang regardless of hardware:

1. `shared_experts_fusion_disable_reason` should refuse fusion when the shared
   expert's quantisation differs from the routed experts'.
2. `process_weights_after_loading` should never emit `inf` from
   `1/weight_global_scale`.
3. `routed_scaling_factor` is applied by nobody for compressed-tensors NVFP4
   MoE: top-k does not fold it in (the method is not in
   `_fuses_routed_scaling_factor_in_topk`) and the scheme passes
   `apply_routed_scaling_factor=False` to the CUTLASS runner.
4. `glm5_next` declares no `hf_to_sglang_mapper`, so the compressed-tensors
   ignore list never gets rewritten and startup fails on any hardware.
5. `_resolve_kpool_tail_backend` tests `device_sm_major >= 10` and routes sm120
   to an sm100-only backend.

## Tests

`bench/test_gather_roundtrip.py` and `bench/test_sparse_attn.py` check the sm120
attention path against fp32 references (3.3% and 4.5% max relative error, both
FP8 rounding). `bench/test_nvfp4_dequant.py` checks expert dequantisation
straight from the checkpoint.
