# Frontier models on 8x RTX 5090

**GLM-5.3-Flash (NVFP4)** — 320B total, 18B active — serving at **37 tok/s** on
eight consumer RTX 5090s. A fork of [SGLang](https://github.com/sgl-project/sglang).

```
"Q: What is 15 times 4?"          ->  " 15 times 4 is 60."
"Q: Name the largest planet..."   ->  " Jupiter is the largest planet in our solar system."
"Q: Who wrote Romeo and Juliet?"  ->  " Romeo and Juliet was written by William Shakespeare."
"Why is the sky blue?"            ->  " ...sunlight is scattered by the atmosphere."
```

No datacenter GPU, no NVLink, no P2P.

## Where the performance came from

| stage | tok/s | change |
|---|---:|---|
| per-expert reference | 2.5 | one kernel launch per expert, ~90 per layer |
| grouped MoE | 6.0 | two grouped GEMMs per layer, however many experts fire |
| + CUDA graphs | 34.1 | capture across 45 layers x 8 ranks |
| **+ adaptive block size** | **36.8** | stop padding 8 routed rows out to 512 |

16 tokens in 435 ms warm, a 14.7x gain. Communication bounds this box near
105 tok/s.

Decode at batch 1 is launch-bound, not compute-bound: the adaptive block size cut
the MoE's padded work 4x but moved end to end only 8%, and a hand-written fused
attention kernel came out *slower* than the reference it replaced (cuBLAS einsum
is hard to beat with a manual reduction). Per-token cost divides roughly as MoE
23 ms, MLA attention 8 ms, all-reduce 9.5 ms measured eagerly -- graphs overlap
much of that.

## The node

| | |
|---|---|
| GPUs | 8x RTX 5090 (sm120), 32 GB each, 256 GB total |
| Interconnect | PCIe Gen4 x16, **no P2P** (CNS on all 28 pairs), dual NUMA |
| Host | 2x EPYC 7K62, 192 threads, 503 GiB RAM |
| Model | `RedHatAI/GLM-5.3-Flash-NVFP4`, 185 GiB |
| Residency | 24.8 GB per GPU, ~62 s to load |

## Run it

```bash
bench/serve-glm53.sh tp8_sm120mla     # serve: TP=8, sm120 sparse MLA, NVFP4 MoE
bench/bench-glm53.sh tp8_sm120mla     # latency + throughput
```

`serve-glm53.sh` carries every workaround below, each commented with the failure
it prevents. The first request after a restart pays ~14 s of Triton JIT.

## What it took to run at all

**Toolchain.** The image ships nvcc 12.8 but SM 12.x needs >= 12.9, so every JIT
kernel failed. `nvidia-cuda-nvcc-cu13` supplies 13.3, which then mismatches the
13.0 runtime headers (CCCL hard-errors), needing
`-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` and a `libcudart.so` soname symlink.

**Shared-expert fusion.** The routed experts are NVFP4 but the shared expert
stays BF16, and the loader fuses it into expert slot `n_routed_experts` anyway.
That slot receives no scale, so `1/weight_global_scale` is `inf` and `inf * 0`
NaNs the layer. Every MoE backend read that poisoned scale, which is why all of
them failed on weights that measured healthy. `--disable-shared-experts-fusion`
is the fix, and this is **not** sm120-specific — the same checkpoint breaks on
Hopper.

**NoPE sparse MLA.** GLM-5.3 is `qk_rope_head_dim=0`; the compiled sm120
sparse-MLA kernel is shaped for `512 + rope 64` and misses on geometry, not on
dispatch. The NoPE route falls back to a layout-independent reference fed by a
flat-latent gather for `DSATokenToKVPool`'s 528 B rows (512 FP8 + 4 FP32 block
scales).

**Tensor-parallel deadlock.** `x[mask] = 0` sizes its write on the host, which
desynchronises ranks and hangs NCCL at 0% GPU with no error at all. `masked_fill`
is the sync-free form.

**Runner responsibilities.** Replacing a MoE runner means taking on everything it
did. Two config-driven behaviours are easy to miss, and both degrade output while
leaving short answers correct: `swiglu_limit` (10.0, clamps both SwiGLU halves)
and `routed_scaling_factor` (2.5, scales the combined routed output).

## Kernels

`sm120_nvfp4_moe_triton.py` fuses NVFP4 dequantisation into the GEMM: e2m1
nibbles are unpacked and per-group FP8 scales applied inside the matmul, so
weights are read once at 4 bits and never materialised in bfloat16. Tiles are
64x64x64 for sm120's 99 KB of shared memory, against sm100's 228 KB.

`sm120_nvfp4_moe_fused.py` removes the per-expert launch. `moe_align_block_size`
sorts routed (token, expert) pairs so a block owns one expert's rows, making the
MoE two grouped GEMMs regardless of how many experts fire. The row tile is 16/32/64
by batch, since every expert is padded up to a whole block.

Against the per-expert reference at GLM-5.3 shapes, 288 experts / top-8, matching
it exactly (`bench/test_moe_fused.py`):

| tokens | experts hit | reference | fused | |
|---:|---:|---:|---:|---|
| 1 | 8 | 3.90 ms | 0.59 ms | **6.6x** |
| 8 | 62 | 29.00 ms | 0.59 ms | **49.2x** |
| 32 | 166 | 76.70 ms | 1.42 ms | **54.1x** |
| 128 | 280 | 129.47 ms | 2.96 ms | **43.7x** |

The single GEMM underneath is 3.4x at decode shapes
(`bench/test_nvfp4_gemm.py`). Read the global scale from a device pointer, not
`float(tensor)` — the latter syncs once per expert, ~90 times per layer.

CUDA graphs need every host sync gone from decode. Boolean-mask indexing is the
trap: `y[valid]` has a data-dependent extent, so PyTorch syncs to size it. Mask
to zero and scatter every row instead.

## Communication

NCCL, unmodified. `bench/pcie_allreduce.py` measures why:

| strategy | vs NCCL |
|---|---|
| NUMA-hierarchical | **1.5-3x slower** — NCCL already reduces topology-aware |
| FP8 inter-group hop | slower still — quantise kernels cost more than bytes saved |
| reduce_scatter + FP8 all_gather | 1.07x, at 4.5e-2 relative error |

An 8 KiB all-reduce costs ~106 us and 64 MiB reaches 3.3 GB/s. That is much
closer to the practical ceiling than PCIe Gen4's ~25 GB/s per-GPU figure suggests,
since an 8-way all-reduce moves `2(N-1)/N` of the data through shared host memory.

## Known limitations

- Attention is still the unfused NoPE reference. A fused Triton replacement was
  tried and abandoned: 2.3x slower, because `tl.sum` over an outer product gives
  up the tensor cores that cuBLAS uses. A `tl.dot`-based flash-decode is the
  version worth writing.
- Long greedy decodes show phrase-level repetition. Short answers, facts,
  arithmetic and chain-of-thought are correct.
- Communication caps this box near 105 tok/s regardless of kernel work.

## Upstream bugs found

Worth reporting to sgl-project/sglang regardless of hardware:

1. `shared_experts_fusion_disable_reason` should refuse fusion when the shared
   expert's quantisation differs from the routed experts'.
2. `process_weights_after_loading` should never emit `inf` from
   `1/weight_global_scale`.
3. `routed_scaling_factor` is applied by nobody for compressed-tensors NVFP4 MoE:
   top-k does not fold it in (the method is absent from
   `_fuses_routed_scaling_factor_in_topk`) and the scheme passes
   `apply_routed_scaling_factor=False` to the CUTLASS runner.
4. `glm5_next` declares no `hf_to_sglang_mapper`, so the compressed-tensors
   ignore list is never rewritten and startup fails on any hardware.
5. `_resolve_kpool_tail_backend` tests `device_sm_major >= 10` and routes sm120
   to an sm100-only backend.

## Tests

| | |
|---|---|
| `bench/test_moe_fused.py` | grouped MoE vs per-expert reference (exact) |
| `bench/test_nvfp4_gemm.py` | fused GEMM vs dequantise + matmul |
| `bench/test_nvfp4_dequant.py` | expert dequantisation from the checkpoint |
| `bench/test_gather_roundtrip.py` | KV gather vs SGLang's own writer |
| `bench/test_sparse_attn.py` | sparse attention vs fp32 reference |
| `bench/pcie_allreduce.py` | all-reduce strategies vs NCCL |
