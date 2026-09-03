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

## Known limitations

- The MoE reference materialises one expert at a time in a Python loop, and CUDA
  graphs are off. **A fused sm120 NVFP4 MoE kernel is the open performance work**;
  the reference exists to validate it against.
- Long greedy decodes still show phrase-level repetition. Short answers, facts,
  arithmetic and chain-of-thought are correct.
- Communication is the ceiling on this box: a 8 KiB all-reduce costs ~106 us and
  64 MiB reaches 3.3 GB/s against PCIe Gen4's ~25 GB/s, capping decode near
  105 tok/s before any compute. See `bench/bench_allreduce.py`.

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
