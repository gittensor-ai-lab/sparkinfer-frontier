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

## Long context

128K works. A needle planted anywhere in a 127,316-token prompt comes back
exactly, in ~118 s end to end at ~1150 tok/s prefill
(`bench/test_long_context.py`, 7/7 across depths 0.05-0.6 and lengths 32K-127K).

It goes further than that. Verified needle recall, all at depth 0.5:

| prompt | 128K | 256K | 512K |
|---|---|---|---|
| end to end | 118 s | 227 s | 456 s |
| | pass | pass | pass |

Prefill holds ~1125-1150 tok/s flat from 128K to 512K, so time is linear in
prompt length and the sparse indexer is not the thing that degrades.

**1M does not fit on this node.** The model would allow it -- GLM-5.3 declares
`max_position_embeddings=1048576`, and being pure NoPE it has no RoPE
extrapolation limit at all -- but the arithmetic does not:

| | per GPU |
|---|---:|
| card | 31.36 GB |
| NVFP4 weights | ~24.3 GB |
| left for KV, state, activations, graphs | ~7.0 GB |
| KV at 6.76 KB/token, 1M tokens | **7.1 GB** |

1M needs the whole remainder for KV alone, leaving nothing to run in. The
measured ceiling is **748,928 tokens** (`--mem-fraction-static 0.925`, 1.34 GB
free), which prefills a 256K prompt correctly. Pushing to 0.95 does allocate
859,648 tokens, and then OOMs on the first forward with 158 MiB free against a
272 MiB request -- capacity that allocates but cannot be used.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EXTRA_ARGS="--mem-fraction-static 0.925 --mamba-full-memory-ratio 0.02 \
            --max-running-requests 1 --cuda-graph-max-bs 2" \
  bench/serve-glm53.sh tp8_sm120mla
```

**Host KV offload does not raise this.** It is the obvious idea -- the box has
451 GiB of free host RAM -- but LMCache and HiCache are prefix-reuse tiers, not
paging for a running sequence. `LMCRadixCache` subclasses `RadixCache`, and
`--hicache-ratio` is documented as the host pool's size *relative to* the device
pool, which it never enlarges. Measured with HiCache attached and an 8x host
pool:

```
max_total_num_tokens = 193408          # identical to HiCache off
Input length (254646 tokens) exceeds the maximum allowed length (193402 tokens)
```

Every token a sequence is currently attending over has to be resident on the
GPU; the host tier only holds finished prefixes so a *later* request can reload
them on a hit. On this model that reload is also the bug above -- with HiCache
on, `test_prefix_cache.py` fails exactly as it did with the plain radix cache
(cold `4817-QUARTZ-9930`, warm `4830`). So host tiering costs correctness here
and buys no context.

`--enable-unified-memory` is a different mechanism -- one buffer split
dynamically between KV and KDA state rather than by a fixed ratio -- but it
requires the Triton attention backend, and sm120 DSA needs
`flashinfer_sparse_mla`. Its VMM arena reserves 256 GiB of *virtual* address
space backed by device memory, not host memory.

Reaching a real 1M means getting weights off these eight cards: TP=16 halves
them and frees ~12 GB/GPU. Quantising further is not the lever -- the weights
are already NVFP4 and the KV already fp8.

Getting there turned up a correctness bug worth stating on its own, because it
is invisible to every short test. Needle recall was failing reproducibly at
depths 0.05 and 0.1 for prompts of 32K and up -- returning a confident near-miss
(`4831`, `6888`) rather than nothing. It looked like sparse-index or KDA
state decay, and it was neither. **Flushing the radix cache before the identical
request makes it pass.**

`bench/test_prefix_cache.py` isolates it to one variable. Same tokens, same
greedy decode, same server; the only difference is whether the request hits a
cached prefix:

```
depth=0.05: cold=ok warm=BAD identical=False
  cold: '4817-QUARTZ-9930...'
  warm: '6888...'
```

GLM-5.3 is hybrid, so a prefix hit has to restore two things at the match point:
paged KV for the 11 MLA layers and a recurrent state for the 34 KDA layers. The
KV half is exact. The recurrent half is not, so a cache hit resumes from a state
that does not correspond to the tokens it is credited with -- and the model
answers fluently from it, which is why this reads as a model quality problem
rather than a cache bug. It reproduces on `extra_buffer` and
`extra_buffer_lazy` alike, and depends on the *match length* rather than the
needle position, which is why the failures looked like a band: at depth 0 there
is nothing cached to match, and past ~0.15 the match lands somewhere benign.

`--disable-radix-cache` is the fix, and it is what `tp8_sm120mla` now sets.
Prefix reuse is a real loss for multi-turn and shared-prefix serving; a wrong
answer is a worse one.

Every reachable way of keeping it was tried first, and all of them lose the
needle on a cache hit while the cold run recalls it:

| strategy | |
|---|---|
| `extra_buffer` (default) | **fail** -- warm `6888` |
| `extra_buffer_lazy` | **fail** -- warm `6888` |
| `--mamba-max-states-per-path 1` | **fail** -- warm `48` |
| `--enable-hierarchical-cache` | **fail** -- warm `4830` |
| `no_buffer` | unreachable: needs `page_size=1`, and `dsa_backend.py:1323` requires `page_size == 64` for the kpool path |

So this is not a tuning problem. Prefix reuse for `glm5_next` needs the state
handoff fixed upstream.

Two things had to change for the memory to be there. GLM-5.3 splits one budget between 34 KDA
layers' recurrent state and 11 MLA layers' paged KV; `--mamba-full-memory-ratio`
defaults to 0.9, which reserves most of it for KDA state and buys concurrency
nobody asked for on a box like this. At 0.25 the KV pool goes from 127,296 to
**193,408 tokens**, and `max_mamba_cache_size` falls from 43 to 17.

The other was mine. The NoPE attention reference gathered every query row's
top-k entries and upcast them in one allocation: at a 2048-row prefill chunk
with `index_topk=2048` that is 4.2M rows x 512 x 4 B = 8 GiB for the dequant
alone, ~17 GB peak, and all eight ranks OOMed. Short prompts never reached it
because the sequence was shorter than the top-k budget, so every earlier test
passed. Rows are independent given their own indices, so the forward now runs in
slices sized to a byte budget, with the slice count derived from shapes so graph
capture stays sync-free.

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
- Prefix caching is off, because a cache hit corrupts the KDA recurrent state;
  see **Long context**. Multi-turn and shared-prefix workloads pay full prefill.
- Prefill is chunked at 2048, so a full 128K prompt costs ~110 s before the
  first token, and 512K about 456 s. Context past ~128K also means
  `--max-running-requests 1`: the KV pool takes the memory concurrency needs.
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
6. A radix-cache prefix hit silently corrupts the KDA recurrent state on
   `glm5_next`: the same prompt answers correctly on a cold cache and wrongly on
   a hit. Every reachable strategy fails -- `extra_buffer`,
   `extra_buffer_lazy`, `--mamba-max-states-per-path 1` and
   `--enable-hierarchical-cache` -- and `no_buffer` cannot be selected at all,
   since it requires `page_size=1` while the DSA kpool path requires 64. Silent,
   and only reachable with prompts long enough to build a substantial cached
   prefix. Reproducer: `bench/test_prefix_cache.py`.

## Tests

| | |
|---|---|
| `bench/test_moe_fused.py` | grouped MoE vs per-expert reference (exact) |
| `bench/test_nvfp4_gemm.py` | fused GEMM vs dequantise + matmul |
| `bench/test_nvfp4_dequant.py` | expert dequantisation from the checkpoint |
| `bench/test_gather_roundtrip.py` | KV gather vs SGLang's own writer |
| `bench/test_sparse_attn.py` | sparse attention vs fp32 reference |
| `bench/test_sparse_attn_chunked.py` | row slicing vs a single pass (fp-reassociation bound) |
| `bench/test_long_context.py` | needle-in-a-haystack at 128K |
| `bench/test_prefix_cache.py` | cache-hit answer vs cold-cache answer |
| `bench/pcie_allreduce.py` | all-reduce strategies vs NCCL |
