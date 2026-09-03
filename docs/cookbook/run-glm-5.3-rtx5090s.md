# Running GLM-5.3-Flash (NVFP4) on 8x RTX 5090

This is the full technical record for serving `RedHatAI/GLM-5.3-Flash-NVFP4` on
an 8x RTX 5090 (sm120, consumer Blackwell) node with this fork. The [top-level
README](../../README.md) is the short version; this document is where the
detail lives — every fix, every measurement, and every dead end, so the next
person (or the next session) does not re-walk them.

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

`serve-glm53.sh` carries every workaround in [What it took to run at
all](#what-it-took-to-run-at-all), each commented with the failure it
prevents. The first request after a restart pays ~14 s of Triton JIT.

For long context, trade concurrency for KV:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EXTRA_ARGS="--mem-fraction-static 0.925 --mamba-full-memory-ratio 0.02 \
            --max-running-requests 1 --cuda-graph-max-bs 2" \
  bench/serve-glm53.sh tp8_sm120mla
```

## Performance

### A correction, stated up front

Earlier numbers in this project (including an earlier README) quoted **37
tok/s** decode, from timing a curl round trip for a 16-token completion on an
~11-token prompt with a bash stopwatch. That number is real but wrong for what
it claims to measure: every request pays a fixed ~190-200 ms of scheduler/IPC
overhead that does not shrink with prompt or output size, and on a 16-token
sample that overhead is most of the wall clock. Dividing total wall time by
total tokens attributes that fixed cost to "decode" and understates it by
roughly 2-3x.

Verified directly, same prompt, both methods, back to back:

```
curl wall-clock, 16 tokens:        437 ms  ->  "37 tok/s" (the old number)
server e2e_latency, 1-token request:      199 ms   (mostly fixed overhead + tiny prefill)
server e2e_latency, 16-token request:     412 ms
  decode: 15 steps in 213 ms = 14.2 ms/tok = 70.5 tok/s
```

The fix is to isolate decode with a subtraction rather than a single wall-clock
timing: request A (`max_new_tokens=1`) measures prefill plus the fixed
overhead, since SGLang produces the first output token inside the prefill/
extend step itself; request B (`max_new_tokens=K`) measures the same prefill
and overhead plus `K-1` decode steps. `B - A` cancels the overhead and the
prefill, leaving decode alone — both timings come from the server's own
`meta_info.e2e_latency`, so there is no client-side jitter either.
`bench/bench_context_sweep.py` implements this; `sglang.bench_serving` could
not be used here (see the tool note below).

### Prefill and decode across context length

Single stream (batch 1), measured with `bench/bench_context_sweep.py`, 128
decode steps per point:

| context | prompt tok | prefill | prefill tok/s | decode ms/tok | decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 1K | 987 | 1.01 s | 974 | 15.4 | 65.1 |
| 4K | 3,979 | 3.62 s | 1098 | 15.6 | 63.9 |
| 16K | 15,913 | 14.67 s | 1085 | 15.7 | 63.8 |
| 32K | 31,825 | 29.33 s | 1085 | 15.7 | 63.6 |
| 64K | 63,649 | 58.78 s | 1083 | 15.4 | 65.1 |
| 128K | 127,297 | 117.73 s | 1081 | 14.8 | 67.7 |
| 256K | 254,627 | 222.98 s | 1142 | 14.9 | 67.1 |
| 512K | 509,287 | 451.82 s | 1127 | 15.1 | 66.1 |

Decode cost does not scale with context length here, and that is architectural,
not incidental: the 34 KDA layers carry an O(1) recurrent state regardless of
sequence length, and the 11 MLA layers read a fixed `index_topk=2048` sparse
set per step rather than the full KV. Prefill is what scales, and it scales
linearly — the sparse indexer does not become the bottleneck as context grows.

**A methodology trap this table avoided.** The first measurement at 256K
returned decode at 180.7 tok/s and 512K at 90.9 tok/s — both far above the flat
~64-68 tok/s band every other point holds. That is not the sparse architecture
doing something clever at long context; it is a one-time Triton
kernel-specialization cost for a context-length shape the freshly-restarted
server had never seen, landing inside the `max_new_tokens=1` baseline request
and getting subtracted out of the decode estimate along with it, which makes
decode look artificially fast. Re-running both points on the now-warmed server
gave 67.1 and 66.1 tok/s — back in line with the rest of the table, which is
what is reported above. The lesson: an isolation method that is correct in
principle (see [the correction above](#a-correction-stated-up-front)) can still
be fooled by a one-time cost that happens to land inside the baseline instead
of the measured phase. Re-run anything that looks like an outlier before
trusting it.

### Where the throughput came from

The historical optimization narrative below predates the correction above and
was measured with the same curl-wall-clock method throughout, so the
**relative** gain (14.7x) is trustworthy even though the **absolute** endpoint
understates true decode speed. Treat this table as "how the work was staged,"
not as today's number — see the corrected figures above for that.

| stage | tok/s (uncorrected) | change |
|---|---:|---|
| per-expert reference | 2.5 | one kernel launch per expert, ~90 per layer |
| grouped MoE | 6.0 | two grouped GEMMs per layer, however many experts fire |
| + CUDA graphs | 34.1 | capture across 45 layers x 8 ranks |
| + adaptive block size | 36.8 | stop padding 8 routed rows out to 512 |

Communication bounds this box near 105 tok/s regardless of kernel work (see
[Communication](#communication)).

Decode at batch 1 is launch-bound, not compute-bound: the adaptive block size
cut the MoE's padded work 4x but moved end to end only 8%, and a hand-written
fused attention kernel came out *slower* than the reference it replaced (cuBLAS
einsum is hard to beat with a manual reduction). At the 36.8 tok/s stage,
per-token cost divided roughly as MoE 23 ms, MLA attention 8 ms, all-reduce
9.5 ms measured eagerly (graphs overlap much of that) — this breakdown is from
the same uncorrected-methodology stage as the table above, not a re-derivation
against the corrected ~15 ms/token total; it is kept for the shape of where the
time went, not as a current number.

### A tool note: `sglang.bench_serving` does not run on this box

`python -m sglang.bench_serving` fails with `ModuleNotFoundError: No module
named 'sglang.benchmark.serving'`, even though that file exists on disk. The
editable install's finder resolves `sglang.__path__` to `/root/sglang` (the
repo root) rather than `/root/sglang/python/sglang`, and the repo root happens
to have its own unrelated top-level `benchmark/` directory (community example
scripts) that shadows the real `sglang.benchmark` package for that one
submodule. Pre-existing on the box, not introduced by this fork's changes, and
out of scope to fix here — `bench_context_sweep.py` drives `/generate`
directly instead.

## Long context

Verified needle recall (`bench/test_long_context.py`), depth 0.5, radix cache
off:

| prompt | 128K | 256K | 512K |
|---|---|---|---|
| end to end | 118 s | 227 s | 456 s |
| | pass | pass | pass |

At 128K it is 7/7 across depths 0.05-0.6 and lengths 32K-127K.

Two things had to change to get here.

**The hybrid pool was misallocated.** GLM-5.3 splits one memory budget between
34 KDA layers' recurrent state and 11 MLA layers' paged KV.
`--mamba-full-memory-ratio` defaults to 0.9, reserving most of it for KDA state
and buying concurrency this box does not need. At 0.25 the KV pool goes from
127,296 to **193,408 tokens** and `max_mamba_cache_size` falls from 43 to 17.

**The other was a bug in this fork's own kernel.** The NoPE attention reference
gathered every query row's top-k entries and upcast them in one allocation: at
a 2048-row prefill chunk with `index_topk=2048` that is 4.2M rows x 512 x 4 B =
8 GiB for the dequant alone, ~17 GB peak, and all eight ranks OOMed. Short
prompts never reached it because the sequence was shorter than the top-k
budget, so every earlier test passed. Rows are independent given their own
indices, so the forward now runs in slices sized from free memory (a share of
`torch.cuda.mem_get_info()`, cached once and skipped entirely during graph
capture so it never bakes in one reading or syncs), with the slice count
derived from shapes so graph capture stays sync-free.

### 1M does not fit on this node

The model would allow it — GLM-5.3 declares `max_position_embeddings=1048576`,
and being pure NoPE it has no RoPE extrapolation limit at all. The arithmetic
does not:

| | per GPU |
|---|---:|
| card | 31.36 GB |
| NVFP4 weights | ~24.3 GB |
| left for KV, state, activations, graphs | ~7.0 GB |
| KV at 6.76 KB/token, 1M tokens | **7.1 GB** |

1M needs the whole remainder for KV alone, leaving nothing to run in. The
measured ceiling is **748,928 tokens** (`--mem-fraction-static 0.925`, 1.34 GB
free), which prefills a 256K prompt correctly. Pushing to 0.95 does allocate
859,648 tokens and then OOMs on the first forward with 158 MiB free against a
272 MiB request — capacity that allocates but cannot be used.

Reaching a real 1M means getting weights off these eight cards: TP=16 halves
them and frees ~12 GB/GPU. Quantising further is not the lever — the weights
are already NVFP4 and the KV already fp8.

### Host KV offload does not help

The obvious idea, given 451 GiB of free host RAM. But LMCache and HiCache are
prefix-reuse tiers, not paging for a running sequence: `LMCRadixCache`
subclasses `RadixCache`, and `--hicache-ratio` sizes the host pool *relative
to* the device pool, which it never enlarges. Measured with HiCache attached
and an 8x host pool:

```
max_total_num_tokens = 193408          # identical to HiCache off
Input length (254646 tokens) exceeds the maximum allowed length (193402 tokens)
```

Every token a sequence is attending over must be GPU-resident; the host tier
only holds finished prefixes for a *later* request to reload. On this model
that reload is the prefix-cache bug below, so host tiering costs correctness
and buys no context.

`--enable-unified-memory` is a different mechanism — one buffer split
dynamically between KV and KDA state rather than by a fixed ratio — but it
requires the Triton attention backend, and sm120 DSA needs
`flashinfer_sparse_mla`. Its VMM arena reserves 256 GiB of *virtual* address
space over device memory, not host.

## The prefix-cache bug (open)

A radix prefix hit on `glm5_next` returns a wrong-but-fluent answer. Cold cache
recalls the needle (`4817-QUARTZ-9930`); a hit returns a confident near-miss
(`4862`, `4830`, `6888`). It is invisible to every short test, because the
prompt has to be long enough to build a cached prefix worth matching.

GLM-5.3 is hybrid, so a prefix hit must restore two things at the match point:
paged KV for the 11 MLA layers and a recurrent state for the 34 KDA layers. The
KV half is exact. The recurrent half is not — and the model answers fluently
from the wrong state, which is why this reads as a model-quality problem
rather than a cache bug.

**Shipped fix: `--disable-radix-cache`**, set by `tp8_sm120mla`. Prefix reuse
is a real loss for multi-turn and shared-prefix serving; a wrong answer is
worse. Throughput and long context are unaffected.

### What is established

Verified with a self-checking cache flush (see the caution below):

| scenario | result |
|---|---|
| identical prompt twice, cache warm between (`test_cache_repeat.py`) | **pass**, byte-identical |
| identical prompt twice, flushed between (control) | **pass** |
| prime shares a prefix then diverges, then the real prompt | **fail** |

So reusing your own identical entry is fine; the corruption needs a
**branching** prefix — another request that shares a prefix and then diverges.
The branching request measures:

```
full_kv_hit=6336  mamba_boundary=0  dev_idx=0  branch=6336
```

It reuses nothing (`dev_idx=0`) yet is still assigned a branch point, with the
needle sitting at that branch.

### What is ruled out

- **A KV/recurrent-state depth gap on the reused path.** Every non-branching
  match measured has `full_kv_hit == mamba_boundary == dev_idx`.
- **Interior vs finished donor.** Suppressing mid-prefill checkpoint donations
  does make the test pass, but only because it also disables KV reuse
  (`cached_tokens` 2624 -> 0) — no better than turning the cache off.
- **Tagging checkpoints with an owning request.** Radix nodes are shared by
  key, not owned by a request, so this makes insert and match disagree and
  trips `new_prefix_len=2048, len(new_indices)=0`.
- **The branching-point checkpoint itself.** Forcing
  `mamba_branching_seqlen = None` does not fix it.
- **No serving-side strategy avoids it.** `extra_buffer`, `extra_buffer_lazy`,
  `--mamba-max-states-per-path 1`, and `--enable-hierarchical-cache` all fail.
  `no_buffer` is unselectable: it asserts `page_size=1`, and the DSA kpool path
  (`dsa_backend.py:1323`) requires `page_size == 64`. The two constraints are
  mutually exclusive for this model.

### A caution about this section's history

`/flush_cache` refuses while any request is running or queued and still
answers 200, with the refusal in the response body. An earlier round of
testing called it fire-and-forget, so flushes silently did nothing and "cold"
runs were served warm — which produced a confident but wrong diagnosis (that
the fault was specifically in-flight vs. finished checkpoint donors). That
conclusion does not survive re-testing and is retracted here rather than left
standing. `flush()` in `bench/test_prefix_cache.py` now retries until the body
confirms the flush and raises if it cannot.

Reproducers: `bench/test_cache_repeat.py` (one prompt twice, nothing else
moving), `bench/test_prefix_cache.py` (`--prime same|shifted`,
`--prime-tokens K`, `--needle-at N`).

## What it took to run at all

**Toolchain.** The image ships nvcc 12.8 but SM 12.x needs >= 12.9, so every
JIT kernel failed. `nvidia-cuda-nvcc-cu13` supplies 13.3, which then mismatches
the 13.0 runtime headers (CCCL hard-errors), needing
`-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` and a `libcudart.so` soname symlink.

**Shared-expert fusion.** The routed experts are NVFP4 but the shared expert
stays BF16, and the loader fuses it into expert slot `n_routed_experts` anyway.
That slot receives no scale, so `1/weight_global_scale` is `inf` and `inf * 0`
NaNs the layer. Every MoE backend read that poisoned scale, which is why all
of them failed on weights that measured healthy. `--disable-shared-experts-fusion`
is the fix, and this is **not** sm120-specific — the same checkpoint breaks on
Hopper.

**NoPE sparse MLA.** GLM-5.3 is `qk_rope_head_dim=0`; the compiled sm120
sparse-MLA kernel is shaped for `512 + rope 64` and misses on geometry, not on
dispatch. The NoPE route falls back to a layout-independent reference fed by a
flat-latent gather for `DSATokenToKVPool`'s 528 B rows (512 FP8 + 4 FP32 block
scales).

**Tensor-parallel deadlock.** `x[mask] = 0` sizes its write on the host, which
desynchronises ranks and hangs NCCL at 0% GPU with no error at all.
`masked_fill` is the sync-free form.

**Runner responsibilities.** Replacing a MoE runner means taking on everything
it did. Two config-driven behaviours are easy to miss, and both degrade output
while leaving short answers correct: `swiglu_limit` (10.0, clamps both SwiGLU
halves) and `routed_scaling_factor` (2.5, scales the combined routed output).

## Kernels

`sm120_nvfp4_moe_triton.py` fuses NVFP4 dequantisation into the GEMM: e2m1
nibbles are unpacked and per-group FP8 scales applied inside the matmul, so
weights are read once at 4 bits and never materialised in bfloat16. Tiles are
64x64x64 for sm120's 99 KB of shared memory, against sm100's 228 KB.

`sm120_nvfp4_moe_fused.py` removes the per-expert launch. `moe_align_block_size`
sorts routed (token, expert) pairs so a block owns one expert's rows, making
the MoE two grouped GEMMs regardless of how many experts fire. The row tile is
16/32/64 by batch, since every expert is padded up to a whole block.

Against the per-expert reference at GLM-5.3 shapes, 288 experts / top-8,
matching it exactly (`bench/test_moe_fused.py`):

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
closer to the practical ceiling than PCIe Gen4's ~25 GB/s per-GPU figure
suggests, since an 8-way all-reduce moves `2(N-1)/N` of the data through
shared host memory.

## Known limitations

- **Prefix caching is off** — a cache hit corrupts the KDA recurrent state.
  Multi-turn and shared-prefix workloads pay full prefill. See [the
  prefix-cache bug](#the-prefix-cache-bug-open).
- **Attention is still the unfused NoPE reference.** A fused Triton
  replacement was tried and abandoned: 2.3x slower, because `tl.sum` over an
  outer product gives up the tensor cores that cuBLAS uses. A `tl.dot`-based
  flash-decode is the version worth writing.
- **Long greedy decodes show phrase-level repetition.** Short answers, facts,
  arithmetic and chain-of-thought are correct.
- **Prefill is chunked at 2048**, so a full 128K prompt costs ~110 s before the
  first token, and 512K about 456 s. Context past ~128K also means
  `--max-running-requests 1`.
- **Communication caps this box near 105 tok/s** regardless of kernel work.

## Upstream bugs found

Worth reporting to sgl-project/sglang regardless of hardware:

1. `shared_experts_fusion_disable_reason` should refuse fusion when the shared
   expert's quantisation differs from the routed experts'.
2. `process_weights_after_loading` should never emit `inf` from
   `1/weight_global_scale`.
3. `routed_scaling_factor` is applied by nobody for compressed-tensors NVFP4
   MoE: top-k does not fold it in (the method is absent from
   `_fuses_routed_scaling_factor_in_topk`) and the scheme passes
   `apply_routed_scaling_factor=False` to the CUTLASS runner.
4. `glm5_next` declares no `hf_to_sglang_mapper`, so the compressed-tensors
   ignore list is never rewritten and startup fails on any hardware.
5. `_resolve_kpool_tail_backend` tests `device_sm_major >= 10` and routes
   sm120 to an sm100-only backend.
6. A radix-cache prefix hit silently corrupts the KDA recurrent state on
   `glm5_next` whenever the tree holds a **branching** prefix. Not yet root
   caused; see [the prefix-cache bug](#the-prefix-cache-bug-open) for what is
   established and what is ruled out. Reproducers: `bench/test_cache_repeat.py`,
   `bench/test_prefix_cache.py`.
7. `/flush_cache` answers 200 with a refusal string in the body when requests
   are in flight, so callers that do not read the body silently keep a
   populated cache. It should return a non-2xx, or block until it can flush.

## Tests

| | |
|---|---|
| `bench/test_moe_fused.py` | grouped MoE vs per-expert reference (exact) |
| `bench/test_nvfp4_gemm.py` | fused GEMM vs dequantise + matmul |
| `bench/test_nvfp4_dequant.py` | expert dequantisation from the checkpoint |
| `bench/test_gather_roundtrip.py` | KV gather vs SGLang's own writer |
| `bench/test_sparse_attn.py` | sparse attention vs fp32 reference |
| `bench/test_sparse_attn_chunked.py` | row slicing vs a single pass (fp-reassociation bound) |
| `bench/test_long_context.py` | needle-in-a-haystack, 32K-512K |
| `bench/test_cache_repeat.py` | one prompt twice, nothing else moving |
| `bench/test_prefix_cache.py` | cache-hit answer vs cold-cache answer |
| `bench/bench_context_sweep.py` | prefill/decode throughput, 1K-512K |
| `bench/pcie_allreduce.py` | all-reduce strategies vs NCCL |
