"""SM120 FlashMLA sparse decode implementation.

On SM120 (Blackwell Desktop / RTX PRO 6000) the flash_mla CUDA kernel
is not available, so this module provides alternative implementations:

- A fused Triton kernel (default, ``SGLANG_SM120_TRITON_FLASHMLA=1``)
- A pure-PyTorch fallback (``SGLANG_SM120_TRITON_FLASHMLA=0``)

The FP8 KV cache uses a page-internal layout where NOPE+ROPE data has
stride (nope_dim + rope_dim*2) per token, and scales are stored in a
separate region at the end of each page.
"""

import logging
import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)
_is_hip = is_hip()

_GLM_DSA_MODEL_ARCHS = (
    "GlmMoeDsaForCausalLM",
    "GlmMoeDsaForCausalLMNextN",
    # GLM-5.3-Flash runs the same DSA sparse-MLA attention. On sm120 no other DSA
    # backend applies: fa3/trtllm are sm100-only, tilelang needs 148 KB smem vs 99 KB.
    "Glm5NextForConditionalGeneration",
    "Glm5NextForConditionalGenerationNextN",
)

# Page layout constants for DSv4-Flash (MODEL1):
#   nope_dim = 448, rope_dim = 64, quantize_block_size = 64
#   nope_rope_stride = 448 + 64*2 = 576 bytes per token
#   scale_stride = ceil(448/64) + 1 = 8 bytes per token (7 scales + 1 pad)
#   bytes_per_token = 448 + 128 + 8 = 584
#   page_bytes = ceil_div(page_size * 584, 576) * 576

_NOPE_DIM = 448
_ROPE_DIM = 64
_NOPE_ROPE_STRIDE = _NOPE_DIM + _ROPE_DIM * 2  # 576
_TILE_SIZE = 64
_NUM_TILES = _NOPE_DIM // _TILE_SIZE  # 7
_SCALE_STRIDE = _NUM_TILES + 1  # 8 (7 scales + 1 pad)
_D = _NOPE_DIM + _ROPE_DIM  # 512


_LATENT_BLOCK = 128  # DSATokenToKVPool.quant_block_size


def _gather_flat_latent(k_cache, indices):
    """Gather from DSATokenToKVPool's flat cache: (num_tokens, 1, D) fp8, no paging."""
    d_total = k_cache.shape[-1]
    flat = k_cache.reshape(-1, d_total)
    gathered = flat[indices.clamp(min=0).reshape(-1)]  # (N, d_total)

    # calculate_mla_kv_cache_dim packs a row as kv_lora_rank FP8 values, then
    # kv_lora_rank//quant_block_size FP32 scales, then rope*2 bytes (0 when NoPE).
    latent = (d_total * _LATENT_BLOCK) // (_LATENT_BLOCK + 4)
    if latent == d_total or latent % _LATENT_BLOCK:
        return gathered.to(torch.bfloat16).reshape(*indices.shape, d_total)

    # quantize_k_cache_separate stores tile/scale_inv with scale_inv =
    # abs(tile).max()/448, so dequant multiplies the FP32 rows back in.
    n_blocks = latent // _LATENT_BLOCK
    vals = gathered[:, :latent].to(torch.float32).view(-1, n_blocks, _LATENT_BLOCK)
    scales = (
        gathered[:, latent:]
        .contiguous()
        .view(torch.uint8)
        .view(torch.float32)[:, :n_blocks]
        .view(-1, n_blocks, 1)
    )
    out = (vals * scales).view(-1, latent).to(torch.bfloat16)
    return out.reshape(*indices.shape, latent)


def _gather_and_dequant(k_cache, indices, page_size):
    """Gather KV entries from the paged buffer using correct page-internal addressing.

    Args:
        k_cache: (num_pages, page_size, 1, bytes_per_token) float8_e4m3fn
                 Non-contiguous view of the raw page buffer.
        indices: (...) int32/int64, token-level indices. -1 = invalid.
        page_size: tokens per page (256)

    Returns:
        kv: (..., _D) bfloat16, dequantized KV vectors
    """
    # DSv4 hands back a 4D paged buffer (num_pages, page_size, 1, bytes_per_token)
    # whose bytes need unpacking. A 3D buffer is an unpacked latent cache
    # (num_tokens, 1, D) -- GLM-5.3's shape -- and gathers directly.
    if k_cache.dim() == 3:
        return _gather_flat_latent(k_cache, indices)

    idx_shape = indices.shape
    flat_idx = indices.reshape(-1)  # (N,)
    N = flat_idx.shape[0]
    device = k_cache.device

    # Page-level addressing
    page_bytes = k_cache.stride(0)  # actual byte stride between pages
    pages = flat_idx // page_size
    offsets = flat_idx % page_size

    # Clamp invalid indices
    safe_pages = pages.clamp(min=0)
    safe_offsets = offsets.clamp(min=0)

    # Access raw buffer as uint8 — use as_strided to get full page view
    num_pages = k_cache.shape[0]
    raw_pages = k_cache.as_strided(
        (num_pages, page_bytes),
        (page_bytes, 1),
    ).view(
        torch.uint8
    )  # (num_pages, page_bytes) uint8
    # Note: float8_e4m3fn and uint8 are both 1 byte, view is safe

    # Compute byte offsets within each page
    # NOPE: page[safe_page, safe_offset * 576 + 0:448]
    # ROPE: page[safe_page, safe_offset * 576 + 448:576]
    # SCALES: page[safe_page, page_size * 576 + safe_offset * 8 + 0:7]

    nope_base = safe_offsets * _NOPE_ROPE_STRIDE  # (N,)
    nope_offsets = nope_base.unsqueeze(-1) + torch.arange(
        _NOPE_DIM, device=device, dtype=torch.long
    )  # (N, 448)

    rope_base = nope_base + _NOPE_DIM  # (N,)
    rope_offsets = rope_base.unsqueeze(-1) + torch.arange(
        _ROPE_DIM * 2, device=device, dtype=torch.long
    )  # (N, 128)

    scale_section_offset = page_size * _NOPE_ROPE_STRIDE  # 147456
    scale_base = scale_section_offset + safe_offsets * _SCALE_STRIDE  # (N,)
    scale_offsets = scale_base.unsqueeze(-1) + torch.arange(
        _NUM_TILES, device=device, dtype=torch.long
    )  # (N, 7)

    # Gather bytes per page — use advanced indexing
    # raw_pages[safe_pages, nope_offsets] → (N, 448)
    page_idx_nope = safe_pages.unsqueeze(-1).expand_as(nope_offsets)
    nope_bytes = raw_pages[page_idx_nope, nope_offsets]  # (N, 448) uint8

    page_idx_rope = safe_pages.unsqueeze(-1).expand_as(rope_offsets)
    rope_bytes = raw_pages[page_idx_rope, rope_offsets]  # (N, 128) uint8

    page_idx_scale = safe_pages.unsqueeze(-1).expand_as(scale_offsets)
    scale_bytes = raw_pages[page_idx_scale, scale_offsets]  # (N, 7) uint8

    # Reinterpret dtypes
    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # (N, 448)
    rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)  # (N, 64)
    scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)  # (N, 7)

    # Dequantize: nope_tile * scale_tile → bf16 (vectorized)
    result = torch.empty(N, _D, dtype=torch.bfloat16, device=device)
    result[:, :_NOPE_DIM] = (
        (
            nope_fp8.view(N, _NUM_TILES, _TILE_SIZE).float()
            * scale_e8m0.view(N, _NUM_TILES, 1).float()
        )
        .view(N, _NOPE_DIM)
        .to(torch.bfloat16)
    )
    result[:, _NOPE_DIM:] = rope_bf16

    return result.reshape(*idx_shape, _D)


# Working-set budget for one slice of query rows. The gather materialises
# topk entries per row and upcasts them to fp32, so a full 2048-row prefill
# chunk at index_topk=2048 asks for ~17 GB and OOMs a 32 GB card. Rows are
# independent given their own indices, so the forward runs in slices.
#
# A fixed budget cannot serve both ends: a 128K-context config leaves GBs free,
# while a 860K one leaves ~0.6 GB, where a 512 MB slice is itself the OOM. So
# take a share of what is actually free, once, and cache it -- free memory is
# stable after the pools are carved, and re-reading it per call would put a
# cudaMemGetInfo in the decode path.
_ROW_CHUNK_MIN = 64 << 20
_ROW_CHUNK_MAX = 512 << 20
_ROW_CHUNK_SHARE = 0.35
_row_chunk_bytes = None


def _row_chunk_budget():
    global _row_chunk_bytes
    if _row_chunk_bytes is None:
        # Capture records a fixed graph, so querying the driver here would both
        # bake in one reading and risk a sync. Decode captures few rows anyway.
        if torch.cuda.is_current_stream_capturing():
            return _ROW_CHUNK_MIN
        free, _ = torch.cuda.mem_get_info()
        _row_chunk_bytes = int(
            max(_ROW_CHUNK_MIN, min(_ROW_CHUNK_MAX, free * _ROW_CHUNK_SHARE))
        )
    return _row_chunk_bytes


def _row_chunk_size(n_rows, topk_total, bytes_per_entry, num_heads, head_dim_v):
    """Rows per slice, derived from shapes alone so graph capture stays sync-free."""
    # Per row: the fp8 gather, its fp32 dequant and bf16 result, the fp32 copy
    # the einsum consumes, and scores/weights over heads.
    latent = max(bytes_per_entry - 16, head_dim_v)
    per_row = topk_total * (bytes_per_entry + latent * 10 + num_heads * 8)
    return max(1, min(n_rows, _row_chunk_budget() // max(per_row, 1)))


def _sparse_attend_rows(
    q_rows,
    k_cache,
    page_size,
    indices_rows,
    invalid_rows,
    extra_k_cache,
    extra_page_size,
    extra_indices_rows,
    attn_sink,
    head_dim_v,
    softmax_scale,
):
    """Attention over a slice of independent query rows: (N, H, D) against (N, T)."""
    H_q = q_rows.shape[1]
    gathered_kv = _gather_and_dequant(k_cache, indices_rows, page_size)

    if extra_indices_rows is not None:
        extra_kv = _gather_and_dequant(
            extra_k_cache, extra_indices_rows, extra_page_size
        )
        gathered_kv = torch.cat([gathered_kv, extra_kv], dim=1)

    # masked_fill, not gathered_kv[mask] = 0: boolean-index assignment syncs to
    # host to size the write, which deadlocks the TP collectives.
    gathered_kv = gathered_kv.masked_fill(invalid_rows.unsqueeze(-1), 0.0)

    kv_f = gathered_kv.float()
    kv_d = kv_f.shape[-1]
    q_f = q_rows.float()
    if q_f.shape[-1] != kv_d:
        q_f = q_f[..., :kv_d]

    scores = torch.einsum("nhd,ntd->nht", q_f, kv_f) * softmax_scale
    scores.masked_fill_(invalid_rows.unsqueeze(1).expand_as(scores), float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)

    if attn_sink is not None:
        lse_for_out = torch.logsumexp(
            torch.stack([lse, attn_sink.view(1, H_q).expand_as(lse)], dim=0), dim=0
        )
    else:
        lse_for_out = lse.clone()

    lonely = lse == float("-inf")
    lse_for_out = lse_for_out.masked_fill(lonely, float("inf"))
    weights = torch.exp(scores - lse_for_out.unsqueeze(-1))
    out = torch.einsum("nht,ntv->nhv", weights, kv_f[..., :head_dim_v])
    out = out.masked_fill(lonely.unsqueeze(-1), 0.0)

    return out.to(torch.bfloat16), lse


_PROBE_INDEX_LAYOUT = os.environ.get("SGLANG_SM120_PROBE_INDEX") == "1"
_probe_seen = 0


def _probe_index_layout(indices):
    """Report whether valid index entries are packed at the front of each row.

    topk_lengths masks positions past the count of non-negative entries, which is
    only equivalent to masking -1 when the valid entries are contiguous. If they
    are not, that mask drops real keys off the tail of long rows.
    """
    global _probe_seen
    if _probe_seen >= 6:
        return
    _probe_seen += 1
    rows = indices.reshape(-1, indices.shape[-1])
    valid = rows >= 0
    counts = valid.sum(-1)
    # Packed <=> the last valid position in a row is exactly count-1.
    pos = torch.arange(rows.shape[-1], device=rows.device).expand_as(rows)
    last_valid = torch.where(valid, pos, torch.full_like(pos, -1)).amax(-1)
    packed = (last_valid == counts - 1) | (counts == 0)
    holes = (counts - 1 - last_valid).abs()
    print(
        f"[probe] rows={rows.shape[0]} topk={rows.shape[-1]} "
        f"count[min/med/max]={int(counts.min())}/{int(counts.median())}/{int(counts.max())} "
        f"packed_rows={int(packed.sum())}/{rows.shape[0]} "
        f"max_tail_gap={int(holes.max())} "
        f"maxidx={int(rows.max())}",
        flush=True,
    )


def _sm120_sparse_decode_fwd(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    extra_k_cache=None,
    extra_indices=None,
    extra_topk_length=None,
):
    B, s_q, H_q, D_qk = q.shape
    if k_cache.dim() == 4:
        num_pages, page_size, H_k, bpt = k_cache.shape
    else:
        # Flat latent cache (num_tokens, H_k, D) -- GLM-5.3's DSATokenToKVPool
        # shape. page_size only feeds the paged gather, which this layout skips.
        num_pages, H_k, bpt = k_cache.shape
        page_size = 1
    topk = indices.shape[-1]

    invalid_mask = indices < 0
    safe_indices = indices.clamp(min=0)

    if topk_length is not None:
        topk_range = torch.arange(topk, device=topk_length.device).view(1, 1, topk)
        invalid_mask = invalid_mask | (topk_range >= topk_length.view(B, 1, 1))

    extra_safe = None
    extra_invalid = None
    extra_page_size = None
    topk_total = topk
    if extra_k_cache is not None and extra_indices is not None:
        extra_topk = extra_indices.shape[-1]
        extra_page_size = extra_k_cache.shape[1]
        extra_invalid = extra_indices < 0
        extra_safe = extra_indices.clamp(min=0)
        if extra_topk_length is not None:
            extra_range = torch.arange(
                extra_topk, device=extra_topk_length.device
            ).view(1, 1, extra_topk)
            extra_invalid = extra_invalid | (
                extra_range >= extra_topk_length.view(B, 1, 1)
            )
        invalid_mask = torch.cat([invalid_mask, extra_invalid], dim=2)
        topk_total += extra_topk

    # (B, s_q) are both row dimensions here -- the NoPE caller passes tokens in B
    # with s_q=1 -- so flatten them and slice uniformly over the result.
    n_rows = B * s_q
    q_rows = q.reshape(n_rows, H_q, D_qk)
    idx_rows = safe_indices.reshape(n_rows, topk)
    invalid_rows = invalid_mask.reshape(n_rows, topk_total)
    extra_rows = None if extra_safe is None else extra_safe.reshape(n_rows, -1)

    step = _row_chunk_size(n_rows, topk_total, bpt, H_q, head_dim_v)
    out = torch.empty(
        (n_rows, H_q, head_dim_v), dtype=torch.bfloat16, device=q.device
    )
    lse = torch.empty((n_rows, H_q), dtype=torch.float32, device=q.device)

    for start in range(0, n_rows, step):
        stop = min(start + step, n_rows)
        out[start:stop], lse[start:stop] = _sparse_attend_rows(
            q_rows[start:stop],
            k_cache,
            page_size,
            idx_rows[start:stop],
            invalid_rows[start:stop],
            extra_k_cache,
            extra_page_size,
            None if extra_rows is None else extra_rows[start:stop],
            attn_sink,
            head_dim_v,
            softmax_scale,
        )

    out = out.reshape(B, s_q, H_q, head_dim_v)
    lse = lse.reshape(B, s_q, H_q)
    return out, lse.permute(0, 2, 1)


# SM120 FlashMLA: default FlashInfer (CUTLASS SM120 sparse MLA decode).
# Override with SGLANG_SM120_FLASHMLA_BACKEND=triton|torch to force fallback.
_sm120_default_backend = envs.SGLANG_SM120_FLASHMLA_BACKEND.get()


def flash_mla_with_kvcache_sm120(**kwargs):
    """SM120 FlashMLA sparse decode entry point.

    Dispatches to FlashInfer (default if available), Triton, or PyTorch fallback.
    """
    q = kwargs["q"]
    k_cache = kwargs["k_cache"]
    indices = kwargs["indices"]
    topk_length = kwargs.get("topk_length")
    attn_sink = kwargs.get("attn_sink")
    head_dim_v = kwargs["head_dim_v"]
    softmax_scale = kwargs.get("softmax_scale")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    extra_k_cache = kwargs.get("extra_k_cache")
    extra_indices = kwargs.get("extra_indices_in_kvcache")
    extra_topk_length = kwargs.get("extra_topk_length")

    if _sm120_default_backend == "flashinfer":
        return _flash_mla_flashinfer(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            head_dim_v,
            softmax_scale,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
        )

    if _sm120_default_backend == "triton":
        from sglang.kernels.ops.attention.flash_mla_sm120_triton import (
            flash_mla_sparse_decode_triton,
        )

        out, lse = flash_mla_sparse_decode_triton(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            head_dim_v,
            softmax_scale,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
        )
        return (out, lse)

    out, lse = _sm120_sparse_decode_fwd(
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink,
        head_dim_v,
        softmax_scale,
        extra_k_cache,
        extra_indices,
        extra_topk_length,
    )
    return (out, lse)


# --- Page-split utilities: pbs=256 → pbs=64 ---
# SGLang SWA KV cache footer layout per 256-token page:
#   [data: 256 * 576 bytes] [scale: 256 * 8 bytes] [padding]
# FlashInfer decode_dsv4 expects per 64-token page:
#   [data: 64 * 576 bytes] [scale: 64 * 8 bytes] [padding to 37440]
_PBS_SRC = 256  # SGLang physical page size
_PBS_DST = 64  # FlashInfer page_block_size
_NOPE_ROPE_STRIDE = 576  # bytes per token for nope+rope
_SCALE_STRIDE = 8  # bytes per token for scale (7 + 1 pad)
_BYTES_PER_DST_PAGE = (
    _PBS_DST * _NOPE_ROPE_STRIDE + _PBS_DST * _SCALE_STRIDE
)  # 64*576 + 64*8 = 37376 + 512 = 37888
# Padded to 576 alignment

_BYTES_PER_DST_PAGE_PADDED = math.ceil(_BYTES_PER_DST_PAGE / 576) * 576  # 37440


@triton.jit
def _page_split_kernel(
    src_ptr,
    dst_ptr,
    N_pages,
    src_stride0: tl.constexpr,
    dst_stride0: tl.constexpr,
    DATA_PER_SUB: tl.constexpr,  # 64 * 576 = 36864
    SCALE_PER_SUB: tl.constexpr,  # 64 * 8 = 512
    SRC_SCALE_OFF: tl.constexpr,  # 256 * 576 = 147456
    DST_SCALE_OFF: tl.constexpr,  # 64 * 576 = 36864
    RATIO: tl.constexpr,  # 4
    BLOCK_SIZE: tl.constexpr,
    mask_ptr,
    HAS_MASK: tl.constexpr,
):
    """Fused page-split: copy data+scale for all sub-pages in one kernel.

    When HAS_MASK is set, only pages flagged in ``mask_ptr`` (int8, 1=touched)
    are copied; untouched pages are skipped so the kernel no longer rewrites the
    entire KV pool every decode step.
    """
    pid = tl.program_id(0)
    page_idx = pid // RATIO
    sub = pid % RATIO

    if page_idx >= N_pages:
        return

    if HAS_MASK:
        if tl.load(mask_ptr + page_idx) == 0:
            return

    src_base = src_ptr + page_idx * src_stride0
    dst_base = dst_ptr + (page_idx * RATIO + sub) * dst_stride0

    # Copy data region: DATA_PER_SUB bytes from src offset sub*DATA_PER_SUB
    data_src_off = sub * DATA_PER_SUB
    for start in tl.range(0, DATA_PER_SUB, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < DATA_PER_SUB
        vals = tl.load(src_base + data_src_off + offs, mask=mask)
        tl.store(dst_base + offs, vals, mask=mask)

    # Copy scale region: SCALE_PER_SUB bytes
    scale_src_off = SRC_SCALE_OFF + sub * SCALE_PER_SUB
    for start in tl.range(0, SCALE_PER_SUB, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < SCALE_PER_SUB
        vals = tl.load(src_base + scale_src_off + offs, mask=mask)
        tl.store(dst_base + DST_SCALE_OFF + offs, vals, mask=mask)


@triton.jit
def _page_mark_kernel(
    indices_ptr,
    mask_ptr,
    N_idx,
    SRC_PBS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Mark touched source pages (1 byte each) from token-level indices.

    ``indices`` are token indices into the pbs=SRC_PBS SWA pool; -1 = invalid.
    Each valid token marks ``mask[token // SRC_PBS] = 1``. Concurrent stores of
    the same value 1 are safe (no atomic needed).
    """
    pid = tl.program_id(0)
    if pid >= N_idx:
        return
    idx = tl.load(indices_ptr + pid)
    if idx < 0:
        return
    page = idx // SRC_PBS
    tl.store(mask_ptr + page, 1)


def _split_kv_pages_to_64(
    kv_u8: torch.Tensor,
    src_pbs: int,
    touched_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Split pbs=N footer-format pages into pbs=64 footer-format pages.

    When ``touched_indices`` (token-level int32 indices into the pbs=src_pbs
    SWA pool, -1 = invalid) is provided, only the source pages that actually
    contain a referenced token are copied. This avoids rewriting the entire KV
    pool on every decode step (only ~2*batch pages are touched vs the full
    pool). The output buffer is persistent and reused across steps; untouched
    dst pages simply retain their (unreferenced) stale data.
    """
    assert src_pbs % _PBS_DST == 0 and src_pbs >= _PBS_DST
    if src_pbs == _PBS_DST:
        return kv_u8

    N = kv_u8.shape[0]
    ratio = src_pbs // _PBS_DST
    num_dst_pages = N * ratio

    from sglang.srt.runtime_context import get_resources

    # Pre-allocated grow-only buffer for page-split output per device.
    dev = kv_u8.device
    buffers = get_resources().buffers
    key = f"flash_mla_sm120_split:{dev}"
    buf = buffers.get(key)
    if buf is None or buf.shape[0] < num_dst_pages:
        # The first allocation can happen under inference mode (autotune), but
        # the buffer is written again during CUDA graph capture outside
        # inference mode, where an inference tensor cannot be mutated.
        with torch.inference_mode(False):
            buf = torch.empty(
                num_dst_pages,
                _BYTES_PER_DST_PAGE_PADDED,
                dtype=torch.uint8,
                device=dev,
            )
        buffers[key] = buf
    out = buf[:num_dst_pages]

    # Get raw 2D view of source
    src_2d = kv_u8
    if src_2d.ndim == 4:
        src_stride0 = src_2d.stride(0)
        src_2d = torch.as_strided(src_2d, (N, src_stride0), (src_stride0, 1))
    else:
        src_stride0 = src_2d.stride(0)

    use_mask = touched_indices is not None and touched_indices.numel() > 0
    mask_ptr = src_2d  # dummy, never dereferenced when HAS_MASK is False
    if use_mask:
        # Persistent per-device int8 mask, zeroed each call (cheap memset,
        # captured cleanly by CUDA graph). 1 = page is referenced this step.
        mkey = f"flash_mla_sm120_mask:{dev}"
        mbuf = buffers.get(mkey)
        if mbuf is None or mbuf.shape[0] < N:
            # The first allocation can happen under inference mode (autotune),
            # but the buffer is zeroed again later during CUDA graph capture
            # outside inference mode -- an inference tensor cannot be mutated
            # there, so force a normal tensor.
            with torch.inference_mode(False):
                mbuf = torch.empty(N, dtype=torch.int8, device=dev)
            buffers[mkey] = mbuf
        mask = mbuf[:N]
        mask.zero_()
        idx_flat = touched_indices.reshape(-1).contiguous()
        if idx_flat.dtype != torch.int32:
            idx_flat = idx_flat.to(torch.int32)
        _page_mark_kernel[(idx_flat.numel(),)](
            idx_flat,
            mask,
            idx_flat.numel(),
            src_pbs,  # SRC_PBS
            1024,  # BLOCK (unused, kept for JIT signature)
        )
        mask_ptr = mask

    grid = (N * ratio,)
    _page_split_kernel[grid](
        src_2d,
        out,
        N,
        src_stride0,
        _BYTES_PER_DST_PAGE_PADDED,
        _PBS_DST * _NOPE_ROPE_STRIDE,  # DATA_PER_SUB = 36864
        _PBS_DST * _SCALE_STRIDE,  # SCALE_PER_SUB = 512
        src_pbs * _NOPE_ROPE_STRIDE,  # SRC_SCALE_OFF = 147456
        _PBS_DST * _NOPE_ROPE_STRIDE,  # DST_SCALE_OFF = 36864
        ratio,  # RATIO = 4
        1024,  # BLOCK_SIZE
        mask_ptr,
        use_mask,  # HAS_MASK
    )

    bpt = _NOPE_ROPE_STRIDE + _SCALE_STRIDE  # 584
    return out.as_strided(
        (num_dst_pages, _PBS_DST, 1, bpt),
        (_BYTES_PER_DST_PAGE_PADDED, bpt, bpt, 1),
    )


def _flash_mla_flashinfer(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    extra_k_cache,
    extra_indices,
    extra_topk_length,
):
    """FlashInfer SM120 sparse MLA via the paged-attention dispatcher.

    SGLang SWA pool uses page_size=256 (footer format: 256*576 bytes data + 256*8 bytes scale).
    FlashInfer decode_dsv4 fast path requires page_block_size=64 (footer: 64*576 + 64*8).
    We split 256-token pages into 4 virtual 64-token pages.
    Token indices are invariant under page-split (identity mapping).
    """
    from flashinfer.mla._sparse_mla_sm120 import (
        _DECODE_MAX_TOKENS as _FI_DECODE_MAX_TOKENS,
    )
    from flashinfer.mla._sparse_mla_sm120 import (
        _sparse_mla_sm120_paged_attention,
    )

    B, _, H, D = q.shape  # (batch, 1, num_heads, head_dim)
    dev = q.device

    # Indices: no remapping needed (page-split preserves token addressing).
    idx = indices.squeeze(1) if indices.dim() == 3 else indices

    # --- Page-split: convert pbs=N kv_cache to pbs=64 view ---
    # Only the SWA pages actually referenced by `idx` are copied (the rest of
    # the persistent dst buffer is left untouched and never read).
    kv_u8 = k_cache.view(torch.uint8) if k_cache.dtype != torch.uint8 else k_cache
    src_pbs = k_cache.shape[1] if k_cache.ndim >= 3 else _PBS_SRC
    kv_64 = (
        _split_kv_pages_to_64(kv_u8, src_pbs, touched_indices=idx)
        if src_pbs != _PBS_DST
        else kv_u8
    )

    extra_kv_u8 = (
        extra_k_cache.view(torch.uint8)
        if extra_k_cache is not None and extra_k_cache.dtype != torch.uint8
        else extra_k_cache
    )
    extra_kv_64 = extra_kv_u8

    extra_idx = (
        extra_indices.squeeze(1)
        if extra_indices is not None and extra_indices.dim() == 3
        else extra_indices
    )

    output = torch.empty(B, H, head_dim_v, dtype=torch.bfloat16, device=dev)
    out_lse = torch.empty(B, H, dtype=torch.float32, device=dev)

    # Use split-K for decode-sized batches and paged attention otherwise.
    if B <= _FI_DECODE_MAX_TOKENS:
        topk = idx.shape[-1]
        extra_topk = extra_idx.shape[-1] if extra_idx is not None else 0
        _BI = 64
        num_splits = (topk + _BI - 1) // _BI + (
            (extra_topk + _BI - 1) // _BI if extra_topk > 0 else 0
        )
        mid_out = torch.empty(
            B, H, num_splits, head_dim_v, dtype=torch.bfloat16, device=dev
        )
        mid_lse = torch.empty(B, H, num_splits, dtype=torch.float32, device=dev)
    else:
        mid_out = None
        mid_lse = None

    _sparse_mla_sm120_paged_attention(
        q.squeeze(1) if q.ndim == 4 else q,
        kv_64,
        idx,
        output,
        out_lse,
        softmax_scale,
        d_v=head_dim_v,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_64,
        extra_indices=extra_idx,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    return (output.unsqueeze(1), None)


def _validate_flashinfer_sparse_mla_backend(
    *,
    model_arch: str,
    device_sm_major: int,
    kv_cache_dtype: torch.dtype,
    prefill_impl: str,
    decode_impl: str,
) -> bool:
    selected = {prefill_impl, decode_impl}
    uses_flashinfer_sparse_mla = "flashinfer_sparse_mla" in selected
    is_glm_sm12_fp8 = (
        model_arch in _GLM_DSA_MODEL_ARCHS
        and device_sm_major == 12
        and kv_cache_dtype == torch.float8_e4m3fn
        and not _is_hip
    )
    if uses_flashinfer_sparse_mla and not is_glm_sm12_fp8:
        raise ValueError(
            "flashinfer_sparse_mla supports only GLM DSA with FP8 KV cache "
            "on NVIDIA SM120/SM121; "
            f"got model_arch={model_arch!r}, sm_major={device_sm_major}, "
            f"kv_cache_dtype={kv_cache_dtype}, prefill_impl={prefill_impl!r}, "
            f"decode_impl={decode_impl!r}."
        )
    if is_glm_sm12_fp8:
        unsupported = selected - {"flashinfer_sparse_mla"}
        if unsupported:
            raise ValueError(
                "GLM DSA with FP8 KV cache on NVIDIA SM120/SM121 supports "
                "only flashinfer_sparse_mla, "
                f"but got {sorted(unsupported)}."
            )
    return uses_flashinfer_sparse_mla


def flashinfer_sparse_mla_forward(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    page_size: int,
    kv_cache_dim: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sm_scale: float,
    skip_softmax_threshold_scale_factor: float | None,
) -> torch.Tensor:
    """Run FlashInfer's SM120 sparse MLA kernel on SGLang's packed DSA cache."""
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    topk = indices.shape[1]

    # The compiled sm120 kernel is shaped for kv_lora_rank=512 with
    # qk_rope_head_dim=64; pure-NoPE models miss it on geometry, not on dispatch.
    if qk_rope_head_dim == 0:
        # The page table is a reused fixed-width buffer, so entries past the
        # current row length can hold stale non-negative indices from an earlier,
        # longer sequence. indices < 0 alone would attend to those; the row length
        # bounds it. (Measured: 190 valid entries for a 19-token prefill, exactly
        # the causal count, so valid entries are packed at the front.)
        topk_lengths = (indices >= 0).sum(dim=-1).to(torch.int32)
        if _PROBE_INDEX_LAYOUT:
            _probe_index_layout(indices)
        out, _ = _sm120_sparse_decode_fwd(
            q.unsqueeze(1),
            kv_cache,
            indices.unsqueeze(1),
            topk_lengths,
            None,
            kv_lora_rank,
            sm_scale,
        )
        out = out.squeeze(1)
        return out

    result = trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1),
        kv_cache=kv_cache.view(torch.uint8)
        .view(-1, page_size, kv_cache_dim)
        .unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=indices.unsqueeze(1),
        seq_lens=seq_lens,
        max_seq_len=topk,
        sparse_mla_top_k=topk,
        bmm1_scale=float(sm_scale),
        bmm2_scale=1.0,
        kv_scale_format="arbitrary_fp32",
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
    )
    return result.squeeze(1)
