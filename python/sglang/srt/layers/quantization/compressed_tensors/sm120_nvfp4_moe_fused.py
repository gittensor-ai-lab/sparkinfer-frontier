"""Fused sm120 NVFP4 MoE: one grouped GEMM per projection, not one per expert.

The reference path launches two kernels for every expert a batch touches -- about
90 experts per layer over 42 MoE layers -- so decode is launch-bound and a faster
GEMM barely shows up. This sorts the routed (token, expert) pairs once with
moe_align_block_size and runs a single grouped GEMM per projection, where each
block reads its own expert id.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.layers.quantization.compressed_tensors.sm120_nvfp4_moe_triton import (
    nvfp4_grouped_gemm,
)

_BLOCK_M = 64


def sm120_nvfp4_moe_fused(
    *,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    routed_scaling_factor: float | None = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    if activation != "silu":
        raise NotImplementedError(
            f"sm120 NVFP4 MoE supports silu/SwiGLU, got {activation!r}"
        )

    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    num_experts = w13_weight.shape[0]
    x = x.to(torch.bfloat16)

    if apply_router_weight_on_input:
        x = x * topk_weights.sum(dim=-1, keepdim=True).to(x.dtype)

    sorted_ids, expert_ids, num_valid = moe_align_block_size(
        topk_ids, _BLOCK_M, num_experts
    )

    # gate+up for every routed pair in one launch, then the same for down.
    h = nvfp4_grouped_gemm(
        x, w13_weight, w13_weight_scale, w13_weight_scale_2,
        sorted_ids, expert_ids, num_valid, top_k, _BLOCK_M,
    )
    gate, up = h.chunk(2, dim=-1)
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    act = (F.silu(gate) * up).to(torch.bfloat16)

    # The second GEMM consumes rows already in sorted order, so it indexes them
    # directly rather than through sorted_ids again.
    identity = torch.arange(
        sorted_ids.shape[0], device=x.device, dtype=sorted_ids.dtype
    ) * top_k
    y = nvfp4_grouped_gemm(
        act, w2_weight, w2_weight_scale, w2_weight_scale_2,
        identity, expert_ids, num_valid, top_k, _BLOCK_M,
    )

    # Scatter the sorted rows back onto their tokens, weighted by the router.
    out = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    # sorted_ids is longer than num_valid and its tail is uninitialised, so bound
    # both ends: negative garbage would otherwise pass an upper-bound-only test
    # and index backwards.
    valid = (sorted_ids >= 0) & (sorted_ids < num_tokens * top_k)
    rows = sorted_ids[valid]
    contrib = y[valid]
    if not apply_router_weight_on_input:
        w = topk_weights.reshape(-1)[rows].unsqueeze(-1).to(contrib.dtype)
        contrib = contrib * w
    out.index_add_(0, (rows // top_k).to(torch.int64), contrib)

    if routed_scaling_factor is not None:
        out = out * routed_scaling_factor
    return out
