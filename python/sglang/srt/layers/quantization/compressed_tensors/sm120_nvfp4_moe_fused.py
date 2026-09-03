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

def _block_m(num_tokens: int, top_k: int) -> int:
    """Pick the row tile so decode does not pad 8 real rows out to 512.

    moe_align_block_size rounds every expert up to a whole block, so at batch 1
    each of the top_k experts owns one row and a 64-row tile wastes 63/64 of the
    work. 16 is the floor where tl.dot still uses tensor cores.
    """
    routed = num_tokens * top_k
    if routed <= 64:
        return 16
    if routed <= 512:
        return 32
    return 64



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

    block_m = _block_m(num_tokens, top_k)
    sorted_ids, expert_ids, num_valid = moe_align_block_size(
        topk_ids, block_m, num_experts
    )

    # gate+up for every routed pair in one launch, then the same for down.
    h = nvfp4_grouped_gemm(
        x, w13_weight, w13_weight_scale, w13_weight_scale_2,
        sorted_ids, expert_ids, num_valid, top_k, block_m,
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
        identity, expert_ids, num_valid, top_k, block_m,
    )

    # Scatter the sorted rows back onto their tokens, weighted by the router.
    out = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    # sorted_ids is longer than num_valid and its tail is uninitialised, so bound
    # both ends: negative garbage would otherwise index backwards.
    valid = (sorted_ids >= 0) & (sorted_ids < num_tokens * top_k)
    # Keep the shape static. Boolean-mask indexing (y[valid]) has a
    # data-dependent extent, which forces a device-to-host sync and cannot be
    # captured in a CUDA graph; masking to zero and scattering every row costs
    # the padded rows but stays capturable.
    safe = torch.where(valid, sorted_ids, torch.zeros_like(sorted_ids)).to(torch.int64)
    keep = valid.unsqueeze(-1).to(y.dtype)
    contrib = y * keep
    if not apply_router_weight_on_input:
        w = topk_weights.reshape(-1)[safe].unsqueeze(-1).to(contrib.dtype)
        contrib = contrib * w
    out.index_add_(0, safe // top_k, contrib)

    if routed_scaling_factor is not None:
        out = out * routed_scaling_factor
    return out
