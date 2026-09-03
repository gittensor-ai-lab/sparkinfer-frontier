"""Reference compressed-tensors NVFP4 MoE for sm120.

The two runners this scheme can build both fail on consumer Blackwell: the
FlashInfer TRT-LLM FP4 MoE is sm100-only, and the FlashInfer CUTLASS path reads
the ModelOpt NVFP4 weight layout, so handing it a compressed-tensors checkpoint
returns NaN rather than an error. Measured on 8x RTX 5090: expert weights load
correctly and the MoE input is clean, yet the layer output is entirely NaN.

Each expert's GEMMs go through nvfp4_gemm, a Triton kernel that unpacks the
e2m1 nibbles and applies the per-group scale inside the matmul, so weights are
never materialised in bfloat16. The loop over the experts a batch touches
remains: that is the next thing to fuse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.compressed_tensors.sm120_nvfp4_moe_triton import (
    nvfp4_gemm,
)


def sm120_nvfp4_moe_forward(
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
    """Run the routed experts for ``x`` and return the combined output.

    ``w13`` is [E, 2I, H//2] packed with gate rows first and up rows second;
    ``w2`` is [E, H, I//2]. Scales are per group of 16 along the input dim, and
    the ``_2`` globals are already reciprocals (process_weights_after_loading
    stores 1/global_scale), so they multiply the per-group scale directly.
    """
    if activation != "silu":
        raise NotImplementedError(
            f"sm120 NVFP4 MoE reference supports silu/SwiGLU, got {activation!r}"
        )

    out = torch.zeros_like(x, dtype=torch.bfloat16)
    if apply_router_weight_on_input:
        x = x * topk_weights.sum(dim=-1, keepdim=True).to(x.dtype)

    # process_weights_after_loading stores 1/weight_global_scale, so an expert
    # whose global scale is 0 -- an unfilled slot in the padded expert dim --
    # arrives as inf. inf * 0 is NaN, and one such expert poisons the whole
    # output through index_add_, so skip experts without a finite scale.
    finite = torch.isfinite(w13_weight_scale_2) & torch.isfinite(w2_weight_scale_2)

    for expert in torch.unique(topk_ids).tolist():
        if not bool(finite[expert]):
            continue
        selected = topk_ids == expert
        tokens = selected.any(dim=-1).nonzero(as_tuple=True)[0]
        if tokens.numel() == 0:
            continue

        # nvfp4_gemm unpacks and scales inside the GEMM, so the expert's weights
        # are read once at 4 bits instead of being materialised in bfloat16.
        rows = x[tokens].to(torch.bfloat16)
        gate, up = nvfp4_gemm(
            rows, w13_weight[expert], w13_weight_scale[expert],
            w13_weight_scale_2[expert],
        ).chunk(2, dim=-1)
        if swiglu_limit is not None:
            # GLM-5.3 clamps both halves before the product (swiglu_limit=10.0);
            # leaving SwiGLU unbounded degrades long generations into repetition.
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        y = nvfp4_gemm(
            (F.silu(gate) * up).to(torch.bfloat16), w2_weight[expert],
            w2_weight_scale[expert], w2_weight_scale_2[expert],
        )

        if not apply_router_weight_on_input:
            weight = (topk_weights * selected).sum(dim=-1)[tokens]
            y = y * weight.unsqueeze(-1).to(y.dtype)
        out.index_add_(0, tokens, y.to(out.dtype))

    # The MoE runners scale the combined routed output (GLM-5.3: 2.5); bypassing
    # them means applying it here or the experts land 2.5x under the residual.
    if routed_scaling_factor is not None:
        out = out * routed_scaling_factor

    return out
