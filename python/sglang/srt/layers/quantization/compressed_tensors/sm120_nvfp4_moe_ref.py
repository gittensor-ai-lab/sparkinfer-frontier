"""Reference compressed-tensors NVFP4 MoE for sm120.

The two runners this scheme can build both fail on consumer Blackwell: the
FlashInfer TRT-LLM FP4 MoE is sm100-only, and the FlashInfer CUTLASS path reads
the ModelOpt NVFP4 weight layout, so handing it a compressed-tensors checkpoint
returns NaN rather than an error. Measured on 8x RTX 5090: expert weights load
correctly and the MoE input is clean, yet the layer output is entirely NaN.

This dequantises to bfloat16 and runs a plain grouped GEMM instead. It is a
correctness path, not a fast one -- it materialises one expert's weights at a
time and loops over the experts a batch touches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4


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

        w13 = dequantize_nvfp4(
            w13_weight[expert], w13_weight_scale[expert], w13_weight_scale_2[expert]
        )
        w2 = dequantize_nvfp4(
            w2_weight[expert], w2_weight_scale[expert], w2_weight_scale_2[expert]
        )

        rows = x[tokens].to(torch.bfloat16)
        gate, up = (rows @ w13.transpose(0, 1)).chunk(2, dim=-1)
        y = (F.silu(gate) * up) @ w2.transpose(0, 1)

        if not apply_router_weight_on_input:
            weight = (topk_weights * selected).sum(dim=-1)[tokens]
            y = y * weight.unsqueeze(-1).to(y.dtype)
        out.index_add_(0, tokens, y.to(out.dtype))

    return out
