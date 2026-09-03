"""Fused NVFP4 expert GEMM for sm120.

The reference path in sm120_nvfp4_moe_ref.py dequantises a whole expert to
bfloat16 before multiplying, so every token that routes to an expert pays for
materialising 2I x H and H x I weight matrices. This unpacks e2m1 nibbles and
applies the per-group scale inside the GEMM instead, so the weights are read
once at 4 bits and never written back out.

sm120 has 99 KB of shared memory per block against sm100's 228 KB, so the tiles
here are sized for the smaller budget rather than inherited from a datacenter
config.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# NVFP4: 4-bit e2m1 values, one FP8 e4m3 scale per group of 16 along K.
_GROUP = 16
# e2m1 has 8 magnitudes; the sign comes from the high bit of the nibble.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@triton.jit
def _nvfp4_gemm_kernel(
    a_ptr, wq_ptr, ws_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_sn, stride_sk,
    stride_om, stride_on,
    gs_ptr,
    lut_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr,
):
    """out[M,N] = a[M,K] @ dequant(wq[N,K/2], ws[N,K/16]).T

    BLOCK_K is a multiple of 16 so a K-tile never straddles a scale group.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    lut = tl.load(lut_ptr + tl.arange(0, 8))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Two e2m1 values share a byte: column k lives in byte k//2, low nibble
        # for even k. Load the byte block once and select per column.
        offs_b = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        packed = tl.load(
            wq_ptr + offs_n[:, None] * stride_wn + offs_b[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_b[None, :] < K // 2),
            other=0,
        ).to(tl.int32)

        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        # Interleave back to K order: [lo0, hi0, lo1, hi1, ...].
        nib = tl.interleave(lo, hi)
        mag = tl.load(lut_ptr + (nib & 0x7).reshape(BLOCK_N * BLOCK_K)).reshape(
            BLOCK_N, BLOCK_K
        )
        w = tl.where((nib & 0x8) != 0, -mag, mag)

        offs_g = (k0 // GROUP) + tl.arange(0, BLOCK_K // GROUP)
        scales = tl.load(
            ws_ptr + offs_n[:, None] * stride_sn + offs_g[None, :] * stride_sk,
            mask=(offs_n[:, None] < N) & (offs_g[None, :] < K // GROUP),
            other=0.0,
        ).to(tl.float32)
        # Broadcast each group's scale across its 16 columns.
        w = w * tl.reshape(
            tl.broadcast_to(scales[:, :, None], (BLOCK_N, BLOCK_K // GROUP, GROUP)),
            (BLOCK_N, BLOCK_K),
        )

        acc += tl.dot(a, tl.trans(w.to(tl.bfloat16)).to(a.dtype), allow_tf32=True)

    # Read the per-expert global scale on device: float(tensor) in the caller
    # would sync once per expert, ~90 times per layer.
    acc = acc * tl.load(gs_ptr)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


_LUT: dict[torch.device, torch.Tensor] = {}


def _lut(device: torch.device) -> torch.Tensor:
    if device not in _LUT:
        _LUT[device] = torch.tensor(_E2M1, dtype=torch.float32, device=device)
    return _LUT[device]


def nvfp4_gemm(
    a: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """a [M,K] bf16 @ dequant(wq [N,K/2] uint8, ws [N,K/16] fp8).T -> [M,N] bf16."""
    M, K = a.shape
    N = wq.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)

    # 64x64x64 keeps a tile inside sm120's 99 KB: the weight tile is the large
    # one and lands at 64*64*4 B in registers/shared after unpack.
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _nvfp4_gemm_kernel[grid](
        a, wq, ws, out,
        M, N, K,
        a.stride(0), a.stride(1),
        wq.stride(0), wq.stride(1),
        ws.stride(0), ws.stride(1),
        out.stride(0), out.stride(1),
        global_scale.reshape(()),
        _lut(a.device),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP=_GROUP,
        num_warps=4, num_stages=3,
    )
    return out
