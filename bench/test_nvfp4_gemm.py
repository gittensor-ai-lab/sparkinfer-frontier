"""Correctness and speed for the fused sm120 NVFP4 GEMM.

Checks the Triton kernel against dequantize_nvfp4 + torch.matmul on the same
packed weights, then times both at GLM-5.3-Flash expert shapes (H=4096,
I=2048, so w13 is 2I x H and w2 is H x I, sharded by TP).

Run:  python test_nvfp4_gemm.py
"""

import torch

from sglang.srt.layers.quantization.compressed_tensors.sm120_nvfp4_moe_triton import (
    nvfp4_gemm,
)
from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

GROUP = 16


def make_weights(n: int, k: int, device: str):
    """Random packed NVFP4 weights with realistic per-group scales."""
    wq = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device)
    # e4m3 scales around 2^-6 keep dequantised weights near 0.02, like the real
    # expert tensors (measured absmax ~0.15, mean_abs ~0.016).
    ws = (torch.rand(n, k // GROUP, device=device) * 0.02 + 0.005).to(
        torch.float8_e4m3fn
    )
    return wq, ws


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    # Matches the checkpoint's stored reciprocal; kept on device so the kernel
    # reads it without a per-call sync.
    global_scale = torch.tensor(1.0 / 17280.0, device=dev)

    for M, N, K in [(64, 512, 4096), (256, 512, 4096), (1024, 4096, 2048)]:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
        wq, ws = make_weights(N, K, dev)

        ref_w = dequantize_nvfp4(wq, ws, None, out_dtype=torch.bfloat16)
        ref = (a.float() @ ref_w.float().T * global_scale.item()).to(torch.bfloat16)
        got = nvfp4_gemm(a, wq, ws, global_scale)

        err = (got.float() - ref.float()).abs().max().item()
        rel = err / (ref.float().abs().max().item() + 1e-9)
        ok = rel < 0.02

        def timed(fn, iters=30):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters

        t_fused = timed(lambda: nvfp4_gemm(a, wq, ws, global_scale))
        t_ref = timed(
            lambda: (
                a @ dequantize_nvfp4(wq, ws, None, out_dtype=torch.bfloat16).T
            ).mul_(global_scale.item())
        )

        print(f"M={M:<5} N={N:<5} K={K:<5} rel_err={rel:.4f} {'OK' if ok else 'FAIL'}"
              f" | dequant+matmul {t_ref:7.3f} ms | fused {t_fused:7.3f} ms"
              f" | {t_ref / t_fused:5.2f}x")


if __name__ == "__main__":
    main()
