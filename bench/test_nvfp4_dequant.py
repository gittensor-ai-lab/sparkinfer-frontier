"""Check the compressed-tensors NVFP4 expert dequant against the checkpoint.

The NVFP4 MoE produces NaN from clean inputs and correctly-loaded weights on
sm120, so before writing a kernel, confirm what the packed expert weights
actually dequantize to. Reads one expert straight out of the safetensors --
no server, no GPU pool -- and applies SGLang's own dequantize_nvfp4.

Layout per compressed_tensors_w4a4_nvfp4_moe.create_weights:
    weight_packed  (2I, H//2)  uint8      two e2m1 values per byte
    weight_scale   (2I, H//16) fp8_e4m3   per group of 16
    weight_global_scale ()     fp32       process_weights_after_loading stores 1/x

Run:  python test_nvfp4_dequant.py [layer] [expert]
"""

import json
import sys

import torch
from safetensors import safe_open

from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

CKPT = "/root/models/GLM-5.3-Flash-NVFP4"
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EXPERT = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def main() -> None:
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    base = f"model.language_model.layers.{LAYER}.mlp.experts.{EXPERT}"

    def load(suffix):
        key = f"{base}.{suffix}"
        if key not in idx:
            cands = [k for k in idx if k.startswith(base)][:6]
            raise KeyError(f"{key} absent; siblings: {cands}")
        with safe_open(f"{CKPT}/{idx[key]}", framework="pt") as f:
            return f.get_tensor(key)

    for proj in ("gate_proj", "up_proj", "down_proj"):
        wq = load(f"{proj}.weight_packed")
        ws = load(f"{proj}.weight_scale")
        wg = load(f"{proj}.weight_global_scale")
        print(f"\n--- {proj} ---")
        print(f"  packed {tuple(wq.shape)} {wq.dtype} | scale {tuple(ws.shape)} {ws.dtype} "
              f"| global {tuple(wg.shape)} {wg.dtype} = {wg.flatten()[:2].tolist()}")

        # process_weights_after_loading stores the reciprocal, so the runtime
        # scale_2 that reaches the kernel is 1/global.
        s2 = (1.0 / wg.float()).reshape(())
        deq = dequantize_nvfp4(wq.cuda(), ws.cuda(), s2.cuda(), out_dtype=torch.bfloat16)
        d = deq.float()
        print(f"  dequant {tuple(deq.shape)}: absmax={d.abs().max():.4f} "
              f"mean_abs={d.abs().mean():.5f} nan={int(d.isnan().sum())} "
              f"inf={int(d.isinf().sum())} zero={100.0 * (d == 0).float().mean():.1f}%")
        # A healthy MoE weight row sits well under 1.0 in magnitude.
        ok = d.isfinite().all() and 0.001 < d.abs().mean() < 1.0
        print(f"  VERDICT: {'plausible' if ok else 'IMPLAUSIBLE'}")


if __name__ == "__main__":
    main()
