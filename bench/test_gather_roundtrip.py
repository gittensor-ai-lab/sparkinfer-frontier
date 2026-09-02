"""B2 -- isolate the sm120 NoPE gather: quantize known tensors, read them back.

Two end-to-end attempts at GLM-5.3 on sm120 produced garbage, one applying the
trailing FP32 scale rows and one ignoring them. Each costs a ~6 minute server
reload to evaluate, which is the wrong loop for a numerics bug. This drives
SGLang's own writer (quantize_k_cache_separate) with known values and checks
_gather_flat_latent against it directly, so a wrong dequant is visible in seconds
and a correct one clears the gather as a suspect.

Run on the GPU box:  python test_gather_roundtrip.py
"""

import torch

from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
from sglang.kernels.ops.attention.flash_mla_sm120 import _gather_flat_latent

KV_LORA_RANK = 512
TILE = 128
NUM_TOKENS = 64


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"

    # Realistic magnitudes: MLA latents are roughly unit-scale activations.
    k_nope = torch.randn(NUM_TOKENS, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=dev)
    k_rope = torch.empty(NUM_TOKENS, 1, 0, dtype=torch.bfloat16, device=dev)

    nope_bytes, _ = quantize_k_cache_separate(k_nope, k_rope, tile_size=TILE)
    print(f"writer output: shape={tuple(nope_bytes.shape)} dtype={nope_bytes.dtype}")

    # The pool stores these bytes as float8_e4m3fn; the gather sees that view.
    cache = nope_bytes.view(torch.float8_e4m3fn).reshape(NUM_TOKENS, 1, -1)
    print(f"cache as pool sees it: {tuple(cache.shape)} {cache.dtype}")

    indices = torch.arange(NUM_TOKENS, device=dev).view(1, NUM_TOKENS)
    got = _gather_flat_latent(cache, indices).float().view(NUM_TOKENS, KV_LORA_RANK)
    want = k_nope.float().view(NUM_TOKENS, KV_LORA_RANK)

    err = (got - want).abs()
    rel = err.max().item() / want.abs().max().item()
    print(f"\nwant  [0, :6] = {want[0, :6].tolist()}")
    print(f"got   [0, :6] = {got[0, :6].tolist()}")
    print(f"\nmax abs err = {err.max().item():.5f}   max rel err = {rel:.5f}")
    print(f"mean abs err = {err.mean().item():.5f}")

    # FP8 e4m3 carries ~2 decimal digits, so per-tile quantisation lands well
    # inside 5% relative error. Anything above that is a layout bug, not rounding.
    print("\nVERDICT:", "gather OK" if rel < 0.05 else "GATHER IS WRONG")


if __name__ == "__main__":
    main()
