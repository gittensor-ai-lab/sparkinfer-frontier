"""B2 -- check _sm120_sparse_decode_fwd against hand-computed sparse attention.

test_gather_roundtrip.py cleared the gather. This clears (or convicts) the layer
above it: drive the same entry point the NoPE route uses, with the same argument
shapes, and compare against softmax(q@k^T * scale) @ v computed in fp32 from the
pre-quantisation tensors. A mismatch here means the call convention is wrong --
softmax_scale, head_dim_v, or the topk_length semantics -- not the dequant.

Run on the GPU box:  python test_sparse_attn.py
"""

import torch

from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
from sglang.kernels.ops.attention.flash_mla_sm120 import _sm120_sparse_decode_fwd

KV_LORA_RANK = 512
QK_NOPE = 256          # GLM-5.3: sm_scale is derived from this, not from 512
NUM_TOKENS = 8         # queries
NUM_KV = 256           # tokens in cache
TOPK = 64
HEADS = 8


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"

    k_nope = torch.randn(NUM_KV, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=dev)
    k_rope = torch.empty(NUM_KV, 1, 0, dtype=torch.bfloat16, device=dev)
    nope_bytes, _ = quantize_k_cache_separate(k_nope, k_rope, tile_size=128)
    cache = nope_bytes.view(torch.float8_e4m3fn).reshape(NUM_KV, 1, -1)

    q = torch.randn(NUM_TOKENS, HEADS, KV_LORA_RANK, dtype=torch.bfloat16, device=dev)

    # Packed valid indices then -1 padding, which is what the DSA page table holds.
    idx = torch.full((NUM_TOKENS, TOPK), -1, dtype=torch.int32, device=dev)
    lens = torch.randint(8, TOPK, (NUM_TOKENS,), device=dev, dtype=torch.int32)
    for i in range(NUM_TOKENS):
        idx[i, : lens[i]] = torch.randperm(NUM_KV, device=dev)[: lens[i]].int()

    sm_scale = QK_NOPE ** -0.5

    got, _ = _sm120_sparse_decode_fwd(
        q.unsqueeze(1), cache, idx.unsqueeze(1),
        (idx >= 0).sum(dim=-1).to(torch.int32), None, KV_LORA_RANK, sm_scale,
    )
    got = got.squeeze(1).float()

    # Reference: dequantise nothing, attend over the ORIGINAL bf16 latents.
    kv = k_nope.view(NUM_KV, KV_LORA_RANK).float()
    want = torch.zeros(NUM_TOKENS, HEADS, KV_LORA_RANK, device=dev)
    for i in range(NUM_TOKENS):
        sel = idx[i, : lens[i]].long()
        k = kv[sel]                                        # (n, 512)
        s = (q[i].float() @ k.T) * sm_scale                # (H, n)
        want[i] = torch.softmax(s, dim=-1) @ k             # (H, 512)

    err = (got - want).abs()
    rel = err.max().item() / want.abs().max().item()
    print(f"want [0,0,:4] = {want[0, 0, :4].tolist()}")
    print(f"got  [0,0,:4] = {got[0, 0, :4].tolist()}")
    print(f"\nmax abs err = {err.max().item():.5f}   max rel err = {rel:.5f}")
    # FP8 KV plus bf16 output rounding; anything past 10% is a real mismatch.
    print("VERDICT:", "attention OK" if rel < 0.10 else "CALL CONVENTION IS WRONG")


if __name__ == "__main__":
    main()
