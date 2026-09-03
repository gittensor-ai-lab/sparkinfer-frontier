"""Row slicing in _sm120_sparse_decode_fwd must not change the result.

The forward slices query rows to bound the gather's working set, which is what
lets a 128K prefill run at all. Slicing is only valid because rows are
independent given their own indices; this asserts that directly by forcing one
slice per row and comparing against a single-pass run on identical inputs.

Run on the GPU box:  python test_sparse_attn_chunked.py
"""

import torch

from sglang.kernels.ops.attention import flash_mla_sm120 as M
from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate

KV_LORA_RANK = 512
NUM_TOKENS = 96
NUM_KV = 512
TOPK = 128
HEADS = 8


def run(cache, q, idx, lens, budget):
    # Seed the cached budget so the forward slices by the amount this test wants
    # rather than by whatever the device happens to have free.
    saved = M._row_chunk_bytes
    M._row_chunk_bytes = budget
    try:
        out, lse = M._sm120_sparse_decode_fwd(
            q.unsqueeze(1), cache, idx.unsqueeze(1), lens, None,
            KV_LORA_RANK, KV_LORA_RANK ** -0.5,
        )
    finally:
        M._row_chunk_bytes = saved
    return out, lse


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"

    k_nope = torch.randn(NUM_KV, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=dev)
    k_rope = torch.empty(NUM_KV, 1, 0, dtype=torch.bfloat16, device=dev)
    nope_bytes, _ = quantize_k_cache_separate(k_nope, k_rope, tile_size=128)
    cache = nope_bytes.view(torch.float8_e4m3fn).reshape(NUM_KV, 1, -1)

    q = torch.randn(NUM_TOKENS, HEADS, KV_LORA_RANK, dtype=torch.bfloat16, device=dev)

    idx = torch.full((NUM_TOKENS, TOPK), -1, dtype=torch.int32, device=dev)
    lens = torch.randint(4, TOPK, (NUM_TOKENS,), device=dev, dtype=torch.int32)
    for i in range(NUM_TOKENS):
        n = int(lens[i])
        idx[i, :n] = torch.randperm(NUM_KV, device=dev)[:n].int()

    # A row with no valid entries at all: the lonely-row masking is the part most
    # likely to differ between a batched and a single-row pass.
    idx[0].fill_(-1)
    lens[0] = 0

    whole = run(cache, q, idx, lens, 1 << 40)
    sliced = run(cache, q, idx, lens, 1)

    steps = M._row_chunk_size(NUM_TOKENS, TOPK, cache.shape[-1], HEADS, KV_LORA_RANK)
    print(f"rows={NUM_TOKENS} topk={TOPK} one_slice_rows={steps}")

    # einsum reassociates its fp32 accumulation differently per batch size, so
    # the bf16 result can land one ULP apart. That is reassociation, not a
    # different computation -- the tell is that the error stays at ULP scale and
    # spreads across rows, where a slicing bug would concentrate on a few.
    ok = True
    for name, a, b in (("out", whole[0], sliced[0]), ("lse", whole[1], sliced[1])):
        a32, b32 = a.float(), b.float()
        same_nonfinite = (torch.isfinite(a32) == torch.isfinite(b32)).all().item()
        finite = torch.isfinite(a32) & torch.isfinite(b32)
        diff = (a32 - b32).abs()
        # The output is a sum of ~topk terms of order max|out|, so reassociation
        # perturbs it by one bf16 ULP *of that scale* -- not of each element.
        # Per-element relative bounds are meaningless where the sum cancels to
        # near zero, which is exactly where these differences show up.
        scale = max(a32[finite].abs().max().item() if finite.any() else 1.0, 1e-6)
        tol = scale * 2.0**-7
        in_tol = ((diff <= tol) | ~finite).all().item()
        n_diff = int((diff[finite] > 0).sum())
        rows_hit = int((diff.reshape(diff.shape[0], -1).amax(1) > 0).sum())
        worst = diff[finite].max().item() if finite.any() else 0.0
        mags = a32[finite][diff[finite] > 0].abs()
        mag_hi = mags.max().item() if mags.numel() else 0.0
        lonely_exact = bool((diff[0] == 0).all()) if diff.shape[0] else True
        good = same_nonfinite and in_tol
        ok &= good
        print(
            f"{name}: max_abs={worst:.3e} tol={tol:.3e} within_tol={in_tol} "
            f"elems_differing={n_diff}/{diff.numel()} rows_touched={rows_hit}/{diff.shape[0]} "
            f"largest_differing_value={mag_hi:.3e} scale={scale:.3e} "
            f"empty_row_exact={lonely_exact} nonfinite_match={same_nonfinite}"
        )

    print("PASS" if ok else "FAIL slicing changed the result beyond fp reassociation")


main()
