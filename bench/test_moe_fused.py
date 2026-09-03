"""Check the fused grouped MoE against the per-expert reference, then time both.

Correctness first: the fused path must match the reference that already produces
correct generations. Then measure the launch saving at GLM-5.3-Flash shapes --
288 experts, top-8, hidden 4096, intermediate 2048 sharded over TP=8.

Run:  python test_moe_fused.py
"""

import torch

from sglang.srt.layers.quantization.compressed_tensors.sm120_nvfp4_moe_fused import (
    sm120_nvfp4_moe_fused,
)
from sglang.srt.layers.quantization.compressed_tensors.sm120_nvfp4_moe_ref import (
    sm120_nvfp4_moe_forward,
)

E, H, I, TOPK = 288, 4096, 256, 8
GROUP = 16


def make_expert_weights(device: str):
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w13_s = (torch.rand(E, 2 * I, H // GROUP, device=device) * 0.02 + 0.005).to(
        torch.float8_e4m3fn
    )
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=device)
    w2_s = (torch.rand(E, H, I // GROUP, device=device) * 0.02 + 0.005).to(
        torch.float8_e4m3fn
    )
    gs13 = torch.full((E,), 1.0 / 17280.0, device=device)
    gs2 = torch.full((E,), 1.0 / 23040.0, device=device)
    return w13, w13_s, gs13, w2, w2_s, gs2


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    w13, w13_s, gs13, w2, w2_s, gs2 = make_expert_weights(dev)

    def timed(fn, iters=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    print(f"{'tokens':>7} {'experts hit':>12} {'ref ms':>9} {'fused ms':>9} "
          f"{'speedup':>8} {'rel err':>9}")
    print("-" * 60)

    for num_tokens in (1, 8, 32, 128):
        x = torch.randn(num_tokens, H, dtype=torch.bfloat16, device=dev) * 0.5
        logits = torch.randn(num_tokens, E, device=dev)
        topk_w, topk_i = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
        topk_i = topk_i.to(torch.int32)

        kw = dict(
            x=x, topk_weights=topk_w, topk_ids=topk_i,
            w13_weight=w13, w13_weight_scale=w13_s, w13_weight_scale_2=gs13,
            w2_weight=w2, w2_weight_scale=w2_s, w2_weight_scale_2=gs2,
            swiglu_limit=10.0, routed_scaling_factor=2.5,
        )

        ref = sm120_nvfp4_moe_forward(**kw)
        got = sm120_nvfp4_moe_fused(**kw)
        rel = (got.float() - ref.float()).abs().max().item() / (
            ref.float().abs().max().item() + 1e-9
        )

        t_ref = timed(lambda: sm120_nvfp4_moe_forward(**kw))
        t_fused = timed(lambda: sm120_nvfp4_moe_fused(**kw))
        hit = int(torch.unique(topk_i).numel())
        print(f"{num_tokens:>7} {hit:>12} {t_ref:>9.2f} {t_fused:>9.2f} "
              f"{t_ref / t_fused:>7.2f}x {rel:>9.4f}")


if __name__ == "__main__":
    main()
