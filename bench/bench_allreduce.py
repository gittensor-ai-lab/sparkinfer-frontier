"""Quantify the TP all-reduce cost on a PCIe-only, dual-NUMA 8x RTX 5090 node.

This is moat 2's business case, measured without needing a model to serve.

GLM-5.3-Flash decode does 2 all-reduces per layer over 45 layers = 90 per token.
At hidden=4096 bf16 that is 8 KiB per all-reduce at batch 1 -- latency-bound, not
bandwidth-bound, which is exactly the regime where routing through host memory
hurts most (P2P reads CNS on all 28 pairs here).

Run:  torchrun --nproc_per_node=8 bench_allreduce.py
"""

import os
import statistics
import torch
import torch.distributed as dist

# Message sizes that matter for GLM-5.3-Flash (hidden_size=4096, bf16).
# Decode at batch B sends B*4096*2 bytes; prefill sends tokens*4096*2.
CASES = [
    ("decode  b=1",  1 * 4096),
    ("decode  b=8",  8 * 4096),
    ("decode  b=32", 32 * 4096),
    ("prefill 1k",   1024 * 4096),
    ("prefill 8k",   8192 * 4096),
]
ITERS, WARMUP = 200, 30
LAYERS, AR_PER_LAYER = 45, 2


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("nccl")

    if rank == 0:
        p = torch.cuda.get_device_properties(0)
        print(f"\n{world}x {p.name}  (sm{p.major}{p.minor})")
        print(f"{'case':<14} {'bytes':>10} {'p50 us':>9} {'p99 us':>9} "
              f"{'GB/s':>8} {'tok/s ceiling':>14}")
        print("-" * 70)

    for name, numel in CASES:
        buf = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
        for _ in range(WARMUP):
            dist.all_reduce(buf)
        torch.cuda.synchronize()

        samples = []
        for _ in range(ITERS):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            dist.all_reduce(buf)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)  # ms -> us

        samples.sort()
        p50 = statistics.median(samples)
        p99 = samples[int(len(samples) * 0.99)]
        nbytes = numel * 2
        # Effective algorithm bandwidth of a ring all-reduce.
        gbps = (nbytes * 2 * (world - 1) / world) / (p50 * 1e-6) / 1e9
        # If all-reduce were the only decode cost, this is the token ceiling.
        ceiling = 1e6 / (p50 * LAYERS * AR_PER_LAYER)

        if rank == 0:
            print(f"{name:<14} {nbytes:>10} {p50:>9.1f} {p99:>9.1f} "
                  f"{gbps:>8.1f} {ceiling:>14.1f}")

    if rank == 0:
        print("\ntok/s ceiling = communication-only bound for GLM-5.3-Flash decode")
        print(f"                ({LAYERS} layers x {AR_PER_LAYER} all-reduces per token)\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
