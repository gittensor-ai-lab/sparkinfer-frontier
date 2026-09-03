"""Topology-aware all-reduce for PCIe-only, multi-NUMA GPU nodes.

On this box P2P reads CNS for all 28 GPU pairs, so every collective stages
through host memory, and half the pairs also cross the inter-socket link. A flat
ring treats all 28 links as equal and pays that crossing repeatedly.

Two strategies, both measured against NCCL in __main__:

* hierarchical -- reduce inside each NUMA group, exchange once between group
  leaders, broadcast back down. Cross-socket traffic drops from every-rank to
  one exchange per collective.
* compressed -- the same shape, with the inter-group exchange carried in FP8.
  PCIe bytes are the bottleneck, so halving the payload of the one hop that
  crosses sockets buys more than it costs in scale/unscale.

Run:  torchrun --nproc_per_node=8 pcie_allreduce.py
"""

from __future__ import annotations

import os
import statistics
from typing import List, Optional

import torch
import torch.distributed as dist

# Anything below this is latency-bound, where an extra kernel launch costs more
# than the bytes it saves; compression only pays above it.
_COMPRESS_MIN_BYTES = 1 << 20
_FP8_MAX = 448.0


class PCIeHierarchicalAllReduce:
    """All-reduce that reduces within a NUMA group before crossing sockets."""

    def __init__(self, group_size: int, compress: bool = False):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        assert self.world % group_size == 0, "world must divide into equal groups"
        self.group_size = group_size
        self.compress = compress
        self.n_groups = self.world // group_size
        self.group_id = self.rank // group_size
        self.is_leader = self.rank % group_size == 0

        # Every rank must build every subgroup, in the same order, or NCCL hangs.
        self.intra: Optional[dist.ProcessGroup] = None
        for g in range(self.n_groups):
            ranks = list(range(g * group_size, (g + 1) * group_size))
            pg = dist.new_group(ranks=ranks)
            if g == self.group_id:
                self.intra = pg

        leaders = list(range(0, self.world, group_size))
        self.leader_group = dist.new_group(ranks=leaders)
        self.is_in_leader_group = self.rank in leaders

    def _exchange_leaders(self, t: torch.Tensor) -> None:
        """All-reduce across group leaders, optionally in FP8."""
        if not (self.compress and t.numel() * t.element_size() >= _COMPRESS_MIN_BYTES):
            dist.all_reduce(t, group=self.leader_group)
            return

        # Per-tensor scale keeps the payload inside e4m3's range; the scale is a
        # single float, so its own all-reduce is free next to the payload.
        amax = t.abs().max().clamp(min=1e-12)
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=self.leader_group)
        scale = _FP8_MAX / amax
        q = (t * scale).to(torch.float8_e4m3fn)
        acc = q.to(torch.float32)
        dist.all_reduce(acc, group=self.leader_group)
        t.copy_((acc / scale).to(t.dtype))

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(t, group=self.intra)
        if self.is_in_leader_group:
            self._exchange_leaders(t)
        dist.broadcast(t, src=self.group_id * self.group_size, group=self.intra)
        return t


class FP8AllGatherAllReduce:
    """reduce_scatter in BF16, then all_gather the shard in FP8.

    The hierarchical variants lose because NCCL already reduces NUMA-aware, so
    re-expressing the same traffic as three collectives only adds launches. The
    remaining lever is bytes: this keeps the reduction exact but halves the
    gather phase, which is the half that is pure bandwidth.
    """

    def __init__(self):
        self.world = dist.get_world_size()

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        n = t.numel()
        if n % self.world or n * t.element_size() < _COMPRESS_MIN_BYTES:
            dist.all_reduce(t)
            return t

        shard = torch.empty(n // self.world, dtype=t.dtype, device=t.device)
        dist.reduce_scatter_tensor(shard, t)

        amax = shard.abs().max().clamp(min=1e-12)
        dist.all_reduce(amax, op=dist.ReduceOp.MAX)
        scale = _FP8_MAX / amax
        q = (shard * scale).to(torch.float8_e4m3fn)

        gathered = torch.empty(n, dtype=torch.float8_e4m3fn, device=t.device)
        dist.all_gather_into_tensor(gathered, q)
        t.copy_((gathered.to(torch.float32) / scale).to(t.dtype))
        return t


def _time(fn, iters: int = 100, warmup: int = 20) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0)
    samples.sort()
    return statistics.median(samples), samples[int(len(samples) * 0.99)]


CASES = [
    ("decode  b=1", 1 * 4096),
    ("decode  b=8", 8 * 4096),
    ("decode  b=32", 32 * 4096),
    ("prefill 1k", 1024 * 4096),
    ("prefill 8k", 8192 * 4096),
]


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    world = dist.get_world_size()

    # GPUs 0-3 sit on NUMA 0 and 4-7 on NUMA 1, so a group is half the node.
    hier = PCIeHierarchicalAllReduce(group_size=world // 2, compress=False)
    comp = PCIeHierarchicalAllReduce(group_size=world // 2, compress=True)
    fp8 = FP8AllGatherAllReduce()

    if rank == 0:
        print(f"\n{world} ranks, groups of {world // 2}")
        print(f"{'case':<13} {'bytes':>9} {'nccl us':>9} {'hier us':>9} "
              f"{'comp us':>9} {'fp8ag us':>9} {'best':>8} {'speedup':>8}")
        print("-" * 82)

    for name, numel in CASES:
        ref = torch.randn(numel, dtype=torch.bfloat16, device="cuda")

        # Correctness before speed: both paths must match NCCL.
        base = ref.clone()
        dist.all_reduce(base)
        a, b = ref.clone(), ref.clone()
        hier.all_reduce(a)
        comp.all_reduce(b)
        err_h = (a.float() - base.float()).abs().max().item()
        denom = base.float().abs().max().item() + 1e-9
        err_c = (b.float() - base.float()).abs().max().item()

        t_nccl, _ = _time(lambda: dist.all_reduce(ref.clone()))
        t_hier, _ = _time(lambda: hier.all_reduce(ref.clone()))
        t_comp, _ = _time(lambda: comp.all_reduce(ref.clone()))
        c = ref.clone(); fp8.all_reduce(c)
        err_f = (c.float() - base.float()).abs().max().item()
        t_fp8, _ = _time(lambda: fp8.all_reduce(ref.clone()))

        if rank == 0:
            times = {"nccl": t_nccl, "hier": t_hier, "comp": t_comp, "fp8ag": t_fp8}
            which = min(times, key=times.get)
            print(f"{name:<13} {numel * 2:>9} {t_nccl:>9.1f} {t_hier:>9.1f} "
                  f"{t_comp:>9.1f} {t_fp8:>9.1f} {which:>8} "
                  f"{t_nccl / times[which]:>7.2f}x")
            print(f"{'':<13} {'':>9} rel err: hier={err_h / denom:.1e} "
                  f"comp={err_c / denom:.1e} fp8ag={err_f / denom:.1e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
