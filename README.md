# SparkInfer Frontier

**GLM-5.3-Flash (NVFP4)** — 320B total, 18B active — serving on eight consumer
RTX 5090s. No datacenter GPU, no NVLink, no P2P. A fork of
[SGLang](https://github.com/sgl-project/sglang).

```
"Q: What is 15 times 4?"          ->  " 15 times 4 is 60."
"Q: Name the largest planet..."   ->  " Jupiter is the largest planet in our solar system."
"Q: Who wrote Romeo and Juliet?"  ->  " Romeo and Juliet was written by William Shakespeare."
"Why is the sky blue?"            ->  " ...sunlight is scattered by the atmosphere."
```

## Performance

Single stream (batch 1), server-measured, prefill and decode isolated by
subtraction so fixed per-request overhead cancels out —
[method](docs/cookbook/run-glm-5.3-rtx5090s.md#performance):

| context | prefill | decode |
|---:|---:|---:|
| 1K | 974 tok/s | 65 tok/s |
| 4K | 1098 tok/s | 64 tok/s |
| 16K | 1085 tok/s | 64 tok/s |
| 32K | 1085 tok/s | 64 tok/s |
| 64K | 1083 tok/s | 65 tok/s |
| 128K | 1081 tok/s | 68 tok/s |
| 256K | 1142 tok/s | 67 tok/s |
| 512K | 1127 tok/s | 66 tok/s |

Decode holds flat — 63-68 tok/s from 1K to 512K, no downward trend. The 34 KDA
layers carry an O(1) recurrent state and the 11 MLA layers read a fixed sparse
top-k regardless of how long the sequence is, so nothing about decode scales
with context. Prefill is what scales, linearly, at ~1080-1140 tok/s throughout.

## Status

| | |
|---|---|
| Context, verified | 128K / 256K / 512K needle recall |
| Context ceiling | 748,928 tokens — [why not 1M](docs/cookbook/run-glm-5.3-rtx5090s.md#1m-does-not-fit-on-this-node) |
| Prefix caching | off — a cache hit corrupts KDA state — [open bug](docs/cookbook/run-glm-5.3-rtx5090s.md#the-prefix-cache-bug-open) |
| Concurrency | 16 up to 128K, 1 beyond (KV takes the memory) |

## Run it

```bash
bench/serve-glm53.sh tp8_sm120mla     # serve: TP=8, sm120 sparse MLA, NVFP4 MoE
bench/bench-glm53.sh tp8_sm120mla     # latency + throughput
```

## The node

8x RTX 5090 (sm120, 32 GB each), PCIe Gen4 x16 with **no P2P**, 2x EPYC 7K62 /
503 GiB RAM. `RedHatAI/GLM-5.3-Flash-NVFP4`, 185 GiB, 24.8 GB/GPU resident.

## More

[**docs/cookbook/run-glm-5.3-rtx5090s.md**](docs/cookbook/run-glm-5.3-rtx5090s.md) —
every workaround this fork carries, the kernels, the long-context and
prefix-cache investigations in full, upstream bugs found, and the test index.
