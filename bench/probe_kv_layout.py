"""B1 -- dump the KV layout GLM-5.3-Flash actually allocates.

The sm120 sparse-MLA kernel (flash_mla_sm120.py) does raw byte addressing against
DeepSeek-V4's packed page format: 448 B FP8 nope + 64x2 B BF16 rope + 8 B scales,
576 B stride. GLM-5.3-Flash is qk_nope_head_dim=256, qk_rope_head_dim=0 (pure
NoPE), kv_lora_rank=512 -- a different pool class entirely. Before writing a
kernel we need the real thing: buffer shapes, dtypes, strides and element sizes,
read off a live pool rather than inferred from config.

Run on the GPU box:  python probe_kv_layout.py [/path/to/checkpoint]
Allocates a deliberately tiny pool (2 layers, 4096 tokens); needs one GPU.
"""

import json
import sys

import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/root/models/GLM-5.3-Flash-NVFP4"


def describe(name, obj, depth=0):
    """Print anything tensor-shaped, recursing one level into lists."""
    pad = "  " * (depth + 1)
    if torch.is_tensor(obj):
        print(f"{pad}{name}: shape={tuple(obj.shape)} dtype={obj.dtype} "
              f"itemsize={obj.element_size()}B strides={obj.stride()} "
              f"contig={obj.is_contiguous()}")
        return True
    if isinstance(obj, (list, tuple)) and obj and depth < 1:
        tensors = [o for o in obj if torch.is_tensor(o)]
        if tensors:
            print(f"{pad}{name}: {len(obj)} entries, showing [0]")
            describe(f"{name}[0]", tensors[0], depth + 1)
            return True
    return False


def main() -> None:
    cfg = json.load(open(f"{CKPT}/config.json"))["text_config"]
    kv_lora_rank = cfg["kv_lora_rank"]
    qk_rope = cfg["qk_rope_head_dim"]
    index_head_dim = cfg["index_head_dim"]
    index_kpool = cfg.get("index_kpool", 1)
    index_kpool_compress = cfg.get("index_kpool_compress", False)

    print("=== GLM-5.3-Flash attention geometry (from config) ===")
    for k in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
              "index_head_dim", "index_topk", "index_kpool", "index_kpool_compress",
              "num_attention_heads"):
        print(f"  {k:24s} {cfg.get(k)}")
    print(f"\n  MLA latent per token = kv_lora_rank + qk_rope_head_dim "
          f"= {kv_lora_rank} + {qk_rope} = {kv_lora_rank + qk_rope}")

    print("\n=== what flash_mla_sm120.py assumes (DeepSeek-V4-Flash) ===")
    print("  nope 448 B FP8 + rope 64x2 B BF16 + 7 scales + 1 pad = 584 B/token")
    print("  page stride 576 B, scales in a separate section at page_size*576")

    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    page_size = 64  # latent-KV attention forces 64
    pool = DSATokenToKVPool(
        size=4096,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=qk_rope,
        layer_num=2,
        device="cuda",
        index_head_dim=index_head_dim,
        enable_memory_saver=False,
        kv_cache_dim=kv_lora_rank + qk_rope,
        index_kpool=index_kpool,
        index_kpool_compress=index_kpool_compress,
        # kpool compress keeps a per-request tail buffer, so the pool refuses to
        # size itself without knowing the concurrency ceiling.
        max_running_requests=16,
    )

    print(f"\n=== live DSATokenToKVPool (size=4096, page_size={page_size}, layers=2) ===")
    for attr in sorted(dir(pool)):
        if attr.startswith("__"):
            continue
        try:
            val = getattr(pool, attr)
        except Exception:
            continue
        if callable(val):
            continue
        if describe(attr, val):
            continue
        if isinstance(val, (int, float, bool, str)) and not attr.startswith("_"):
            print(f"    {attr}: {val}")

    # The kernel's real contract: what does one token's KV look like in memory?
    try:
        buf = pool.get_key_buffer(0)
        print("\n=== get_key_buffer(0) ===")
        describe("key_buffer", buf)
        print(f"    bytes per token = {buf[0].numel() * buf.element_size()}")
    except Exception as exc:
        print(f"\nget_key_buffer(0) unavailable: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
