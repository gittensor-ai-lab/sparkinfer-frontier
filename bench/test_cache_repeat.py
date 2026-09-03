"""Send one prompt twice. The second must answer like the first.

The smallest statement of the KDA prefix-cache bug, with nothing else moving:
no second prompt, no shifted needle, no token slicing. Run 1 goes in against a
verified-empty cache; run 2 is the identical prompt, so it is served from run 1's
radix entry. Same tokens, same greedy decode -- only the cache differs.

`--no-flush-between` is the default and is the point. `--flush-between` reruns
both cold, which is the control: if that passes and the default fails, the cache
is what changed the answer.

Run on the GPU box:  python test_cache_repeat.py
"""

import argparse
import os
import sys

from test_prefix_cache import ANSWER, build, flush, generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--depth", type=float, default=0.05)
    ap.add_argument("--flush-between", action="store_true")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "/root/models/glm53"))
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prompt = build(tok, a.tokens, a.depth)
    print(f"prompt_tokens={len(tok(prompt).input_ids)} depth={a.depth} "
          f"flush_between={a.flush_between}", flush=True)

    flush(a.url)
    first = generate(a.url, prompt)
    if a.flush_between:
        flush(a.url)
    second = generate(a.url, prompt)

    first_ok, second_ok = ANSWER in first, ANSWER in second
    print(f"first ={'ok ' if first_ok else 'BAD'} {first[:70]!r}")
    print(f"second={'ok ' if second_ok else 'BAD'} {second[:70]!r}")
    print(f"identical={first == second}")

    ok = first_ok and second_ok
    print("PASS" if ok else "FAIL the repeat lost the needle")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
