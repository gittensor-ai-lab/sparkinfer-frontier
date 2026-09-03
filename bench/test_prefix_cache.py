"""Reusing a cached prefix must not change the answer.

GLM-5.3 is hybrid: 11 MLA layers keep paged KV, 34 KDA layers keep a recurrent
state. A radix-cache hit has to restore both at the match point. The KV half is
exact; the recurrent half is what this checks.

The test sends the same needle prompt twice. The first send is preceded by a
cache flush, so it computes the whole sequence. The second is preceded by a
priming request that shares a long filler prefix, so the real request resumes
from a restored KDA state partway in. Same tokens, same greedy decode -- any
difference in the answer is the state handoff.

Failures concentrate 5-10% into the prompt at >=32K, which is why the sweep
covers those depths rather than a single midpoint.

Run on the GPU box:  python test_prefix_cache.py
"""

import argparse
import json
import os
import sys
import time
import urllib.request

NEEDLE = "The secret passcode for the vault is 4817-QUARTZ-9930."
ANSWER = "4817-QUARTZ-9930"
QUESTION = (
    "\n\nQ: What is the secret passcode for the vault? "
    "Answer with the code only.\nA:"
)
FILLER = (
    "The quarterly logistics review noted that regional depot throughput remained "
    "within the expected band, and that no material deviation was recorded against "
    "the prior reporting period. Routine maintenance proceeded on schedule. "
)
HEAD = "You are reading an archived report. Answer questions about it.\n\n"


def build(tok, target_tokens, depth, needle=NEEDLE, needle_at=None):
    budget = target_tokens - len(tok(QUESTION + HEAD + needle).input_ids)
    per = len(tok(FILLER).input_ids)
    n = max(1, budget // per)
    if needle_at is not None:
        # Approximate token offset -> filler repeat count. Coarse (one FILLER is
        # ~per tokens) but enough to straddle a 2048 boundary across runs.
        head_tok = len(tok(HEAD).input_ids)
        at = max(0, min(n, round((needle_at - head_tok) / per)))
    else:
        at = int(n * depth)
    return HEAD + FILLER * at + needle + " " + FILLER * (n - at) + QUESTION


def post(url, path, payload, timeout=2400):
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def generate(url, text, max_new_tokens=24):
    out = post(url, "/generate", {
        "text": text,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
    })
    return (out["text"] if isinstance(out, dict) else out[0]["text"]).strip()


def generate_ids(url, ids, max_new_tokens=24):
    out = post(url, "/generate", {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
    })
    return (out["text"] if isinstance(out, dict) else out[0]["text"]).strip()


def flush(url, attempts=20):
    """Flush and confirm it happened.

    /flush_cache refuses while any request is running or queued and answers 200
    with a refusal string, so a fire-and-forget call silently leaves the cache
    populated -- which quietly turns a cold run into a warm one.
    """
    for _ in range(attempts):
        try:
            body = urllib.request.urlopen(
                urllib.request.Request(url + "/flush_cache", data=b""), timeout=60
            ).read().decode()
        except Exception:
            body = ""
        if "flushed" in body.lower():
            time.sleep(1)
            return
        time.sleep(2)
    raise RuntimeError("could not flush the radix cache; results would be confounded")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--depths", default="0.05,0.1,0.5")
    # "shifted" primes with the needle moved deeper, so the prompts diverge at
    # the needle and the hit resumes right at real branching. "same" primes with
    # the identical prompt: any failure then indicts the state handoff itself,
    # since the restored state encodes exactly what the warm run would compute.
    ap.add_argument("--prime", choices=["shifted", "same"], default="shifted")
    # Prime with exactly this many tokens of the warm prompt (token-level slice,
    # sent as input_ids so the prefix is exact). Picks WHICH checkpoint the warm
    # run restores: a multiple of 2048 makes the prime's deepest checkpoint the
    # aligned final-state track, anything else the unaligned h track.
    ap.add_argument("--prime-tokens", type=int, default=None)
    # Place the needle at an exact token offset (overrides depth). Lets the
    # branch point land on or off a 2048 chunk boundary, which selects whether
    # the reused checkpoint is an aligned final state or an interior h state.
    ap.add_argument("--needle-at", type=int, default=None)
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "/root/models/glm53"))
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    failures = 0
    for depth in [float(d) for d in a.depths.split(",")]:
        prompt = build(tok, a.tokens, depth)

        if a.prime_tokens is not None:
            # Exact-prefix priming: cold and warm both go in as token ids so the
            # shared prefix is byte-for-byte the prime, not a re-tokenization.
            ids = tok(prompt).input_ids
            flush(a.url)
            cold = generate_ids(a.url, ids)
            flush(a.url)
            generate_ids(a.url, ids[: a.prime_tokens], max_new_tokens=1)
            warm = generate_ids(a.url, ids)
        else:
            flush(a.url)
            cold = generate(a.url, prompt)

            # Prime with the same needle placed later, so the two prompts share
            # a long pure-filler prefix and the real request lands on a cache hit.
            flush(a.url)
            if a.prime == "same":
                generate(a.url, prompt, max_new_tokens=1)
            else:
                generate(a.url, build(tok, a.tokens, min(depth + 0.4, 0.95)), max_new_tokens=1)
            warm = generate(a.url, prompt)

        cold_ok, warm_ok = ANSWER in cold, ANSWER in warm
        # Only recall decides. Two runs of the same prompt need not be
        # bitwise-equal on a live server -- batch composition reorders float
        # reductions and can flip a token well after the answer -- so `identical`
        # is reported but never fails the test. What must not happen is the cold
        # run recalling the needle and the warm run not.
        ok = cold_ok and warm_ok
        failures += not ok
        print(
            f"depth={depth}: cold={'ok' if cold_ok else 'BAD'} "
            f"warm={'ok' if warm_ok else 'BAD'} identical={cold == warm}"
        )
        if not ok:
            print(f"  cold: {cold[:80]!r}")
            print(f"  warm: {warm[:80]!r}")

    print("PASS" if not failures else f"FAIL {failures} depth(s) lost the needle on a cache hit")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
