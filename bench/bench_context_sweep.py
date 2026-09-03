"""Prefill and decode throughput across context length, isolated cleanly.

sglang.bench_serving is unusable on this box (a stray repo-root benchmark/
directory shadows sglang.benchmark.serving under the editable install's
finder), so this drives /generate directly and separates the two phases with
one trick: request A (max_new_tokens=1) measures prefill alone, since the
first token is produced inside the prefill/extend step itself; request B
(max_new_tokens=K) measures prefill + (K-1) decode steps against the SAME
prompt. Subtracting isolates decode without any client-side timing jitter --
both e2e_latency values are server-measured. The radix cache is off in every
serving mode this fork ships, so B never gets a free ride on A's KV.

Run on the GPU box:  python bench_context_sweep.py --lengths 1024,4096,...
"""

import argparse
import json
import os
import sys
import urllib.request

FILLER = (
    "The quarterly logistics review noted that regional depot throughput remained "
    "within the expected band, and that no material deviation was recorded against "
    "the prior reporting period. Routine maintenance proceeded on schedule. "
)


def build_prompt(tok, target_tokens):
    per = len(tok(FILLER).input_ids)
    n = max(1, target_tokens // per)
    return FILLER * n


def generate(url, text, max_new_tokens):
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps({
            "text": text,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    out = json.load(urllib.request.urlopen(req, timeout=3600))
    return out["meta_info"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="1024,4096,16384,32768,65536,131072")
    ap.add_argument("--decode-tokens", type=int, default=129)
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "/root/models/glm53"))
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    print(f"{'target':>8} {'prompt_tok':>10} {'prefill_s':>10} {'prefill_tok/s':>14} "
          f"{'decode_ms/tok':>14} {'decode_tok/s':>13}", flush=True)

    for target in [int(x) for x in a.lengths.split(",")]:
        prompt = build_prompt(tok, target)
        try:
            meta_a = generate(a.url, prompt, max_new_tokens=1)
            meta_b = generate(a.url, prompt, max_new_tokens=a.decode_tokens)
        except Exception as e:
            print(f"{target:>8} FAILED: {e}", flush=True)
            continue

        prompt_tokens = meta_a["prompt_tokens"]
        # meta_a["e2e_latency"] is prefill compute PLUS a fixed per-request
        # overhead (scheduler/IPC round trip) that does not shrink with prompt
        # size -- confirmed by hand: a 10-token prompt alone measured ~199ms of
        # e2e_latency, far more than ~1000 tok/s prefill implies for 10 tokens.
        # It is real and worth knowing (it sets interactive TTFT), but it would
        # bias prefill_tok_s high-error at small N if not disclosed; the decode
        # figure is unaffected because it comes from a subtraction where the
        # constant overhead cancels between the two same-prompt requests.
        prefill_s = meta_a["e2e_latency"]
        decode_steps = meta_b["completion_tokens"] - 1
        decode_s = meta_b["e2e_latency"] - prefill_s

        prefill_tok_s = prompt_tokens / prefill_s if prefill_s > 0 else float("nan")
        decode_ms_tok = (decode_s / decode_steps * 1000) if decode_steps > 0 else float("nan")
        decode_tok_s = (decode_steps / decode_s) if decode_s > 0 else float("nan")

        print(f"{target:>8} {prompt_tokens:>10} {prefill_s:>10.2f} {prefill_tok_s:>14.1f} "
              f"{decode_ms_tok:>14.2f} {decode_tok_s:>13.2f}", flush=True)


if __name__ == "__main__":
    main()
