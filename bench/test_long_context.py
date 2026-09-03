"""Needle-in-a-haystack at 128K tokens.

Allocating a 128K KV pool proves nothing about whether attention reads it. This
plants a fact at a known depth, buries it in filler, and asks for it back --
exercising the KDA state pool and the sparse-MLA index (index_topk=2048) at a
range far past anything the short-prompt tests touch.
"""
import argparse, json, os, sys, time, urllib.request

NEEDLE = "The secret passcode for the vault is 4817-QUARTZ-9930."
QUESTION = "\n\nQ: What is the secret passcode for the vault? Answer with the code only.\nA:"
FILLER = (
    "The quarterly logistics review noted that regional depot throughput remained "
    "within the expected band, and that no material deviation was recorded against "
    "the prior reporting period. Routine maintenance proceeded on schedule. "
)


def build(tok, target_tokens, depth):
    head = "You are reading an archived report. Answer questions about it.\n\n"
    q_len = len(tok(QUESTION).input_ids) + len(tok(head).input_ids)
    budget = target_tokens - q_len - len(tok(NEEDLE).input_ids)
    per = len(tok(FILLER).input_ids)
    n = max(1, budget // per)
    at = int(n * depth)
    body = FILLER * at + NEEDLE + " " + FILLER * (n - at)
    return head + body + QUESTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128 * 1024)
    ap.add_argument("--depth", type=float, default=0.6)
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "/root/models/glm53"))
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prompt = build(tok, a.tokens, a.depth)
    n_tok = len(tok(prompt).input_ids)
    print(f"prompt_tokens={n_tok} depth={a.depth}", flush=True)

    req = urllib.request.Request(
        a.url + "/generate",
        data=json.dumps({
            "text": prompt,
            "sampling_params": {"max_new_tokens": 24, "temperature": 0.0},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        out = json.load(urllib.request.urlopen(req, timeout=2400))
    except Exception as e:
        body = getattr(e, "file", None)
        print(f"FAIL request: {e}" + (f" :: {body.read()[:400]}" if body else ""))
        sys.exit(1)
    dt = time.time() - t0

    text = out["text"] if isinstance(out, dict) else out[0]["text"]
    print(f"prefill+decode_s={dt:.1f}")
    print(f"TEXT: {text!r}")
    print("PASS" if "4817-QUARTZ-9930" in text else "FAIL needle not recalled")


main()
