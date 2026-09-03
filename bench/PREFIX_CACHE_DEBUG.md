# KDA prefix-cache corruption: what is established

## Symptom
A radix prefix hit on `glm5_next` returns a wrong-but-fluent answer. Cold cache
recalls the needle (`4817-QUARTZ-9930`); a hit returns a near-miss (`4862`,
`4830`, `6888`). KV is exact; the KDA recurrent state is not.

## Read this first: an earlier round of results was invalid
`/flush_cache` refuses while any request is running or queued and still answers
200, with a refusal string in the body. The test called it fire-and-forget, so
flushes silently did nothing and "cold" runs were served warm. Every conclusion
drawn before `flush()` verified its result is untrustworthy -- in particular the
claim that the fault was **in-flight vs finished donors**, which does not
survive re-testing. `flush()` now retries until the body says flushed and raises
if it cannot.

## Established with verified flushes

| scenario | result |
|---|---|
| identical prompt sent twice, cache warm between (`test_cache_repeat.py`) | **pass**, byte-identical |
| identical prompt twice, flushed between (control) | **pass** |
| prime with a prompt that shares a prefix then diverges, then the real prompt (`test_prefix_cache.py --prime shifted`) | **fail**, both depths |

So reusing your own identical entry is fine. The corruption needs a **branching**
prefix: another request that shares a prefix and then diverges.

## The branching request's match

Instrumenting `finalize_match_result_in_tree_core` and logging only matches that
branch or leave KV ahead of the recurrent state (a request's own chunked
continuation shows neither and floods the log otherwise):

```
full_kv_hit=6336  mamba_boundary=0  dev_idx=0  branch=6336
```

KV matched 6336 tokens, no mamba checkpoint was usable (`mamba_boundary=0`, so
`dev_idx=0` -- nothing reused), and the request is nonetheless assigned
`mamba_branching_seqlen=6336`. The needle sits at ~6.3K, right at that branch.

## Ruled out
- **A KV / recurrent-state depth gap on the reused path.** Every non-branching
  match measured has `full_kv_hit == mamba_boundary == dev_idx`.
- **Interior vs finished donor.** Suppressing mid-prefill donations does make
  the test pass, but only because it also disables KV reuse
  (`cached_tokens` 2624 -> 0); it is no better than turning the cache off. The
  distinction itself rests on pre-flush-fix data.
- **Tagging checkpoints with an owning request.** Radix nodes are shared by key,
  not owned by a request, so this makes insert and match disagree and trips
  `new_prefix_len=2048, len(new_indices)=0`.
- **The branching-point checkpoint.** Forcing `mamba_branching_seqlen = None`
  does not fix it (re-tested after the flush fix).

## Shipped
`--disable-radix-cache`. Correct, at the cost of prefix reuse. Upstream bug #6.

## Tools
- `bench/test_cache_repeat.py` -- one prompt twice, nothing else moving.
- `bench/test_prefix_cache.py` -- `--prime same|shifted`, `--prime-tokens K`,
  `--needle-at N`. Both now guard `main()` so importing one does not run it.
