# KDA prefix-cache corruption: root-cause diagnosis

## Symptom
A radix prefix hit on `glm5_next` returns a wrong-but-fluent answer. Cold cache
recalls the needle (`4817-QUARTZ-9930`); a hit returns a near-miss (`4830`,
`6888`, `48`). KV is exact; the KDA recurrent state restored at the match point
is not. Reproducer: `bench/test_prefix_cache.py`.

## Root cause (localized, measured)
A KDA checkpoint donated by an **in-flight (unfinished / chunked-prefill)**
request misresumes on a later partial-prefix hit. A checkpoint donated when a
request **finishes** is exact.

Measured on the running server, radix cache on:

| donor | reuse | result |
|---|---|---|
| finished request, new request extends it (multi-turn) | `cached_tokens=2624`, needle recalled | **correct** |
| finished short donor, needle recomputed past it (`--prime-tokens 4096`, needle at ~6.3K) | needle recomputed | **correct** |
| long in-flight donor, shared prefix ends at an interior checkpoint (`--prime shifted`) | interior checkpoint reused | **wrong** |

So the fault is specific to reusing a checkpoint that a **chunked, not-yet-finished**
prefill donated at an interior grid boundary. `test_prefix_cache.py --prime same`
(warm resumes from a checkpoint that encodes exactly the warm prompt) passes,
which rules out the donate/restore copy itself (`MambaPool.copy_from` copies both
conv and temporal correctly).

## Where it is
`cache_unfinished_req` donates the request's current KDA state as a reusable
checkpoint at the last grid boundary of each chunk. During chunked prefill the
donated state does not correspond to the depth it is filed under, so a second
request that matches that depth and recomputes the tail resumes from the wrong
state. The write path is
`hybrid_linear_attn_backend.py::_track_mamba_state_extend` (aligned path copies
the live chunk-end `ssm_states`; `_init_track_ssm_indices` keys aligned/unaligned
on `state_chunk_size`, not on the chunk's actual end) feeding
`mamba_component.py::prepare_for_caching_req` (unfinished branch).

## Why the obvious fix does not land here
"Donate a reusable checkpoint only when the request finishes" is directionally
correct and makes `test_prefix_cache.py` pass. But suppressing the unfinished
donation breaks a tighter invariant: `cache_unfinished_req` re-matches the
request's **own** just-inserted prefix to continue its next chunk, and that
re-match is gated on a mamba checkpoint existing on the node
(`create_match_validator` requires `component_data[MAMBA].value`). Remove the
checkpoint and the re-match returns zero indices:

```
AssertionError: new_prefix_len=2048, len(new_indices)=0   (unified_radix_cache.py)
```

A request's own chunked continuation and cross-request reuse share the same
checkpoint. A correct fix has to separate them -- e.g. mark an interior
checkpoint "usable by this request's continuation only, not a cross-request
reuse boundary," or fix the interior state capture so the donated state matches
its filed depth. Both are upstream protocol changes, out of scope for a runtime
workaround.

## Shipped
`--disable-radix-cache`. Correct, at the cost of prefix reuse. This is upstream
bug #6 in the README.

## Reproducer knobs (`bench/test_prefix_cache.py`)
- `--prime same` -- prime with the identical prompt; isolates the handoff.
- `--prime shifted` -- prime with the needle moved deeper; the failing case.
- `--prime-tokens K` -- prime with the first K tokens (finished donor at K).
- `--needle-at N` -- place the needle near token N (straddle a chunk boundary).
