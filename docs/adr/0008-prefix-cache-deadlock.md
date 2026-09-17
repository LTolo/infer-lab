# ADR-0008: The prefix cache must yield memory under pressure

**Status:** Accepted · **Date:** 2026-09 · **Supersedes part of:** ADR-0006

## Context

This decision was forced by a live failure, found by the dashboard rather than
by the test suite.

Under sustained load the engine stopped serving. The symptoms:

| Metric | Observed |
|---|---|
| `infer_lab_engine_steps_total` | 173,545,000 |
| `infer_lab_running_sequences` | 0 |
| `infer_lab_waiting_sequences` | 23 |
| `infer_lab_kv_blocks_free` | 4 of 512 |
| radix `cached_blocks` | 508 of 512 |
| `tokens_per_step` | 0 |
| `requests_total{outcome="timeout"}` | 106 |

One CPU core was pinned at 100%. Zero tokens were produced.

## Root cause

Blocks retained by the RadixAttention prefix cache carry refcount 1. The
allocator therefore counts them as *in use*, even though no sequence is reading
them. Nothing ever asked the cache to give memory back, so it grew until it
owned the pool. From there:

1. a new request could not be admitted -- no free blocks;
2. the scheduler tried to preempt, but `running` was empty, so there was **no
   victim**;
3. `schedule()` returned an empty result, the engine recorded a step and
   immediately looped again, at full speed, for ever.

The forward-progress guarantee from ADR-0006 did not cover this. That guarantee
assumes a *running* sequence exists to advance. Here nothing was running at all:
the queue was full, the pool was full, and neither could release the other.

The cache behaved exactly as specified. The specification was wrong. A cache
that cannot be evicted under pressure is not a cache -- it is a leak with good
intentions.

## Decision

Four changes, in order of how directly they address the cause:

1. **Reclaim from the cache before preempting** (`engine/scheduler.py`). Cached
   blocks are unreferenced by any sequence; dropping one costs a future cache
   miss. Preempting costs a live sequence its entire accumulated KV. The cheaper
   resource goes first, always.
2. **Break the stall when nothing is running** (`engine/scheduler.py`). If a step
   would be empty, nothing is running and requests are queued, evict cached
   blocks until the head of the queue can start. This is the case ADR-0006 left
   open.
3. **Do not spin on unproductive steps** (`server/engine_runner.py`). A step that
   emits nothing means the engine is blocked on memory rather than working.
   Sleep briefly instead of looping at full speed.
4. **Cap the cache at 80% of the pool** (`engine/llm_engine.py`). Defence in
   depth: prefix caching is an optimisation, admitting requests is the job, and
   an optimisation must never be able to starve the job.

## Consequences

Measured on the same workload, before and after:

| Metric | Before | After |
|---|---|---|
| Engine steps | 173,545,000 | 2,586 |
| Running sequences | 0 | 8 |
| Waiting sequences | 23 | 0 |
| `tokens_per_step` | 0 | 24.96 |
| Successful requests | 42 | 313 |
| Timeouts | 106 | 0 |
| Preemptions | 6 | 0 |
| Prefix-cache hit rate | -- | 0.965 |

Roughly 67,000x fewer steps while serving about 7x more requests. Note the hit
rate: bounding the cache did not neuter it. Note also that preemptions fell to
zero -- change 1 means the engine now reaches for the cheap memory first and
rarely needs the expensive option at all.

## The part worth remembering

**The test suite reported 27 passed both before and after this bug.**

It was invisible to the suite for a structural reason: unit tests are short, and
the cache only swallows the pool under sustained load against a small pool. The
failure needed *time* and *memory pressure*, and the tests supplied neither.

Two consequences followed:

- `tests/test_kv_pressure.py` now reproduces the pressure conditions explicitly
  -- undersized pool, shared prefixes, sustained load -- and asserts bounded
  steps per request. Those tests fail on the unpatched engine.
- The lesson generalises: correctness tests verify *what* the engine computes;
  they say nothing about whether it keeps making progress. Liveness needs its
  own tests, and resource-exhaustion paths need to be exercised on purpose,
  because production will exercise them whether or not we do.

It is also worth stating plainly that observability is what found this. No
exception was raised and no test failed. The engine simply went quiet, and only
`engine_steps_total` climbing into the hundreds of millions next to
`running_sequences 0` made the cause legible. That pair of numbers is the whole
argument for `/metrics` existing.
