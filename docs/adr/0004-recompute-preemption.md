# ADR-0004: Preempt by recomputation, not by swapping

**Status:** Accepted · **Date:** 2026-09

## Context
When the KV pool is exhausted, a running sequence must give up its memory. Two
options: **swap** its KV to host memory and copy it back later, or **discard**
it and recompute the prefill on resume.

## Decision
Default to `recompute` (`engine/request.py::reset_for_recompute`), with the mode
left configurable in `EngineConfig.preemption_mode`.

## Rationale
Recompute costs one prefill pass, which is compute-bound and highly parallel.
Swap costs a round trip over PCIe of `2 · layers · kv_heads · head_dim · len`
bytes, which is bandwidth-bound and serialises against the very transfers the
engine needs. For short and medium prompts recompute wins outright; swapping
only pays off once the prompt is long enough that prefill dominates the PCIe
transfer, which is exactly the regime where prefix caching usually rescues you
anyway.

## Consequences
- Generated tokens are preserved across preemption and replayed as part of the
  next prefill, so preemption is invisible in the output — only in latency.
  `test_preemption_under_memory_pressure_preserves_output` pins this.
- `RequestMetrics.recomputed_tokens` makes the wasted work measurable, and
  sustained preemption is an alert (`observability/alerts.yml`).
