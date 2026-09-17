# ADR-0003: PagedAttention instead of contiguous per-sequence KV buffers

**Status:** Accepted · **Date:** 2026-09

## Context
Reserving `max_seq_len` of KV per sequence wastes most of the pool to internal
fragmentation, because almost no sequence reaches the maximum length. KV memory
is what limits concurrency, so that waste is directly lost throughput.

## Decision
Carve the pool into fixed-size blocks and give each sequence a **block table**
(`kv/paged_cache.py`, `kv/block_allocator.py`), with reference counting and
copy-on-write.

## Rationale
- Fragmentation drops to at most one partially-filled block per sequence.
- Reference counting is what makes prefix sharing possible at all (ADR-0004).
- Copy-on-write lets two sequences share a prefix block until one of them writes.

## Consequences
- **Negative:** the NumPy attention kernel needs a `gather` to materialise a
  contiguous K/V view. On a GPU this gather does not exist — the paged attention
  kernel indexes through the block table directly — so our gather is an artefact
  of the reference implementation, not of the design.
- **Positive:** preemption becomes cheap and precise (free a block list).
- Block size is a real tuning knob: smaller means less waste but longer block
  tables and more indexing overhead.
