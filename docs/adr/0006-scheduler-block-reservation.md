# ADR-0006: The scheduler reserves KV blocks within a step

**Status:** Accepted · **Date:** 2026-09 · **Supersedes:** naive per-request checks

## Context
This decision was forced by a real bug found during development. The scheduler
checked "does this request have room for one more block?" per request. With
several decodes scheduled into the *same* step, each check independently saw the
same free blocks. Execution then allocated for all of them and the step aborted
with `OutOfBlocks` — intermittently, and only under memory pressure.

A second, related bug: a request already committed to the step could be chosen
as a preemption victim later in the same step, releasing KV that the step was
about to write into.

## Decision
`Scheduler.schedule()` maintains a `reserved` counter of blocks that scheduled
work will consume, and a `scheduled` set of request ids that are **not eligible
for preemption** in that step.

## Rationale
Admission decisions must be made against *remaining* capacity, not total
capacity. This is ordinary transactional reasoning: decide against the state you
will actually execute in.

## Consequences
- No `OutOfBlocks` can escape the scheduler into execution.
- Because the scheduler now throttles instead of over-committing, a new failure
  mode appeared: if every running sequence is blocked on memory, nothing is
  scheduled and the engine spins. This is resolved by the explicit
  **forward-progress guarantee** — if a step would otherwise be empty, evict the
  newest sequences until the *oldest* one can take its next block. The oldest
  sequence always advances, which is what keeps the system out of livelock.
- `tests/test_engine.py::test_preemption_under_memory_pressure_preserves_output`
  exercises this path and asserts zero KV leakage afterwards.
