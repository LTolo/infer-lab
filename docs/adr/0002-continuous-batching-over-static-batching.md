# ADR-0002: Iteration-level scheduling instead of static batching

**Status:** Accepted · **Date:** 2026-09

## Context
Output lengths in real traffic vary by an order of magnitude. With static
batching the whole batch is held hostage by its longest sequence, so accelerator
utilisation collapses and short requests inherit the latency of long ones.

## Decision
Re-decide batch composition on **every** forward pass (`engine/scheduler.py`).
Finished sequences leave immediately; queued ones join in the same step.

## Rationale
Decode is memory-bound: the cost of a step is dominated by reading the weights,
which is nearly independent of batch size. Any sequence that could have been in
the batch and was not is throughput thrown away.

## Consequences
- Throughput rises roughly linearly with batch size until the KV pool saturates.
- The scheduler becomes the most complex component and needs the strongest tests
  — hence `test_batched_results_equal_sequential_results`, which asserts that
  batching never changes any sequence's output.
- Per-step bookkeeping (block reservation) is mandatory; see ADR-0006.
