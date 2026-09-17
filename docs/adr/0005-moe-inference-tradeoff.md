# ADR-0005: Ship an MoE variant, and treat it as an inference problem

**Status:** Accepted · **Date:** 2026-09

## Context
Mixture-of-Experts is presented as "more parameters at constant FLOPs". That
framing is a *training* framing. For inference it changes the problem shape.

## Decision
Implement top-k routing with a grouped scatter/gather FFN (`model/moe.py`) and
account for it correctly in the roofline model (`fleet/roofline.py::_active_params`).

## Rationale
Per token, an MoE layer computes only `top_k / num_experts` of the FFN FLOPs but
must still have *all* expert weights resident in memory. Arithmetic intensity
therefore falls, and the layer becomes memory-bound earlier than a dense FFN of
the same quality. `test_moe_activates_fewer_params_than_it_stores` encodes this.

## Consequences
- Router load imbalance is an **operational** signal, not just a training one: a
  skewed router creates a straggler expert that sets the latency of the whole
  batch. Exported as `router_cv` via `model/moe.py::load_balance_loss`.
- Expert-parallel placement is out of scope here; single-device grouped GEMM is
  implemented instead.
