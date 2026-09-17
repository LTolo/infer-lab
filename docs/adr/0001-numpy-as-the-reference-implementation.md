# ADR-0001: NumPy is the reference implementation, PyTorch is a mirror

**Status:** Accepted · **Date:** 2026-09 · **Deciders:** engine team

## Context
The project must run and be fully testable on a locked-down laptop with no GPU
and no guaranteed PyTorch install, while still demonstrating real engine
internals. We had to pick which implementation defines correctness.

## Decision
The **NumPy** implementation is normative. Every other backend — C++/ctypes,
pybind11, nanobind, Triton, PyTorch — is verified *against* it in
`tests/test_kernels.py`.

## Rationale
- NumPy has no optional dependency, so the correctness suite runs everywhere.
- It forces us to write attention, RoPE and paging explicitly rather than
  delegating to a framework that hides the mechanism we are trying to show.
- A single normative reference makes multi-backend testing a loop rather than a
  matrix of special cases.

## Consequences
- **Positive:** the suite is green on any machine; new backends are cheap to add.
- **Negative:** NumPy is slow, so absolute throughput numbers are not comparable
  to a real engine. This is acceptable — the benchmark harness exists to compare
  *configurations against each other*, not against production hardware.
- **Negative:** float32 only. FP16/BF16 numerics are modelled via the fp16
  overflow check in `debug/instability.py` rather than executed.
