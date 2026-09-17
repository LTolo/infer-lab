# ADR-0007: No Docker, no Kubernetes

**Status:** Accepted · **Date:** 2026-09

## Context
Docker is marked *Prohibited* on the target machine. Kubernetes requires a
container runtime and is therefore excluded transitively.

## Decision
Every component runs as a plain local process, supervised by
`scripts/run_stack.py`. Prometheus and Grafana are consumed as native binaries;
k6 is a single static binary.

## Rationale
- The constraint is real and not negotiable.
- More importantly, containers and K8s solve *infrastructure orchestration*.
  The interesting scheduling problem in inference is not "which pod runs where"
  but "which request runs on which heterogeneous accelerator pool, at what
  predicted latency and cost" — which is implemented for real in
  `fleet/multi_pool.py` against roofline-derived predictions.

## Consequences
- Setup is simpler: two commands, no daemon, no admin rights.
- The project cannot demonstrate container-level deployment concerns. That is an
  accepted, documented gap rather than a hidden one.
