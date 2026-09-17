"""Heterogeneous multi-pool placement.

A fleet with three accelerator generations serving one traffic mix has to answer
a concrete question for every request: *which pool, and is it worth it?*

The policy here is explicitly multi-objective, because the naive answers are both
wrong:

* "always fastest hardware" burns budget on requests whose SLO is loose anyway
* "always cheapest hardware" misses latency SLOs on long prompts and gets you
  queueing collapse

So each candidate pool is scored on predicted latency (from the roofline model,
not a guess), SLO feasibility, current queue depth and relative cost, and the
best feasible pool wins.  Requests that no pool can serve within SLO are
reported as such instead of being silently accepted -- admission control is part
of the design, not an afterthought.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from infer_lab.config import ModelConfig
from infer_lab.fleet.hardware import HardwareProfile, get_profile
from infer_lab.fleet.roofline import analyze_decode, analyze_prefill, kv_cache_capacity
from infer_lab.utils.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class WorkloadRequest:
    request_id: str
    prompt_len: int
    max_tokens: int
    slo_ms: float = 2000.0
    priority: int = 0          # lower is more important


@dataclass
class Pool:
    """A homogeneous group of devices within the heterogeneous fleet."""

    name: str
    hardware: HardwareProfile
    num_devices: int
    max_concurrent_seqs: int = 64
    active_seqs: int = 0
    queued_tokens: int = 0
    assigned: list[str] = field(default_factory=list)

    @property
    def load(self) -> float:
        capacity = self.max_concurrent_seqs * self.num_devices
        return self.active_seqs / capacity if capacity else 1.0

    @property
    def has_capacity(self) -> bool:
        return self.active_seqs < self.max_concurrent_seqs * self.num_devices

    def snapshot(self) -> dict[str, object]:
        return {
            "name": self.name,
            "hardware": self.hardware.name,
            "devices": self.num_devices,
            "active_seqs": self.active_seqs,
            "load": round(self.load, 3),
            "assigned": len(self.assigned),
        }


@dataclass
class PlacementDecision:
    request_id: str
    pool: str | None
    predicted_latency_ms: float
    predicted_cost: float
    meets_slo: bool
    reason: str
    considered: dict[str, float] = field(default_factory=dict)


class MultiPoolScheduler:
    """Latency/cost-aware placement across heterogeneous pools."""

    def __init__(self, model_config: ModelConfig, pools: list[Pool],
                 *, cost_weight: float = 0.3, queue_weight: float = 0.2) -> None:
        if not pools:
            raise ValueError("at least one pool is required")
        self.model_config = model_config
        self.pools = {p.name: p for p in pools}
        self.cost_weight = cost_weight
        self.queue_weight = queue_weight
        self.decisions: list[PlacementDecision] = []

    # ------------------------------------------------------------------ modelling
    def predict_latency_ms(self, pool: Pool, request: WorkloadRequest) -> float:
        """Roofline-predicted end-to-end latency, inflated by current queue depth."""
        batch = max(1, pool.active_seqs)
        prefill = analyze_prefill(self.model_config, pool.hardware, 1, request.prompt_len)
        avg_ctx = request.prompt_len + request.max_tokens / 2
        decode = analyze_decode(self.model_config, pool.hardware, batch, int(avg_ctx))
        decode_ms = (request.max_tokens / max(decode.tokens_per_s, 1e-9)) * 1000.0
        queueing = 1.0 + self.queue_weight * pool.load * 10.0
        return (prefill.time_s * 1000.0 + decode_ms) * queueing

    def predict_cost(self, pool: Pool, latency_ms: float) -> float:
        return pool.hardware.relative_cost * (latency_ms / 1000.0)

    def fits_in_memory(self, pool: Pool) -> bool:
        cap = kv_cache_capacity(self.model_config, pool.hardware)
        return cap["max_kv_tokens"] > 0

    # ------------------------------------------------------------------ placement
    def place(self, request: WorkloadRequest) -> PlacementDecision:
        considered: dict[str, float] = {}
        scored: list[tuple[float, str, float, float, bool]] = []

        for pool in self.pools.values():
            if not pool.has_capacity or not self.fits_in_memory(pool):
                continue
            latency = self.predict_latency_ms(pool, request)
            cost = self.predict_cost(pool, latency)
            meets = latency <= request.slo_ms
            considered[pool.name] = round(latency, 2)
            # Normalised objective: latency headroom + weighted cost.
            score = (latency / request.slo_ms) + self.cost_weight * cost
            if not meets:
                score += 100.0        # heavy penalty, but still rankable
            scored.append((score, pool.name, latency, cost, meets))

        if not scored:
            decision = PlacementDecision(
                request.request_id, None, float("inf"), float("inf"), False,
                "no pool has capacity or sufficient memory", considered,
            )
            self.decisions.append(decision)
            logger.warning("placement failed", extra={"request_id": request.request_id})
            return decision

        heapq.heapify(scored)
        _, name, latency, cost, meets = heapq.heappop(scored)
        pool = self.pools[name]
        pool.active_seqs += 1
        pool.assigned.append(request.request_id)

        reason = (f"lowest combined latency/cost score on {pool.hardware.name}"
                  if meets else "no pool meets the SLO; placed on the fastest feasible pool")
        decision = PlacementDecision(request.request_id, name, round(latency, 2),
                                     round(cost, 4), meets, reason, considered)
        self.decisions.append(decision)
        return decision

    def release(self, request_id: str) -> bool:
        for pool in self.pools.values():
            if request_id in pool.assigned:
                pool.assigned.remove(request_id)
                pool.active_seqs = max(0, pool.active_seqs - 1)
                return True
        return False

    # ------------------------------------------------------------------ reporting
    def slo_attainment(self) -> float:
        if not self.decisions:
            return 1.0
        return sum(1 for d in self.decisions if d.meets_slo) / len(self.decisions)

    def total_cost(self) -> float:
        return sum(d.predicted_cost for d in self.decisions if d.pool)

    def snapshot(self) -> dict[str, object]:
        return {
            "pools": [p.snapshot() for p in self.pools.values()],
            "placements": len(self.decisions),
            "slo_attainment": round(self.slo_attainment(), 4),
            "total_predicted_cost": round(self.total_cost(), 4),
            "unplaced": sum(1 for d in self.decisions if d.pool is None),
        }


def default_fleet(devices_per_pool: int = 8) -> list[Pool]:
    return [
        Pool(name=f"pool-{name}", hardware=get_profile(name), num_devices=devices_per_pool)
        for name in ("accel-a", "accel-b", "accel-c")
    ]
