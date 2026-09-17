"""Pipeline parallelism (GPipe-style micro-batching).

Tensor parallelism splits *within* a layer and pays an all-reduce per layer, so
it only makes sense inside a node where NVLink bandwidth is plentiful.  Pipeline
parallelism splits *across* layers and only ships the activation tensor at the
stage boundary -- far less traffic, which is why it is the one that crosses
InfiniBand between nodes.

The cost of PP is the **bubble**: while stage 0 works on micro-batch 0, stages
1..P-1 idle.  Splitting the batch into ``M`` micro-batches shrinks the bubble to

    bubble_fraction = (P - 1) / (M + P - 1)

which is the single formula that decides whether a pipeline config is sane.
This module implements the schedule and measures the bubble empirically so the
formula can be checked rather than trusted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PipelineStage:
    """One contiguous slice of the layer stack, pinned to one device."""

    stage_id: int
    layer_ids: list[int]
    fn: Callable[[np.ndarray], np.ndarray]
    busy_steps: int = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.busy_steps += 1
        return self.fn(x)


@dataclass
class PipelineSchedule:
    """GPipe forward-only schedule (inference has no backward pass)."""

    stages: list[PipelineStage]
    num_micro_batches: int = 4
    timeline: list[tuple[int, int, int]] = field(default_factory=list)  # (t, stage, micro)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def total_slots(self) -> int:
        return self.num_stages * (self.num_micro_batches + self.num_stages - 1)

    def theoretical_bubble_fraction(self) -> float:
        p, m = self.num_stages, self.num_micro_batches
        return (p - 1) / (m + p - 1)

    def run(self, micro_batches: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Execute the 1F (forward-only) pipeline and record the occupancy timeline."""
        if len(micro_batches) != self.num_micro_batches:
            raise ValueError(
                f"expected {self.num_micro_batches} micro-batches, got {len(micro_batches)}"
            )
        p, m = self.num_stages, self.num_micro_batches
        buffers: dict[tuple[int, int], np.ndarray] = {
            (0, i): np.asarray(x) for i, x in enumerate(micro_batches)
        }
        outputs: list[np.ndarray | None] = [None] * m
        self.timeline = []

        # Wavefront: at time t, stage s processes micro-batch (t - s) if it exists.
        for t in range(m + p - 1):
            for s in range(p):
                micro = t - s
                if not 0 <= micro < m:
                    continue
                x = buffers.pop((s, micro))
                y = self.stages[s](x)
                self.timeline.append((t, s, micro))
                if s + 1 < p:
                    buffers[(s + 1, micro)] = y
                else:
                    outputs[micro] = y

        assert all(o is not None for o in outputs)
        return [o for o in outputs if o is not None]

    def measured_bubble_fraction(self) -> float:
        busy = len(self.timeline)
        return 1.0 - (busy / self.total_slots) if self.total_slots else 0.0

    def activation_bytes_per_boundary(self, micro_batch: np.ndarray) -> int:
        """Only this crosses the wire per stage boundary -- compare against TP."""
        return int(np.asarray(micro_batch).nbytes)

    def report(self, micro_batch: np.ndarray | None = None) -> dict[str, float | int]:
        report: dict[str, float | int] = {
            "stages": self.num_stages,
            "micro_batches": self.num_micro_batches,
            "theoretical_bubble": round(self.theoretical_bubble_fraction(), 4),
            "measured_bubble": round(self.measured_bubble_fraction(), 4),
            "timeline_slots": self.total_slots,
            "busy_slots": len(self.timeline),
        }
        if micro_batch is not None:
            per_boundary = self.activation_bytes_per_boundary(micro_batch)
            report["bytes_per_boundary"] = per_boundary
            report["total_transfer_bytes"] = per_boundary * (self.num_stages - 1) \
                * self.num_micro_batches
        return report


def split_layers(num_layers: int, num_stages: int) -> list[list[int]]:
    """Balanced contiguous layer assignment.

    Contiguous matters: a non-contiguous assignment would send activations
    backwards across the pipeline and serialise everything.
    """
    if num_stages > num_layers:
        raise ValueError("cannot have more stages than layers")
    base, extra = divmod(num_layers, num_stages)
    out, start = [], 0
    for s in range(num_stages):
        take = base + (1 if s < extra else 0)
        out.append(list(range(start, start + take)))
        start += take
    return out
