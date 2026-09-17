"""Timing helpers shared by the benchmark harness and the server."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile.

    Deliberately dependency-free (no numpy) so that it can be used inside the
    server hot path without importing heavy modules.
    """
    if not values:
        return float("nan")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    if q == 0:
        return ordered[0]
    rank = max(1, min(len(ordered), int(-(-q / 100.0 * len(ordered) // 1))))
    return ordered[rank - 1]


@dataclass
class Timer:
    """Context manager that records wall-clock durations in milliseconds."""

    name: str = "block"
    samples: list[float] = field(default_factory=list)
    _start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.samples.append((time.perf_counter() - self._start) * 1000.0)

    @property
    def last_ms(self) -> float:
        return self.samples[-1] if self.samples else float("nan")

    @property
    def total_ms(self) -> float:
        return sum(self.samples)

    def summary(self) -> dict[str, float]:
        return {
            "count": float(len(self.samples)),
            "total_ms": self.total_ms,
            "mean_ms": self.total_ms / len(self.samples) if self.samples else float("nan"),
            "p50_ms": percentile(self.samples, 50),
            "p90_ms": percentile(self.samples, 90),
            "p99_ms": percentile(self.samples, 99),
        }
