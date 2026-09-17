"""Numeric instability debugger.

Numeric bugs in inference are uniquely nasty: the model does not crash, it just
quietly produces worse tokens.  By the time a NaN reaches the output, the actual
cause is dozens of layers upstream.

This detector hooks every layer's activations, records per-layer statistics and
flags the *first* layer where something goes wrong -- NaN, Inf, an activation
magnitude that would overflow fp16, or a distribution that has silently
collapsed (a dead layer). ``inject_fault`` deliberately corrupts a weight matrix
so the detector can be tested against a known-bad model.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from infer_lab.utils.logging_conf import get_logger

logger = get_logger(__name__)

FP16_MAX = 65504.0


class NumericIssue(enum.Enum):
    NAN = "nan"
    INF = "inf"
    FP16_OVERFLOW = "fp16_overflow"
    UNDERFLOW = "underflow"
    DEAD_ACTIVATION = "dead_activation"
    MAGNITUDE_SPIKE = "magnitude_spike"


@dataclass
class LayerReport:
    name: str
    layer_index: int
    shape: tuple[int, ...]
    min: float
    max: float
    mean: float
    std: float
    abs_max: float
    nan_count: int
    inf_count: int
    issues: list[NumericIssue] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "layer": self.layer_index,
            "shape": list(self.shape),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "abs_max": round(self.abs_max, 6),
            "nan": self.nan_count,
            "inf": self.inf_count,
            "issues": [i.value for i in self.issues],
        }


class InstabilityDetector:
    """Forward-hook style activation monitor."""

    def __init__(self, *, fp16_check: bool = True, spike_factor: float = 50.0,
                 dead_std: float = 1e-8) -> None:
        self.fp16_check = fp16_check
        self.spike_factor = spike_factor
        self.dead_std = dead_std
        self.reports: list[LayerReport] = []
        self._running_abs_max = 0.0
        self.enabled = True

    # ------------------------------------------------------------------- hooking
    def observe(self, name: str, tensor: np.ndarray, layer_index: int = -1) -> LayerReport:
        """Record statistics for one activation tensor and classify any problem."""
        arr = np.asarray(tensor, dtype=np.float32)
        finite_mask = np.isfinite(arr)
        finite = arr[finite_mask]
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        abs_max = float(np.abs(finite).max()) if finite.size else float("inf")

        report = LayerReport(
            name=name,
            layer_index=layer_index,
            shape=tuple(arr.shape),
            min=float(finite.min()) if finite.size else float("nan"),
            max=float(finite.max()) if finite.size else float("nan"),
            mean=float(finite.mean()) if finite.size else float("nan"),
            std=float(finite.std()) if finite.size else float("nan"),
            abs_max=abs_max,
            nan_count=nan_count,
            inf_count=inf_count,
        )

        if nan_count:
            report.issues.append(NumericIssue.NAN)
        if inf_count:
            report.issues.append(NumericIssue.INF)
        if self.fp16_check and np.isfinite(abs_max) and abs_max > FP16_MAX:
            report.issues.append(NumericIssue.FP16_OVERFLOW)
        if finite.size and report.std < self.dead_std and abs(report.mean) < self.dead_std:
            report.issues.append(NumericIssue.DEAD_ACTIVATION)
        if (self._running_abs_max > 0 and np.isfinite(abs_max)
                and abs_max > self._running_abs_max * self.spike_factor):
            report.issues.append(NumericIssue.MAGNITUDE_SPIKE)

        if np.isfinite(abs_max):
            self._running_abs_max = max(self._running_abs_max, abs_max)

        self.reports.append(report)
        if report.issues:
            logger.error("numeric instability detected",
                         extra={"layer": name, "issues": [i.value for i in report.issues],
                                "abs_max": abs_max, "nan": nan_count, "inf": inf_count})
        return report

    def hook(self, name: str, layer_index: int = -1):
        """Return a callable usable as a forward hook: ``hook(name)(tensor)``."""
        def _hook(tensor: np.ndarray) -> np.ndarray:
            if self.enabled:
                self.observe(name, tensor, layer_index)
            return tensor
        return _hook

    # ------------------------------------------------------------------ analysis
    @property
    def has_issues(self) -> bool:
        return any(r.issues for r in self.reports)

    def first_bad_layer(self) -> LayerReport | None:
        """The actual root cause -- everything downstream is collateral damage."""
        return next((r for r in self.reports if r.issues), None)

    def issue_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for report in self.reports:
            for issue in report.issues:
                counts[issue.value] = counts.get(issue.value, 0) + 1
        return counts

    def summary(self) -> dict[str, object]:
        first = self.first_bad_layer()
        return {
            "layers_observed": len(self.reports),
            "healthy": not self.has_issues,
            "issue_counts": self.issue_counts(),
            "first_bad_layer": first.as_dict() if first else None,
            "max_abs_activation": round(self._running_abs_max, 6),
        }

    def reset(self) -> None:
        self.reports.clear()
        self._running_abs_max = 0.0


def inject_fault(weights, layer_index: int = 0, kind: str = "overflow",
                 scale: float = 1e4):
    """Corrupt one weight matrix in place so the detector has something to find.

    kinds: ``overflow`` (huge magnitudes), ``nan``, ``inf``, ``dead`` (all zeros).
    """
    layer = weights.layers[layer_index]
    target = layer.wq
    if kind == "overflow":
        target[:] = target * scale
    elif kind == "nan":
        target[0, 0] = np.nan
    elif kind == "inf":
        target[0, 0] = np.inf
    elif kind == "dead":
        target[:] = 0.0
    else:
        raise ValueError(f"unknown fault kind {kind!r}")
    return weights
