"""Performance regression tracking.

Benchmarks that nobody compares against yesterday are decoration.  This tracker
persists every run to a JSON lines file, compares a new run against the rolling
baseline and fails loudly when a metric degrades beyond tolerance.

Noise handling matters: a single slower run on a laptop means nothing.  The
baseline is the *median* of the last ``window`` runs, which is robust to the
occasional outlier, and the tolerance is relative, not absolute.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Direction = Literal["lower_is_better", "higher_is_better"]

DEFAULT_METRICS: dict[str, Direction] = {
    "output_throughput_tok_s": "higher_is_better",
    "ttft_p50_ms": "lower_is_better",
    "ttft_p99_ms": "lower_is_better",
    "e2e_p50_ms": "lower_is_better",
    "e2e_p99_ms": "lower_is_better",
}


@dataclass
class MetricVerdict:
    metric: str
    current: float
    baseline: float
    delta_pct: float
    regressed: bool
    improved: bool


@dataclass
class RegressionVerdict:
    name: str
    ok: bool
    metrics: list[MetricVerdict] = field(default_factory=list)
    note: str = ""

    def regressions(self) -> list[MetricVerdict]:
        return [m for m in self.metrics if m.regressed]

    def report(self) -> str:
        lines = [f"regression check '{self.name}': {'PASS' if self.ok else 'FAIL'}"]
        if self.note:
            lines.append(f"  note: {self.note}")
        for m in self.metrics:
            flag = "REGRESSED" if m.regressed else ("improved" if m.improved else "ok")
            lines.append(f"  {m.metric:<28} {m.current:>12.3f} vs baseline "
                         f"{m.baseline:>12.3f}  ({m.delta_pct:+.1f}%)  {flag}")
        return "\n".join(lines)


def flatten_result(result) -> dict[str, float]:
    """Extract the tracked scalar metrics from a BenchmarkResult."""
    d = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    return {
        "output_throughput_tok_s": float(d["output_throughput_tok_s"]),
        "ttft_p50_ms": float(d["ttft"]["p50_ms"]),
        "ttft_p99_ms": float(d["ttft"]["p99_ms"]),
        "e2e_p50_ms": float(d["e2e"]["p50_ms"]),
        "e2e_p99_ms": float(d["e2e"]["p99_ms"]),
    }


class RegressionTracker:
    """Append-only JSONL history with median-baseline comparison."""

    def __init__(self, path: str | Path = "artifacts/bench_history.jsonl", *,
                 tolerance_pct: float = 15.0, window: int = 5,
                 metrics: dict[str, Direction] | None = None) -> None:
        self.path = Path(path)
        self.tolerance_pct = tolerance_pct
        self.window = window
        self.metrics = metrics or DEFAULT_METRICS

    # -------------------------------------------------------------------- history
    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue      # a truncated last line must not break the tool
        return records

    def history_for(self, name: str) -> list[dict]:
        return [r for r in self.load() if r.get("name") == name]

    def record(self, name: str, values: dict[str, float], *,
               metadata: dict | None = None) -> dict:
        entry = {
            "name": name,
            "timestamp": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "values": values,
            "metadata": metadata or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    # -------------------------------------------------------------------- compare
    def baseline(self, name: str) -> dict[str, float] | None:
        history = self.history_for(name)
        if not history:
            return None
        recent = history[-self.window:]
        out: dict[str, float] = {}
        for metric in self.metrics:
            samples = [r["values"][metric] for r in recent
                       if metric in r.get("values", {})]
            if samples:
                out[metric] = statistics.median(samples)
        return out or None

    def check(self, name: str, values: dict[str, float]) -> RegressionVerdict:
        base = self.baseline(name)
        if base is None:
            return RegressionVerdict(name, True, [], "no baseline yet -- run recorded")

        verdicts: list[MetricVerdict] = []
        for metric, direction in self.metrics.items():
            if metric not in values or metric not in base:
                continue
            current, reference = values[metric], base[metric]
            if reference == 0:
                continue
            delta_pct = (current - reference) / abs(reference) * 100.0
            if direction == "higher_is_better":
                regressed = delta_pct < -self.tolerance_pct
                improved = delta_pct > self.tolerance_pct
            else:
                regressed = delta_pct > self.tolerance_pct
                improved = delta_pct < -self.tolerance_pct
            verdicts.append(MetricVerdict(metric, current, reference,
                                          delta_pct, regressed, improved))

        ok = not any(v.regressed for v in verdicts)
        return RegressionVerdict(name, ok, verdicts)

    def check_and_record(self, name: str, values: dict[str, float], *,
                         metadata: dict | None = None) -> RegressionVerdict:
        verdict = self.check(name, values)
        self.record(name, values, metadata=metadata)
        return verdict

    def prune(self, keep: int = 200) -> int:
        records = self.load()
        if len(records) <= keep:
            return 0
        kept = records[-keep:]
        self.path.write_text("\n".join(json.dumps(r) for r in kept) + "\n",
                             encoding="utf-8")
        return len(records) - keep
