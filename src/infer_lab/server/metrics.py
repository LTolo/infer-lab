"""Prometheus metrics in the text exposition format -- no client library needed.

Keeping this dependency-free is deliberate: ``prometheus_client`` would be one
more thing to install on a locked-down machine, and the text format is a dozen
lines of code.  The output is byte-compatible with what Prometheus scrapes.

Histogram bucket choice matters more than people think: buckets must bracket
your SLO, otherwise ``histogram_quantile`` interpolates inside a bucket that is
wider than the thing you are trying to measure.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

LATENCY_BUCKETS_MS = (1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


@dataclass
class Counter:
    name: str
    help: str
    labels: tuple[str, ...] = ()
    values: dict[tuple[str, ...], float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, **label_values: str) -> None:
        key = tuple(str(label_values.get(k, "")) for k in self.labels)
        self.values[key] = self.values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        if not self.values:
            out.append(f"{self.name} 0")
        for key, value in sorted(self.values.items()):
            out.append(f"{self.name}{_labels(self.labels, key)} {value:g}")
        return out


@dataclass
class Gauge:
    name: str
    help: str
    labels: tuple[str, ...] = ()
    values: dict[tuple[str, ...], float] = field(default_factory=dict)

    def set(self, value: float, **label_values: str) -> None:
        key = tuple(str(label_values.get(k, "")) for k in self.labels)
        self.values[key] = float(value)

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        if not self.values:
            out.append(f"{self.name} 0")
        for key, value in sorted(self.values.items()):
            out.append(f"{self.name}{_labels(self.labels, key)} {value:g}")
        return out


@dataclass
class Histogram:
    name: str
    help: str
    buckets: tuple[float, ...] = LATENCY_BUCKETS_MS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        if value != value:      # NaN
            return
        self.total += value
        self.count += 1
        for i, upper in enumerate(self.buckets):
            if value <= upper:
                self.counts[i] += 1

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        cumulative = 0
        for upper, c in zip(self.buckets, self.counts):
            cumulative = max(cumulative, c)
            out.append(f'{self.name}_bucket{{le="{upper:g}"}} {c}')
        out.append(f'{self.name}_bucket{{le="+Inf"}} {self.count}')
        out.append(f"{self.name}_sum {self.total:g}")
        out.append(f"{self.name}_count {self.count}")
        return out


def _labels(names: tuple[str, ...], values: tuple[str, ...]) -> str:
    if not names:
        return ""
    pairs = ",".join(f'{n}="{v}"' for n, v in zip(names, values))
    return "{" + pairs + "}"


class MetricsRegistry:
    """The metric surface an inference server is expected to expose."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.start_time = time.time()

        self.requests_total = Counter(
            "infer_lab_requests_total", "Total requests by outcome.", ("outcome",))
        self.tokens_total = Counter(
            "infer_lab_tokens_total", "Tokens processed by phase.", ("phase",))
        self.preemptions_total = Counter(
            "infer_lab_preemptions_total", "Sequences preempted under KV pressure.")
        self.prefix_cache_hits = Counter(
            "infer_lab_prefix_cache_hit_tokens_total", "Tokens served from the prefix cache.")
        self.engine_steps_total = Counter(
            "infer_lab_engine_steps_total", "Engine scheduler iterations executed.")

        self.running_sequences = Gauge(
            "infer_lab_running_sequences", "Sequences currently decoding.")
        self.waiting_sequences = Gauge(
            "infer_lab_waiting_sequences", "Sequences queued for admission.")
        self.kv_cache_utilization = Gauge(
            "infer_lab_kv_cache_utilization_ratio", "Fraction of KV blocks in use.")
        self.kv_blocks_free = Gauge(
            "infer_lab_kv_blocks_free", "Free KV blocks.")
        self.batch_size = Gauge(
            "infer_lab_batch_size", "Tokens batched in the most recent engine step.")
        self.build_info = Gauge(
            "infer_lab_build_info", "Build metadata.", ("version", "backend"))

        self.ttft_ms = Histogram("infer_lab_ttft_milliseconds", "Time to first token.")
        self.e2e_ms = Histogram("infer_lab_e2e_milliseconds", "End-to-end request latency.")
        self.tpot_ms = Histogram(
            "infer_lab_tpot_milliseconds", "Time per output token.",
            buckets=(0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500))
        self.step_ms = Histogram(
            "infer_lab_engine_step_milliseconds", "Duration of one engine step.",
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250))

    # ------------------------------------------------------------------- helpers
    def observe_request(self, *, outcome: str, ttft_ms: float, e2e_ms: float,
                        tpot_ms: float, prompt_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.requests_total.inc(outcome=outcome)
            self.tokens_total.inc(prompt_tokens, phase="prefill")
            self.tokens_total.inc(output_tokens, phase="decode")
            self.ttft_ms.observe(ttft_ms)
            self.e2e_ms.observe(e2e_ms)
            self.tpot_ms.observe(tpot_ms)

    def observe_engine(self, engine) -> None:
        with self._lock:
            sched = engine.scheduler
            self.running_sequences.set(sched.num_running)
            self.waiting_sequences.set(sched.num_waiting)
            self.kv_cache_utilization.set(engine.kv_cache.allocator.utilization)
            self.kv_blocks_free.set(engine.kv_cache.allocator.num_free)
            if engine.step_history:
                last = engine.step_history[-1]
                self.batch_size.set(last.num_prefill_tokens + last.num_decode_tokens)
                self.step_ms.observe(last.duration_ms)

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for metric in (
                self.requests_total, self.tokens_total, self.preemptions_total,
                self.prefix_cache_hits, self.engine_steps_total,
                self.running_sequences, self.waiting_sequences,
                self.kv_cache_utilization, self.kv_blocks_free, self.batch_size,
                self.build_info,
                self.ttft_ms, self.e2e_ms, self.tpot_ms, self.step_ms,
            ):
                lines.extend(metric.render())
            lines.append("# HELP infer_lab_uptime_seconds Process uptime.")
            lines.append("# TYPE infer_lab_uptime_seconds gauge")
            lines.append(f"infer_lab_uptime_seconds {time.time() - self.start_time:g}")
            return "\n".join(lines) + "\n"


_REGISTRY: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = MetricsRegistry()
    return _REGISTRY


def reset_metrics() -> MetricsRegistry:
    global _REGISTRY
    _REGISTRY = MetricsRegistry()
    return _REGISTRY
