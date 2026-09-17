"""Benchmark harness.

Reports the four numbers that actually describe an inference system, not just
"tokens/sec":

* **TTFT**  time to first token -- dominated by prefill and queueing
* **TPOT**  time per output token -- the steady-state decode cost
* **throughput** output tokens/sec across all concurrent sequences
* **tail latency** p90/p99 -- the SLO that gets violated first

Means are reported but never used for judgement: a p99 that is 8x the mean is a
queueing problem, and the mean hides it completely.
"""

from __future__ import annotations

import gc
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.engine.llm_engine import LLMEngine
from infer_lab.engine.request import SamplingParams
from infer_lab.utils.timing import percentile


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    @staticmethod
    def from_samples(samples: Sequence[float]) -> LatencyStats:
        clean = [s for s in samples if s == s and s not in (float("inf"), float("-inf"))]
        if not clean:
            nan = float("nan")
            return LatencyStats(0, nan, nan, nan, nan, nan, nan, nan)
        return LatencyStats(
            count=len(clean),
            mean_ms=round(statistics.fmean(clean), 4),
            p50_ms=round(percentile(clean, 50), 4),
            p90_ms=round(percentile(clean, 90), 4),
            p99_ms=round(percentile(clean, 99), 4),
            min_ms=round(min(clean), 4),
            max_ms=round(max(clean), 4),
            stdev_ms=round(statistics.pstdev(clean), 4) if len(clean) > 1 else 0.0,
        )

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @property
    def tail_ratio(self) -> float:
        """p99/p50 -- the queueing health indicator."""
        return self.p99_ms / self.p50_ms if self.p50_ms else float("nan")


@dataclass
class BenchmarkResult:
    name: str
    num_requests: int
    prompt_len: int
    max_tokens: int
    wall_ms: float
    total_output_tokens: int
    output_throughput: float          # tokens/s
    request_throughput: float         # requests/s
    ttft: LatencyStats
    e2e: LatencyStats
    tpot: LatencyStats
    engine_stats: dict[str, object] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "num_requests": self.num_requests,
            "prompt_len": self.prompt_len,
            "max_tokens": self.max_tokens,
            "wall_ms": round(self.wall_ms, 3),
            "total_output_tokens": self.total_output_tokens,
            "output_throughput_tok_s": round(self.output_throughput, 3),
            "request_throughput_req_s": round(self.request_throughput, 4),
            "ttft": self.ttft.as_dict(),
            "e2e": self.e2e.as_dict(),
            "tpot": self.tpot.as_dict(),
            "tail_ratio_e2e": round(self.e2e.tail_ratio, 3),
            "engine": self.engine_stats,
            "environment": self.environment,
        }

    def summary_line(self) -> str:
        return (f"{self.name}: {self.output_throughput:7.1f} tok/s | "
                f"TTFT p50 {self.ttft.p50_ms:7.2f} p99 {self.ttft.p99_ms:7.2f} ms | "
                f"E2E p50 {self.e2e.p50_ms:7.2f} p99 {self.e2e.p99_ms:7.2f} ms")


def environment_report() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
    }


def make_prompts(num_requests: int, prompt_len: int, *, shared_prefix: int = 0,
                 seed: int = 0, vocab: int = 1000) -> list[list[int]]:
    """Synthetic prompts with an optional shared prefix (to exercise RadixAttention)."""
    rng = np.random.default_rng(seed)
    prefix = rng.integers(1, vocab, size=shared_prefix).tolist() if shared_prefix else []
    unique_len = max(1, prompt_len - shared_prefix)
    return [prefix + rng.integers(1, vocab, size=unique_len).tolist()
            for _ in range(num_requests)]


def run_benchmark(*, name: str = "default", num_requests: int = 16, prompt_len: int = 64,
                  max_tokens: int = 32, shared_prefix: int = 0,
                  model_config: ModelConfig | None = None,
                  engine_config: EngineConfig | None = None,
                  warmup: int = 1, seed: int = 0) -> BenchmarkResult:
    """Closed-loop benchmark: submit everything at once, drain the engine."""
    model_config = model_config or ModelConfig()
    engine_config = engine_config or EngineConfig()
    engine = LLMEngine(model_config, engine_config)
    sampling = SamplingParams(max_tokens=max_tokens, ignore_eos=True)

    # Token ids must stay inside the model's vocabulary -- deriving the bound from
    # the config rather than hardcoding it keeps the harness usable for any model.
    vocab = model_config.vocab_size

    if warmup:
        engine.generate(
            make_prompts(warmup, min(prompt_len, 16), seed=seed + 999, vocab=vocab),
            sampling,
        )

    prompts = make_prompts(num_requests, prompt_len, shared_prefix=shared_prefix,
                           seed=seed, vocab=vocab)
    gc.collect()

    started = time.perf_counter()
    outputs = engine.generate(prompts, sampling)
    wall_ms = (time.perf_counter() - started) * 1000.0

    ttft = [o.metrics["ttft_ms"] for o in outputs]
    e2e = [o.metrics["e2e_ms"] for o in outputs]
    tpot = [o.metrics["tpot_ms"] for o in outputs]
    total_tokens = sum(o.num_generated for o in outputs)

    return BenchmarkResult(
        name=name,
        num_requests=num_requests,
        prompt_len=prompt_len,
        max_tokens=max_tokens,
        wall_ms=wall_ms,
        total_output_tokens=total_tokens,
        output_throughput=total_tokens / (wall_ms / 1000.0) if wall_ms else 0.0,
        request_throughput=len(outputs) / (wall_ms / 1000.0) if wall_ms else 0.0,
        ttft=LatencyStats.from_samples(ttft),
        e2e=LatencyStats.from_samples(e2e),
        tpot=LatencyStats.from_samples(tpot),
        engine_stats=engine.stats(),
        environment=environment_report(),
    )


def run_kernel_benchmark(fn: Callable[[], object], *, iterations: int = 200,
                         warmup: int = 20) -> LatencyStats:
    """Micro-benchmark a single callable. Used for the binding-overhead comparison."""
    for _ in range(warmup):
        fn()
    gc.collect()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return LatencyStats.from_samples(samples)
