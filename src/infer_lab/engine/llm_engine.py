"""The engine loop: schedule -> execute -> sample -> retire.

``step()`` performs exactly one iteration and is the unit that continuous
batching is built on.  Everything above it (the HTTP server, the benchmark
harness, the CLI) just drives ``step()`` and consumes the outputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.engine.request import (
    FinishReason,
    Request,
    RequestState,
    SamplingParams,
)
from infer_lab.engine.sampling import sample_token
from infer_lab.engine.scheduler import Scheduler
from infer_lab.kv.paged_cache import PagedKVCache
from infer_lab.kv.radix_cache import RadixCache
from infer_lab.model.numpy_model import NumpyTransformer
from infer_lab.model.tokenizer import EOS_ID, ByteTokenizer
from infer_lab.model.weights import ModelWeights
from infer_lab.utils.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class EngineOutput:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    finished: bool
    finish_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)


@dataclass
class StepStats:
    step_index: int
    num_prefill_tokens: int
    num_decode_tokens: int
    num_running: int
    num_waiting: int
    duration_ms: float
    kv_utilization: float
    preemptions: int


class LLMEngine:
    """Owns the model, the KV memory and the scheduler."""

    def __init__(self, model_config: ModelConfig | None = None,
                 engine_config: EngineConfig | None = None,
                 weights: ModelWeights | None = None,
                 *, attention_impl: str = "flash") -> None:
        self.model_config = model_config or ModelConfig()
        self.engine_config = engine_config or EngineConfig()
        self.weights = weights or ModelWeights.random(self.model_config,
                                                      seed=self.engine_config.seed)
        self.model = NumpyTransformer(self.weights, attention_impl=attention_impl)
        self.kv_cache = PagedKVCache(self.model_config, self.engine_config)
        # Defence in depth: cap the cache so it can never crowd live sequences out
        # of the pool.  Prefix caching is an optimisation; admitting requests is
        # the actual job, and an optimisation must never starve the job.
        self.radix_cache = (
            RadixCache(
                self.engine_config.block_size,
                self.kv_cache.allocator,
                max_blocks=max(1, int(self.engine_config.num_gpu_blocks * 0.8)),
            )
            if self.engine_config.enable_prefix_caching else None
        )
        self.scheduler = Scheduler(self.engine_config, self.kv_cache, self.radix_cache)
        self.tokenizer = ByteTokenizer()
        self._rng = np.random.default_rng(self.engine_config.seed)
        self._requests: dict[str, Request] = {}
        self._step_index = 0
        self.step_history: list[StepStats] = []

    # ---------------------------------------------------------------- submission
    def add_request(self, prompt_token_ids: list[int],
                    sampling: SamplingParams | None = None,
                    request_id: str | None = None) -> str:
        # Validate at the boundary, before any KV is allocated.
        #
        # An out-of-range id would otherwise surface as an IndexError from the
        # embedding lookup -- inside the engine thread, after admission, where
        # the only safe recovery is to fail every in-flight request in that
        # step. One malformed client request must never be able to do that.
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if not all(isinstance(t, int) and not isinstance(t, bool)
                   for t in prompt_token_ids):
            raise ValueError("prompt_token_ids must contain only integers")
        smallest = min(prompt_token_ids)
        largest = max(prompt_token_ids)
        if smallest < 0:
            raise ValueError(f"token id {smallest} is negative")
        if largest >= self.model_config.vocab_size:
            raise ValueError(
                f"token id {largest} is out of range for vocab_size "
                f"{self.model_config.vocab_size}"
            )
        max_len = self.engine_config.max_model_len
        sampling = sampling or SamplingParams(max_tokens=self.engine_config.default_max_tokens)
        if len(prompt_token_ids) + sampling.max_tokens > max_len:
            raise ValueError(
                f"prompt ({len(prompt_token_ids)}) + max_tokens ({sampling.max_tokens}) "
                f"exceeds max_model_len ({max_len})"
            )
        request = Request(prompt_token_ids=list(prompt_token_ids), sampling=sampling)
        if request_id:
            request.request_id = request_id
        self._requests[request.request_id] = request
        self.scheduler.add_request(request)
        return request.request_id

    def abort_request(self, request_id: str) -> bool:
        request = self._requests.get(request_id)
        if request is None or request.is_finished():
            return False
        self.scheduler.abort(request_id)
        request.finish(FinishReason.ABORT)
        return True

    @property
    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_work

    # --------------------------------------------------------------------- step
    def step(self) -> list[EngineOutput]:
        """Run exactly one scheduler iteration."""
        started = time.perf_counter()
        scheduled = self.scheduler.schedule()
        outputs: list[EngineOutput] = []

        if scheduled.is_empty:
            self._record_step(scheduled, started)
            return outputs

        # ---- prefill (each chunk is one sequence; only the last chunk samples)
        for chunk in scheduled.prefill_chunks:
            request = chunk.request
            assert request.kv is not None
            logits = self.model.forward_prefill(
                np.asarray(chunk.token_ids, dtype=np.int64), request.kv,
                start_pos=chunk.start_pos,
            )
            request.num_computed_tokens = chunk.start_pos + chunk.num_tokens
            if chunk.is_final_chunk:
                request.state = RequestState.RUNNING
                if request.num_generated == 0:
                    self._emit(request, logits)
                    outputs.append(self._output_for(request))

        # ---- decode (all running sequences in ONE batched forward)
        decodes = [r for r in scheduled.decode_requests
                   if not r.is_finished() and r.kv is not None
                   and r.state is RequestState.RUNNING]
        if decodes:
            tokens = np.array([r.last_token() for r in decodes], dtype=np.int64)
            positions = np.array([r.next_position for r in decodes], dtype=np.int64)
            kvs = [r.kv for r in decodes]
            assert all(kv is not None for kv in kvs)
            batch_logits = self.model.forward_decode_batch(tokens, kvs, positions)  # type: ignore[arg-type]
            for request, logits in zip(decodes, batch_logits):
                request.num_computed_tokens += 1
                self._emit(request, logits)
                outputs.append(self._output_for(request))

        # ---- retire finished sequences immediately (that is the whole point)
        for request in list(self.scheduler.running):
            if request.is_finished():
                self.scheduler.retire(request)

        self._record_step(scheduled, started)
        return outputs

    def _emit(self, request: Request, logits: np.ndarray) -> None:
        token = sample_token(logits, request.sampling, self._rng)
        request.append_token(token)
        if not request.sampling.ignore_eos and token == EOS_ID:
            request.finish(FinishReason.EOS)
        elif request.num_generated >= request.sampling.max_tokens:
            request.finish(FinishReason.LENGTH)

    def _output_for(self, request: Request) -> EngineOutput:
        metrics = {
            "queue_ms": request.metrics.queue_ms,
            "ttft_ms": request.metrics.ttft_ms,
            "e2e_ms": request.metrics.e2e_ms,
            "tpot_ms": request.metrics.tpot_ms(request.num_generated),
            "preemptions": float(request.metrics.preemption_count),
            "cached_prefix_tokens": float(request.metrics.cached_prefix_tokens),
            "recomputed_tokens": float(request.metrics.recomputed_tokens),
        }
        return EngineOutput(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=list(request.output_token_ids),
            finished=request.is_finished(),
            finish_reason=request.finish_reason.value if request.finish_reason else None,
            metrics=metrics,
        )

    def _record_step(self, scheduled, started: float) -> None:
        self._step_index += 1
        stats = StepStats(
            step_index=self._step_index,
            num_prefill_tokens=sum(c.num_tokens for c in scheduled.prefill_chunks),
            num_decode_tokens=len(scheduled.decode_requests),
            num_running=self.scheduler.num_running,
            num_waiting=self.scheduler.num_waiting,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            kv_utilization=self.kv_cache.allocator.utilization,
            preemptions=len(scheduled.preempted),
        )
        self.step_history.append(stats)
        if len(self.step_history) > 10_000:      # bounded memory for long runs
            del self.step_history[:5_000]

    # ------------------------------------------------------------ offline helper
    def generate(self, prompts: list[list[int]],
                 sampling: SamplingParams | None = None,
                 *, max_steps: int = 100_000) -> list[EngineOutput]:
        """Blocking batch API -- submit everything, drive the loop until drained."""
        ids = [self.add_request(p, sampling) for p in prompts]
        final: dict[str, EngineOutput] = {}
        steps = 0
        while self.has_unfinished_requests and steps < max_steps:
            for out in self.step():
                if out.finished:
                    final[out.request_id] = out
            steps += 1
        if steps >= max_steps:
            raise RuntimeError(f"engine did not drain within {max_steps} steps")
        return [final[i] for i in ids if i in final]

    def generate_text(self, prompts: list[str],
                      sampling: SamplingParams | None = None) -> list[str]:
        encoded = [self.tokenizer.encode(p) for p in prompts]
        return [self.tokenizer.decode(o.output_token_ids)
                for o in self.generate(encoded, sampling)]

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict[str, object]:
        recent = self.step_history[-200:]
        step_ms = [s.duration_ms for s in recent]
        return {
            "model": self.model_config.to_dict(),
            "params": self.model_config.param_count(),
            "engine": self.engine_config.to_dict(),
            "scheduler": self.scheduler.snapshot(),
            "kv_cache": self.kv_cache.snapshot(),
            "prefix_cache": self.radix_cache.snapshot() if self.radix_cache else None,
            "model_stats": self.model.stats.as_dict(),
            "steps_recorded": len(self.step_history),
            "avg_step_ms": round(sum(step_ms) / len(step_ms), 4) if step_ms else 0.0,
        }
