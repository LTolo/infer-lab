"""Request objects and their lifecycle."""

from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field

from infer_lab.kv.paged_cache import SequenceKV

_COUNTER = itertools.count(1)


class RequestState(enum.Enum):
    WAITING = "waiting"        # admitted to the engine, no KV allocated yet
    PREFILL = "prefill"        # prompt is being consumed (possibly in chunks)
    RUNNING = "running"        # generating, one token per engine step
    PREEMPTED = "preempted"    # KV reclaimed under pressure, will be restarted
    FINISHED = "finished"
    ABORTED = "aborted"


class FinishReason(enum.Enum):
    LENGTH = "length"
    EOS = "eos"
    ABORT = "abort"


@dataclass
class SamplingParams:
    max_tokens: int = 32
    temperature: float = 0.0      # 0.0 -> greedy/deterministic
    top_p: float = 1.0
    top_k: int = 0                # 0 -> disabled
    seed: int | None = None
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")


@dataclass
class RequestMetrics:
    arrival_time: float = field(default_factory=time.perf_counter)
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    finished_time: float | None = None
    preemption_count: int = 0
    recomputed_tokens: int = 0
    cached_prefix_tokens: int = 0
    accepted_draft_tokens: int = 0
    proposed_draft_tokens: int = 0

    @property
    def queue_ms(self) -> float:
        if self.first_scheduled_time is None:
            return float("nan")
        return (self.first_scheduled_time - self.arrival_time) * 1000.0

    @property
    def ttft_ms(self) -> float:
        """Time to first token -- the latency users actually feel."""
        if self.first_token_time is None:
            return float("nan")
        return (self.first_token_time - self.arrival_time) * 1000.0

    @property
    def e2e_ms(self) -> float:
        if self.finished_time is None:
            return float("nan")
        return (self.finished_time - self.arrival_time) * 1000.0

    def tpot_ms(self, num_output_tokens: int) -> float:
        """Time per output token, excluding the prefill."""
        if self.first_token_time is None or self.finished_time is None or num_output_tokens < 2:
            return float("nan")
        return ((self.finished_time - self.first_token_time) * 1000.0) / (num_output_tokens - 1)


@dataclass
class Request:
    prompt_token_ids: list[int]
    sampling: SamplingParams = field(default_factory=SamplingParams)
    request_id: str = field(default_factory=lambda: f"req-{next(_COUNTER):06d}")

    state: RequestState = RequestState.WAITING
    finish_reason: FinishReason | None = None
    output_token_ids: list[int] = field(default_factory=list)
    metrics: RequestMetrics = field(default_factory=RequestMetrics)

    kv: SequenceKV | None = None
    num_computed_tokens: int = 0   # prompt tokens whose KV is resident

    # ------------------------------------------------------------------ helpers
    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.num_generated

    @property
    def prefill_target_len(self) -> int:
        """How many tokens must be KV-resident before decoding can resume.

        Normally the prompt.  After a *recompute* preemption the already-emitted
        tokens must be replayed too -- all but the last one, because that last
        token is precisely what the next decode step will feed.
        """
        return self.prompt_len + max(0, self.num_generated - 1)

    @property
    def prefill_token_ids(self) -> list[int]:
        return self.all_token_ids()[: self.prefill_target_len]

    @property
    def prefill_done(self) -> bool:
        return self.num_computed_tokens >= self.prefill_target_len

    @property
    def remaining_prefill(self) -> int:
        return max(0, self.prefill_target_len - self.num_computed_tokens)

    @property
    def next_position(self) -> int:
        """Position index of the token that will be fed next.

        Invariant: KV holds positions ``[0, num_computed_tokens)``, so the next
        token is written exactly at ``num_computed_tokens``.
        """
        return self.num_computed_tokens

    def all_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.output_token_ids]

    def last_token(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]

    def is_finished(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.ABORTED)

    def append_token(self, token_id: int) -> None:
        if self.metrics.first_token_time is None:
            self.metrics.first_token_time = time.perf_counter()
        self.output_token_ids.append(token_id)

    def finish(self, reason: FinishReason) -> None:
        self.state = RequestState.ABORTED if reason is FinishReason.ABORT else RequestState.FINISHED
        self.finish_reason = reason
        self.metrics.finished_time = time.perf_counter()

    def reset_for_recompute(self) -> None:
        """Preemption by recomputation: drop KV, keep generated tokens.

        Generated tokens are *not* thrown away -- they are re-prefilled together
        with the prompt when the request is rescheduled, so the user never sees
        the preemption in the output, only in the latency.
        """
        if self.kv is not None:
            self.kv.release()
            self.kv = None
        self.metrics.recomputed_tokens += self.num_computed_tokens
        self.num_computed_tokens = 0
        self.state = RequestState.PREEMPTED

    def snapshot(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "state": self.state.value,
            "prompt_len": self.prompt_len,
            "generated": self.num_generated,
            "computed": self.num_computed_tokens,
            "preemptions": self.metrics.preemption_count,
            "cached_prefix": self.metrics.cached_prefix_tokens,
            "finish_reason": self.finish_reason.value if self.finish_reason else None,
        }
