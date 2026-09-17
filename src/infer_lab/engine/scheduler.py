"""Iteration-level (continuous batching) scheduler.

Static batching waits for the slowest sequence in a batch before admitting new
work, so GPU utilisation collapses whenever output lengths differ -- which is
always.  Continuous batching (Orca, then vLLM) instead re-decides the batch
composition *every* forward pass: finished sequences leave immediately and
queued ones join in the same step.

This scheduler implements the three mechanisms that make that practical:

1. **Token budget** -- prefill is compute-bound and decode is memory-bound, so a
   single unbounded prefill can stall every running decode.  A per-step budget
   (``max_num_batched_tokens``) bounds that head-of-line blocking.
2. **Chunked prefill** -- a prompt longer than the budget is sliced, so long
   prompts never monopolise a step and decodes keep flowing alongside them.
3. **Preemption** -- KV memory is finite; when a running sequence cannot get its
   next block, the newest sequence is evicted (recompute or swap) so that older
   ones can finish.  Evicting newest-first keeps the system out of livelock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from infer_lab.config import EngineConfig
from infer_lab.engine.request import Request, RequestState
from infer_lab.kv.block_allocator import OutOfBlocks
from infer_lab.kv.paged_cache import PagedKVCache
from infer_lab.kv.radix_cache import RadixCache
from infer_lab.utils.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class PrefillChunk:
    request: Request
    start_pos: int
    token_ids: list[int]
    is_final_chunk: bool

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class SchedulerOutput:
    prefill_chunks: list[PrefillChunk] = field(default_factory=list)
    decode_requests: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    finished: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_chunks and not self.decode_requests

    @property
    def num_batched_tokens(self) -> int:
        return sum(c.num_tokens for c in self.prefill_chunks) + len(self.decode_requests)


@dataclass
class SchedulerStats:
    steps: int = 0
    scheduled_prefill_tokens: int = 0
    scheduled_decode_tokens: int = 0
    preemptions: int = 0
    prefix_cache_hit_tokens: int = 0
    admitted: int = 0
    finished: int = 0
    max_running: int = 0
    mixed_steps: int = 0          # steps that ran prefill and decode together

    def as_dict(self) -> dict[str, int | float]:
        total = self.scheduled_prefill_tokens + self.scheduled_decode_tokens
        return {
            "steps": self.steps,
            "prefill_tokens": self.scheduled_prefill_tokens,
            "decode_tokens": self.scheduled_decode_tokens,
            "tokens_per_step": round(total / self.steps, 3) if self.steps else 0.0,
            "preemptions": self.preemptions,
            "prefix_cache_hit_tokens": self.prefix_cache_hit_tokens,
            "admitted": self.admitted,
            "finished": self.finished,
            "max_running": self.max_running,
            "mixed_steps": self.mixed_steps,
        }


class Scheduler:
    """FCFS with preemption, chunked prefill and optional prefix caching."""

    def __init__(self, config: EngineConfig, kv_cache: PagedKVCache,
                 radix_cache: RadixCache | None = None) -> None:
        self.config = config
        self.kv = kv_cache
        self.radix = radix_cache
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.stats = SchedulerStats()

    # ------------------------------------------------------------------- queueing
    def add_request(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self.waiting.append(request)
        self.stats.admitted += 1

    def abort(self, request_id: str) -> bool:
        for queue in (self.waiting, self.running):
            for req in list(queue):
                if req.request_id == request_id:
                    self._release(req)
                    queue.remove(req)
                    return True
        return False

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    # ------------------------------------------------------------------ watermark
    def _reserved_blocks(self) -> int:
        return int(self.kv.allocator.num_blocks * self.config.watermark)

    def _free_blocks(self) -> int:
        return self.kv.allocator.num_free - self._reserved_blocks()

    def _make_room(self, blocks_needed: int, protected: set[str]) -> list[Request]:
        """Free KV by preempting the newest running sequences.

        ``protected`` holds the requests already committed to this step.  Evicting
        one of those would release KV that the very same step is about to write
        into -- the kind of bug that surfaces as a rare OutOfBlocks under load
        rather than as a clean failure.  Newest-first eviction otherwise: the
        youngest sequence has the least accumulated work to lose.
        """
        # Reclaim from the prefix cache FIRST.  Those blocks are held only by the
        # cache (refcount 1) and nobody is reading them, so dropping one costs a
        # future cache miss -- while preempting costs a live sequence its entire
        # accumulated KV.  Cheapest memory always goes first.
        if self.radix is not None:
            shortfall = blocks_needed - self._free_blocks()
            if shortfall > 0:
                self.radix.evict(shortfall)

        preempted: list[Request] = []
        if not self.config.enable_preemption:
            return preempted
        while self._free_blocks() < blocks_needed:
            victim = next((r for r in reversed(self.running)
                           if r.request_id not in protected), None)
            if victim is None:
                break
            self._preempt(victim)
            preempted.append(victim)
        return preempted

    def _preempt(self, request: Request) -> None:
        # Recompute is cheaper than swap whenever prompt recomputation is faster
        # than moving KV over PCIe -- true for short/medium prompts, which is the
        # common case. ADR-0004 covers when swapping wins.
        request.reset_for_recompute()
        request.metrics.preemption_count += 1
        self.stats.preemptions += 1
        self.running.remove(request)
        self.waiting.insert(0, request)   # oldest-first fairness on retry
        logger.warning("preempted request under KV pressure",
                       extra={"request_id": request.request_id,
                              "mode": self.config.preemption_mode,
                              "kv_utilization": round(self.kv.allocator.utilization, 3)})

    def _release(self, request: Request) -> None:
        if request.kv is not None:
            request.kv.release()
            request.kv = None

    # -------------------------------------------------------------------- prefix
    def _try_prefix_cache(self, request: Request) -> None:
        if self.radix is None or not self.config.enable_prefix_caching:
            return
        blocks, matched = self.radix.match(request.prompt_token_ids)
        if not matched:
            return
        assert request.kv is not None
        request.kv.adopt_prefix(blocks, matched)
        request.num_computed_tokens = matched
        request.metrics.cached_prefix_tokens = matched
        self.stats.prefix_cache_hit_tokens += matched
        logger.info("prefix cache hit", extra={"request_id": request.request_id,
                                               "matched_tokens": matched})

    def commit_prefix(self, request: Request) -> None:
        """Publish a finished sequence's full blocks into the radix cache."""
        if self.radix is None or not self.config.enable_prefix_caching or request.kv is None:
            return
        self.radix.insert(request.all_token_ids(), request.kv.full_blocks())

    # ------------------------------------------------------------------ scheduling
    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens
        self.stats.steps += 1

        # Blocks the sequences scheduled *in this step* will consume once executed.
        # Without this reservation every request would individually see the same
        # free blocks and the step would abort mid-execution with OutOfBlocks.
        reserved = 0
        # Requests already committed to this step -- never eligible for eviction.
        scheduled: set[str] = set()

        def available() -> int:
            return self._free_blocks() - reserved

        # ---- 1. decodes first: running sequences have priority over new arrivals
        for request in list(self.running):
            if budget <= 0:
                break
            if request.state is RequestState.PREEMPTED or not request.prefill_done:
                continue          # still prefilling (or just evicted), handled below
            assert request.kv is not None
            needed = request.kv.blocks_needed(request.num_computed_tokens + 1)
            if needed > available():
                out.preempted.extend(
                    self._make_room(needed + reserved, scheduled | {request.request_id})
                )
                if request.state is RequestState.PREEMPTED:
                    continue      # this very request got evicted
                # _make_room may have run out of victims. Skipping this step is
                # always safe; allocating anyway would abort a live request.
                if needed > available():
                    continue
            reserved += needed
            scheduled.add(request.request_id)
            out.decode_requests.append(request)
            budget -= 1

        # ---- 2. in-flight chunked prefills
        for request in list(self.running):
            if budget <= 0:
                break
            if request.prefill_done or request.state is RequestState.PREEMPTED:
                continue
            chunk = self._make_chunk(request, budget, available())
            if chunk is None:
                continue
            assert request.kv is not None
            reserved += request.kv.blocks_needed(chunk.start_pos + chunk.num_tokens)
            scheduled.add(request.request_id)
            out.prefill_chunks.append(chunk)
            budget -= chunk.num_tokens

        # ---- 3. admit new requests while budget and memory allow
        while self.waiting and budget > 0 and len(self.running) < self.config.max_num_seqs:
            request = self.waiting[0]
            if not self._admit(request, available()):
                break
            self.waiting.pop(0)
            self.running.append(request)
            chunk = self._make_chunk(request, budget, available())
            if chunk is None:
                # cannot even fit one token right now -- roll back the admission
                self.running.remove(request)
                self.waiting.insert(0, request)
                self._release(request)
                request.state = RequestState.WAITING
                break
            assert request.kv is not None
            reserved += request.kv.blocks_needed(chunk.start_pos + chunk.num_tokens)
            scheduled.add(request.request_id)
            out.prefill_chunks.append(chunk)
            budget -= chunk.num_tokens

        # ---- 4. forward-progress guarantee
        #
        # If nothing could be scheduled but sequences are running, every one of
        # them is blocked on memory and the engine would spin forever making no
        # progress. The contract is that the OLDEST sequence always advances, so
        # we evict the newest until the head of the line can take its next block.
        # This is the invariant that keeps the system out of livelock.
        # Nothing running, nothing scheduled, but requests are queued.  This is the
        # deadlock: the prefix cache owns the pool, so admission fails, and there
        # is no running sequence to preempt.  Without this branch the engine spins
        # for ever, burning a core and inflating the step counter into the
        # millions while serving exactly zero tokens.
        if out.is_empty and not self.running and self.waiting and self.radix is not None:
            head = self.waiting[0]
            needed = -(-(head.num_computed_tokens + 1) // self.kv.block_size)
            shortfall = max(1, needed - self.kv.allocator.num_free)
            reclaimed = self.radix.evict(shortfall)
            if reclaimed:
                logger.warning(
                    "reclaimed prefix-cache blocks to break a KV stall",
                    extra={"reclaimed_blocks": reclaimed,
                           "waiting": len(self.waiting),
                           "free_blocks": self.kv.allocator.num_free},
                )

        if out.is_empty and self.running:
            head = self.running[0]
            if head.prefill_done and head.kv is not None:
                needed = head.kv.blocks_needed(head.num_computed_tokens + 1)
                if needed > self.kv.allocator.num_free:
                    out.preempted.extend(self._make_room(needed, {head.request_id}))
                if head.kv.blocks_needed(head.num_computed_tokens + 1) \
                        <= self.kv.allocator.num_free:
                    scheduled.add(head.request_id)
                    out.decode_requests.append(head)

        # ---- bookkeeping
        self.stats.scheduled_prefill_tokens += sum(c.num_tokens for c in out.prefill_chunks)
        self.stats.scheduled_decode_tokens += len(out.decode_requests)
        self.stats.max_running = max(self.stats.max_running, len(self.running))
        if out.prefill_chunks and out.decode_requests:
            self.stats.mixed_steps += 1
        return out

    def _admit(self, request: Request, available: int) -> bool:
        if request.kv is None:
            request.kv = self.kv.new_sequence()
            request.metrics.first_scheduled_time = (
                request.metrics.first_scheduled_time or time.perf_counter()
            )
            self._try_prefix_cache(request)
        # need at least one block's worth of headroom to make progress
        if request.kv.blocks_needed(request.num_computed_tokens + 1) > available:
            return False
        request.state = RequestState.PREFILL
        return True

    def _make_chunk(self, request: Request, budget: int,
                    available: int) -> PrefillChunk | None:
        remaining = request.remaining_prefill
        if remaining <= 0:
            return None
        take = min(remaining, budget)
        if not self.config.enable_chunked_prefill:
            if remaining > budget:
                return None       # all-or-nothing prefill
            take = remaining
        if take <= 0:
            return None
        assert request.kv is not None
        start = request.num_computed_tokens
        need = request.kv.blocks_needed(start + take)
        if need > available:
            # shrink the chunk to what memory allows, rather than preempting
            capacity = request.kv.capacity + max(0, available) * self.kv.block_size
            take = max(0, min(take, capacity - start))
            if take <= 0:
                return None
        return PrefillChunk(
            request=request,
            start_pos=start,
            token_ids=request.prefill_token_ids[start:start + take],
            is_final_chunk=(start + take) >= request.prefill_target_len,
        )

    # -------------------------------------------------------------------- retire
    def retire(self, request: Request) -> None:
        self.commit_prefix(request)
        self._release(request)
        if request in self.running:
            self.running.remove(request)
        self.stats.finished += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            **self.stats.as_dict(),
        }
