"""KV-pressure regression tests.

These exist because of a real production-shaped failure: under sustained load
the RadixAttention prefix cache grew until it owned the entire KV pool. Cached
blocks carry refcount 1, so the allocator counted them as *in use* even though
no sequence needed them. Once that happened:

  1. no request could be admitted -- no free blocks;
  2. the scheduler tried to preempt, but nothing was running, so there was no
     victim to evict;
  3. the step produced no work, and the engine immediately looped again.

The engine burned a full core and drove ``engine_steps_total`` past 170 million
while serving zero tokens. Every request timed out.

The point of this file: the pre-existing suite passed 27/27 both before and
after that bug. It was invisible because unit tests are short, and the cache
only swallows the pool under *sustained* load against a *small* pool. Each test
below is built to fail on the unpatched engine.

See docs/adr/0008-prefix-cache-deadlock.md.
"""

from __future__ import annotations

import pytest

from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.engine.llm_engine import LLMEngine
from infer_lab.engine.request import SamplingParams
from infer_lab.kv.block_allocator import BlockAllocator
from infer_lab.kv.radix_cache import RadixCache


# --------------------------------------------------------------------- helpers
def tiny_model() -> ModelConfig:
    return ModelConfig(vocab_size=300, hidden_size=64, intermediate_size=128,
                       num_layers=2, num_heads=4, num_kv_heads=2,
                       max_position_embeddings=512)


def pressured_engine(*, num_gpu_blocks: int = 24,
                     prefix_caching: bool = True) -> LLMEngine:
    """An engine whose KV pool is deliberately far too small for the workload."""
    return LLMEngine(
        tiny_model(),
        EngineConfig(block_size=8, num_gpu_blocks=num_gpu_blocks, max_num_seqs=8,
                     max_num_batched_tokens=64, max_model_len=256,
                     enable_prefix_caching=prefix_caching,
                     enable_chunked_prefill=True, enable_preemption=True),
    )


def shared_prefix_prompts(count: int, prefix_len: int = 40) -> list[list[int]]:
    """Prompts sharing a long prefix -- the workload that fills the radix cache."""
    prefix = list(range(1, prefix_len + 1))
    return [prefix + [200 + i, 201 + i, 202 + i] for i in range(count)]


def greedy(max_tokens: int = 8) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)


# ------------------------------------------------------- the deadlock itself
def test_prefix_cache_never_starves_admission():
    """The regression test for the deadlock.

    On the unpatched engine the cache consumes the pool, admission fails for
    ever and ``generate`` never drains -- surfacing here as a RuntimeError from
    the step budget rather than as a hang.
    """
    engine = pressured_engine()
    prompts = shared_prefix_prompts(12)

    outputs = engine.generate(prompts, greedy(8), max_steps=5_000)

    assert len(outputs) == len(prompts), "not every request completed"
    assert all(o.num_generated == 8 for o in outputs)


def test_engine_does_not_busy_spin_under_kv_pressure():
    """A step that schedules nothing must not repeat unboundedly.

    Bounded work per request is the invariant. The observed failure was ~170M
    steps for a few dozen requests; any sane bound catches that by orders of
    magnitude, so this assertion is robust rather than brittle.
    """
    engine = pressured_engine()
    prompts = shared_prefix_prompts(10)

    engine.generate(prompts, greedy(8), max_steps=5_000)

    steps_per_request = engine.scheduler.stats.steps / len(prompts)
    assert steps_per_request < 150, (
        f"{steps_per_request:.0f} steps per request indicates the scheduler is "
        f"spinning without making progress"
    )


def test_cached_blocks_can_never_own_the_whole_pool():
    """The structural guarantee: the cache is capped below the pool size.

    Prefix caching is an optimisation; admitting requests is the actual job.
    An optimisation must never be able to starve the job.
    """
    engine = pressured_engine(num_gpu_blocks=32)
    engine.generate(shared_prefix_prompts(10), greedy(6), max_steps=5_000)

    assert engine.radix_cache is not None
    allocator = engine.kv_cache.allocator
    assert engine.radix_cache.num_cached_blocks < allocator.num_blocks, (
        "the prefix cache holds the entire KV pool -- nothing can be admitted"
    )
    assert allocator.num_free > 0, "no free blocks left for a new request"


def test_engine_recovers_after_sustained_load():
    """A second wave must still be servable after the first filled the cache.

    This is the shape of the real incident: the engine worked, then stopped
    working, and never recovered on its own.
    """
    engine = pressured_engine()

    first = engine.generate(shared_prefix_prompts(8), greedy(6), max_steps=5_000)
    assert len(first) == 8

    # Fresh prompts -- no prefix overlap, so the cache cannot help and must
    # instead get out of the way.
    second = [[100 + i, 101 + i, 102 + i, 103 + i] * 5 for i in range(6)]
    assert len(engine.generate(second, greedy(6), max_steps=5_000)) == 6


def test_prefix_caching_still_pays_off_under_pressure():
    """The fix must not amount to "disable the cache".

    Reclaiming blocks under pressure is only correct if the cache still serves
    its purpose the rest of the time.
    """
    engine = pressured_engine(num_gpu_blocks=48)
    engine.generate(shared_prefix_prompts(10), greedy(6), max_steps=5_000)

    assert engine.scheduler.stats.prefix_cache_hit_tokens > 0, (
        "no prefix reuse at all -- the cache has been neutered rather than bounded"
    )


def test_cache_reclamation_is_preferred_over_preemption():
    """Evicting an unused cached block is cheaper than preempting a live sequence.

    A cached block costs a future cache miss. A preempted sequence loses all of
    its accumulated KV and must recompute. The cheaper resource must go first.
    """
    engine = pressured_engine(num_gpu_blocks=40)
    engine.generate(shared_prefix_prompts(8), greedy(6), max_steps=5_000)

    assert engine.scheduler.stats.preemptions == 0, (
        "sequences were preempted while reclaimable cached blocks were available"
    )


def test_no_kv_leak_after_draining_under_pressure():
    """Every block must be accounted for: free, or deliberately held by the cache."""
    engine = pressured_engine()
    engine.generate(shared_prefix_prompts(10), greedy(6), max_steps=5_000)

    allocator = engine.kv_cache.allocator
    cached = engine.radix_cache.num_cached_blocks if engine.radix_cache else 0
    assert allocator.num_free + cached == allocator.num_blocks, (
        f"KV accounting mismatch: {allocator.num_free} free + {cached} cached "
        f"!= {allocator.num_blocks} total"
    )


# ------------------------------------------------------ radix cache unit level
def test_radix_cache_respects_its_block_cap():
    allocator = BlockAllocator(num_blocks=64, block_size=4)
    cache = RadixCache(4, allocator, max_blocks=8)

    for i in range(12):
        tokens = list(range(i * 100, i * 100 + 16))
        blocks = allocator.allocate_many(4)
        cache.insert(tokens, blocks)
        allocator.free_many(blocks)      # the owning sequence finishes

    assert cache.num_cached_blocks <= 8


def test_evict_returns_blocks_to_the_free_list():
    allocator = BlockAllocator(num_blocks=16, block_size=4)
    cache = RadixCache(4, allocator)

    blocks = allocator.allocate_many(4)
    cache.insert(list(range(16)), blocks)
    allocator.free_many(blocks)          # only the cache holds them now

    free_before = allocator.num_free
    evicted = cache.evict(2)
    assert evicted == 2
    assert allocator.num_free == free_before + 2


def test_evict_never_takes_blocks_a_sequence_is_using():
    allocator = BlockAllocator(num_blocks=16, block_size=4)
    cache = RadixCache(4, allocator)

    blocks = allocator.allocate_many(2)
    cache.insert(list(range(8)), blocks)   # refcount now 2: sequence + cache

    assert cache.evict(2) == 0, "evicted a block that is still in use"


@pytest.mark.parametrize("prefix_caching", [True, False])
def test_output_is_identical_with_and_without_the_cache(prefix_caching):
    """Reclamation must not change results.

    This is the project's core rule: an optimisation is only legitimate if the
    system behaves as though it were not there.
    """
    prompts = shared_prefix_prompts(4)
    engine = pressured_engine(num_gpu_blocks=64, prefix_caching=prefix_caching)
    outputs = engine.generate(prompts, greedy(6), max_steps=5_000)

    reference = LLMEngine(
        tiny_model(),
        EngineConfig(block_size=8, num_gpu_blocks=512, max_num_seqs=8,
                     max_num_batched_tokens=512, max_model_len=256,
                     enable_prefix_caching=False, enable_preemption=False),
    ).generate(prompts, greedy(6), max_steps=5_000)

    assert [o.output_token_ids for o in outputs] == \
           [o.output_token_ids for o in reference]
