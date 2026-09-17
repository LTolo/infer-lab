"""RadixAttention prefix caching."""

from __future__ import annotations

from infer_lab.kv.block_allocator import BlockAllocator
from infer_lab.kv.radix_cache import RadixCache


def make_cache(num_blocks: int = 32, block_size: int = 4):
    alloc = BlockAllocator(num_blocks, block_size)
    return RadixCache(block_size, alloc), alloc


def test_miss_on_empty_cache():
    cache, _ = make_cache()
    blocks, matched = cache.match([1, 2, 3, 4, 5])
    assert blocks == [] and matched == 0


def test_exact_prefix_is_reused():
    cache, alloc = make_cache()
    tokens = list(range(12))
    blocks = alloc.allocate_many(3)
    cache.insert(tokens, blocks)

    reused, matched = cache.match(tokens + [99])
    assert matched == 12 and reused == blocks


def test_partial_prefix_matches_only_full_blocks():
    cache, alloc = make_cache(block_size=4)
    shared = list(range(8))
    cache.insert(shared + [100, 101, 102, 103], alloc.allocate_many(3))

    _, matched = cache.match(shared + [200, 201, 202, 203, 204])
    assert matched == 8, "divergent block must not be reused"


def test_last_block_is_never_matched():
    """A sequence must retain at least one token to produce a logit from."""
    cache, alloc = make_cache(block_size=4)
    tokens = list(range(8))
    cache.insert(tokens, alloc.allocate_many(2))
    _, matched = cache.match(tokens)
    assert matched < len(tokens)


def test_insert_takes_a_reference_so_blocks_are_not_reclaimed():
    cache, alloc = make_cache()
    blocks = alloc.allocate_many(2)
    cache.insert(list(range(8)), blocks)
    for block in blocks:
        assert alloc.ref_count(block) == 2
    alloc.free_many(blocks)          # the "owning sequence" finishes
    for block in blocks:
        assert alloc.ref_count(block) == 1, "cache must keep the block alive"


def test_lru_eviction_frees_blocks():
    cache, alloc = make_cache(num_blocks=16, block_size=4)
    cache.insert(list(range(0, 8)), alloc.allocate_many(2))
    cache.insert(list(range(100, 108)), alloc.allocate_many(2))
    alloc.free_many([b for b in range(4)])   # sequences finished; cache holds ref 1

    free_before = alloc.num_free
    evicted = cache.evict(2)
    assert evicted == 2 and alloc.num_free == free_before + 2


def test_eviction_skips_blocks_still_in_use():
    cache, alloc = make_cache(num_blocks=16, block_size=4)
    blocks = alloc.allocate_many(2)
    cache.insert(list(range(8)), blocks)
    # blocks still referenced by the running sequence (ref == 2)
    assert cache.evict(2) == 0


def test_hit_rate_statistics():
    cache, alloc = make_cache()
    tokens = list(range(12))
    cache.insert(tokens, alloc.allocate_many(3))
    cache.match(tokens + [1])
    cache.match([999, 998, 997, 996, 995])
    snap = cache.snapshot()
    assert snap["lookups"] == 2 and snap["hits"] == 1 and snap["hit_rate"] == 0.5


def test_reset_releases_everything():
    cache, alloc = make_cache()
    blocks = alloc.allocate_many(3)
    cache.insert(list(range(12)), blocks)
    alloc.free_many(blocks)
    cache.reset()
    assert cache.num_cached_blocks == 0 and alloc.num_free == alloc.num_blocks
