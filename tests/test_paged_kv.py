"""PagedAttention: allocator, block tables, copy-on-write."""

from __future__ import annotations

import numpy as np
import pytest

from infer_lab.kv.block_allocator import BlockAllocator, OutOfBlocks


def test_allocate_and_free_roundtrip():
    alloc = BlockAllocator(num_blocks=4, block_size=8)
    blocks = alloc.allocate_many(4)
    assert alloc.num_free == 0 and sorted(blocks) == [0, 1, 2, 3]
    with pytest.raises(OutOfBlocks):
        alloc.allocate()
    alloc.free_many(blocks)
    assert alloc.num_free == 4


def test_all_or_nothing_allocation_leaves_no_partial_state():
    alloc = BlockAllocator(num_blocks=3, block_size=8)
    with pytest.raises(OutOfBlocks):
        alloc.allocate_many(4)
    assert alloc.num_free == 3, "a failed allocation must not leak blocks"


def test_refcounting_defers_free_until_last_referent():
    alloc = BlockAllocator(num_blocks=2, block_size=8)
    block = alloc.allocate()
    alloc.incref(block)
    assert alloc.ref_count(block) == 2 and alloc.is_shared(block)
    alloc.free(block)
    assert alloc.num_free == 1, "block must stay allocated while a referent remains"
    alloc.free(block)
    assert alloc.num_free == 2


def test_double_free_is_rejected():
    alloc = BlockAllocator(num_blocks=1, block_size=8)
    block = alloc.allocate()
    alloc.free(block)
    with pytest.raises(ValueError):
        alloc.free(block)


def test_copy_on_write_allocates_private_block_only_when_shared():
    alloc = BlockAllocator(num_blocks=4, block_size=8)
    block = alloc.allocate()
    same, copied = alloc.prepare_for_write(block)
    assert same == block and not copied

    alloc.incref(block)
    new_block, copied = alloc.prepare_for_write(block)
    assert copied and new_block != block
    assert alloc.ref_count(block) == 1 and alloc.stats.cow_copies == 1


def test_sequence_write_and_gather_roundtrip(kv_cache, rng):
    seq = kv_cache.new_sequence()
    n = 20
    k = rng.normal(size=(n, kv_cache.num_kv_heads, kv_cache.head_dim)).astype(np.float32)
    v = rng.normal(size=(n, kv_cache.num_kv_heads, kv_cache.head_dim)).astype(np.float32)
    seq.write(0, 0, k, v)
    seq.length = n
    gk, gv = seq.gather(0)
    np.testing.assert_array_equal(gk, k)
    np.testing.assert_array_equal(gv, v)


def test_gather_spans_multiple_non_contiguous_blocks(kv_cache, rng):
    """The whole point of paging: physical blocks need not be adjacent."""
    decoy = kv_cache.new_sequence()
    decoy.ensure_capacity(kv_cache.block_size)      # fragment the pool

    seq = kv_cache.new_sequence()
    n = kv_cache.block_size * 3 + 1
    k = rng.normal(size=(n, kv_cache.num_kv_heads, kv_cache.head_dim)).astype(np.float32)
    seq.write(0, 0, k, k)
    seq.length = n
    assert len(seq.block_table) == 4
    np.testing.assert_array_equal(seq.gather(0)[0], k)


def test_incremental_writes_equal_one_shot_write(kv_cache, rng):
    n = kv_cache.block_size * 2 + 3
    data = rng.normal(size=(n, kv_cache.num_kv_heads, kv_cache.head_dim)).astype(np.float32)

    one = kv_cache.new_sequence()
    one.write(0, 0, data, data)
    one.length = n

    incremental = kv_cache.new_sequence()
    for i in range(n):
        incremental.write(0, i, data[i:i + 1], data[i:i + 1])
    incremental.length = n

    np.testing.assert_array_equal(one.gather(0)[0], incremental.gather(0)[0])


def test_release_returns_every_block(kv_cache):
    before = kv_cache.allocator.num_free
    seq = kv_cache.new_sequence()
    seq.ensure_capacity(kv_cache.block_size * 5)
    assert kv_cache.allocator.num_free < before
    seq.release()
    assert kv_cache.allocator.num_free == before


def test_bytes_per_token_matches_analytic_formula(kv_cache, model_config):
    expected = 2 * model_config.num_layers * model_config.num_kv_heads \
        * model_config.head_dim * 4
    assert kv_cache.bytes_per_token() == expected
