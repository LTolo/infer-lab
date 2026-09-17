"""Paged KV cache: physical pool + per-sequence block tables.

Layout
------
``k_pool`` / ``v_pool`` have shape

    (num_layers, num_blocks, block_size, num_kv_heads, head_dim)

which is the same layout vLLM uses on GPU.  A sequence never owns a contiguous
range -- it owns a *block table*, and logical token ``t`` lives at

    block = block_table[t // block_size]
    slot  = t %  block_size

The ``gather`` step below materialises a contiguous view for the NumPy attention
kernel.  On a GPU this gather does not exist: the paged attention kernel reads
through the block table directly, which is precisely why PagedAttention is
(near) zero-overhead in production.  ADR-0003 documents that trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.kv.block_allocator import BlockAllocator, OutOfBlocks

DTYPE = np.float32


@dataclass
class SequenceKV:
    """A sequence's view onto the paged pool."""

    cache: PagedKVCache
    block_table: list[int] = field(default_factory=list)
    length: int = 0                 # number of tokens whose KV has been written
    num_cached_prefix: int = 0      # tokens served from the prefix cache (no recompute)

    # ------------------------------------------------------------------ capacity
    @property
    def capacity(self) -> int:
        return len(self.block_table) * self.cache.block_size

    def blocks_needed(self, total_tokens: int) -> int:
        bs = self.cache.block_size
        return max(0, -(-total_tokens // bs) - len(self.block_table))

    def ensure_capacity(self, total_tokens: int) -> None:
        """Grow the block table so that ``total_tokens`` fit. Raises OutOfBlocks."""
        need = self.blocks_needed(total_tokens)
        if need:
            self.block_table.extend(self.cache.allocator.allocate_many(need))

    # --------------------------------------------------------------------- write
    def write(self, layer: int, start_pos: int, k: np.ndarray, v: np.ndarray) -> None:
        """Write KV for tokens [start_pos, start_pos + n).

        k, v: (n, num_kv_heads, head_dim)
        """
        n = k.shape[0]
        if n == 0:
            return
        self.ensure_capacity(start_pos + n)
        bs = self.cache.block_size
        written = 0
        while written < n:
            pos = start_pos + written
            bidx, slot = divmod(pos, bs)
            block = self._writable_block(bidx)
            take = min(bs - slot, n - written)
            self.cache.k_pool[layer, block, slot:slot + take] = k[written:written + take]
            self.cache.v_pool[layer, block, slot:slot + take] = v[written:written + take]
            written += take

    def _writable_block(self, block_index: int) -> int:
        """Copy-on-write: never append into a block that another sequence shares."""
        block = self.block_table[block_index]
        new_block, copied = self.cache.allocator.prepare_for_write(block)
        if copied:
            self.cache.k_pool[:, new_block] = self.cache.k_pool[:, block]
            self.cache.v_pool[:, new_block] = self.cache.v_pool[:, block]
            self.block_table[block_index] = new_block
        return new_block

    # ---------------------------------------------------------------------- read
    def gather(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Materialise (length, num_kv_heads, head_dim) K and V for one layer."""
        if self.length == 0 or not self.block_table:
            h, d = self.cache.num_kv_heads, self.cache.head_dim
            empty = np.zeros((0, h, d), dtype=DTYPE)
            return empty, empty.copy()
        nblocks = -(-self.length // self.cache.block_size)
        table = np.asarray(self.block_table[:nblocks], dtype=np.int64)
        k = self.cache.k_pool[layer, table].reshape(-1, self.cache.num_kv_heads,
                                                    self.cache.head_dim)
        v = self.cache.v_pool[layer, table].reshape(-1, self.cache.num_kv_heads,
                                                    self.cache.head_dim)
        return k[:self.length], v[:self.length]

    # --------------------------------------------------------------- prefix reuse
    def adopt_prefix(self, blocks: list[int], num_tokens: int) -> None:
        """Reuse cached prefix blocks instead of recomputing them.

        The blocks are *shared*, not copied: we only bump their refcount.  The
        first write into the last shared block triggers copy-on-write.
        """
        for block in blocks:
            self.cache.allocator.incref(block)
        self.block_table.extend(blocks)
        self.length = num_tokens
        self.num_cached_prefix = num_tokens

    def release(self) -> None:
        self.cache.allocator.free_many(self.block_table)
        self.block_table = []
        self.length = 0

    def full_blocks(self) -> list[int]:
        """Block ids that are completely filled -- the only ones safe to cache."""
        return self.block_table[: self.length // self.cache.block_size]


class PagedKVCache:
    """Owns the physical KV memory and hands out per-sequence views."""

    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig) -> None:
        self.model_config = model_config
        self.block_size = engine_config.block_size
        self.num_blocks = engine_config.num_gpu_blocks
        self.num_layers = model_config.num_layers
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim

        shape = (self.num_layers, self.num_blocks, self.block_size,
                 self.num_kv_heads, self.head_dim)
        self.k_pool = np.zeros(shape, dtype=DTYPE)
        self.v_pool = np.zeros(shape, dtype=DTYPE)
        self.allocator = BlockAllocator(self.num_blocks, self.block_size)

    # ------------------------------------------------------------------ lifecycle
    def new_sequence(self) -> SequenceKV:
        return SequenceKV(cache=self)

    # ------------------------------------------------------------------ reporting
    @property
    def nbytes(self) -> int:
        return self.k_pool.nbytes + self.v_pool.nbytes

    def bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 4

    def max_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def snapshot(self) -> dict[str, float | int]:
        snap = self.allocator.snapshot()
        snap.update({
            "pool_mib": round(self.nbytes / (1024 ** 2), 3),
            "bytes_per_token": self.bytes_per_token(),
            "max_tokens": self.max_tokens(),
        })
        return snap


__all__ = ["PagedKVCache", "SequenceKV", "OutOfBlocks"]
