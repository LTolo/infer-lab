"""Physical KV block allocator -- the heart of PagedAttention.

Classic inference engines reserve a contiguous ``max_seq_len`` KV buffer per
sequence, wasting 60-80% of HBM to internal fragmentation.  PagedAttention
instead carves the KV pool into fixed-size *blocks* and gives each sequence a
*block table* (a list of physical block ids), exactly like OS virtual memory.

Reference counting is what makes prefix sharing possible: two sequences with the
same system prompt point at the *same* physical blocks, and a block is only
returned to the free list when the last referent releases it.
"""

from __future__ import annotations

from dataclasses import dataclass


class OutOfBlocks(RuntimeError):
    """Raised when the KV pool is exhausted; the scheduler turns this into a preemption."""


@dataclass
class AllocatorStats:
    allocated: int = 0
    freed: int = 0
    peak_used: int = 0
    cow_copies: int = 0


class BlockAllocator:
    """Free-list allocator over ``num_blocks`` physical KV pages."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))  # pop() -> lowest id
        self._ref: list[int] = [0] * num_blocks
        self.stats = AllocatorStats()

    # ------------------------------------------------------------------ core ops
    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks(f"KV pool exhausted ({self.num_blocks} blocks all in use)")
        block = self._free.pop()
        self._ref[block] = 1
        self.stats.allocated += 1
        self.stats.peak_used = max(self.stats.peak_used, self.num_used)
        return block

    def allocate_many(self, count: int) -> list[int]:
        """All-or-nothing allocation -- avoids half-allocated sequences on OOM."""
        if count > self.num_free:
            raise OutOfBlocks(f"need {count} blocks, only {self.num_free} free")
        return [self.allocate() for _ in range(count)]

    def incref(self, block: int) -> int:
        self._check(block)
        if self._ref[block] == 0:
            raise ValueError(f"cannot incref free block {block}")
        self._ref[block] += 1
        return self._ref[block]

    def free(self, block: int) -> int:
        self._check(block)
        if self._ref[block] == 0:
            raise ValueError(f"double free of block {block}")
        self._ref[block] -= 1
        if self._ref[block] == 0:
            self._free.append(block)
            self.stats.freed += 1
        return self._ref[block]

    def free_many(self, blocks: list[int]) -> None:
        for block in blocks:
            self.free(block)

    # ------------------------------------------------------- copy-on-write support
    def is_shared(self, block: int) -> bool:
        self._check(block)
        return self._ref[block] > 1

    def prepare_for_write(self, block: int) -> tuple[int, bool]:
        """Return (block_to_write, needs_copy).

        If the block is shared (e.g. a cached prefix block that a second sequence
        is now about to append into), allocate a private copy and drop one
        reference to the shared original.  The caller performs the data copy.
        """
        if not self.is_shared(block):
            return block, False
        new_block = self.allocate()
        self.free(block)
        self.stats.cow_copies += 1
        return new_block, True

    # ---------------------------------------------------------------- accessors
    def ref_count(self, block: int) -> int:
        self._check(block)
        return self._ref[block]

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    @property
    def utilization(self) -> float:
        return self.num_used / self.num_blocks

    def evictable_blocks(self) -> list[int]:
        """Blocks held only by the prefix cache (ref==1) are eviction candidates."""
        return [b for b in range(self.num_blocks) if self._ref[b] == 1]

    def snapshot(self) -> dict[str, float | int]:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "used": self.num_used,
            "free": self.num_free,
            "utilization": round(self.utilization, 4),
            "peak_used": self.stats.peak_used,
            "cow_copies": self.stats.cow_copies,
        }

    def _check(self, block: int) -> None:
        if not 0 <= block < self.num_blocks:
            raise IndexError(f"block id {block} out of range [0, {self.num_blocks})")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"BlockAllocator(blocks={self.num_blocks}, size={self.block_size}, "
                f"used={self.num_used}, free={self.num_free})")
