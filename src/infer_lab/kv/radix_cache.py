"""RadixAttention -- SGLang-style automatic prefix caching.

Motivation
----------
In agentic / few-shot / chat workloads a huge fraction of every request is a
*repeat* of a previous one: the same system prompt, the same tool schema, the
same conversation history.  Recomputing that prefix is pure waste.

RadixAttention keeps a radix tree over already-computed token prefixes.  Each
edge carries exactly ``block_size`` tokens and owns one physical KV block, so a
match can be adopted by simply splicing block ids into the new sequence's block
table and bumping their refcounts -- zero data movement.

Eviction is LRU over *leaves*: an interior node cannot be evicted while a longer
cached continuation still depends on it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from infer_lab.kv.block_allocator import BlockAllocator


@dataclass
class RadixNode:
    tokens: tuple[int, ...] = ()                 # the block_size tokens on the incoming edge
    block_id: int | None = None                  # physical KV block for those tokens
    parent: "RadixNode | None" = None
    children: dict[tuple[int, ...], "RadixNode"] = field(default_factory=dict)
    last_access: int = 0
    depth: int = 0                               # number of cached tokens up to and incl. this node

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    hit_tokens: int = 0
    inserted_blocks: int = 0
    evicted_blocks: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class RadixCache:
    """Block-aligned radix tree over token prefixes."""

    def __init__(self, block_size: int, allocator: BlockAllocator,
                 max_blocks: int | None = None) -> None:
        self.block_size = block_size
        self.allocator = allocator
        self.max_blocks = max_blocks
        self.root = RadixNode()
        self.stats = CacheStats()
        self._clock = itertools.count(1)
        self._cached_blocks = 0

    # ------------------------------------------------------------------- lookup
    def match(self, tokens: Sequence[int]) -> tuple[list[int], int]:
        """Longest block-aligned prefix match.

        Returns ``(block_ids, num_matched_tokens)``.  We never match the *final*
        block of the query: a sequence must keep at least one token to actually
        run a forward pass through, otherwise there is no logit to sample from.
        """
        self.stats.lookups += 1
        usable = max(0, len(tokens) - 1)
        max_chunks = usable // self.block_size

        node = self.root
        blocks: list[int] = []
        now = next(self._clock)
        for chunk_idx in range(max_chunks):
            start = chunk_idx * self.block_size
            key = tuple(tokens[start:start + self.block_size])
            child = node.children.get(key)
            if child is None or child.block_id is None:
                break
            child.last_access = now
            blocks.append(child.block_id)
            node = child

        matched = len(blocks) * self.block_size
        if matched:
            self.stats.hits += 1
            self.stats.hit_tokens += matched
        return blocks, matched

    # ------------------------------------------------------------------- insert
    def insert(self, tokens: Sequence[int], block_ids: Sequence[int]) -> int:
        """Register ``block_ids`` as the cached KV for full blocks of ``tokens``.

        Returns how many *new* blocks were registered.  Newly registered blocks
        get one extra reference (held by the cache itself).
        """
        num_chunks = min(len(block_ids), len(tokens) // self.block_size)
        node = self.root
        now = next(self._clock)
        added = 0
        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.block_size
            key = tuple(tokens[start:start + self.block_size])
            child = node.children.get(key)
            if child is None:
                block = block_ids[chunk_idx]
                self.allocator.incref(block)          # the cache now holds a reference
                child = RadixNode(tokens=key, block_id=block, parent=node,
                                  last_access=now, depth=node.depth + self.block_size)
                node.children[key] = child
                self._cached_blocks += 1
                self.stats.inserted_blocks += 1
                added += 1
            else:
                child.last_access = now
            node = child
        if self.max_blocks is not None:
            self.evict_to(self.max_blocks)
        return added

    # ------------------------------------------------------------------- evict
    def evict(self, num_blocks: int) -> int:
        """Evict up to ``num_blocks`` LRU leaves whose blocks nobody else uses."""
        freed = 0
        while freed < num_blocks:
            victim = self._lru_evictable_leaf()
            if victim is None:
                break
            parent = victim.parent
            assert parent is not None
            del parent.children[victim.tokens]
            if victim.block_id is not None:
                self.allocator.free(victim.block_id)
            self._cached_blocks -= 1
            self.stats.evicted_blocks += 1
            freed += 1
        return freed

    def evict_to(self, target_blocks: int) -> int:
        excess = self._cached_blocks - target_blocks
        return self.evict(excess) if excess > 0 else 0

    def _lru_evictable_leaf(self) -> RadixNode | None:
        best: RadixNode | None = None
        for node in self._walk(self.root):
            if node is self.root or not node.is_leaf or node.block_id is None:
                continue
            # ref==1 means only the cache holds it; a running sequence would make it >1
            if self.allocator.ref_count(node.block_id) != 1:
                continue
            if best is None or node.last_access < best.last_access:
                best = node
        return best

    def _walk(self, node: RadixNode) -> Iterator[RadixNode]:
        yield node
        for child in list(node.children.values()):
            yield from self._walk(child)

    # ------------------------------------------------------------------- report
    @property
    def num_cached_blocks(self) -> int:
        return self._cached_blocks

    def num_nodes(self) -> int:
        return sum(1 for _ in self._walk(self.root)) - 1

    def reset(self) -> None:
        for node in list(self._walk(self.root)):
            if node is not self.root and node.block_id is not None:
                self.allocator.free(node.block_id)
        self.root = RadixNode()
        self._cached_blocks = 0

    def snapshot(self) -> dict[str, float | int]:
        return {
            "cached_blocks": self._cached_blocks,
            "nodes": self.num_nodes(),
            "lookups": self.stats.lookups,
            "hits": self.stats.hits,
            "hit_rate": round(self.stats.hit_rate, 4),
            "hit_tokens": self.stats.hit_tokens,
            "evicted_blocks": self.stats.evicted_blocks,
        }
