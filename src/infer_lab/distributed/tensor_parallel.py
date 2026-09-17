"""Tensor parallelism (Megatron-LM style).

A transformer block needs exactly **two** collectives per layer if you shard it
correctly, and zero if you shard it wrongly but get the wrong answer:

* the QKV/gate/up projections are **column-parallel** -- split the *output*
  dimension, every rank keeps a slice of the activations, no communication.
* the output/down projections are **row-parallel** -- split the *input*
  dimension so each rank consumes its own slice, then **all-reduce** the partial
  sums.

Pairing them this way means the intermediate activation never has to be
gathered.  Attention heads are assigned whole to ranks, which is why
``num_heads`` must be divisible by the TP degree -- and with GQA, ``num_kv_heads``
becomes the real constraint.

This module simulates the ranks in-process with NumPy so the *math* can be
verified exactly against the unsharded reference; the actual wire transport is
implemented for real in ``ring_allreduce.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DTYPE = np.float32


def shard_tensor(x: np.ndarray, world_size: int, axis: int) -> list[np.ndarray]:
    if x.shape[axis] % world_size:
        raise ValueError(
            f"dimension {axis} of size {x.shape[axis]} is not divisible by "
            f"world_size {world_size}"
        )
    return [np.ascontiguousarray(s) for s in np.split(x, world_size, axis=axis)]


def all_reduce(partials: list[np.ndarray]) -> np.ndarray:
    """Sum across ranks. The one collective row-parallel layers cannot avoid."""
    out = partials[0].astype(np.float32).copy()
    for p in partials[1:]:
        out += p
    return out.astype(DTYPE)


def all_gather(shards: list[np.ndarray], axis: int = -1) -> np.ndarray:
    return np.concatenate(shards, axis=axis).astype(DTYPE)


@dataclass
class ColumnParallelLinear:
    """y = x @ W, W split along the output dimension. No collective."""

    weight: np.ndarray
    world_size: int

    def __post_init__(self) -> None:
        self.shards = shard_tensor(self.weight, self.world_size, axis=1)

    def forward_rank(self, x: np.ndarray, rank: int) -> np.ndarray:
        return (x @ self.shards[rank]).astype(DTYPE)

    def forward(self, x: np.ndarray, *, gather: bool = True) -> np.ndarray:
        outs = [self.forward_rank(x, r) for r in range(self.world_size)]
        return all_gather(outs, axis=-1) if gather else outs  # type: ignore[return-value]

    @property
    def bytes_communicated(self) -> int:
        return 0


@dataclass
class RowParallelLinear:
    """y = x @ W, W split along the input dimension. Requires one all-reduce."""

    weight: np.ndarray
    world_size: int

    def __post_init__(self) -> None:
        self.shards = shard_tensor(self.weight, self.world_size, axis=0)
        self._last_bytes = 0

    def forward_rank(self, x_shard: np.ndarray, rank: int) -> np.ndarray:
        return (x_shard @ self.shards[rank]).astype(DTYPE)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_shards = shard_tensor(np.ascontiguousarray(x), self.world_size, axis=-1)
        partials = [self.forward_rank(x_shards[r], r) for r in range(self.world_size)]
        # Ring all-reduce moves 2*(N-1)/N * bytes per rank -- the cost model that
        # decides whether TP is worth it over a given interconnect.
        elem_bytes = partials[0].nbytes
        self._last_bytes = int(2 * (self.world_size - 1) / self.world_size * elem_bytes)
        return all_reduce(partials)

    @property
    def bytes_communicated(self) -> int:
        return self._last_bytes


@dataclass
class TensorParallelAttention:
    """Head-parallel attention: each rank owns whole heads, one all-reduce at the end."""

    num_heads: int
    num_kv_heads: int
    world_size: int

    def __post_init__(self) -> None:
        if self.num_heads % self.world_size:
            raise ValueError("num_heads must be divisible by world_size")
        if self.num_kv_heads % self.world_size:
            raise ValueError(
                "num_kv_heads must be divisible by world_size -- with GQA this is the "
                "binding constraint on the TP degree (replicate KV heads otherwise)"
            )

    @property
    def heads_per_rank(self) -> int:
        return self.num_heads // self.world_size

    @property
    def kv_heads_per_rank(self) -> int:
        return self.num_kv_heads // self.world_size

    def head_slice(self, rank: int) -> slice:
        h = self.heads_per_rank
        return slice(rank * h, (rank + 1) * h)

    def kv_cache_bytes_per_rank(self, total_bytes: int) -> int:
        """TP also shards the KV cache -- often the real reason to use it."""
        return total_bytes // self.world_size


def verify_equivalence(x: np.ndarray, w_col: np.ndarray, w_row: np.ndarray,
                       world_size: int) -> dict[str, float]:
    """Sharded column->row composition must equal the unsharded computation."""
    reference = (x @ w_col) @ w_row
    col = ColumnParallelLinear(w_col, world_size)
    row = RowParallelLinear(w_row, world_size)
    sharded = row.forward(col.forward(x, gather=True))
    diff = np.abs(reference - sharded)
    return {
        "max_abs_err": float(diff.max()),
        "rel_err": float(np.linalg.norm(diff) / (np.linalg.norm(reference) or 1.0)),
        "allreduce_bytes": float(row.bytes_communicated),
    }
