"""Tensor parallelism, pipeline parallelism, ring all-reduce, failure detection."""

from __future__ import annotations

import numpy as np
import pytest

from infer_lab.distributed.health import HealthMonitor, PeerState
from infer_lab.distributed.pipeline_parallel import (
    PipelineSchedule, PipelineStage, split_layers,
)
from infer_lab.distributed.ring_allreduce import RingError, run_ring_allreduce
from infer_lab.distributed.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear, TensorParallelAttention,
    shard_tensor, verify_equivalence,
)


# ------------------------------------------------------------- tensor parallel
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_sharded_matmul_equals_unsharded(world_size, rng):
    """TP is only correct if it is bit-for-bit equivalent modulo float assoc."""
    x = rng.normal(size=(8, 64)).astype(np.float32)
    w_col = rng.normal(size=(64, 128)).astype(np.float32)
    w_row = rng.normal(size=(128, 64)).astype(np.float32)
    stats = verify_equivalence(x, w_col, w_row, world_size)
    assert stats["rel_err"] < 1e-5


def test_column_parallel_needs_no_communication(rng):
    w = rng.normal(size=(32, 64)).astype(np.float32)
    layer = ColumnParallelLinear(w, 4)
    assert layer.bytes_communicated == 0


def test_row_parallel_allreduce_volume_matches_ring_formula(rng):
    x = rng.normal(size=(4, 64)).astype(np.float32)
    w = rng.normal(size=(64, 32)).astype(np.float32)
    layer = RowParallelLinear(w, 4)
    out = layer.forward(x)
    expected = int(2 * 3 / 4 * out.nbytes)
    assert layer.bytes_communicated == expected


def test_shard_rejects_indivisible_dimension(rng):
    with pytest.raises(ValueError, match="divisible"):
        shard_tensor(rng.normal(size=(10, 7)).astype(np.float32), 4, axis=1)


def test_gqa_kv_heads_constrain_tp_degree():
    TensorParallelAttention(num_heads=32, num_kv_heads=8, world_size=8)
    with pytest.raises(ValueError, match="num_kv_heads"):
        TensorParallelAttention(num_heads=32, num_kv_heads=4, world_size=8)


def test_tp_shards_the_kv_cache():
    attn = TensorParallelAttention(num_heads=32, num_kv_heads=8, world_size=4)
    assert attn.kv_cache_bytes_per_rank(8000) == 2000


# ----------------------------------------------------------- pipeline parallel
def test_pipeline_preserves_values_and_order(rng):
    stages = [PipelineStage(i, layers, lambda a: a + 1.0)
              for i, layers in enumerate(split_layers(8, 4))]
    schedule = PipelineSchedule(stages, num_micro_batches=4)
    micro = [np.full((2, 3), float(i), dtype=np.float32) for i in range(4)]
    outputs = schedule.run(micro)
    for i, out in enumerate(outputs):
        np.testing.assert_allclose(out, float(i) + 4.0)


@pytest.mark.parametrize("stages,micro", [(2, 4), (4, 8), (8, 32)])
def test_measured_bubble_matches_analytic_formula(stages, micro):
    schedule = PipelineSchedule(
        [PipelineStage(i, ls, lambda a: a) for i, ls in enumerate(split_layers(32, stages))],
        num_micro_batches=micro,
    )
    schedule.run([np.ones((1, 2), dtype=np.float32) for _ in range(micro)])
    assert schedule.measured_bubble_fraction() == pytest.approx(
        schedule.theoretical_bubble_fraction(), rel=1e-9
    )


def test_more_micro_batches_shrink_the_bubble():
    def bubble(m: int) -> float:
        s = PipelineSchedule(
            [PipelineStage(i, ls, lambda a: a) for i, ls in enumerate(split_layers(16, 4))],
            num_micro_batches=m)
        s.run([np.ones((1, 2), dtype=np.float32) for _ in range(m)])
        return s.measured_bubble_fraction()

    assert bubble(2) > bubble(8) > bubble(32)


def test_layer_split_is_balanced_and_contiguous():
    parts = split_layers(10, 4)
    assert [len(p) for p in parts] == [3, 3, 2, 2]
    assert [i for p in parts for i in p] == list(range(10))


# ----------------------------------------------------------------- ring reduce
@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_ring_allreduce_sums_across_ranks(world_size):
    tensors = [np.full(128, r + 1, dtype=np.float32) for r in range(world_size)]
    results, _ = run_ring_allreduce(tensors)
    expected = sum(range(1, world_size + 1))
    assert len(results) == world_size
    for result in results:
        np.testing.assert_allclose(result, expected)


def test_ring_allreduce_is_bandwidth_optimal():
    """Every rank must move exactly 2*(N-1)/N*S bytes -- the ring's whole point."""
    world_size = 4
    tensors = [np.arange(1024, dtype=np.float32) for _ in range(world_size)]
    _, stats = run_ring_allreduce(tensors)
    expected = int(2 * (world_size - 1) / world_size * tensors[0].nbytes)
    for s in stats:
        assert s.bytes_sent == expected
        assert s.bytes_received == expected
        assert s.steps == 2 * (world_size - 1)


def test_ring_allreduce_handles_multidimensional_tensors(rng):
    tensors = [rng.normal(size=(8, 16)).astype(np.float32) for _ in range(3)]
    results, _ = run_ring_allreduce(tensors)
    # atol is required, not optional: the ring accumulates in a different order
    # than a sequential sum, so results agree only to float32 epsilon.
    np.testing.assert_allclose(results[0], sum(tensors), rtol=1e-5, atol=1e-5)
    assert results[0].shape == (8, 16)


def test_ring_allreduce_is_not_bitwise_identical_to_sequential_sum(rng):
    """Reduction order changes the result -- a real distributed-inference gotcha.

    The same model on the same input can produce different logits at different TP
    degrees purely because the all-reduce sums partials in a different order.
    Tests that assert bitwise equality across world sizes will flake for ever;
    this test pins the *expected* magnitude of that divergence instead.
    """
    tensors = [rng.normal(size=1024).astype(np.float32) * 1000 for _ in range(4)]
    results, _ = run_ring_allreduce(tensors)
    diff = float(np.max(np.abs(results[0] - sum(tensors))))
    assert diff < 1e-2, "divergence must stay at float32 epsilon scale"


def test_ring_rejects_single_rank():
    with pytest.raises(ValueError):
        run_ring_allreduce([np.ones(4, dtype=np.float32)])


def test_ring_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        run_ring_allreduce([np.ones(4, dtype=np.float32), np.ones(8, dtype=np.float32)])


# ------------------------------------------------------------------- health
def test_dead_peer_is_detected():
    monitor = HealthMonitor(world_size=3, interval_s=0.001, failure_threshold=2)
    monitor.heartbeat(0)
    monitor.heartbeat(1)
    import time
    time.sleep(0.02)
    monitor.heartbeat(0)
    states = monitor.evaluate()
    assert states[0] is PeerState.HEALTHY
    assert states[1] is PeerState.DEAD
    assert states[2] is PeerState.UNKNOWN
    assert not monitor.is_quorate()


def test_straggler_detection():
    monitor = HealthMonitor(world_size=4)
    for rank in range(3):
        monitor.heartbeat(rank, step=100)
    monitor.heartbeat(3, step=17)
    assert monitor.stragglers() == [3]
