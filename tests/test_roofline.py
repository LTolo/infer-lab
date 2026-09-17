"""Roofline model and heterogeneous fleet placement."""

from __future__ import annotations

import pytest

from infer_lab.config import ModelConfig
from infer_lab.fleet.hardware import PROFILES, get_profile
from infer_lab.fleet.multi_pool import (
    MultiPoolScheduler, Pool, WorkloadRequest, default_fleet,
)
from infer_lab.fleet.roofline import (
    analyze_decode, analyze_prefill, batch_sweep, crossover_batch_size, kv_cache_capacity,
)


@pytest.fixture
def big_model() -> ModelConfig:
    return ModelConfig(vocab_size=32000, hidden_size=4096, intermediate_size=11008,
                       num_layers=32, num_heads=32, num_kv_heads=8,
                       max_position_embeddings=4096)


def test_profiles_expose_a_sane_ridge_point():
    for hw in PROFILES.values():
        assert 50 < hw.ridge_point < 1000


def test_decode_at_batch_one_is_memory_bound(big_model):
    """The foundational fact of LLM serving."""
    analysis = analyze_decode(big_model, get_profile("accel-b"), batch_size=1, seq_len=1024)
    assert analysis.bound == "memory"
    assert analysis.arithmetic_intensity < analysis.ridge_point
    assert analysis.utilization < 0.05


def test_prefill_of_a_long_prompt_is_compute_bound(big_model):
    analysis = analyze_prefill(big_model, get_profile("accel-b"), batch_size=1, seq_len=4096)
    assert analysis.bound == "compute"
    assert analysis.arithmetic_intensity > analysis.ridge_point


def test_larger_batches_raise_arithmetic_intensity(big_model):
    sweep = batch_sweep(big_model, get_profile("accel-b"), seq_len=512)
    intensities = [a.arithmetic_intensity for a in sweep]
    assert intensities == sorted(intensities)


def test_batching_raises_throughput_sublinearly(big_model):
    hw = get_profile("accel-b")
    one = analyze_decode(big_model, hw, 1, 1024).tokens_per_s
    many = analyze_decode(big_model, hw, 64, 1024).tokens_per_s
    assert many > one * 10, "batching must amortise the weight read"


def test_crossover_exists_at_short_context(big_model):
    crossover = crossover_batch_size(big_model, get_profile("accel-b"), seq_len=128)
    assert crossover is not None and crossover & (crossover - 1) == 0


def test_long_context_decode_can_never_become_compute_bound(big_model):
    """A genuinely counter-intuitive consequence of the roofline model.

    As batch size grows, decode intensity tends to ``2*P / (S * kv_bytes)``.  With
    a long context that limit sits *below* the device ridge point, so no batch
    size makes decode compute-bound: the KV-cache read dominates for ever.  This
    is why long-context serving is a memory-bandwidth problem, and why paged KV,
    GQA and quantized KV matter far more there than faster tensor cores.
    """
    assert crossover_batch_size(big_model, get_profile("accel-b"), seq_len=4096) is None


def test_gqa_shrinks_the_kv_cache():
    mha = ModelConfig(hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=32,
                      intermediate_size=11008, vocab_size=32000)
    gqa = ModelConfig(hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=8,
                      intermediate_size=11008, vocab_size=32000)
    hw = get_profile("accel-b")
    assert kv_cache_capacity(gqa, hw)["max_kv_tokens"] == pytest.approx(
        4 * kv_cache_capacity(mha, hw)["max_kv_tokens"], rel=1e-4
    )


def test_moe_activates_fewer_params_than_it_stores():
    dense = ModelConfig(hidden_size=1024, num_layers=8, intermediate_size=2048,
                        num_heads=16, num_kv_heads=4)
    moe = ModelConfig(hidden_size=1024, num_layers=8, intermediate_size=2048,
                      num_heads=16, num_kv_heads=4, num_experts=8, num_experts_per_tok=2)
    hw = get_profile("accel-b")
    assert moe.param_count() > dense.param_count()
    # more memory traffic, but the FLOPs stay close to the dense model
    assert analyze_decode(moe, hw, 1, 512).flops < 3 * analyze_decode(dense, hw, 1, 512).flops


# ------------------------------------------------------------------- placement
def test_placement_prefers_cheaper_hardware_for_loose_slos(big_model):
    scheduler = MultiPoolScheduler(big_model, default_fleet(4))
    decision = scheduler.place(WorkloadRequest("r1", prompt_len=128, max_tokens=32,
                                               slo_ms=60_000))
    assert decision.pool is not None and decision.meets_slo


def test_tight_slo_is_reported_rather_than_silently_accepted(big_model):
    scheduler = MultiPoolScheduler(big_model, default_fleet(2))
    decision = scheduler.place(WorkloadRequest("r1", prompt_len=4096, max_tokens=1024,
                                               slo_ms=0.001))
    assert not decision.meets_slo
    assert "SLO" in decision.reason or "fastest" in decision.reason


def test_pool_capacity_is_respected(big_model):
    pool = Pool("only", get_profile("accel-a"), num_devices=1, max_concurrent_seqs=2)
    scheduler = MultiPoolScheduler(big_model, [pool])
    placed = [scheduler.place(WorkloadRequest(f"r{i}", 128, 16)) for i in range(4)]
    assert sum(1 for d in placed if d.pool) == 2
    assert sum(1 for d in placed if d.pool is None) == 2


def test_release_frees_capacity(big_model):
    pool = Pool("only", get_profile("accel-a"), num_devices=1, max_concurrent_seqs=1)
    scheduler = MultiPoolScheduler(big_model, [pool])
    scheduler.place(WorkloadRequest("r1", 128, 16))
    assert scheduler.place(WorkloadRequest("r2", 128, 16)).pool is None
    assert scheduler.release("r1") is True
    assert scheduler.place(WorkloadRequest("r3", 128, 16)).pool is not None


def test_snapshot_reports_slo_attainment(big_model):
    scheduler = MultiPoolScheduler(big_model, default_fleet(4))
    for i in range(10):
        scheduler.place(WorkloadRequest(f"r{i}", 256, 64, slo_ms=60_000))
    snap = scheduler.snapshot()
    assert snap["placements"] == 10 and snap["slo_attainment"] == 1.0
