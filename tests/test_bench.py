"""Benchmark harness and regression tracking."""

from __future__ import annotations

from infer_lab.bench.harness import LatencyStats, make_prompts, run_benchmark
from infer_lab.bench.regression import RegressionTracker, flatten_result
from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.utils.timing import percentile


def small_configs():
    return (ModelConfig(vocab_size=300, hidden_size=64, intermediate_size=128,
                        num_layers=2, num_heads=4, num_kv_heads=2,
                        max_position_embeddings=256),
            EngineConfig(block_size=8, num_gpu_blocks=128, max_num_seqs=8,
                         max_num_batched_tokens=64, max_model_len=256))


def test_percentiles_are_ordered():
    samples = list(range(1, 101))
    assert percentile(samples, 50) <= percentile(samples, 90) <= percentile(samples, 99)


def test_latency_stats_handle_nan_and_empty():
    stats = LatencyStats.from_samples([float("nan"), 1.0, 2.0])
    assert stats.count == 2
    assert LatencyStats.from_samples([]).count == 0


def test_make_prompts_share_the_requested_prefix():
    prompts = make_prompts(4, prompt_len=20, shared_prefix=8)
    first = prompts[0][:8]
    assert all(p[:8] == first for p in prompts)
    assert all(len(p) == 20 for p in prompts)


def test_benchmark_produces_consistent_totals():
    model, engine = small_configs()
    result = run_benchmark(name="test", num_requests=4, prompt_len=16, max_tokens=5,
                           model_config=model, engine_config=engine, warmup=0)
    assert result.total_output_tokens == 4 * 5
    assert result.output_throughput > 0
    assert result.ttft.p50_ms <= result.ttft.p99_ms
    assert result.e2e.count == 4


def test_benchmark_result_is_json_serialisable():
    import json
    model, engine = small_configs()
    result = run_benchmark(name="json", num_requests=2, prompt_len=8, max_tokens=3,
                           model_config=model, engine_config=engine, warmup=0)
    json.dumps(result.as_dict(), default=str)


# ----------------------------------------------------------------- regression
def test_first_run_has_no_baseline(tmp_path):
    tracker = RegressionTracker(tmp_path / "h.jsonl")
    verdict = tracker.check_and_record("bench", {"output_throughput_tok_s": 100.0})
    assert verdict.ok and "no baseline" in verdict.note


def test_regression_is_detected(tmp_path):
    tracker = RegressionTracker(tmp_path / "h.jsonl", tolerance_pct=10.0)
    for _ in range(3):
        tracker.record("bench", {"output_throughput_tok_s": 100.0, "e2e_p99_ms": 50.0})
    verdict = tracker.check("bench", {"output_throughput_tok_s": 50.0, "e2e_p99_ms": 50.0})
    assert not verdict.ok
    assert verdict.regressions()[0].metric == "output_throughput_tok_s"


def test_latency_increase_is_a_regression(tmp_path):
    tracker = RegressionTracker(tmp_path / "h.jsonl", tolerance_pct=10.0)
    for _ in range(3):
        tracker.record("bench", {"e2e_p99_ms": 50.0})
    assert not tracker.check("bench", {"e2e_p99_ms": 100.0}).ok


def test_noise_within_tolerance_passes(tmp_path):
    tracker = RegressionTracker(tmp_path / "h.jsonl", tolerance_pct=20.0)
    for _ in range(3):
        tracker.record("bench", {"output_throughput_tok_s": 100.0})
    assert tracker.check("bench", {"output_throughput_tok_s": 92.0}).ok


def test_baseline_is_a_median_so_one_outlier_does_not_move_it(tmp_path):
    tracker = RegressionTracker(tmp_path / "h.jsonl", tolerance_pct=10.0, window=5)
    for value in (100.0, 100.0, 1000.0, 100.0, 100.0):
        tracker.record("bench", {"output_throughput_tok_s": value})
    assert tracker.baseline("bench")["output_throughput_tok_s"] == 100.0


def test_corrupt_history_line_is_skipped(tmp_path):
    path = tmp_path / "h.jsonl"
    path.write_text('{"name": "b", "values": {"e2e_p99_ms": 1.0}}\n{broken\n')
    assert len(RegressionTracker(path).load()) == 1


def test_flatten_result_extracts_tracked_metrics():
    model, engine = small_configs()
    result = run_benchmark(name="f", num_requests=2, prompt_len=8, max_tokens=3,
                           model_config=model, engine_config=engine, warmup=0)
    flat = flatten_result(result)
    assert set(flat) == {"output_throughput_tok_s", "ttft_p50_ms", "ttft_p99_ms",
                         "e2e_p50_ms", "e2e_p99_ms"}
