"""infer-lab command line interface.

    infer-lab serve          start the FastAPI server
    infer-lab bench          run the latency/throughput benchmark
    infer-lab roofline       roofline + batch sweep across hardware profiles
    infer-lab kernels        report available kernel backends and compare them
    infer-lab fleet          heterogeneous multi-pool placement simulation
    infer-lab debug          numeric instability demonstration
    infer-lab distributed    tensor/pipeline parallel + ring all-reduce demo
    infer-lab spec           speculative decoding demonstration
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from infer_lab import __version__
from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.utils.logging_conf import setup_logging


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from infer_lab.server.api import create_app

    app = create_app(
        ModelConfig(),
        EngineConfig(max_num_seqs=args.max_num_seqs,
                     max_num_batched_tokens=args.max_num_batched_tokens,
                     num_gpu_blocks=args.num_gpu_blocks),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level,
                access_log=False)
    return 0


# --------------------------------------------------------------------------- bench
def cmd_bench(args: argparse.Namespace) -> int:
    from infer_lab.bench.harness import run_benchmark
    from infer_lab.bench.regression import RegressionTracker, flatten_result

    result = run_benchmark(
        name=args.name,
        num_requests=args.requests,
        prompt_len=args.prompt_len,
        max_tokens=args.max_tokens,
        shared_prefix=args.shared_prefix,
        engine_config=EngineConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            num_gpu_blocks=args.num_gpu_blocks,
            enable_prefix_caching=not args.no_prefix_cache,
            enable_chunked_prefill=not args.no_chunked_prefill,
        ),
    )
    print(result.summary_line())
    if args.json:
        _print(result.as_dict())

    if args.track:
        tracker = RegressionTracker(args.history, tolerance_pct=args.tolerance)
        verdict = tracker.check_and_record(args.name, flatten_result(result),
                                           metadata=result.environment)
        print()
        print(verdict.report())
        return 0 if verdict.ok else 1
    return 0


# ------------------------------------------------------------------------ roofline
def cmd_roofline(args: argparse.Namespace) -> int:
    from infer_lab.fleet.hardware import PROFILES
    from infer_lab.fleet.roofline import (
        analyze_decode, analyze_prefill, batch_sweep, crossover_batch_size,
        kv_cache_capacity,
    )

    config = ModelConfig(
        hidden_size=args.hidden, num_layers=args.layers,
        intermediate_size=args.hidden * 4, num_heads=args.heads,
        num_kv_heads=max(1, args.heads // 4), vocab_size=args.vocab,
    )
    print(f"model: {config.param_count() / 1e6:.1f}M params, "
          f"{config.num_layers} layers, GQA {config.num_heads}/{config.num_kv_heads}\n")

    for name, hw in PROFILES.items():
        print(f"--- {name} (ridge point {hw.ridge_point:.0f} FLOP/byte)")
        decode = analyze_decode(config, hw, args.batch, args.seq_len)
        prefill = analyze_prefill(config, hw, 1, args.seq_len)
        print(f"  prefill S={args.seq_len:<6} {prefill.bound:<7} "
              f"intensity {prefill.arithmetic_intensity:9.1f}  "
              f"util {prefill.utilization * 100:5.1f}%")
        print(f"  decode  B={args.batch:<6} {decode.bound:<7} "
              f"intensity {decode.arithmetic_intensity:9.1f}  "
              f"util {decode.utilization * 100:5.1f}%  "
              f"{decode.tokens_per_s:,.0f} tok/s")
        crossover = crossover_batch_size(config, hw, args.seq_len)
        print(f"  memory->compute crossover at batch size: {crossover}")
        print(f"  kv capacity: {kv_cache_capacity(config, hw)}")
        print()

    if args.sweep:
        hw = PROFILES[args.hardware]
        print(f"batch sweep on {hw.name} (seq_len={args.seq_len}):")
        print(f"  {'batch':>6} {'bound':>8} {'intensity':>11} {'util%':>7} {'tok/s':>12}")
        for a in batch_sweep(config, hw, args.seq_len):
            print(f"  {a.batch_size:>6} {a.bound:>8} {a.arithmetic_intensity:>11.1f} "
                  f"{a.utilization * 100:>7.2f} {a.tokens_per_s:>12,.0f}")
    return 0


# ------------------------------------------------------------------------- kernels
def cmd_kernels(args: argparse.Namespace) -> int:
    from infer_lab.bench.harness import run_kernel_benchmark
    from infer_lab.kernels.numpy_kernels import rms_norm as np_rms_norm
    from infer_lab.kernels.registry import get_registry

    registry = get_registry()
    report = registry.report()
    print("kernel backends:")
    for name, info in report.items():
        mark = "available" if info["available"] else "unavailable"
        print(f"  {name:<10} {mark:<12} {info['reason']}")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(args.rows, args.cols)).astype(np.float32)
    w = rng.normal(size=(args.cols,)).astype(np.float32)
    reference = np_rms_norm(x, w)

    print(f"\nrms_norm benchmark ({args.rows}x{args.cols}, {args.iterations} iters):")
    print(f"  {'backend':<10} {'p50 ms':>9} {'p99 ms':>9} {'max abs err':>13}")
    for backend in registry.backends_for("rms_norm"):
        fn = registry.fn("rms_norm", backend)
        try:
            out = np.asarray(fn(x, w))
            err = float(np.max(np.abs(out - reference)))
            stats = run_kernel_benchmark(lambda f=fn: f(x, w), iterations=args.iterations)
            print(f"  {backend:<10} {stats.p50_ms:>9.4f} {stats.p99_ms:>9.4f} {err:>13.2e}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {backend:<10} failed: {type(exc).__name__}: {exc}")

    print("\nbinding call overhead (empty function call):")
    from infer_lab.kernels import native
    for binding in ("ctypes", "pybind11", "nanobind"):
        try:
            mod = native.load(binding)
            stats = run_kernel_benchmark(mod.noop, iterations=args.iterations * 5)
            print(f"  {binding:<10} p50 {stats.p50_ms * 1000:>8.3f} us")
        except Exception as exc:  # noqa: BLE001
            print(f"  {binding:<10} unavailable ({type(exc).__name__})")
    return 0


# --------------------------------------------------------------------------- fleet
def cmd_fleet(args: argparse.Namespace) -> int:
    from infer_lab.fleet.multi_pool import MultiPoolScheduler, WorkloadRequest, default_fleet

    rng = np.random.default_rng(args.seed)
    scheduler = MultiPoolScheduler(ModelConfig(hidden_size=4096, num_layers=32,
                                               intermediate_size=11008, num_heads=32,
                                               num_kv_heads=8, vocab_size=32000),
                                   default_fleet(args.devices))
    for i in range(args.requests):
        request = WorkloadRequest(
            request_id=f"w-{i:04d}",
            prompt_len=int(rng.integers(64, 4096)),
            max_tokens=int(rng.integers(16, 512)),
            slo_ms=float(rng.choice([500.0, 2000.0, 10000.0])),
        )
        decision = scheduler.place(request)
        if args.verbose:
            print(f"  {request.request_id} prompt={request.prompt_len:<5} "
                  f"slo={request.slo_ms:<8.0f} -> {decision.pool or 'UNPLACED':<12} "
                  f"{decision.predicted_latency_ms:>10.1f} ms  "
                  f"{'ok' if decision.meets_slo else 'SLO MISS'}")
    _print(scheduler.snapshot())
    return 0


# --------------------------------------------------------------------------- debug
def cmd_debug(args: argparse.Namespace) -> int:
    from infer_lab.debug.instability import InstabilityDetector, inject_fault
    from infer_lab.kv.paged_cache import PagedKVCache
    from infer_lab.model.numpy_model import NumpyTransformer
    from infer_lab.model.weights import ModelWeights

    config = ModelConfig()
    engine_config = EngineConfig()
    tokens = np.arange(1, 17)

    for kind in ("healthy", args.fault):
        weights = ModelWeights.random(config, seed=0)
        if kind != "healthy":
            inject_fault(weights, layer_index=args.layer, kind=kind)
        detector = InstabilityDetector()
        model = NumpyTransformer(weights)
        cache = PagedKVCache(config, engine_config)
        seq = cache.new_sequence()

        # Instrument: observe the embedding and every layer's weights-driven output.
        detector.observe("embed", weights.embed[tokens], -1)
        for i, layer in enumerate(weights.layers):
            detector.observe(f"layer{i}.wq", layer.wq, i)
        logits = model.forward_prefill(tokens, seq)
        detector.observe("logits", logits, -1)

        print(f"\n=== {kind} ===")
        _print(detector.summary())
    return 0


# --------------------------------------------------------------------- distributed
def cmd_distributed(args: argparse.Namespace) -> int:
    from infer_lab.distributed import (
        HealthMonitor, PipelineSchedule, PipelineStage, run_ring_allreduce,
        split_layers, verify_equivalence,
    )

    rng = np.random.default_rng(0)
    print(f"--- tensor parallel (world_size={args.world_size})")
    x = rng.normal(size=(args.tokens, 256)).astype(np.float32)
    wc = rng.normal(size=(256, 512)).astype(np.float32)
    wr = rng.normal(size=(512, 256)).astype(np.float32)
    _print(verify_equivalence(x, wc, wr, args.world_size))

    print(f"\n--- ring all-reduce over TCP (world_size={args.world_size})")
    tensors = [np.full(args.elements, r + 1, dtype=np.float32) for r in range(args.world_size)]
    results, stats = run_ring_allreduce(tensors)
    expected = sum(range(1, args.world_size + 1))
    _print({
        "correct": bool(all(np.allclose(r, expected) for r in results)),
        "expected_value": expected,
        "bytes_sent_per_rank": stats[0].bytes_sent,
        "theoretical_bytes": int(2 * (args.world_size - 1) / args.world_size
                                 * tensors[0].nbytes),
        "steps_per_rank": stats[0].steps,
    })

    print(f"\n--- pipeline parallel (stages={args.world_size})")
    stages = [PipelineStage(i, ls, lambda a: a)
              for i, ls in enumerate(split_layers(args.layers, args.world_size))]
    schedule = PipelineSchedule(stages, num_micro_batches=args.micro_batches)
    micro = [np.ones((2, 256), dtype=np.float32) for _ in range(args.micro_batches)]
    schedule.run(micro)
    _print(schedule.report(micro[0]))

    print("\n--- failure detection")
    monitor = HealthMonitor(args.world_size)
    for r in range(args.world_size - 1):
        monitor.heartbeat(r, step=100)
    monitor.heartbeat(args.world_size - 1, step=42)
    _print(monitor.snapshot())
    return 0


# ---------------------------------------------------------------------------- spec
def cmd_spec(args: argparse.Namespace) -> int:
    from infer_lab.engine.request import SamplingParams
    from infer_lab.engine.speculative import SpeculativeDecoder
    from infer_lab.kv.paged_cache import PagedKVCache
    from infer_lab.model.numpy_model import NumpyTransformer
    from infer_lab.model.weights import ModelWeights

    target_config = ModelConfig()
    draft_config = ModelConfig(num_layers=1)     # the cheap proposer
    engine_config = EngineConfig()

    target_weights = ModelWeights.random(target_config, seed=0)
    draft_weights = ModelWeights.random(draft_config, seed=0)
    # Make the draft a genuine approximation of the target: share the layers it has.
    for i in range(draft_config.num_layers):
        draft_weights.layers[i] = target_weights.layers[i]
    draft_weights.embed = target_weights.embed
    draft_weights.lm_head = target_weights.lm_head

    decoder = SpeculativeDecoder(
        target=NumpyTransformer(target_weights),
        draft=NumpyTransformer(draft_weights),
        target_cache=PagedKVCache(target_config, engine_config),
        draft_cache=PagedKVCache(draft_config, engine_config),
        num_draft_tokens=args.draft_tokens,
    )
    prompt = list(range(1, args.prompt_len + 1))
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    tokens = decoder.generate(prompt, params)
    print(f"generated {len(tokens)} tokens with k={args.draft_tokens}")
    _print(decoder.stats.as_dict())
    return 0


# ---------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="infer-lab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"infer-lab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="start the HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log-level", default="info")
    s.add_argument("--max-num-seqs", type=int, default=32)
    s.add_argument("--max-num-batched-tokens", type=int, default=512)
    s.add_argument("--num-gpu-blocks", type=int, default=512)
    s.set_defaults(func=cmd_serve)

    b = sub.add_parser("bench", help="latency/throughput benchmark")
    b.add_argument("--name", default="default")
    b.add_argument("--requests", type=int, default=16)
    b.add_argument("--prompt-len", type=int, default=64)
    b.add_argument("--max-tokens", type=int, default=32)
    b.add_argument("--shared-prefix", type=int, default=0)
    b.add_argument("--max-num-seqs", type=int, default=32)
    b.add_argument("--max-num-batched-tokens", type=int, default=512)
    b.add_argument("--num-gpu-blocks", type=int, default=512)
    b.add_argument("--no-prefix-cache", action="store_true")
    b.add_argument("--no-chunked-prefill", action="store_true")
    b.add_argument("--json", action="store_true")
    b.add_argument("--track", action="store_true", help="compare against the baseline")
    b.add_argument("--history", default="artifacts/bench_history.jsonl")
    b.add_argument("--tolerance", type=float, default=15.0)
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("roofline", help="roofline analysis")
    r.add_argument("--hidden", type=int, default=4096)
    r.add_argument("--layers", type=int, default=32)
    r.add_argument("--heads", type=int, default=32)
    r.add_argument("--vocab", type=int, default=32000)
    r.add_argument("--seq-len", type=int, default=2048)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--hardware", default="accel-b")
    r.add_argument("--sweep", action="store_true")
    r.set_defaults(func=cmd_roofline)

    k = sub.add_parser("kernels", help="kernel backend report and benchmark")
    k.add_argument("--rows", type=int, default=512)
    k.add_argument("--cols", type=int, default=512)
    k.add_argument("--iterations", type=int, default=100)
    k.set_defaults(func=cmd_kernels)

    f = sub.add_parser("fleet", help="heterogeneous placement simulation")
    f.add_argument("--requests", type=int, default=50)
    f.add_argument("--devices", type=int, default=8)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--verbose", action="store_true")
    f.set_defaults(func=cmd_fleet)

    d = sub.add_parser("debug", help="numeric instability demonstration")
    d.add_argument("--fault", default="overflow",
                   choices=["overflow", "nan", "inf", "dead"])
    d.add_argument("--layer", type=int, default=0)
    d.set_defaults(func=cmd_debug)

    dist = sub.add_parser("distributed", help="tensor/pipeline/collective demo")
    dist.add_argument("--world-size", type=int, default=4)
    dist.add_argument("--elements", type=int, default=4096)
    dist.add_argument("--tokens", type=int, default=16)
    dist.add_argument("--layers", type=int, default=32)
    dist.add_argument("--micro-batches", type=int, default=8)
    dist.set_defaults(func=cmd_distributed)

    sp = sub.add_parser("spec", help="speculative decoding demonstration")
    sp.add_argument("--draft-tokens", type=int, default=4)
    sp.add_argument("--prompt-len", type=int, default=16)
    sp.add_argument("--max-tokens", type=int, default=32)
    sp.set_defaults(func=cmd_spec)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(plain=True)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
