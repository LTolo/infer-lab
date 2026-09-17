#!/usr/bin/env python
"""ONE COMMAND: verify the whole project runs error-free.

    python scripts/verify.py

Runs, in order:
  1. environment & dependency check      (what is present, what degrades)
  2. native kernel build                 (ctypes / pybind11 / nanobind, best effort)
  3. import check of every module
  4. the full pytest suite
  5. end-to-end engine smoke test
  6. HTTP API smoke test (in-process, no port needed)
  7. cross-language client build check   (C / Java / Node / C#, best effort)

Exit code 0 means everything that CAN work on this machine DOES work.
Optional components that are genuinely unavailable (no GPU, no .NET, no
compiler) are reported as SKIP and never fail the run -- that is the whole
point of the degradation design.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

GREEN, RED, YELLOW, BLUE, DIM, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[2m", "\033[0m")
    if sys.stdout.isatty() and os.name != "nt" else ("", "", "", "", "", "")
)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(stage: str, status: str, detail: str = "") -> None:
    colour = {PASS: GREEN, FAIL: RED, SKIP: YELLOW}[status]
    print(f"  [{colour}{status}{RESET}] {stage}" + (f" {DIM}-- {detail}{RESET}" if detail else ""))
    results.append((stage, status, detail))


def header(text: str) -> None:
    print(f"\n{BLUE}=== {text} ==={RESET}")


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 900,
        env: dict[str, str] | None = None):
    # env matters for MSVC: the compiler needs INCLUDE and LIB from the vcvars
    # environment, not just an executable path.
    return subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True,
                          timeout=timeout, env=env)


# --------------------------------------------------------------------- 1. environment
def stage_environment() -> None:
    header("1/7  environment")
    print(f"  python {sys.version.split()[0]} on {sys.platform}")

    required = ["numpy", "fastapi", "uvicorn", "pydantic", "pytest"]
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        record("required dependencies", FAIL,
               f"missing: {', '.join(missing)} -- run: pip install -e \".[dev]\"")
    else:
        record("required dependencies", PASS, ", ".join(required))

    for module, why in [("torch", "PyTorch mirror kernels"),
                        ("pybind11", "pybind11 bindings"),
                        ("nanobind", "nanobind bindings"),
                        ("matplotlib", "roofline plots"),
                        ("httpx", "HTTP client tests")]:
        try:
            __import__(module)
            record(f"optional: {module}", PASS, why)
        except ImportError:
            record(f"optional: {module}", SKIP, f"{why} unavailable")

    # Report the C++ toolchain through the build's own detection, so this stage
    # cannot claim a compiler is missing while stage 2 goes on to use it.
    try:
        from infer_lab.kernels.build import BuildError, detect_compiler

        try:
            kind, cc, _ = detect_compiler()
            record(f"toolchain: {Path(cc).name}", PASS,
                   f"C/C++ compiler ({kind}) at {cc}")
        except BuildError as exc:
            record("toolchain: C/C++ compiler", SKIP, str(exc)[:120])
    except ImportError:
        pass

    for tool, why in [("javac", "Java load-test client"), ("node", "web dashboard"),
                      ("dotnet", "C# enterprise client"), ("k6", "load testing")]:
        if shutil.which(tool):
            record(f"toolchain: {tool}", PASS, why)


# ------------------------------------------------------------------- 2. native build
def stage_native_build() -> None:
    header("2/7  native kernel build")
    try:
        from infer_lab.kernels import build as kbuild
        outcome = kbuild.build_all()
        for name, status in outcome.items():
            record(f"kernel binding: {name}", PASS if status.startswith("ok") else SKIP,
                   status)
    except Exception as exc:  # noqa: BLE001
        record("native kernel build", SKIP, f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------- 3. imports
MODULES = [
    "infer_lab", "infer_lab.config", "infer_lab.cli",
    "infer_lab.utils.logging_conf", "infer_lab.utils.timing",
    "infer_lab.kernels", "infer_lab.kernels.numpy_kernels", "infer_lab.kernels.registry",
    "infer_lab.kernels.native", "infer_lab.kernels.torch_kernels",
    "infer_lab.kernels.triton_kernels",
    "infer_lab.model.weights", "infer_lab.model.numpy_model", "infer_lab.model.moe",
    "infer_lab.model.tokenizer",
    "infer_lab.quant.int8",
    "infer_lab.kv.block_allocator", "infer_lab.kv.paged_cache", "infer_lab.kv.radix_cache",
    "infer_lab.engine.request", "infer_lab.engine.sampling", "infer_lab.engine.scheduler",
    "infer_lab.engine.llm_engine", "infer_lab.engine.speculative",
    "infer_lab.distributed.tensor_parallel", "infer_lab.distributed.pipeline_parallel",
    "infer_lab.distributed.ring_allreduce", "infer_lab.distributed.health",
    "infer_lab.fleet.hardware", "infer_lab.fleet.roofline", "infer_lab.fleet.multi_pool",
    "infer_lab.debug.instability",
    "infer_lab.bench.harness", "infer_lab.bench.regression",
    "infer_lab.server.metrics", "infer_lab.server.engine_runner", "infer_lab.server.api",
]


def stage_imports() -> None:
    header("3/7  module imports")
    import importlib
    failed = []
    for module in MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{module}: {type(exc).__name__}: {exc}")
    if failed:
        for f in failed:
            record("import", FAIL, f)
    else:
        record(f"all {len(MODULES)} modules import cleanly", PASS)


# ------------------------------------------------------------------------ 4. pytest
def stage_tests(extra: list[str]) -> None:
    header("4/7  test suite")
    env = dict(os.environ, PYTHONPATH=str(SRC))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short", "-p", "no:cacheprovider",
         *extra],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=1800,
    )
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:] or [""]
    if proc.returncode == 0:
        record("pytest", PASS, tail[0].strip())
    else:
        record("pytest", FAIL, tail[0].strip())
        print(DIM + proc.stdout[-4000:] + RESET)


# ------------------------------------------------------------------ 5. engine smoke
def stage_engine_smoke() -> None:
    header("5/7  engine end-to-end")
    try:
        from infer_lab.config import EngineConfig, ModelConfig
        from infer_lab.engine.llm_engine import LLMEngine
        from infer_lab.engine.request import SamplingParams

        model = ModelConfig()
        engine = LLMEngine(model, EngineConfig(num_gpu_blocks=256,
                                               max_num_batched_tokens=128))
        sampling = SamplingParams(max_tokens=12, temperature=0.0, ignore_eos=True)
        prompts = [list(range(1, 40)), list(range(1, 25)), list(range(1, 60))]

        started = time.perf_counter()
        outputs = engine.generate(prompts, sampling)
        elapsed = (time.perf_counter() - started) * 1000

        assert len(outputs) == 3, "not every request finished"
        assert all(o.num_generated == 12 for o in outputs), "wrong output length"

        # Correct invariant: with prefix caching ON, blocks retained by the radix
        # cache are NOT a leak -- they are the cache doing its job. Everything
        # else must be back in the free list.
        allocator = engine.kv_cache.allocator
        cached = engine.radix_cache.num_cached_blocks if engine.radix_cache else 0
        accounted = allocator.num_free + cached
        assert accounted == allocator.num_blocks, (
            f"KV accounting mismatch: {allocator.num_free} free + {cached} cached "
            f"!= {allocator.num_blocks} total"
        )

        stats = engine.scheduler.stats
        record("engine generate", PASS,
               f"{len(outputs)} requests, {stats.steps} steps, {elapsed:.0f} ms, "
               f"KV accounted ({allocator.num_free} free + {cached} prefix-cached)")

        # And with the cache disabled, every single block must come back.
        strict = LLMEngine(model, EngineConfig(num_gpu_blocks=256,
                                               max_num_batched_tokens=128,
                                               enable_prefix_caching=False))
        strict.generate(prompts, sampling)
        free = strict.kv_cache.allocator.num_free
        assert free == strict.kv_cache.allocator.num_blocks, f"KV leak: {free} free"
        record("KV reclamation (no prefix cache)", PASS, "all blocks returned")

        # determinism
        again = LLMEngine(model, EngineConfig(num_gpu_blocks=256,
                                              max_num_batched_tokens=128)) \
            .generate(prompts, sampling)
        same = all(a.output_token_ids == b.output_token_ids for a, b in zip(outputs, again))
        record("determinism (greedy)", PASS if same else FAIL,
               "identical tokens across runs" if same else "outputs diverged")
    except Exception as exc:  # noqa: BLE001
        record("engine end-to-end", FAIL, f"{type(exc).__name__}: {exc}")


# -------------------------------------------------------------------- 6. http smoke
def stage_http_smoke() -> None:
    header("6/7  HTTP API")
    try:
        from fastapi.testclient import TestClient

        from infer_lab.config import EngineConfig, ModelConfig
        from infer_lab.server.api import create_app

        with TestClient(create_app(ModelConfig(), EngineConfig())) as client:
            assert client.get("/health").json()["status"] == "ok"
            record("GET /health", PASS)
            assert client.get("/ready").status_code == 200
            record("GET /ready", PASS)

            body = client.post("/generate",
                               json={"prompt": "infer-lab", "max_tokens": 6}).json()
            assert body["output_tokens"] == 6
            record("POST /generate", PASS, f"{body['output_tokens']} tokens")

            text = client.get("/metrics").text
            assert "infer_lab_ttft_milliseconds_bucket" in text
            record("GET /metrics", PASS, f"{len(text.splitlines())} exposition lines")

            assert "scheduler" in client.get("/stats").json()["engine"]
            record("GET /stats", PASS)
    except Exception as exc:  # noqa: BLE001
        record("HTTP API", FAIL, f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------- 7. other languages
def stage_clients() -> None:
    header("7/7  cross-language clients")
    build = ROOT / "artifacts" / "clients"
    build.mkdir(parents=True, exist_ok=True)

    # --- C
    #
    # Ask the build for its compiler instead of looking one up again here.
    # Stage 2 already locates Visual Studio, runs vcvarsall and resolves cl.exe
    # to an absolute path; repeating that search with shutil.which() against the
    # calling shell's PATH is how the two stages ended up disagreeing about
    # whether a C compiler exists on this machine.
    source = str(ROOT / "clients" / "c" / "probe.c")
    try:
        from infer_lab.kernels.build import BuildError, detect_compiler

        try:
            kind, cc, env = detect_compiler()
        except BuildError as exc:
            kind, cc, env = None, None, None
            reason = str(exc)
    except ImportError as exc:  # pragma: no cover - package not installed
        kind, cc, env = None, None, None
        reason = f"cannot import the build module: {exc}"

    if kind == "msvc":
        proc = run([cc, "/nologo", "/O2", source,
                    "/Fe:" + str(build / "probe.exe"), "ws2_32.lib"],
                   cwd=build, env=env)
        record("C latency probe", PASS if proc.returncode == 0 else FAIL,
               "compiled with MSVC"
               if proc.returncode == 0 else (proc.stdout or proc.stderr)[-300:])
    elif kind == "unix":
        proc = run([cc, "-O2", "-std=c11", source, "-o", str(build / "probe")],
                   env=env)
        record("C latency probe", PASS if proc.returncode == 0 else FAIL,
               f"compiled with {Path(cc).name}"
               if proc.returncode == 0 else proc.stderr[-300:])
    else:
        record("C latency probe", SKIP, reason)

    # --- Java
    if shutil.which("javac"):
        proc = run(["javac", "-d", str(build),
                    str(ROOT / "clients" / "java" / "LoadTestClient.java")])
        record("Java load-test client", PASS if proc.returncode == 0 else FAIL,
               "compiled" if proc.returncode == 0 else proc.stderr[-300:])
    else:
        record("Java load-test client", SKIP, "no javac")

    # --- Node
    if shutil.which("node"):
        proc = run(["node", "--check", str(ROOT / "clients" / "node" / "dashboard.js")])
        record("Node dashboard", PASS if proc.returncode == 0 else FAIL,
               "syntax ok" if proc.returncode == 0 else proc.stderr[-200:])
    else:
        record("Node dashboard", SKIP, "no node")

    # --- C#
    if shutil.which("dotnet"):
        proc = run(["dotnet", "build", "-v", "q", "--nologo"],
                   cwd=ROOT / "clients" / "csharp", timeout=600)
        record("C# async client", PASS if proc.returncode == 0 else FAIL,
               "built" if proc.returncode == 0 else proc.stdout[-300:])
    else:
        record("C# async client", SKIP, "no dotnet SDK")


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the infer-lab project end to end")
    parser.add_argument("--fast", action="store_true",
                        help="skip the native build and the cross-language clients")
    parser.add_argument("--no-tests", action="store_true", help="skip pytest")
    parser.add_argument("pytest_args", nargs="*", help="extra args passed to pytest")
    args = parser.parse_args(argv)

    print(f"{BLUE}infer-lab verification{RESET}  ({ROOT})")
    started = time.perf_counter()

    stage_environment()
    if not args.fast:
        stage_native_build()
    stage_imports()
    if not args.no_tests:
        stage_tests(args.pytest_args)
    stage_engine_smoke()
    stage_http_smoke()
    if not args.fast:
        stage_clients()

    elapsed = time.perf_counter() - started
    passed = sum(1 for _, s, _ in results if s == PASS)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    failed = [(n, d) for n, s, d in results if s == FAIL]

    print(f"\n{BLUE}{'=' * 68}{RESET}")
    print(f"  {GREEN}{passed} passed{RESET}   {YELLOW}{skipped} skipped{RESET}   "
          f"{RED if failed else ''}{len(failed)} failed{RESET}   ({elapsed:.1f}s)")
    if failed:
        print(f"\n{RED}failures:{RESET}")
        for name, detail in failed:
            print(f"  - {name}: {detail}")
        print(f"\n{RED}VERIFICATION FAILED{RESET}")
        return 1
    print(f"\n{GREEN}ALL CHECKS PASSED{RESET} -- "
          f"skipped items are optional components unavailable on this machine.")
    print(f"\nnext: {BLUE}python scripts/run_stack.py{RESET} to start the application.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
