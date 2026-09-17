# Requirement → Implementation Mapping

Every line of the role description, mapped to the code that addresses it, with an
honest status. Three statuses are used:

- **Implemented** — real, executing, tested code.
- **Modelled** — the algorithm/analysis is real and tested, but the physical
  substrate is simulated (no NVIDIA GPU, no InfiniBand fabric, no 70B weights).
- **Reference** — real code that cannot execute on this machine (Triton kernels).

Nothing in the role description is silently missing. Where something is Modelled
or Reference, the reason is physical, not effort.

---

## "This role could be a great match for you if you…"

| Requirement | Where | Status |
|---|---|---|
| Understand modern generative AI architectures and optimise them for inference | `model/numpy_model.py` (RMSNorm, RoPE, GQA, SwiGLU), `model/moe.py` (top-k MoE) | Implemented |
| Familiar with internals of vLLM and SGLang | `kv/paged_cache.py` + `kv/block_allocator.py` (PagedAttention), `kv/radix_cache.py` (RadixAttention), `engine/scheduler.py` (continuous batching, chunked prefill, preemption) | Implemented |
| Value clear communication, team process, supportive teamwork | `docs/adr/`, `CONTRIBUTING.md`, `docs/POSTMORTEM_TEMPLATE.md`, `docs/CULTURE.md` | Implemented |
| Results-oriented, bias to action, own problems end-to-end | Repo spans kernels → engine → server → observability → clients in one owned stack | Implemented |
| Modern Python and tooling (uv, pybind/nanobind, FastAPI) | `pyproject.toml`, `kernels/csrc/pybind_module.cpp`, `kernels/csrc/nanobind_module.cpp`, `server/api.py` | Implemented |
| PyTorch | `kernels/torch_kernels.py` (mirror kernels + SDPA wrapper) | Implemented (optional dep) |
| Nvidia GPU kernel programming and optimisation | `kernels/triton_kernels.py` (fused RMSNorm, SwiGLU) | Reference — needs a CUDA device |
| Infiniband and NVLink | `distributed/ring_allreduce.py` over real TCP sockets; bandwidth cost model in `fleet/hardware.py` | Modelled — real protocol, loopback transport |

## Responsibilities

| Responsibility | Where | Status |
|---|---|---|
| Implement frontier research ideas alongside researchers | Speculative decoding, RadixAttention, chunked prefill, MoE — each implemented from the paper's mechanism | Implemented |
| Introduce systems/tools/techniques improving inference performance | `engine/scheduler.py`, `engine/speculative.py`, `quant/int8.py`, `kernels/numpy_kernels.py::flash_attention` | Implemented |
| Build tools to debug performance bottlenecks | `bench/harness.py` (TTFT/TPOT/p99), `fleet/roofline.py` (bound analysis), `cli.py kernels` (backend comparison) | Implemented |
| Build tools to debug numeric instabilities | `debug/instability.py` + `inject_fault()` for verification | Implemented |
| Build tools to debug distributed systems issues | `distributed/health.py` (dead peers, stragglers, quorum) | Implemented |
| Build tools/processes for team productivity | `scripts/verify.py`, `scripts/run_stack.py`, `bench/regression.py`, CI workflow | Implemented |
| Deliver quickly and iteratively | One-command verify, one-command run, no container prerequisites | Implemented |

## Required qualifications

| Requirement | Where | Status |
|---|---|---|
| Coding in C, C++, C#, Java, JavaScript, Python | `clients/c/probe.c`, `kernels/csrc/*.cpp`, `clients/csharp/Program.cs`, `clients/java/LoadTestClient.java`, `clients/node/dashboard.js`, `src/infer_lab/**` | Implemented |

## Preferred qualifications

| Requirement | Where | Status |
|---|---|---|
| Experience with generative AI | Full decoder + MoE + sampling stack | Implemented |
| Distributed computing | `distributed/tensor_parallel.py`, `pipeline_parallel.py`, `ring_allreduce.py` | Implemented (modelled substrate) |
| Python ecosystem (uv, pybind/nanobind, FastAPI) | `pyproject.toml` (uv-compatible), both binding modules, FastAPI server | Implemented |
| Large scale production inference | Continuous batching, admission control, preemption, graceful shutdown, Prometheus metrics, alert rules, load testing | Implemented at engine scale, not fleet scale |
| GPU kernel programming | `kernels/triton_kernels.py` | Reference |
| Benchmarking, profiling, optimising PyTorch generative AI models | `bench/harness.py`, `bench/regression.py`, `kernels/torch_kernels.py` | Implemented |
| Open source inference frameworks (vLLM, SGLang) | PagedAttention, RadixAttention, continuous batching, chunked prefill — reimplemented, not wrapped | Implemented |
| JAX scaling book material | `fleet/roofline.py` (arithmetic intensity, ridge point, batch-size crossover, KV capacity) | Implemented |

---

## What is simulated, and why

| Simulated | Reason |
|---|---|
| GPU kernel execution (Triton/CUDA) | No NVIDIA hardware available. Kernels are written to be correct, not to be decorative. |
| InfiniBand / NVLink transport | Physical fabric cannot be reproduced. The ring all-reduce **protocol** is real — real sockets, real framing, real partial-read and dead-peer handling — over loopback. |
| Billion-parameter scale | Would require paid compute. The architecture is identical; only the parameter count differs, and `fleet/roofline.py` computes the real-scale numbers analytically. |
| Heterogeneous fleet hardware | Multiple accelerator generations are modelled as roofline profiles in `fleet/hardware.py`; the placement policy operating on them is real code. |

## Deliberately excluded

| Excluded | Reason |
|---|---|
| Docker / Kubernetes | Docker is prohibited on the target machine, and K8s requires a container runtime. More importantly, K8s solves infrastructure orchestration, not inference-engine internals; `fleet/multi_pool.py` addresses the part of that problem that is actually about inference. |
| C#/Java/C as *engine* languages | Real inference engines are Python + C++ + CUDA. The language list in the role description is a hiring filter, not a stack proposal. Writing the engine four times would dilute depth instead of demonstrating it. |
