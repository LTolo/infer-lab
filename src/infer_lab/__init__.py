"""infer-lab: a vertically integrated LLM inference engine laboratory.

Layers (bottom-up):
    kernels/      fused compute primitives (NumPy, C++, ctypes/pybind11/nanobind, Triton)
    model/        Llama-style decoder + MoE variant (NumPy reference, optional PyTorch mirror)
    quant/        INT8 per-channel weight quantization
    kv/           PagedAttention block allocator + RadixAttention prefix cache
    engine/       continuous batching scheduler, chunked prefill, speculative decoding
    distributed/  tensor parallelism, pipeline parallelism, ring all-reduce over TCP
    fleet/        heterogeneous multi-pool placement + roofline model
    debug/        numeric instability detection, distributed failure detection
    bench/        latency/throughput harness + performance regression tracking
    server/       FastAPI serving layer with Prometheus metrics
"""

__version__ = "0.1.0"

from infer_lab.config import ModelConfig, EngineConfig  # noqa: E402,F401

__all__ = ["ModelConfig", "EngineConfig", "__version__"]
