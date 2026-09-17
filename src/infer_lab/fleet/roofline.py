"""Roofline analysis for transformer inference.

The model is the one from the JAX scaling book / Williams' roofline:

    t_compute = FLOPs / peak_FLOPs
    t_memory  = bytes_moved / bandwidth
    t         = max(t_compute, t_memory)

and the regime is decided by the **arithmetic intensity** ``FLOPs / bytes``
compared against the device ridge point.

The two phases sit on opposite sides of that line, which is why they need
different optimisations and why chunked prefill (mixing them in one step) is
such a good idea:

* **prefill** processes ``S`` tokens against one set of weights -> intensity
  scales with S -> compute-bound -> wants better kernels and more FLOPs.
* **decode** processes 1 token per sequence -> the whole weight matrix is read
  to produce ``B`` tokens -> intensity ~ B -> memory-bound -> wants bigger
  batches, fewer weight bytes (quantization) and a smaller KV cache (GQA).
"""

from __future__ import annotations

from dataclasses import dataclass

from infer_lab.config import ModelConfig
from infer_lab.fleet.hardware import HardwareProfile

BYTES_PER_PARAM = 2      # bf16 weights
BYTES_PER_KV_ELEM = 2    # bf16 KV cache


@dataclass
class RooflineAnalysis:
    phase: str
    batch_size: int
    seq_len: int
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    ridge_point: float
    bound: str               # "memory" | "compute"
    time_s: float
    achieved_flops: float
    utilization: float       # achieved / peak
    tokens_per_s: float

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "phase": self.phase,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "gflops": round(self.flops / 1e9, 3),
            "mib_moved": round(self.bytes_moved / 1024**2, 3),
            "arithmetic_intensity": round(self.arithmetic_intensity, 3),
            "ridge_point": round(self.ridge_point, 1),
            "bound": self.bound,
            "time_ms": round(self.time_s * 1000, 4),
            "utilization_pct": round(self.utilization * 100, 2),
            "tokens_per_s": round(self.tokens_per_s, 1),
        }


def _kv_bytes_per_token(config: ModelConfig) -> float:
    return 2 * config.num_layers * config.num_kv_heads * config.head_dim * BYTES_PER_KV_ELEM


def _active_params(config: ModelConfig) -> float:
    """MoE only activates ``top_k`` of ``E`` experts, so FLOPs != parameter count."""
    total = config.param_count()
    if not config.is_moe:
        return float(total)
    h, i, L = config.hidden_size, config.intermediate_size, config.num_layers
    inactive_experts = config.num_experts - config.num_experts_per_tok
    return float(total - L * 3 * h * i * inactive_experts)


def _finish(phase: str, flops: float, bytes_moved: float, hw: HardwareProfile,
            batch_size: int, seq_len: int, tokens: float) -> RooflineAnalysis:
    intensity = flops / bytes_moved if bytes_moved else float("inf")
    t_compute = flops / hw.peak_flops_bf16
    t_memory = bytes_moved / hw.memory_bandwidth
    time_s = max(t_compute, t_memory)
    return RooflineAnalysis(
        phase=phase,
        batch_size=batch_size,
        seq_len=seq_len,
        flops=flops,
        bytes_moved=bytes_moved,
        arithmetic_intensity=intensity,
        ridge_point=hw.ridge_point,
        bound="compute" if t_compute >= t_memory else "memory",
        time_s=time_s,
        achieved_flops=flops / time_s if time_s else 0.0,
        utilization=(flops / time_s) / hw.peak_flops_bf16 if time_s else 0.0,
        tokens_per_s=tokens / time_s if time_s else 0.0,
    )


def analyze_decode(config: ModelConfig, hw: HardwareProfile, batch_size: int,
                   seq_len: int) -> RooflineAnalysis:
    """One decode step: B sequences, 1 token each, KV length ``seq_len``."""
    params = _active_params(config)
    flops = 2.0 * params * batch_size
    # Weights are read once and amortised over the batch -- that IS the batching win.
    weight_bytes = params * BYTES_PER_PARAM
    kv_bytes = batch_size * seq_len * _kv_bytes_per_token(config)
    return _finish("decode", flops, weight_bytes + kv_bytes, hw, batch_size, seq_len,
                   tokens=float(batch_size))


def analyze_prefill(config: ModelConfig, hw: HardwareProfile, batch_size: int,
                    seq_len: int) -> RooflineAnalysis:
    """Prefill of ``batch_size`` prompts of length ``seq_len``."""
    params = _active_params(config)
    tokens = float(batch_size * seq_len)
    gemm_flops = 2.0 * params * tokens
    # Attention is quadratic in S and is NOT covered by the parameter count.
    attn_flops = (4.0 * config.num_layers * config.num_heads * config.head_dim
                  * seq_len * seq_len * batch_size)
    weight_bytes = params * BYTES_PER_PARAM
    kv_write_bytes = tokens * _kv_bytes_per_token(config)
    return _finish("prefill", gemm_flops + attn_flops, weight_bytes + kv_write_bytes,
                   hw, batch_size, seq_len, tokens=tokens)


def batch_sweep(config: ModelConfig, hw: HardwareProfile, seq_len: int = 1024,
                batch_sizes: list[int] | None = None) -> list[RooflineAnalysis]:
    """Sweep batch size to locate the memory->compute crossover empirically."""
    batch_sizes = batch_sizes or [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    return [analyze_decode(config, hw, b, seq_len) for b in batch_sizes]


def crossover_batch_size(config: ModelConfig, hw: HardwareProfile,
                         seq_len: int = 1024, max_batch: int = 4096) -> int | None:
    """Smallest batch at which decode stops being memory-bound. None if never."""
    b = 1
    while b <= max_batch:
        if analyze_decode(config, hw, b, seq_len).bound == "compute":
            return b
        b *= 2
    return None


def kv_cache_capacity(config: ModelConfig, hw: HardwareProfile,
                      weight_fraction: float = 0.5) -> dict[str, float]:
    """How many tokens fit in HBM after the weights -- the real concurrency limit."""
    weight_bytes = config.param_count() * BYTES_PER_PARAM
    available = hw.memory_bytes * (1 - weight_fraction) if weight_bytes < hw.memory_bytes \
        else 0.0
    per_token = _kv_bytes_per_token(config)
    return {
        "weight_gib": round(weight_bytes / 1024**3, 3),
        "kv_budget_gib": round(available / 1024**3, 3),
        "kv_bytes_per_token": per_token,
        "max_kv_tokens": int(available // per_token) if per_token else 0,
    }
