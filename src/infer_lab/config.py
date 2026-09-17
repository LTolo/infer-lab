"""Central configuration objects.

Everything downstream is driven by these two dataclasses so that a benchmark,
a unit test and the HTTP server all describe the *same* system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Llama-style decoder configuration.

    Defaults are deliberately tiny so that the full engine runs in milliseconds
    on a CPU-only laptop.  The *architecture* is identical to production models;
    only the parameter count differs.
    """

    vocab_size: int = 1024
    hidden_size: int = 256
    intermediate_size: int = 688          # ~= 8/3 * hidden, rounded to a multiple of 16
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 2                 # grouped-query attention
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    # Mixture-of-Experts (only used by the MoE model variant)
    num_experts: int = 0
    num_experts_per_tok: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_experts and self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok must be <= num_experts")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    def param_count(self) -> int:
        """Analytic parameter count -- used by the roofline model."""
        h, i, L = self.hidden_size, self.intermediate_size, self.num_layers
        kv = self.num_kv_heads * self.head_dim
        attn = h * h + 2 * h * kv + h * h          # q, k, v, o
        ffn_experts = max(self.num_experts, 1)
        ffn = 3 * h * i * ffn_experts
        router = h * self.num_experts
        norms = 2 * h
        embed = self.vocab_size * h * (1 if self.tie_word_embeddings else 2)
        return L * (attn + ffn + router + norms) + embed + h

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineConfig:
    """Runtime/scheduler configuration (the vLLM-style knobs)."""

    block_size: int = 16                  # KV tokens per page
    num_gpu_blocks: int = 512             # size of the paged KV pool
    max_num_seqs: int = 32                # max concurrently *running* sequences
    max_num_batched_tokens: int = 512     # prefill token budget per engine step
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True    # SGLang-style RadixAttention
    enable_preemption: bool = True
    preemption_mode: str = "recompute"    # "recompute" | "swap"
    watermark: float = 0.01               # keep a few blocks free to avoid thrashing
    max_model_len: int = 512

    # speculative decoding
    speculative_num_draft_tokens: int = 0  # 0 disables speculation

    # sampling defaults
    default_max_tokens: int = 32
    seed: int = 0

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.preemption_mode not in ("recompute", "swap"):
            raise ValueError("preemption_mode must be 'recompute' or 'swap'")
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a positive power of two")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
