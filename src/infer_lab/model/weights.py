"""Deterministic synthetic weights.

infer-lab is an *engine* laboratory: the point is scheduling, paging, kernels and
distribution, not model quality.  Weights are generated from a fixed seed with
sane init scales so activations stay in a numerically healthy range -- which
also makes the instability debugger's job meaningful (we can inject a bad
matrix and watch it get caught).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from infer_lab.config import ModelConfig

DTYPE = np.float32


@dataclass
class LayerWeights:
    attn_norm: np.ndarray      # (h,)
    wq: np.ndarray             # (h, n_heads * head_dim)
    wk: np.ndarray             # (h, n_kv * head_dim)
    wv: np.ndarray             # (h, n_kv * head_dim)
    wo: np.ndarray             # (n_heads * head_dim, h)
    ffn_norm: np.ndarray       # (h,)
    w_gate: np.ndarray         # (E, h, i) -- E == 1 for dense
    w_up: np.ndarray           # (E, h, i)
    w_down: np.ndarray         # (E, i, h)
    router: np.ndarray | None  # (h, E) for MoE, else None

    def nbytes(self) -> int:
        total = 0
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                total += value.nbytes
        return total


@dataclass
class ModelWeights:
    config: ModelConfig
    embed: np.ndarray          # (vocab, h)
    layers: list[LayerWeights]
    final_norm: np.ndarray     # (h,)
    lm_head: np.ndarray        # (h, vocab)

    @staticmethod
    def random(config: ModelConfig, seed: int = 0) -> ModelWeights:
        rng = np.random.default_rng(seed)
        h = config.hidden_size
        i = config.intermediate_size
        kv = config.num_kv_heads * config.head_dim
        q = config.num_heads * config.head_dim
        experts = max(config.num_experts, 1)

        def normal(*shape: int, fan_in: int) -> np.ndarray:
            # He-ish init: keeps per-layer activation variance ~constant
            return rng.normal(0.0, (1.0 / fan_in) ** 0.5, size=shape).astype(DTYPE)

        layers = [
            LayerWeights(
                attn_norm=np.ones(h, dtype=DTYPE),
                wq=normal(h, q, fan_in=h),
                wk=normal(h, kv, fan_in=h),
                wv=normal(h, kv, fan_in=h),
                wo=normal(q, h, fan_in=q),
                ffn_norm=np.ones(h, dtype=DTYPE),
                w_gate=normal(experts, h, i, fan_in=h),
                w_up=normal(experts, h, i, fan_in=h),
                w_down=normal(experts, i, h, fan_in=i),
                router=normal(h, config.num_experts, fan_in=h) if config.is_moe else None,
            )
            for _ in range(config.num_layers)
        ]

        embed = normal(config.vocab_size, h, fan_in=h)
        lm_head = embed.T.copy() if config.tie_word_embeddings else normal(
            h, config.vocab_size, fan_in=h
        )
        return ModelWeights(config, embed, layers, np.ones(h, dtype=DTYPE), lm_head)

    def nbytes(self) -> int:
        return (self.embed.nbytes + self.lm_head.nbytes + self.final_norm.nbytes
                + sum(layer.nbytes() for layer in self.layers))
