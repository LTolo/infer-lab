"""Llama-style decoder, NumPy reference implementation.

Deliberately written against the *paged* KV cache rather than a dense per-sequence
buffer, because that is the interface a real engine has to satisfy.

Two entry points mirror the two phases every inference engine distinguishes:

``forward_prefill``      one sequence, many tokens, compute-bound
``forward_decode_batch`` many sequences, one token each, memory-bound

The decode path is genuinely batched: all running sequences are stacked into a
single (B, hidden) tensor so every projection and the whole FFN are *one* GEMM.
Only the attention itself loops per sequence, because each sequence has a
different KV length -- exactly the varlen pattern that FlashAttention's
``cu_seqlens`` interface solves on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from infer_lab.config import ModelConfig
from infer_lab.kernels.numpy_kernels import (
    apply_rope,
    flash_attention,
    naive_attention,
    rms_norm,
    rope_cos_sin,
    swiglu,
)
from infer_lab.kv.paged_cache import SequenceKV
from infer_lab.model.moe import load_balance_loss, moe_ffn
from infer_lab.model.weights import LayerWeights, ModelWeights

DTYPE = np.float32


@dataclass
class ForwardStats:
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_calls: int = 0
    decode_calls: int = 0
    expert_load: list[int] = field(default_factory=list)
    router_cv: float = 0.0

    def as_dict(self) -> dict[str, float | int | list[int]]:
        return {
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "prefill_calls": self.prefill_calls,
            "decode_calls": self.decode_calls,
            "expert_load": self.expert_load,
            "router_cv": round(self.router_cv, 4),
        }


class NumpyTransformer:
    """The reference engine-facing model."""

    def __init__(self, weights: ModelWeights, *, attention_impl: str = "flash") -> None:
        self.w = weights
        self.config: ModelConfig = weights.config
        if attention_impl not in ("flash", "naive"):
            raise ValueError("attention_impl must be 'flash' or 'naive'")
        self.attention_impl = attention_impl
        self.stats = ForwardStats()
        # RoPE tables are position-only: precompute once for the whole context window
        self._cos, self._sin = rope_cos_sin(
            self.config.max_position_embeddings, self.config.head_dim, self.config.rope_theta
        )

    # --------------------------------------------------------------------- utils
    def _attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                   causal_offset: int) -> np.ndarray:
        fn = flash_attention if self.attention_impl == "flash" else naive_attention
        return fn(q, k, v, causal_offset=causal_offset)

    def _repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """Expand (S, num_kv_heads, D) -> (S, num_heads, D) for grouped-query attention.

        GQA is a pure *inference* optimisation: it shrinks the KV cache by
        ``kv_group_size`` with almost no quality cost, which directly raises the
        number of sequences that fit in the KV pool.
        """
        g = self.config.kv_group_size
        return x if g == 1 else np.repeat(x, g, axis=1)

    def _project_qkv(self, h: np.ndarray, layer: LayerWeights, positions: np.ndarray):
        n = h.shape[0]
        cfg = self.config
        q = (h @ layer.wq).reshape(n, cfg.num_heads, cfg.head_dim)
        k = (h @ layer.wk).reshape(n, cfg.num_kv_heads, cfg.head_dim)
        v = (h @ layer.wv).reshape(n, cfg.num_kv_heads, cfg.head_dim)
        cos = self._cos[positions]
        sin = self._sin[positions]
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def _ffn(self, h: np.ndarray, layer: LayerWeights) -> np.ndarray:
        if self.config.is_moe and layer.router is not None:
            out, routing = moe_ffn(h, layer.w_gate, layer.w_up, layer.w_down,
                                   layer.router, self.config.num_experts_per_tok)
            self.stats.expert_load = routing.load.tolist()
            self.stats.router_cv = load_balance_loss(routing.load)
            return out
        return swiglu(h @ layer.w_gate[0], h @ layer.w_up[0]) @ layer.w_down[0]

    def _logits(self, h: np.ndarray) -> np.ndarray:
        return rms_norm(h, self.w.final_norm, self.config.rms_norm_eps) @ self.w.lm_head

    # ------------------------------------------------------------------- prefill
    def forward_prefill(self, token_ids: np.ndarray, seq_kv: SequenceKV,
                        start_pos: int = 0, *, return_all_logits: bool = False) -> np.ndarray:
        """Run ``len(token_ids)`` tokens of one sequence and append their KV.

        ``start_pos`` > 0 is what makes *chunked prefill* work: a long prompt is
        fed through in slices, each slice attending to everything written before it.
        """
        tokens = np.asarray(token_ids, dtype=np.int64)
        n = int(tokens.shape[0])
        if n == 0:
            raise ValueError("forward_prefill called with zero tokens")

        positions = np.arange(start_pos, start_pos + n, dtype=np.int64)
        x = self.w.embed[tokens].astype(DTYPE)

        for layer_idx, layer in enumerate(self.w.layers):
            h = rms_norm(x, layer.attn_norm, self.config.rms_norm_eps)
            q, k, v = self._project_qkv(h, layer, positions)

            seq_kv.write(layer_idx, start_pos, k, v)
            if layer_idx == 0:
                seq_kv.length = start_pos + n
            k_all, v_all = seq_kv.gather(layer_idx)

            attn = self._attention(q, self._repeat_kv(k_all), self._repeat_kv(v_all),
                                   causal_offset=start_pos)
            x = x + attn.reshape(n, -1) @ layer.wo
            x = x + self._ffn(rms_norm(x, layer.ffn_norm, self.config.rms_norm_eps), layer)

        self.stats.prefill_tokens += n
        self.stats.prefill_calls += 1
        logits = self._logits(x if return_all_logits else x[-1:])
        return logits if return_all_logits else logits[0]

    # -------------------------------------------------------------------- decode
    def forward_decode_batch(self, token_ids: np.ndarray, seq_kvs: list[SequenceKV],
                             positions: np.ndarray) -> np.ndarray:
        """One step for ``B`` sequences at once. Returns (B, vocab) logits.

        This is the batched-decode core: every linear layer sees a (B, hidden)
        tensor, so B sequences cost one GEMM instead of B GEMMs.  That is the
        entire reason continuous batching raises throughput -- the weights are
        loaded from memory once and amortised across the batch.
        """
        tokens = np.asarray(token_ids, dtype=np.int64)
        batch = int(tokens.shape[0])
        if batch == 0:
            return np.zeros((0, self.config.vocab_size), dtype=DTYPE)
        if len(seq_kvs) != batch or positions.shape[0] != batch:
            raise ValueError("token_ids, seq_kvs and positions must have equal length")

        x = self.w.embed[tokens].astype(DTYPE)          # (B, h)

        for layer_idx, layer in enumerate(self.w.layers):
            h = rms_norm(x, layer.attn_norm, self.config.rms_norm_eps)
            q, k, v = self._project_qkv(h, layer, np.asarray(positions, dtype=np.int64))

            attn_out = np.empty((batch, self.config.num_heads, self.config.head_dim),
                                dtype=DTYPE)
            for b, seq_kv in enumerate(seq_kvs):
                pos = int(positions[b])
                seq_kv.write(layer_idx, pos, k[b:b + 1], v[b:b + 1])
                if layer_idx == 0:
                    seq_kv.length = pos + 1
                k_all, v_all = seq_kv.gather(layer_idx)
                attn_out[b] = self._attention(
                    q[b:b + 1], self._repeat_kv(k_all), self._repeat_kv(v_all),
                    causal_offset=pos,
                )[0]

            x = x + attn_out.reshape(batch, -1) @ layer.wo
            x = x + self._ffn(rms_norm(x, layer.ffn_norm, self.config.rms_norm_eps), layer)

        self.stats.decode_tokens += batch
        self.stats.decode_calls += 1
        return self._logits(x)

    # ---------------------------------------------------------------- bookkeeping
    def reset_stats(self) -> None:
        self.stats = ForwardStats()
