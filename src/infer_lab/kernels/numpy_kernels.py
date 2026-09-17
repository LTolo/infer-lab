"""Pure-NumPy reference kernels.

These define the *numerical contract* of infer-lab: every other backend
(C++/ctypes, pybind11, nanobind, Triton, PyTorch) is verified against these.
"""

from __future__ import annotations

import numpy as np

DTYPE = np.float32


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Root-mean-square layer norm (no mean subtraction, no bias)."""
    x = x.astype(np.float32, copy=False)
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * inv) * weight


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (max-subtraction)."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU activation: silu(gate) * up."""
    return silu(gate) * up


def rope_cos_sin(seq_len: int, head_dim: int, theta: float = 10000.0,
                 offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Precompute rotary embedding tables for positions [offset, offset+seq_len)."""
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(offset, offset + seq_len, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)
    return np.cos(freqs).astype(DTYPE), np.sin(freqs).astype(DTYPE)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply RoPE to `x` of shape (seq, heads, head_dim).

    Uses the interleaved-halves convention (same as Llama/HF).
    """
    seq, heads, hd = x.shape
    half = hd // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).astype(DTYPE)


def naive_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    causal_offset: int | None = None) -> np.ndarray:
    """Materialise the full (Sq x Sk) score matrix. O(S^2) memory -- the baseline.

    q: (Sq, H, D), k/v: (Sk, H, D) -> (Sq, H, D)
    """
    sq, h, d = q.shape
    sk = k.shape[0]
    scale = 1.0 / np.sqrt(d)
    scores = np.einsum("qhd,khd->hqk", q, k).astype(np.float32) * scale
    if causal_offset is not None:
        qi = np.arange(sq)[:, None] + causal_offset
        ki = np.arange(sk)[None, :]
        scores = np.where(ki <= qi, scores, -np.inf)
    probs = softmax(scores, axis=-1)
    return np.einsum("hqk,khd->qhd", probs, v).astype(DTYPE)


def flash_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    causal_offset: int | None = None,
                    block_q: int = 32, block_k: int = 32) -> np.ndarray:
    """Tiled / online-softmax attention (FlashAttention-style).

    Never materialises the full score matrix: the (m, l) running max/sum
    statistics are carried across key tiles.  Peak score memory is
    O(block_q * block_k) instead of O(Sq * Sk).
    """
    sq, h, d = q.shape
    sk = k.shape[0]
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((sq, h, d), dtype=np.float32)

    for q0 in range(0, sq, block_q):
        q1 = min(q0 + block_q, sq)
        qt = q[q0:q1].astype(np.float32)                     # (bq, H, D)
        m = np.full((q1 - q0, h), -np.inf, dtype=np.float32)  # running max
        l = np.zeros((q1 - q0, h), dtype=np.float32)          # running sum
        acc = np.zeros((q1 - q0, h, d), dtype=np.float32)

        for k0 in range(0, sk, block_k):
            k1 = min(k0 + block_k, sk)
            if causal_offset is not None and k0 > (q1 - 1) + causal_offset:
                break  # entire tile is masked out
            kt = k[k0:k1].astype(np.float32)
            vt = v[k0:k1].astype(np.float32)
            s = np.einsum("qhd,khd->qhk", qt, kt) * scale     # (bq, H, bk)
            if causal_offset is not None:
                qi = np.arange(q0, q1)[:, None] + causal_offset
                ki = np.arange(k0, k1)[None, :]
                mask = (ki <= qi)[:, None, :]                 # (bq, 1, bk)
                s = np.where(mask, s, -np.inf)
            tile_max = np.max(s, axis=-1)                      # (bq, H)
            new_m = np.maximum(m, tile_max)
            # guard against a fully-masked tile (all -inf)
            safe_m = np.where(np.isfinite(new_m), new_m, 0.0)
            p = np.exp(s - safe_m[..., None])
            p = np.where(np.isfinite(s), p, 0.0)
            alpha = np.exp(np.where(np.isfinite(m), m, -np.inf) - safe_m)
            alpha = np.where(np.isfinite(alpha), alpha, 0.0)
            l = l * alpha + np.sum(p, axis=-1)
            acc = acc * alpha[..., None] + np.einsum("qhk,khd->qhd", p, vt)
            m = new_m

        denom = np.where(l > 0, l, 1.0)
        out[q0:q1] = acc / denom[..., None]

    return out.astype(DTYPE)
