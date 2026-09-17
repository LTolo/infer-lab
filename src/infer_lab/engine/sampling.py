"""Token sampling.

Greedy by default: a deterministic engine is a testable engine, and every
correctness test in this repo depends on being able to assert exact token
sequences across preemption, prefix caching and speculative decoding.
"""

from __future__ import annotations

import numpy as np

from infer_lab.engine.request import SamplingParams
from infer_lab.kernels.numpy_kernels import softmax


def _apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -top_k)[-top_k]
    return np.where(logits >= kth, logits, -np.inf)


def _apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return logits
    order = np.argsort(-logits)
    probs = softmax(logits[order])
    cumulative = np.cumsum(probs)
    # keep the smallest set whose mass exceeds top_p (always keep the argmax)
    cutoff = int(np.searchsorted(cumulative, top_p)) + 1
    keep = order[:cutoff]
    masked = np.full_like(logits, -np.inf)
    masked[keep] = logits[keep]
    return masked


def token_probs(logits: np.ndarray, params: SamplingParams) -> np.ndarray:
    """Full post-processing pipeline -> probability vector (used by speculation)."""
    if params.temperature == 0.0:
        probs = np.zeros_like(logits, dtype=np.float32)
        probs[int(np.argmax(logits))] = 1.0
        return probs
    scaled = logits.astype(np.float32) / params.temperature
    scaled = _apply_top_k(scaled, params.top_k)
    scaled = _apply_top_p(scaled, params.top_p)
    return softmax(scaled)


def sample_token(logits: np.ndarray, params: SamplingParams,
                 rng: np.random.Generator | None = None) -> int:
    if params.temperature == 0.0:
        return int(np.argmax(logits))
    probs = token_probs(logits, params)
    rng = rng or np.random.default_rng(params.seed)
    return int(rng.choice(probs.shape[-1], p=probs))
