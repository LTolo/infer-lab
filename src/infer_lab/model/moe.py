"""Mixture-of-Experts routing.

Architecture co-design in practice: an MoE layer buys you more parameters at
roughly constant FLOPs-per-token, but it changes the *inference* problem
completely -- the FFN becomes a scatter/gather over a token-dependent set of
weight matrices, so the arithmetic intensity collapses and the layer becomes
memory-bound much earlier than a dense FFN.  ADR-0005 works through the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from infer_lab.kernels.numpy_kernels import softmax, swiglu

DTYPE = np.float32


@dataclass
class RoutingResult:
    expert_ids: np.ndarray      # (n, k) int64
    expert_weights: np.ndarray  # (n, k) float32, normalised over k
    load: np.ndarray            # (E,) int64 -- tokens assigned per expert


def route(x: np.ndarray, router: np.ndarray, top_k: int) -> RoutingResult:
    """Top-k softmax routing (Mixtral/Switch style)."""
    logits = x @ router                                   # (n, E)
    num_experts = logits.shape[-1]
    top_k = min(top_k, num_experts)
    # argpartition gives the top-k without a full sort
    idx = np.argpartition(-logits, kth=top_k - 1, axis=-1)[:, :top_k]
    gathered = np.take_along_axis(logits, idx, axis=-1)
    order = np.argsort(-gathered, axis=-1)
    expert_ids = np.take_along_axis(idx, order, axis=-1)
    weights = softmax(np.take_along_axis(gathered, order, axis=-1), axis=-1).astype(DTYPE)
    load = np.bincount(expert_ids.ravel(), minlength=num_experts)
    return RoutingResult(expert_ids=expert_ids, expert_weights=weights, load=load)


def moe_ffn(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray,
            router: np.ndarray, top_k: int) -> tuple[np.ndarray, RoutingResult]:
    """Sparse FFN.

    Implemented as a grouped-by-expert scatter/gather rather than a dense
    per-token loop -- the same structure a real grouped-GEMM kernel uses.
    """
    result = route(x, router, top_k)
    out = np.zeros_like(x, dtype=DTYPE)
    num_experts = router.shape[-1]

    for expert in range(num_experts):
        rows, slots = np.nonzero(result.expert_ids == expert)
        if rows.size == 0:
            continue
        xe = x[rows]
        hidden = swiglu(xe @ w_gate[expert], xe @ w_up[expert])
        ye = hidden @ w_down[expert]
        scale = result.expert_weights[rows, slots][:, None]
        np.add.at(out, rows, (ye * scale).astype(DTYPE))

    return out, result


def load_balance_loss(load: np.ndarray) -> float:
    """Coefficient of variation of expert load.

    Not a training loss here -- an *observability* signal.  A skewed router turns
    one expert into a straggler and destroys batch latency, so the engine exports
    this as a metric.
    """
    load = load.astype(np.float64)
    mean = load.mean()
    return float(load.std() / mean) if mean > 0 else 0.0
