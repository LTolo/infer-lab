"""INT8 per-channel weight quantization (W8A16-style).

Why per-channel and not per-tensor
----------------------------------
Weight matrices in transformers have wildly different dynamic ranges per output
channel.  A single per-tensor scale is dominated by the largest channel and
throws away precision everywhere else.  Per-channel (one scale per output
column) costs one extra float per column -- negligible -- and typically cuts the
quantization error by an order of magnitude.

Why this matters for an inference engine: decode is *memory-bound*.  Halving the
bytes moved per weight roughly halves decode latency at small batch sizes, long
before any FLOP limit is reached.  The roofline module makes that explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DTYPE = np.float32
INT8_MAX = 127.0


@dataclass
class QuantizedMatrix:
    """Symmetric INT8 weights plus one FP32 scale per output channel."""

    qweight: np.ndarray   # (in_features, out_features) int8
    scales: np.ndarray    # (out_features,) float32
    original_shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return self.qweight.nbytes + self.scales.nbytes

    def compression_ratio(self) -> float:
        original = int(np.prod(self.original_shape)) * 4
        return original / self.nbytes

    def dequantize(self) -> np.ndarray:
        return dequantize(self)


def quantize_per_channel(weight: np.ndarray, axis: int = 0) -> QuantizedMatrix:
    """Quantize a 2-D weight matrix along ``axis`` (0 = per output column)."""
    w = np.ascontiguousarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("quantize_per_channel expects a 2-D weight matrix")
    amax = np.max(np.abs(w), axis=axis, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)          # all-zero channel -> scale 1
    scales = (amax / INT8_MAX).astype(DTYPE)
    q = np.clip(np.rint(w / scales), -127, 127).astype(np.int8)
    return QuantizedMatrix(qweight=q, scales=scales.reshape(-1), original_shape=w.shape)


def dequantize(qm: QuantizedMatrix) -> np.ndarray:
    return (qm.qweight.astype(np.float32) * qm.scales[None, :]).astype(DTYPE)


def quantized_matmul(x: np.ndarray, qm: QuantizedMatrix) -> np.ndarray:
    """y = x @ dequant(W), computed as (x @ W_int8) * scales.

    The int8 GEMM is accumulated in int32/float32 and rescaled once at the end --
    the same fusion a real INT8 kernel performs, so the numerics match what you
    would see on hardware rather than a dequantize-then-GEMM approximation.
    """
    acc = x.astype(np.float32) @ qm.qweight.astype(np.float32)
    return (acc * qm.scales[None, :]).astype(DTYPE)


def quantization_error(weight: np.ndarray, qm: QuantizedMatrix) -> dict[str, float]:
    """Relative error metrics -- used by the regression tests to pin quality."""
    w = weight.astype(np.float32)
    recon = dequantize(qm)
    diff = w - recon
    denom = float(np.linalg.norm(w)) or 1.0
    return {
        "max_abs_err": float(np.max(np.abs(diff))),
        "rel_fro_err": float(np.linalg.norm(diff) / denom),
        "cosine_sim": float(
            np.dot(w.ravel(), recon.ravel())
            / ((np.linalg.norm(w.ravel()) * np.linalg.norm(recon.ravel())) or 1.0)
        ),
        "compression_ratio": qm.compression_ratio(),
    }
