"""INT8 per-channel quantization."""

from __future__ import annotations

import numpy as np
import pytest

from infer_lab.quant.int8 import (
    dequantize, quantization_error, quantize_per_channel, quantized_matmul,
)


def test_compression_is_close_to_four_x(rng):
    w = rng.normal(size=(256, 512)).astype(np.float32)
    qm = quantize_per_channel(w)
    assert 3.9 < qm.compression_ratio() < 4.01


def test_roundtrip_error_is_small(rng):
    w = rng.normal(size=(128, 256)).astype(np.float32)
    stats = quantization_error(w, quantize_per_channel(w))
    assert stats["rel_fro_err"] < 0.01
    assert stats["cosine_sim"] > 0.9999


def test_per_channel_beats_per_tensor_on_skewed_weights(rng):
    """The scenario per-channel scales exist for: one channel dominates the range."""
    w = rng.normal(size=(64, 8)).astype(np.float32)
    w[:, 0] *= 1000.0

    per_channel = quantization_error(w, quantize_per_channel(w))["rel_fro_err"]

    amax = np.abs(w).max()
    scale = amax / 127.0
    per_tensor = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    per_tensor_err = float(np.linalg.norm(w - per_tensor * scale) / np.linalg.norm(w))

    assert per_channel < per_tensor_err


def test_quantized_matmul_matches_dequantized_matmul(rng):
    x = rng.normal(size=(8, 64)).astype(np.float32)
    w = rng.normal(size=(64, 32)).astype(np.float32)
    qm = quantize_per_channel(w)
    np.testing.assert_allclose(quantized_matmul(x, qm), x @ dequantize(qm),
                               rtol=1e-4, atol=1e-4)


def test_quantized_matmul_is_close_to_fp32(rng):
    x = rng.normal(size=(16, 128)).astype(np.float32)
    w = rng.normal(size=(128, 64)).astype(np.float32)
    out = quantized_matmul(x, quantize_per_channel(w))
    reference = x @ w
    rel = np.linalg.norm(out - reference) / np.linalg.norm(reference)
    assert rel < 0.02


def test_zero_channel_does_not_divide_by_zero():
    w = np.zeros((8, 4), dtype=np.float32)
    w[:, 1] = 1.0
    qm = quantize_per_channel(w)
    assert np.isfinite(qm.scales).all()
    np.testing.assert_allclose(dequantize(qm)[:, 0], 0.0)


def test_rejects_non_2d_input(rng):
    with pytest.raises(ValueError):
        quantize_per_channel(rng.normal(size=(4, 4, 4)).astype(np.float32))
