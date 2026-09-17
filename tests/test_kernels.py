"""Kernel correctness: every backend must agree with the NumPy reference."""

from __future__ import annotations

import numpy as np
import pytest

from infer_lab.kernels import native
from infer_lab.kernels.numpy_kernels import (
    apply_rope, flash_attention, naive_attention, rms_norm, rope_cos_sin, softmax, swiglu,
)
from infer_lab.kernels.registry import get_registry


# --------------------------------------------------------------------- reference
@pytest.mark.parametrize("shape", [(1, 16), (8, 64), (33, 128)])
def test_rms_norm_matches_definition(shape, rng):
    x = rng.normal(size=shape).astype(np.float32)
    w = rng.normal(size=shape[-1]).astype(np.float32)
    out = rms_norm(x, w, eps=1e-5)
    expected = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * w
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_softmax_is_stable_for_large_inputs():
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    out = softmax(x)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(-1), 1.0, rtol=1e-6)


def test_swiglu_matches_silu_times_up(rng):
    g = rng.normal(size=(4, 16)).astype(np.float32)
    u = rng.normal(size=(4, 16)).astype(np.float32)
    np.testing.assert_allclose(swiglu(g, u), (g / (1 + np.exp(-g))) * u, rtol=1e-6)


# ------------------------------------------------------------------- attention
@pytest.mark.parametrize("sq,sk,offset", [(16, 16, 0), (33, 33, 0), (1, 40, 39), (7, 64, 57)])
def test_flash_attention_equals_naive_causal(sq, sk, offset, rng):
    """The tiled/online-softmax kernel must be numerically identical to the baseline."""
    q = rng.normal(size=(sq, 4, 16)).astype(np.float32)
    k = rng.normal(size=(sk, 4, 16)).astype(np.float32)
    v = rng.normal(size=(sk, 4, 16)).astype(np.float32)
    ref = naive_attention(q, k, v, causal_offset=offset)
    out = flash_attention(q, k, v, causal_offset=offset, block_q=8, block_k=8)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_flash_attention_non_causal(rng):
    q = rng.normal(size=(12, 2, 8)).astype(np.float32)
    k = rng.normal(size=(20, 2, 8)).astype(np.float32)
    v = rng.normal(size=(20, 2, 8)).astype(np.float32)
    np.testing.assert_allclose(
        flash_attention(q, k, v, block_q=5, block_k=7),
        naive_attention(q, k, v), rtol=1e-4, atol=1e-5,
    )


def test_flash_attention_tiling_is_invariant_to_block_size(rng):
    q = rng.normal(size=(24, 2, 8)).astype(np.float32)
    k = rng.normal(size=(24, 2, 8)).astype(np.float32)
    v = rng.normal(size=(24, 2, 8)).astype(np.float32)
    a = flash_attention(q, k, v, causal_offset=0, block_q=4, block_k=4)
    b = flash_attention(q, k, v, causal_offset=0, block_q=16, block_k=32)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------------ rope
def test_rope_preserves_norm(rng):
    """RoPE is a rotation, so it must not change vector magnitude."""
    x = rng.normal(size=(10, 4, 16)).astype(np.float32)
    cos, sin = rope_cos_sin(10, 16)
    out = apply_rope(x, cos, sin)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1),
                               np.linalg.norm(x, axis=-1), rtol=1e-5)


def test_rope_relative_position_property(rng):
    """<RoPE(q,m), RoPE(k,n)> must depend only on (m-n), not on absolute positions."""
    head_dim = 16
    q = rng.normal(size=(1, 1, head_dim)).astype(np.float32)
    k = rng.normal(size=(1, 1, head_dim)).astype(np.float32)

    def dot(m: int, n: int) -> float:
        cq, sq = rope_cos_sin(1, head_dim, offset=m)
        ck, sk = rope_cos_sin(1, head_dim, offset=n)
        return float(np.sum(apply_rope(q, cq, sq) * apply_rope(k, ck, sk)))

    assert dot(5, 3) == pytest.approx(dot(105, 103), rel=1e-4)


# -------------------------------------------------------------------- backends
def test_registry_always_has_numpy():
    registry = get_registry()
    assert "numpy" in registry.available()


@pytest.mark.parametrize("backend", ["ctypes", "pybind11", "nanobind", "torch"])
def test_native_backends_match_reference(backend, rng):
    """Skips cleanly when a toolchain is absent -- the suite must stay green anywhere."""
    registry = get_registry()
    info = registry.get(backend)
    if not info.available:
        pytest.skip(f"{backend} unavailable: {info.reason}")

    x = rng.normal(size=(16, 32)).astype(np.float32)
    w = rng.normal(size=(32,)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(registry.fn("rms_norm", backend)(x, w)),
                               rms_norm(x, w), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(registry.fn("softmax", backend)(x)),
                               softmax(x), rtol=1e-4, atol=1e-6)

    g = rng.normal(size=(8, 16)).astype(np.float32)
    u = rng.normal(size=(8, 16)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(registry.fn("swiglu", backend)(g, u)),
                               swiglu(g, u), rtol=1e-4, atol=1e-5)


def test_ctypes_abi_version_is_checked():
    try:
        lib = native.load("ctypes")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ctypes extension not built: {exc}")
    assert lib._lib.il_abi_version() == 1
