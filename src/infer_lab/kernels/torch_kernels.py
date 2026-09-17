"""PyTorch mirror of the reference kernels.

Imported lazily -- infer-lab never *requires* torch.  When torch is present the
registry exposes these so that the benchmark can compare a vectorised BLAS
backend against NumPy and against the hand-written C++ kernels.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability probe
    import torch
    import torch.nn.functional as F
    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False


def _require() -> None:
    if not _HAVE_TORCH:
        raise RuntimeError("PyTorch is not installed (pip install torch)")


def _to_t(x):
    return torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32))


def rms_norm(x, weight, eps: float = 1e-5):
    _require()
    t = _to_t(x)
    w = _to_t(weight)
    inv = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
    return (t * inv * w).numpy()


def softmax(x, axis: int = -1):
    _require()
    return torch.softmax(_to_t(x), dim=axis).numpy()


def swiglu(gate, up):
    _require()
    return (F.silu(_to_t(gate)) * _to_t(up)).numpy()


def sdpa_attention(q, k, v, causal_offset: int | None = None):
    """scaled_dot_product_attention wrapper.

    This is the call that dispatches to FlashAttention/memory-efficient kernels
    on a real GPU; on CPU it falls back to the math backend.  Inputs use the
    infer-lab layout (S, H, D) and are permuted to torch's (B, H, S, D).
    """
    _require()
    qt = _to_t(q).permute(1, 0, 2).unsqueeze(0)
    kt = _to_t(k).permute(1, 0, 2).unsqueeze(0)
    vt = _to_t(v).permute(1, 0, 2).unsqueeze(0)
    attn_mask = None
    if causal_offset is not None:
        sq, sk = qt.shape[-2], kt.shape[-2]
        qi = torch.arange(sq).unsqueeze(1) + causal_offset
        ki = torch.arange(sk).unsqueeze(0)
        attn_mask = (ki <= qi)
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask)
    return out.squeeze(0).permute(1, 0, 2).contiguous().numpy()


def available() -> bool:
    return _HAVE_TORCH


def device_report() -> dict[str, object]:
    if not _HAVE_TORCH:
        return {"torch": False}
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
