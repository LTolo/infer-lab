"""Triton GPU kernels.

These are *real* kernels, not pseudocode -- but Triton only compiles and runs on
an NVIDIA GPU.  On a CPU-only machine ``probe()`` reports why, and the registry
marks the backend unavailable rather than crashing the import.

The fused RMSNorm kernel below is the canonical shape of a production kernel:
one program per row, the whole row held in registers/SRAM, a single load and a
single store, and BLOCK_SIZE chosen as the next power of two so the mask is free.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability probe
    import torch
    import triton
    import triton.language as tl
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 # pragma: no cover
    _IMPORT_ERROR = exc
    triton = None  # type: ignore[assignment]


def probe() -> tuple[bool, str]:
    """Return (available, human readable reason)."""
    if _IMPORT_ERROR is not None:
        return False, f"triton/torch import failed: {_IMPORT_ERROR}"
    if not torch.cuda.is_available():
        return False, "no CUDA device available (Triton requires an NVIDIA GPU)"
    return True, f"cuda device: {torch.cuda.get_device_name(0)}"


if _IMPORT_ERROR is None:  # pragma: no cover - requires GPU to execute

    @triton.jit
    def _rms_norm_fwd(x_ptr, w_ptr, out_ptr, stride, n_cols, eps,
                      BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        x_row = x_ptr + row * stride
        out_row = out_ptr + row * stride
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row + cols, x * rstd * w, mask=mask)

    @triton.jit
    def _swiglu_fwd(g_ptr, u_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, (g * tl.sigmoid(g)) * u, mask=mask)


def rms_norm(x, weight, eps: float = 1e-5):  # pragma: no cover - requires GPU
    ok, reason = probe()
    if not ok:
        raise RuntimeError(reason)
    xt = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device="cuda")
    if xt.ndim == 1:
        xt = xt[None, :]
    wt = torch.as_tensor(np.ascontiguousarray(weight, dtype=np.float32), device="cuda")
    out = torch.empty_like(xt)
    n_cols = xt.shape[1]
    block = triton.next_power_of_2(n_cols)
    _rms_norm_fwd[(xt.shape[0],)](xt, wt, out, xt.stride(0), n_cols, eps,
                                  BLOCK_SIZE=block, num_warps=4)
    return out.cpu().numpy().reshape(np.shape(x))


def swiglu(gate, up):  # pragma: no cover - requires GPU
    ok, reason = probe()
    if not ok:
        raise RuntimeError(reason)
    g = torch.as_tensor(np.ascontiguousarray(gate, dtype=np.float32), device="cuda").ravel()
    u = torch.as_tensor(np.ascontiguousarray(up, dtype=np.float32), device="cuda").ravel()
    out = torch.empty_like(g)
    n = g.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # noqa: E731
    _swiglu_fwd[grid](g, u, out, n, BLOCK_SIZE=1024)
    return out.cpu().numpy().reshape(np.shape(gate))
