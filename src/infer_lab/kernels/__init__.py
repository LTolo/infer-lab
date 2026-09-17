"""Compute kernels and the three Python<->C++ binding strategies.

Everything degrades gracefully: if no C++ toolchain, no PyTorch and no GPU is
available, the pure-NumPy reference kernels are used and every test still runs.
"""

from infer_lab.kernels.numpy_kernels import (
    apply_rope,
    flash_attention,
    naive_attention,
    rms_norm,
    rope_cos_sin,
    softmax,
    swiglu,
)
from infer_lab.kernels.registry import KernelRegistry, get_registry

__all__ = [
    "rms_norm", "softmax", "swiglu", "naive_attention", "flash_attention",
    "rope_cos_sin", "apply_rope", "KernelRegistry", "get_registry",
]
