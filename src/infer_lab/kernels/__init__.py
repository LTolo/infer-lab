"""Compute kernels and the three Python<->C++ binding strategies.

Everything degrades gracefully: if no C++ toolchain, no PyTorch and no GPU is
available, the pure-NumPy reference kernels are used and every test still runs.
"""

from infer_lab.kernels.numpy_kernels import (
    rms_norm,
    softmax,
    swiglu,
    naive_attention,
    flash_attention,
    rope_cos_sin,
    apply_rope,
)
from infer_lab.kernels.registry import KernelRegistry, get_registry

__all__ = [
    "rms_norm", "softmax", "swiglu", "naive_attention", "flash_attention",
    "rope_cos_sin", "apply_rope", "KernelRegistry", "get_registry",
]
