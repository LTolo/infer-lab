"""Kernel backend registry.

A single place that answers "which implementations of this primitive are
actually available on this machine?".  The server, the benchmark harness and
the tests all consult the registry instead of doing their own try/except
import dances.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable

from infer_lab.utils.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class Backend:
    name: str
    available: bool
    reason: str = ""
    fns: dict[str, Callable] = field(default_factory=dict)


class KernelRegistry:
    """Discovers kernel backends lazily and caches the result."""

    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}
        self._discovered = False

    # ---------------------------------------------------------------- discovery
    def discover(self) -> dict[str, Backend]:
        if self._discovered:
            return self._backends

        from infer_lab.kernels import numpy_kernels as npk

        self._backends["numpy"] = Backend(
            "numpy", True, "reference implementation",
            {"rms_norm": npk.rms_norm, "softmax": npk.softmax, "swiglu": npk.swiglu},
        )

        # --- ctypes / pybind11 / nanobind native extensions
        from infer_lab.kernels import native

        for name in ("ctypes", "pybind11", "nanobind"):
            try:
                mod = native.load(name)
                self._backends[name] = Backend(
                    name, True, "native extension loaded",
                    {"rms_norm": mod.rms_norm, "softmax": mod.softmax, "swiglu": mod.swiglu},
                )
            except Exception as exc:  # noqa: BLE001 - availability probe
                self._backends[name] = Backend(name, False, f"{type(exc).__name__}: {exc}")

        # --- PyTorch mirror
        try:
            import torch  # noqa: F401
            from infer_lab.kernels import torch_kernels as tk

            self._backends["torch"] = Backend(
                "torch", True, "torch imported",
                {"rms_norm": tk.rms_norm, "softmax": tk.softmax, "swiglu": tk.swiglu},
            )
        except Exception as exc:  # noqa: BLE001
            self._backends["torch"] = Backend("torch", False, f"{type(exc).__name__}: {exc}")

        # --- Triton (requires an actual NVIDIA GPU)
        try:
            from infer_lab.kernels import triton_kernels as trk

            ok, reason = trk.probe()
            self._backends["triton"] = Backend(
                "triton", ok, reason,
                {"rms_norm": trk.rms_norm} if ok else {},
            )
        except Exception as exc:  # noqa: BLE001
            self._backends["triton"] = Backend("triton", False, f"{type(exc).__name__}: {exc}")

        self._discovered = True
        logger.info(
            "kernel backends discovered",
            extra={"available": [b.name for b in self._backends.values() if b.available]},
        )
        return self._backends

    # ---------------------------------------------------------------- accessors
    def available(self) -> list[str]:
        return [b.name for b in self.discover().values() if b.available]

    def get(self, backend: str) -> Backend:
        b = self.discover().get(backend)
        if b is None:
            raise KeyError(f"unknown backend {backend!r}; known: {sorted(self.discover())}")
        return b

    def fn(self, op: str, backend: str = "numpy") -> Callable:
        b = self.get(backend)
        if not b.available:
            raise RuntimeError(f"backend {backend!r} unavailable: {b.reason}")
        if op not in b.fns:
            raise KeyError(f"backend {backend!r} does not implement {op!r}")
        return b.fns[op]

    def backends_for(self, op: str) -> list[str]:
        return [n for n, b in self.discover().items() if b.available and op in b.fns]

    def report(self) -> dict[str, dict[str, object]]:
        return {
            n: {"available": b.available, "reason": b.reason, "ops": sorted(b.fns)}
            for n, b in self.discover().items()
        }


@functools.lru_cache(maxsize=1)
def get_registry() -> KernelRegistry:
    return KernelRegistry()
