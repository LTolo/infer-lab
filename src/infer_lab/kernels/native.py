"""Loader for the three native binding flavours.

``load(kind)`` returns an object exposing ``rms_norm`` / ``softmax`` / ``swiglu``
with the *same* Python signature regardless of how the C++ was bound, so the
benchmark can compare them without special-casing.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"
SHARED_SUFFIX = ".dll" if IS_WINDOWS else (".dylib" if sys.platform == "darwin" else ".so")

_F32 = np.float32


def _as_2d(x: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=_F32)
    return a if a.ndim == 2 else a.reshape(1, -1) if a.ndim == 1 else a.reshape(-1, a.shape[-1])


# --------------------------------------------------------------------------- ctypes
def _load_ctypes() -> SimpleNamespace:
    candidates = sorted(HERE.glob(f"libinfer_lab_kernels*{SHARED_SUFFIX}"))
    if not candidates:
        raise FileNotFoundError(
            f"no ctypes shared library in {HERE}; run "
            "`python -m infer_lab.kernels.build --only ctypes`"
        )
    lib = ctypes.CDLL(str(candidates[0]))

    lib.il_abi_version.restype = ctypes.c_int
    if lib.il_abi_version() != 1:
        raise RuntimeError("ctypes kernel library ABI mismatch (expected 1)")

    f32p = ctypes.POINTER(ctypes.c_float)
    lib.il_rms_norm_f32.argtypes = [f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_size_t,
                                    ctypes.c_float]
    lib.il_rms_norm_f32.restype = None
    lib.il_softmax_f32.argtypes = [f32p, f32p, ctypes.c_size_t, ctypes.c_size_t]
    lib.il_softmax_f32.restype = None
    lib.il_swiglu_f32.argtypes = [f32p, f32p, f32p, ctypes.c_size_t]
    lib.il_swiglu_f32.restype = None

    def ptr(a: np.ndarray):
        return a.ctypes.data_as(f32p)

    def rms_norm(x, weight, eps: float = 1e-5):
        xa = _as_2d(x)
        wa = np.ascontiguousarray(weight, dtype=_F32)
        out = np.empty_like(xa)
        lib.il_rms_norm_f32(ptr(xa), ptr(wa), ptr(out), xa.shape[0], xa.shape[1], eps)
        return out.reshape(np.shape(x))

    def softmax(x):
        xa = _as_2d(x)
        out = np.empty_like(xa)
        lib.il_softmax_f32(ptr(xa), ptr(out), xa.shape[0], xa.shape[1])
        return out.reshape(np.shape(x))

    def swiglu(gate, up):
        ga = np.ascontiguousarray(gate, dtype=_F32)
        ua = np.ascontiguousarray(up, dtype=_F32)
        out = np.empty_like(ga)
        lib.il_swiglu_f32(ptr(ga), ptr(ua), ptr(out), ga.size)
        return out

    def noop():
        lib.il_abi_version()

    return SimpleNamespace(binding="ctypes", rms_norm=rms_norm, softmax=softmax,
                           swiglu=swiglu, noop=noop, _lib=lib)


# ------------------------------------------------------------------ python extensions
def _load_extension(module_name: str, binding: str) -> SimpleNamespace:
    spec = importlib.util.find_spec(f"infer_lab.kernels.{module_name}")
    if spec is None:
        matches = sorted(HERE.glob(f"{module_name}*"))
        if not matches:
            raise FileNotFoundError(
                f"extension {module_name} not built; run "
                f"`python -m infer_lab.kernels.build --only {binding}`"
            )
        spec = importlib.util.spec_from_file_location(module_name, matches[0])
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    def rms_norm(x, weight, eps: float = 1e-5):
        xa = _as_2d(x)
        out = mod.rms_norm(xa, np.ascontiguousarray(weight, dtype=_F32), eps)
        return np.asarray(out).reshape(np.shape(x))

    def softmax(x):
        xa = _as_2d(x)
        return np.asarray(mod.softmax(xa)).reshape(np.shape(x))

    def swiglu(gate, up):
        ga = _as_2d(gate)
        ua = _as_2d(up)
        return np.asarray(mod.swiglu(ga, ua)).reshape(np.shape(gate))

    return SimpleNamespace(binding=binding, rms_norm=rms_norm, softmax=softmax,
                           swiglu=swiglu, noop=mod.noop, _mod=mod)


_LOADERS = {
    "ctypes": _load_ctypes,
    "pybind11": lambda: _load_extension("_infer_lab_pybind", "pybind11"),
    "nanobind": lambda: _load_extension("_infer_lab_nanobind", "nanobind"),
}

_CACHE: dict[str, SimpleNamespace] = {}


def load(kind: str) -> SimpleNamespace:
    if kind not in _LOADERS:
        raise KeyError(f"unknown binding {kind!r}; known: {sorted(_LOADERS)}")
    if kind not in _CACHE:
        _CACHE[kind] = _LOADERS[kind]()
    return _CACHE[kind]


def available() -> dict[str, bool]:
    out = {}
    for kind in _LOADERS:
        try:
            load(kind)
            out[kind] = True
        except Exception:  # noqa: BLE001
            out[kind] = False
    return out
