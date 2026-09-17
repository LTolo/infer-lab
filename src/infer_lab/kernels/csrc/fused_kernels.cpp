// Fused CPU kernels -- the numerical twin of infer_lab/kernels/numpy_kernels.py.
//
// "Fused" here means each kernel makes a single pass over a row and keeps
// intermediates in registers instead of allocating a temporary array per
// arithmetic step, which is exactly what the NumPy version cannot do.
#include "fused_kernels.h"

#include <algorithm>
#include <cmath>

namespace infer_lab {

void rms_norm_f32(const float* x, const float* weight, float* out,
                  std::size_t rows, std::size_t cols, float eps) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    float* orow = out + r * cols;
    // pass 1: sum of squares (accumulated in double to match NumPy's pairwise
    // summation closely enough for a 1e-6 tolerance)
    double acc = 0.0;
    for (std::size_t c = 0; c < cols; ++c) acc += static_cast<double>(xr[c]) * xr[c];
    const float inv = static_cast<float>(1.0 / std::sqrt(acc / static_cast<double>(cols) + eps));
    // pass 2: scale + affine, fused
    for (std::size_t c = 0; c < cols; ++c) orow[c] = xr[c] * inv * weight[c];
  }
}

void softmax_f32(const float* x, float* out, std::size_t rows, std::size_t cols) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    float* orow = out + r * cols;
    float m = -INFINITY;
    for (std::size_t c = 0; c < cols; ++c) m = std::max(m, xr[c]);
    double sum = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const float e = std::exp(xr[c] - m);
      orow[c] = e;
      sum += e;
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (std::size_t c = 0; c < cols; ++c) orow[c] *= inv;
  }
}

void swiglu_f32(const float* gate, const float* up, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const float g = gate[i];
    out[i] = (g / (1.0f + std::exp(-g))) * up[i];
  }
}

}  // namespace infer_lab

// ---------------------------------------------------------------------------
// C ABI used by the ctypes binding (no name mangling, no C++ runtime needed).
// ---------------------------------------------------------------------------
extern "C" {

#if defined(_WIN32)
#define INFER_LAB_EXPORT __declspec(dllexport)
#else
#define INFER_LAB_EXPORT __attribute__((visibility("default")))
#endif

INFER_LAB_EXPORT void il_rms_norm_f32(const float* x, const float* w, float* out,
                                      std::size_t rows, std::size_t cols, float eps) {
  infer_lab::rms_norm_f32(x, w, out, rows, cols, eps);
}

INFER_LAB_EXPORT void il_softmax_f32(const float* x, float* out,
                                     std::size_t rows, std::size_t cols) {
  infer_lab::softmax_f32(x, out, rows, cols);
}

INFER_LAB_EXPORT void il_swiglu_f32(const float* gate, const float* up, float* out,
                                    std::size_t n) {
  infer_lab::swiglu_f32(gate, up, out, n);
}

INFER_LAB_EXPORT int il_abi_version(void) { return 1; }

}  // extern "C"
