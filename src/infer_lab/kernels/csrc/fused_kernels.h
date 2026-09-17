// Shared declarations for the infer-lab fused CPU kernels.
//
// The same three kernels are exposed through *three* different Python binding
// strategies (ctypes / pybind11 / nanobind) so that the call-overhead
// trade-off can be measured instead of argued about.
#pragma once

#include <cstddef>

namespace infer_lab {

// out[r, :] = (x[r, :] * rsqrt(mean(x[r, :]^2) + eps)) * weight[:]
void rms_norm_f32(const float* x, const float* weight, float* out,
                  std::size_t rows, std::size_t cols, float eps);

// row-wise numerically stable softmax (max subtraction, single fused pass set)
void softmax_f32(const float* x, float* out, std::size_t rows, std::size_t cols);

// out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
void swiglu_f32(const float* gate, const float* up, float* out, std::size_t n);

}  // namespace infer_lab
