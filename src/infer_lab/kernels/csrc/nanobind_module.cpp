// nanobind binding -- the modern, leaner successor to pybind11.
// Smaller binaries and lower per-call overhead; API is deliberately similar so
// the benchmark comparison is apples-to-apples.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstring>
#include <stdexcept>

#include "fused_kernels.h"

namespace nb = nanobind;
using RArr = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

static nb::ndarray<nb::numpy, float> make_out(float* data, size_t rows, size_t cols) {
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
  size_t shape[2] = {rows, cols};
  return nb::ndarray<nb::numpy, float>(data, 2, shape, owner);
}

static nb::ndarray<nb::numpy, float> rms_norm(RArr x, RArr weight, float eps) {
  if (x.ndim() != 2) throw std::runtime_error("x must be 2-D (rows, cols)");
  if (weight.ndim() != 1 || weight.shape(0) != x.shape(1))
    throw std::runtime_error("weight must be 1-D with length == x.shape[1]");
  const size_t rows = x.shape(0), cols = x.shape(1);
  float* out = new float[rows * cols];
  infer_lab::rms_norm_f32(x.data(), weight.data(), out, rows, cols, eps);
  return make_out(out, rows, cols);
}

static nb::ndarray<nb::numpy, float> softmax(RArr x) {
  if (x.ndim() != 2) throw std::runtime_error("x must be 2-D (rows, cols)");
  const size_t rows = x.shape(0), cols = x.shape(1);
  float* out = new float[rows * cols];
  infer_lab::softmax_f32(x.data(), out, rows, cols);
  return make_out(out, rows, cols);
}

static nb::ndarray<nb::numpy, float> swiglu(RArr gate, RArr up) {
  if (gate.size() != up.size()) throw std::runtime_error("gate and up must have equal size");
  const size_t rows = gate.ndim() == 2 ? gate.shape(0) : 1;
  const size_t cols = gate.size() / rows;
  float* out = new float[gate.size()];
  infer_lab::swiglu_f32(gate.data(), up.data(), out, gate.size());
  return make_out(out, rows, cols);
}

NB_MODULE(_infer_lab_nanobind, m) {
  m.doc() = "infer-lab fused kernels (nanobind binding)";
  m.attr("binding") = "nanobind";
  m.def("rms_norm", &rms_norm, nb::arg("x"), nb::arg("weight"), nb::arg("eps") = 1e-5f);
  m.def("softmax", &softmax, nb::arg("x"));
  m.def("swiglu", &swiglu, nb::arg("gate"), nb::arg("up"));
  m.def("noop", []() {}, "empty call -- measures pure binding overhead");
}
