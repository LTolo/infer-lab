// pybind11 binding -- the established standard.
// Zero-copy: numpy arrays are accessed through buffer protocol, no conversion.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fused_kernels.h"

namespace py = pybind11;
using Arr = py::array_t<float, py::array::c_style | py::array::forcecast>;

static Arr rms_norm(Arr x, Arr weight, float eps) {
  auto xb = x.request(), wb = weight.request();
  if (xb.ndim != 2) throw std::runtime_error("x must be 2-D (rows, cols)");
  if (wb.ndim != 1 || wb.shape[0] != xb.shape[1])
    throw std::runtime_error("weight must be 1-D with length == x.shape[1]");
  Arr out(std::vector<py::ssize_t>{xb.shape[0], xb.shape[1]});
  infer_lab::rms_norm_f32(static_cast<const float*>(xb.ptr),
                          static_cast<const float*>(wb.ptr),
                          static_cast<float*>(out.request().ptr),
                          static_cast<std::size_t>(xb.shape[0]),
                          static_cast<std::size_t>(xb.shape[1]), eps);
  return out;
}

static Arr softmax(Arr x) {
  auto xb = x.request();
  if (xb.ndim != 2) throw std::runtime_error("x must be 2-D (rows, cols)");
  Arr out(std::vector<py::ssize_t>{xb.shape[0], xb.shape[1]});
  infer_lab::softmax_f32(static_cast<const float*>(xb.ptr),
                         static_cast<float*>(out.request().ptr),
                         static_cast<std::size_t>(xb.shape[0]),
                         static_cast<std::size_t>(xb.shape[1]));
  return out;
}

static Arr swiglu(Arr gate, Arr up) {
  auto gb = gate.request(), ub = up.request();
  if (gb.size != ub.size) throw std::runtime_error("gate and up must have equal size");
  Arr out(gb.shape);
  infer_lab::swiglu_f32(static_cast<const float*>(gb.ptr),
                        static_cast<const float*>(ub.ptr),
                        static_cast<float*>(out.request().ptr),
                        static_cast<std::size_t>(gb.size));
  return out;
}

PYBIND11_MODULE(_infer_lab_pybind, m) {
  m.doc() = "infer-lab fused kernels (pybind11 binding)";
  m.attr("binding") = "pybind11";
  m.def("rms_norm", &rms_norm, py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-5f);
  m.def("softmax", &softmax, py::arg("x"));
  m.def("swiglu", &swiglu, py::arg("gate"), py::arg("up"));
  m.def("noop", []() {}, "empty call -- measures pure binding overhead");
}
