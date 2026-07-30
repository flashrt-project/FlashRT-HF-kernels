// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "padded_fp8_producers.cuh"

namespace py = pybind11;

void adaptive(
    uintptr_t input, uintptr_t weight, uintptr_t gamma, uintptr_t beta,
    uintptr_t scale, uintptr_t output, int batch, int rows, int padded_rows,
    int dim, float eps) {
  flash_rt::padded_fp8::adaptive_rms_norm_quant_fp8_padded_bf16(
      reinterpret_cast<const __nv_bfloat16*>(input),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<const __nv_bfloat16*>(gamma),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<__nv_fp8_e4m3*>(output), batch, rows, padded_rows, dim,
      eps, at::cuda::getDefaultCUDAStream().stream());
}

void swiglu(
    uintptr_t gate, uintptr_t up, uintptr_t scale, uintptr_t output,
    int rows, int padded_rows, int dim) {
  flash_rt::padded_fp8::swiglu_quant_fp8_padded_bf16(
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<const __nv_bfloat16*>(up),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<__nv_fp8_e4m3*>(output), rows, padded_rows, dim,
      at::cuda::getDefaultCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("adaptive", &adaptive);
  module.def("swiglu", &swiglu);
}
