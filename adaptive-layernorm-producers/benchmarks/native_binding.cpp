// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "adaln_modulation6.cuh"

void adaln_modulation6(
    uintptr_t params, uintptr_t modulation, uintptr_t out0, uintptr_t out1,
    uintptr_t out2, uintptr_t out3, uintptr_t out4, uintptr_t out5,
    int batch, int sequence, int dim) {
  flash_rt::adaln_producers::adaln_modulation6_bf16(
      reinterpret_cast<const float*>(params),
      reinterpret_cast<const float*>(modulation),
      reinterpret_cast<void*>(out0), reinterpret_cast<void*>(out1),
      reinterpret_cast<void*>(out2), reinterpret_cast<void*>(out3),
      reinterpret_cast<void*>(out4), reinterpret_cast<void*>(out5),
      batch, sequence, dim, at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("adaln_modulation6", &adaln_modulation6);
}
