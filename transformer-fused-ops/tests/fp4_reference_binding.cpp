// SPDX-License-Identifier: Apache-2.0
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include "quantize_fp4_sfa_bf16.cuh"

namespace {

void quantize(torch::Tensor const& x, torch::Tensor& packed,
              torch::Tensor& sfa) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                  x.scalar_type() == torch::kBFloat16,
              "x must be contiguous CUDA bfloat16 with shape (rows,dim)");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int status = flash_rt::fp4::quantize_fp4_dynamic_sfa_bf16_vec(
      x.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
      static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), false, stream);
  TORCH_CHECK(status == 0, "reference quantizer failed with status ", status);
}

}  // namespace

TORCH_LIBRARY(transformer_fused_fp4_reference, ops) {
  ops.def("quantize(Tensor x, Tensor! packed, Tensor! sfa) -> ()");
}

TORCH_LIBRARY_IMPL(transformer_fused_fp4_reference, CUDA, ops) {
  ops.impl("quantize", &quantize);
}
