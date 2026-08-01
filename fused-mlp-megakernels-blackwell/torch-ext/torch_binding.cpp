// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#endif

#include "registration.h"
#include "torch_binding.h"

#if defined(CUDA_KERNEL)
extern "C" int flashrt_megakernel_geglu_fp16(
    void*, void*, void*, void*, void*, int, int, int, cudaStream_t);
#endif

namespace {
void check_fp16_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype torch.float16");
}
}  // namespace

void fp16_geglu_fused_out(
    torch::Tensor const& input,
    torch::Tensor const& gate_weight,
    torch::Tensor const& up_weight,
    torch::Tensor& gate_scratch,
    torch::Tensor& output) {
  check_fp16_cuda_contiguous(input, "input");
  check_fp16_cuda_contiguous(gate_weight, "gate_weight");
  check_fp16_cuda_contiguous(up_weight, "up_weight");
  check_fp16_cuda_contiguous(gate_scratch, "gate_scratch");
  check_fp16_cuda_contiguous(output, "output");
  TORCH_CHECK(input.dim() == 2 && gate_weight.dim() == 2 &&
                  up_weight.dim() == 2,
              "input and weights must be rank-2 tensors");
  TORCH_CHECK(gate_weight.sizes() == up_weight.sizes(),
              "gate_weight and up_weight must have the same [N,K] shape");
  TORCH_CHECK(input.size(1) == gate_weight.size(1),
              "input K must match weight K");
  TORCH_CHECK(output.sizes() ==
                  torch::IntArrayRef({input.size(0), gate_weight.size(0)}),
              "output must have shape [M,N]");
  TORCH_CHECK(gate_scratch.sizes() == output.sizes(),
              "gate_scratch must have shape [M,N]");
  TORCH_CHECK(input.get_device() == gate_weight.get_device() &&
                  input.get_device() == up_weight.get_device() &&
                  input.get_device() == gate_scratch.get_device() &&
                  input.get_device() == output.get_device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(input.size(0) > 0 && gate_weight.size(0) > 0 && input.size(1) > 0,
              "M, N, and K must be positive");

#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  const int capability = properties->major * 10 + properties->minor;
  TORCH_CHECK(capability == 100 || capability == 103 || capability == 110,
              "fp16_geglu_fused_out requires SM100, SM103, or SM110");
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  int rc = flashrt_megakernel_geglu_fp16(
      input.data_ptr(), gate_weight.data_ptr(), up_weight.data_ptr(),
      gate_scratch.data_ptr(), output.data_ptr(),
      static_cast<int>(input.size(0)), static_cast<int>(gate_weight.size(0)),
      static_cast<int>(input.size(1)), stream);
  TORCH_CHECK(rc == 0, "FlashRT FP16 GeGLU megakernel failed with status ", rc);
#else
  TORCH_CHECK(false, "CUDA support was not built");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp16_geglu_fused_out(Tensor input, Tensor gate_weight, Tensor "
          "up_weight, Tensor! gate_scratch, Tensor! output) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fp16_geglu_fused_out", torch::kCUDA, &fp16_geglu_fused_out);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
