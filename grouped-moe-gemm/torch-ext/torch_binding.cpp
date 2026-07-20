// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include <torch/library.h>
#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "moe_blocktile_mma_sm120.cuh"
#include "moe_m16_mma_sm120.cuh"
#include "moe_m64_mma_sm120.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {
void check(torch::Tensor const &t, c10::ScalarType dtype, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name,
              " must be contiguous CUDA");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has incorrect dtype");
}
} // namespace

void grouped_nvfp4_gemm_bf16_out(
    torch::Tensor const &input, torch::Tensor const &weight,
    torch::Tensor const &input_scale, torch::Tensor const &weight_scale,
    torch::Tensor const &alpha, torch::Tensor const &tile_expert,
    int64_t tile_rows, int64_t input_scale_stride, int64_t weight_stride,
    int64_t weight_scale_stride, torch::Tensor &output) {
  check(input, torch::kUInt8, "input");
  check(weight, torch::kUInt8, "weight");
  check(input_scale, torch::kUInt8, "input_scale");
  check(weight_scale, torch::kUInt8, "weight_scale");
  check(alpha, torch::kFloat32, "alpha");
  check(tile_expert, torch::kInt32, "tile_expert");
  check(output, torch::kBFloat16, "output");
  TORCH_CHECK(tile_rows == 16 || tile_rows == 64, "tile_rows must be 16 or 64");
  TORCH_CHECK(
      input.dim() == 2 && weight.dim() == 3 && output.dim() == 2,
      "input/output must be matrices and weight must be (experts,N,K/2)");
  int64_t num_tiles = tile_expert.numel(), K = input.size(1) * 2,
          N = weight.size(1);
  TORCH_CHECK(num_tiles > 0 && input.size(0) == num_tiles * tile_rows,
              "input rows must equal num_tiles * tile_rows");
  TORCH_CHECK(weight.size(2) * 2 == K && K % 64 == 0,
              "packed K mismatch or K not divisible by 64");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({input.size(0), N}),
              "output shape mismatch");
  TORCH_CHECK(alpha.numel() >= weight.size(0), "alpha must cover all experts");
  TORCH_CHECK(input_scale_stride >= 0 && weight_stride > 0 &&
                  weight_scale_stride > 0,
              "invalid byte strides");
  TORCH_CHECK(input.get_device() == output.get_device(),
              "all tensors must share one device");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  auto s = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  int rc;
  if (tile_rows == 16) {
    TORCH_CHECK(N % 8 == 0, "M16 path requires N divisible by 8");
    rc = flash_rt::gemm::moe_m16_mma_sm120_bf16(
        input.data_ptr(), weight.data_ptr(), input_scale.data_ptr(),
        weight_scale.data_ptr(), output.data_ptr(), alpha.data_ptr(),
        tile_expert.data_ptr(), num_tiles, N, K, input_scale_stride,
        weight_stride, weight_scale_stride, s);
  } else if (N % 64 == 0) {
    rc = flash_rt::gemm::moe_blocktile_mma_sm120_bf16(
        input.data_ptr(), weight.data_ptr(), input_scale.data_ptr(),
        weight_scale.data_ptr(), output.data_ptr(), alpha.data_ptr(),
        tile_expert.data_ptr(), num_tiles, N, K, input_scale_stride,
        weight_stride, weight_scale_stride, s);
  } else {
    TORCH_CHECK(N % 16 == 0, "M64 path requires N divisible by 16");
    rc = flash_rt::gemm::moe_m64_mma_sm120_bf16(
        input.data_ptr(), weight.data_ptr(), input_scale.data_ptr(),
        weight_scale.data_ptr(), output.data_ptr(), alpha.data_ptr(),
        tile_expert.data_ptr(), num_tiles, N, K, input_scale_stride,
        weight_stride, weight_scale_stride, s);
  }
  TORCH_CHECK(rc == 0, "grouped NVFP4 GEMM failed with rc=", rc);
#else
  TORCH_CHECK(false, "grouped-moe-gemm was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("grouped_nvfp4_gemm_bf16_out(Tensor input, Tensor weight, Tensor "
          "input_scale, Tensor weight_scale, Tensor alpha, Tensor tile_expert, "
          "int tile_rows, int input_scale_stride, int weight_stride, int "
          "weight_scale_stride, Tensor! output) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("grouped_nvfp4_gemm_bf16_out", torch::kCUDA,
           &grouped_nvfp4_gemm_bf16_out);
#endif
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
