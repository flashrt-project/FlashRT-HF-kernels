// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>
#include <sstream>
#include <string>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "fp8_gemv_m1_sm120.cuh"
#include "fp8_smallM_handtuned_ldmatrix_sm120.cuh"
#include "fp8_smallM_handtuned_sm120.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

using KernelFn = int (*)(const void*, const void*, void*, int, int, int, float, cudaStream_t);

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_fp8_matrix(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (rows, cols)");
  TORCH_CHECK(tensor.size(0) > 0 && tensor.size(1) > 0,
              name, " dimensions must be positive");
}

void check_bf16_matrix(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (rows, cols)");
}

int checked_positive_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

void check_common(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& out) {
  check_fp8_matrix(input, "input");
  check_fp8_matrix(weight, "weight");
  check_bf16_matrix(out, "out");
  TORCH_CHECK(input.get_device() == weight.get_device(),
              "input and weight must be on the same CUDA device");
  TORCH_CHECK(input.get_device() == out.get_device(),
              "input and out must be on the same CUDA device");
  TORCH_CHECK(input.size(1) == weight.size(1),
              "input.shape[1] must equal weight.shape[1]");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({input.size(0), weight.size(0)}),
              "out must have shape (input.shape[0], weight.shape[0])");
  TORCH_CHECK(input.size(1) % 32 == 0,
              "K must be divisible by 32 for FP8 tensor-core/GEMV kernels");
  TORCH_CHECK(input.size(0) == 1 || input.size(0) <= 64,
              "only M=1 decode or 2 <= M <= 64 small-M rows are supported");
}

std::string tile_name_for_shape(int M, int N, int K, int variant) {
  if (M == 1) {
    if (variant == 4) return "gemv_fp8_m1_w4";
    if (variant == 8) return "gemv_fp8_m1_w8";
    if (variant == 16) return "gemv_fp8_m1_w16";
    TORCH_CHECK(variant == 0, "M=1 variant must be 0, 4, 8, or 16");
    if (N <= 2048) return "gemv_fp8_m1_w4";
    if (N <= 8192) return "gemv_fp8_m1_w8";
    return "gemv_fp8_m1_w16";
  }

  TORCH_CHECK(variant == 0,
              "small-M public dispatcher currently supports variant=0 only; "
              "use benchmark scripts for tile sweeps before promoting a forced variant");
  if (M <= 16) {
    if (K % 256 == 0) {
      if (N % 128 == 0) return "ld_fp8_gemm_16x128x256_w4";
      return "ld_fp8_gemm_16x64x256_w4";
    }
    if (N % 256 == 0) return "ld_fp8_gemm_16x256x128_w8";
    if (N % 192 == 0) return "ld_fp8_gemm_16x192x128_w4";
    if (N % 128 == 0) return "ld_fp8_gemm_16x128x128_w4";
    return "ld_fp8_gemm_16x64x128_w4";
  }
  if (M <= 32) {
    if (K % 256 == 0) {
      if (N % 128 == 0) return "ld_fp8_gemm_32x128x256_w4";
      return "ld_fp8_gemm_32x64x256_w4";
    }
    if (N % 192 == 0) return "ld_fp8_gemm_32x192x128_w4";
    if (N % 128 == 0) return "ld_fp8_gemm_32x128x128_w4";
    return "ld_fp8_gemm_32x64x128_w4";
  }
  if (M <= 64) {
    if (K % 256 == 0) {
      if (N % 128 == 0) return "ld_fp8_gemm_64x128x256_w4";
      return "ld_fp8_gemm_64x64x256_w4";
    }
    if (N % 128 == 0) return "ld_fp8_gemm_64x128x128_w4";
    return "ld_fp8_gemm_64x64x128_w4";
  }
  TORCH_CHECK(false, "M > 64 is not exposed in fp8-gemm v1; pending tile tuning");
  TORCH_CHECK(false, "unsupported M");
}

KernelFn kernel_for_tile(std::string const& tile, bool residual) {
#if defined(CUDA_KERNEL)
  namespace gemv = flash_rt::gemm::gemv_m1;
  namespace hand = flash_rt::gemm::smallM_hand;
  namespace ld = flash_rt::gemm::smallM_ld;
  if (tile == "gemv_fp8_m1_w4") return residual ? gemv::gemv_fp8_m1_resadd_w4 : gemv::gemv_fp8_m1_w4;
  if (tile == "gemv_fp8_m1_w8") return residual ? gemv::gemv_fp8_m1_resadd_w8 : gemv::gemv_fp8_m1_w8;
  if (tile == "gemv_fp8_m1_w16") {
    TORCH_CHECK(!residual, "residual path supports only GEMV w4/w8 variants");
    return gemv::gemv_fp8_m1_w16;
  }
  TORCH_CHECK(!residual, "residual path supports M=1 GEMV only");
  if (tile == "ld_fp8_gemm_16x64x128_w4") return ld::ld_fp8_gemm_16x64x128_w4;
  if (tile == "ld_fp8_gemm_16x128x128_w4") return ld::ld_fp8_gemm_16x128x128_w4;
  if (tile == "ld_fp8_gemm_16x256x128_w8") return ld::ld_fp8_gemm_16x256x128_w8;
  if (tile == "ld_fp8_gemm_16x192x128_w4") return ld::ld_fp8_gemm_16x192x128_w4;
  if (tile == "ld_fp8_gemm_16x64x256_w4") return ld::ld_fp8_gemm_16x64x256_w4;
  if (tile == "ld_fp8_gemm_16x128x256_w4") return ld::ld_fp8_gemm_16x128x256_w4;
  if (tile == "ld_fp8_gemm_32x64x128_w4") return ld::ld_fp8_gemm_32x64x128_w4;
  if (tile == "ld_fp8_gemm_32x128x128_w4") return ld::ld_fp8_gemm_32x128x128_w4;
  if (tile == "ld_fp8_gemm_32x192x128_w4") return ld::ld_fp8_gemm_32x192x128_w4;
  if (tile == "ld_fp8_gemm_32x64x256_w4") return ld::ld_fp8_gemm_32x64x256_w4;
  if (tile == "ld_fp8_gemm_32x128x256_w4") return ld::ld_fp8_gemm_32x128x256_w4;
  if (tile == "ld_fp8_gemm_64x64x128_w4") return ld::ld_fp8_gemm_64x64x128_w4;
  if (tile == "ld_fp8_gemm_64x128x128_w4") return ld::ld_fp8_gemm_64x128x128_w4;
  if (tile == "ld_fp8_gemm_64x64x256_w4") return ld::ld_fp8_gemm_64x64x256_w4;
  if (tile == "ld_fp8_gemm_64x128x256_w4") return ld::ld_fp8_gemm_64x128x256_w4;
#else
  (void)tile;
  (void)residual;
#endif
  TORCH_CHECK(false, "unsupported FP8 GEMM tile: ", tile);
}

void launch(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    double alpha,
    int64_t variant64,
    torch::Tensor& out,
    bool residual) {
  check_common(input, weight, out);
  const int M = checked_positive_int(input.size(0), "M");
  const int K = checked_positive_int(input.size(1), "K");
  const int N = checked_positive_int(weight.size(0), "N");
  const int variant = static_cast<int>(variant64);
  if (residual) {
    TORCH_CHECK(M == 1, "fp8_linear_residual_bf16 supports only M=1");
  }
  const std::string tile = tile_name_for_shape(M, N, K, variant);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  KernelFn fn = kernel_for_tile(tile, residual);
  const int rc = fn(
      input.data_ptr(),
      weight.data_ptr(),
      out.data_ptr(),
      M,
      N,
      K,
      static_cast<float>(alpha),
      stream);
  TORCH_CHECK(rc == 0, tile, " failed with rc=", rc);
#else
  TORCH_CHECK(false, "fp8-gemm was not built with CUDA support");
#endif
}

}  // namespace

void fp8_linear_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    double alpha,
    int64_t variant,
    torch::Tensor& out) {
  launch(input, weight, alpha, variant, out, false);
}

void fp8_linear_residual_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    double alpha,
    int64_t variant,
    torch::Tensor& residual) {
  launch(input, weight, alpha, variant, residual, true);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp8_linear_bf16(Tensor input, Tensor weight, float alpha, int variant, Tensor! out) -> ()");
  ops.def("fp8_linear_residual_bf16(Tensor input, Tensor weight, float alpha, int variant, Tensor! residual) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fp8_linear_bf16", torch::kCUDA, &fp8_linear_bf16);
  ops.impl("fp8_linear_residual_bf16", torch::kCUDA, &fp8_linear_residual_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
