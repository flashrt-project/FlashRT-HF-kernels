// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "gemm/bf16_gemv_m1_sm120.cuh"
#include "kernels/nexn2_bf16_gemv.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

using GemvFn = int (*)(const void*, const void*, void*, int, int, int, float, cudaStream_t);

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

int checked_positive_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

void check_common(torch::Tensor const& x, torch::Tensor const& weight, torch::Tensor const& out) {
  check_bf16(x, "x");
  check_bf16(weight, "weight");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 1 || (x.dim() == 2 && x.size(0) == 1),
              "x must have shape (K,) or (1,K)");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape (N,K)");
  const int64_t k = x.dim() == 1 ? x.size(0) : x.size(1);
  TORCH_CHECK(weight.size(1) == k, "x and weight K mismatch");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({weight.size(0)}),
              "out must have shape (N,)");
  TORCH_CHECK(k % 8 == 0, "K must be divisible by 8");
  TORCH_CHECK(x.get_device() == weight.get_device(), "x and weight must be on same CUDA device");
  TORCH_CHECK(x.get_device() == out.get_device(), "x and out must be on same CUDA device");
}

GemvFn select_variant(int64_t variant, int64_t n) {
#if defined(CUDA_KERNEL)
  namespace gemv = flash_rt::gemm::gemv_m1;
  if (variant == 4) return gemv::gemv_bf16_m1_w4;
  if (variant == 8) return gemv::gemv_bf16_m1_w8;
  if (variant == 16) return gemv::gemv_bf16_m1_w16;
  TORCH_CHECK(variant == 0, "variant must be 0, 4, 8, or 16");
  if (n <= 2048) return gemv::gemv_bf16_m1_w4;
  if (n <= 8192) return gemv::gemv_bf16_m1_w8;
  return gemv::gemv_bf16_m1_w16;
#else
  (void)variant;
  (void)n;
  return nullptr;
#endif
}

}  // namespace

void bf16_decode_gemv_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    double alpha,
    int64_t variant,
    torch::Tensor& out) {
  check_common(x, weight, out);
  const int n = checked_positive_int(weight.size(0), "N");
  const int k = checked_positive_int(weight.size(1), "K");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  GemvFn fn = select_variant(variant, n);
  const int rc = fn(
      x.data_ptr(), weight.data_ptr(), out.data_ptr(),
      1, n, k, static_cast<float>(alpha), stream);
  TORCH_CHECK(rc == 0, "bf16_decode_gemv_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "bf16-linear-gemv was not built with CUDA support");
#endif
}

void bf16_decode_gemv_unrolled_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    torch::Tensor& out) {
  check_common(x, weight, out);
  const int n = checked_positive_int(weight.size(0), "N");
  const int k = checked_positive_int(weight.size(1), "K");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::kernels::nexn2_bf16_matvec_bf16(
      x.data_ptr(), weight.data_ptr(), out.data_ptr(), n, k, stream);
  TORCH_CHECK(rc == 0, "bf16_decode_gemv_unrolled_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "bf16-linear-gemv was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("bf16_decode_gemv_bf16(Tensor x, Tensor weight, float alpha, int variant, Tensor! out) -> ()");
  ops.def("bf16_decode_gemv_unrolled_bf16(Tensor x, Tensor weight, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("bf16_decode_gemv_bf16", torch::kCUDA, &bf16_decode_gemv_bf16);
  ops.impl("bf16_decode_gemv_unrolled_bf16", torch::kCUDA, &bf16_decode_gemv_unrolled_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
