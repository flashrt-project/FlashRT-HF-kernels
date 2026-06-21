// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "kernels/nexn2_moe_grouped_w4a16.cuh"
#include "kernels/nexn2_w4a16_gemv.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be torch.bfloat16");
}

void check_u8(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kUInt8, name, " must be torch.uint8");
}

void check_f32(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be torch.float32");
}

void check_i32(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be torch.int32");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

long checked_long(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0, name, " must be non-negative");
  return static_cast<long>(value);
}

}  // namespace

void w4a16_decode_gemv_bf16(torch::Tensor const& x_bf16,
                            torch::Tensor const& weight_packed,
                            torch::Tensor const& sfb,
                            double alpha,
                            torch::Tensor& out) {
  check_bf16(x_bf16, "x_bf16");
  check_u8(weight_packed, "weight_packed");
  check_u8(sfb, "sfb");
  check_bf16(out, "out");
  TORCH_CHECK(x_bf16.dim() == 1 || (x_bf16.dim() == 2 && x_bf16.size(0) == 1),
              "x_bf16 must have shape (K,) or (1,K)");
  const int64_t k = x_bf16.dim() == 1 ? x_bf16.size(0) : x_bf16.size(1);
  TORCH_CHECK(weight_packed.dim() == 2 && weight_packed.size(1) == k / 2,
              "weight_packed must have shape (N,K/2)");
  TORCH_CHECK(k % 16 == 0, "K must be divisible by 16");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({weight_packed.size(0)}), "out shape mismatch");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x_bf16.device());
  auto stream = at::cuda::getCurrentCUDAStream(x_bf16.get_device()).stream();
  const int rc = flash_rt::kernels::nexn2_w4a16_matvec_bf16(
      x_bf16.data_ptr(), weight_packed.data_ptr(), sfb.data_ptr(), out.data_ptr(),
      checked_int(weight_packed.size(0), "N"), checked_int(k, "K"),
      static_cast<float>(alpha), stream);
  TORCH_CHECK(rc == 0, "w4a16_decode_gemv_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "grouped-moe-gemv was not built with CUDA support");
#endif
}

void grouped_w4a16_gemv_bf16(torch::Tensor const& activations,
                             torch::Tensor const& weight_stack,
                             torch::Tensor const& sfb_stack,
                             torch::Tensor const& alpha_stack,
                             torch::Tensor const& expert_idx,
                             int64_t w_stride,
                             int64_t sfb_stride,
                             torch::Tensor& out) {
  check_bf16(activations, "activations");
  check_u8(weight_stack, "weight_stack");
  check_u8(sfb_stack, "sfb_stack");
  check_f32(alpha_stack, "alpha_stack");
  check_i32(expert_idx, "expert_idx");
  check_bf16(out, "out");
  TORCH_CHECK(activations.dim() == 2, "activations must have shape (slots,K)");
  TORCH_CHECK(weight_stack.dim() >= 2, "weight_stack must be a flat or 3D uint8 stack");
  const int64_t slots = activations.size(0);
  const int64_t k = activations.size(1);
  const int64_t n = out.size(1);
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({slots, n}), "out must have shape (slots,N)");
  TORCH_CHECK(expert_idx.sizes() == torch::IntArrayRef({slots}), "expert_idx must have shape (slots,)");
  TORCH_CHECK(k % 16 == 0, "K must be divisible by 16");
  TORCH_CHECK(w_stride > 0 && sfb_stride > 0, "w_stride and sfb_stride must be positive byte strides");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(activations.device());
  auto stream = at::cuda::getCurrentCUDAStream(activations.get_device()).stream();
  const int rc = flash_rt::kernels::nexn2_moe_grouped_w4a16_bf16(
      activations.data_ptr(), weight_stack.data_ptr(), sfb_stack.data_ptr(),
      alpha_stack.data_ptr(), expert_idx.data_ptr(), out.data_ptr(),
      checked_int(slots, "slots"), checked_int(n, "N"), checked_int(k, "K"),
      checked_long(k, "a_stride"), checked_long(w_stride, "w_stride"),
      checked_long(sfb_stride, "sfb_stride"), stream);
  TORCH_CHECK(rc == 0, "grouped_w4a16_gemv_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "grouped-moe-gemv was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("w4a16_decode_gemv_bf16(Tensor x_bf16, Tensor weight_packed, Tensor sfb, float alpha, Tensor! out) -> ()");
  ops.def("grouped_w4a16_gemv_bf16(Tensor activations, Tensor weight_stack, Tensor sfb_stack, Tensor alpha_stack, Tensor expert_idx, int w_stride, int sfb_stride, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("w4a16_decode_gemv_bf16", torch::kCUDA, &w4a16_decode_gemv_bf16);
  ops.impl("grouped_w4a16_gemv_bf16", torch::kCUDA, &grouped_w4a16_gemv_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
