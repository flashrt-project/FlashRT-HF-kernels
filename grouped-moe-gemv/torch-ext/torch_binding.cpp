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
#include "kernels/grouped_w4a4_gemv_sm120.cuh"
#include "kernels/quantize_activations_nvfp4.cuh"
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

int64_t swizzled_bytes(int64_t rows, int64_t dim) {
  const int64_t row_supers = (rows + 127) / 128;
  const int64_t col_supers = ((dim / 16) + 3) / 4;
  return row_supers * col_supers * 512;
}

void check_same_device(torch::Tensor const& reference,
                       torch::Tensor const& value,
                       const char* name) {
  TORCH_CHECK(reference.device() == value.device(), name, " must be on ",
              reference.device());
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

void quantize_activations_nvfp4_bf16(torch::Tensor const& activations,
                                     torch::Tensor& packed,
                                     torch::Tensor& sfa) {
  check_bf16(activations, "activations");
  check_u8(packed, "packed");
  check_u8(sfa, "sfa");
  TORCH_CHECK(activations.dim() == 2, "activations must have shape (M,K)");
  const int64_t m = activations.size(0);
  const int64_t k = activations.size(1);
  TORCH_CHECK(k % 16 == 0, "K must be divisible by 16");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({m, k / 2}),
              "packed must have shape (M,K/2)");
  TORCH_CHECK(sfa.numel() >= swizzled_bytes(m, k),
              "sfa is too small for the CUTLASS SFA layout");
  check_same_device(activations, packed, "packed");
  check_same_device(activations, sfa, "sfa");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(activations.device());
  auto stream = at::cuda::getCurrentCUDAStream(activations.get_device()).stream();
  const int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_bf16(
      activations.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
      checked_int(m, "M"), checked_int(k, "K"), false, stream);
  TORCH_CHECK(rc == 0, "quantize_activations_nvfp4_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "grouped-moe-gemv was not built with CUDA support");
#endif
}

void quantize_weights_nvfp4_bf16(torch::Tensor const& weights,
                                 torch::Tensor& packed,
                                 torch::Tensor& sfb) {
  check_bf16(weights, "weights");
  check_u8(packed, "packed");
  check_u8(sfb, "sfb");
  TORCH_CHECK(weights.dim() == 2, "weights must have shape (N,K)");
  const int64_t n = weights.size(0);
  const int64_t k = weights.size(1);
  TORCH_CHECK(k % 16 == 0, "K must be divisible by 16");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({n, k / 2}),
              "packed must have shape (N,K/2)");
  TORCH_CHECK(sfb.numel() >= swizzled_bytes(n, k),
              "sfb is too small for the CUTLASS SFB layout");
  check_same_device(weights, packed, "packed");
  check_same_device(weights, sfb, "sfb");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(weights.device());
  auto stream = at::cuda::getCurrentCUDAStream(weights.get_device()).stream();
  const int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_bf16(
      weights.data_ptr(), packed.data_ptr(), sfb.data_ptr(),
      checked_int(n, "N"), checked_int(k, "K"), true, stream);
  TORCH_CHECK(rc == 0, "quantize_weights_nvfp4_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "grouped-moe-gemv was not built with CUDA support");
#endif
}

void grouped_w4a4_gemv_bf16(torch::Tensor const& activations_packed,
                             torch::Tensor const& weight_stack,
                             torch::Tensor const& sfa,
                             torch::Tensor const& sfb_stack,
                             torch::Tensor const& alpha_stack,
                             torch::Tensor const& expert_idx,
                             torch::Tensor& out) {
  check_u8(activations_packed, "activations_packed");
  check_u8(weight_stack, "weight_stack");
  check_u8(sfa, "sfa");
  check_u8(sfb_stack, "sfb_stack");
  check_f32(alpha_stack, "alpha_stack");
  check_i32(expert_idx, "expert_idx");
  check_bf16(out, "out");
  TORCH_CHECK(activations_packed.dim() == 2,
              "activations_packed must have shape (M,K/2)");
  TORCH_CHECK(weight_stack.dim() == 3,
              "weight_stack must have shape (E,N,K/2)");
  TORCH_CHECK(expert_idx.dim() == 2,
              "expert_idx must have shape (M,top_k)");
  const int64_t m = activations_packed.size(0);
  const int64_t k_half = activations_packed.size(1);
  const int64_t k = k_half * 2;
  const int64_t experts = weight_stack.size(0);
  const int64_t n = weight_stack.size(1);
  const int64_t top_k = expert_idx.size(1);
  TORCH_CHECK(k % 16 == 0, "K must be divisible by 16");
  TORCH_CHECK(weight_stack.size(2) == k_half,
              "weight_stack K does not match activations_packed");
  TORCH_CHECK(expert_idx.size(0) == m,
              "expert_idx first dimension must equal M");
  TORCH_CHECK(alpha_stack.numel() == experts,
              "alpha_stack must have shape (E,)");
  TORCH_CHECK(sfa.numel() >= swizzled_bytes(m, k),
              "sfa is too small for the CUTLASS SFA layout");
  TORCH_CHECK(sfb_stack.dim() == 2 && sfb_stack.size(0) == experts &&
                  sfb_stack.size(1) >= swizzled_bytes(n, k),
              "sfb_stack must have shape (E,sfb_bytes) with sufficient storage");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({m, top_k, n}),
              "out must have shape (M,top_k,N)");
  check_same_device(activations_packed, weight_stack, "weight_stack");
  check_same_device(activations_packed, sfa, "sfa");
  check_same_device(activations_packed, sfb_stack, "sfb_stack");
  check_same_device(activations_packed, alpha_stack, "alpha_stack");
  check_same_device(activations_packed, expert_idx, "expert_idx");
  check_same_device(activations_packed, out, "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(activations_packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(activations_packed.get_device()).stream();
  const int rc = flash_rt::gemm::grouped_w4a4_gemv_sm120_bf16(
      activations_packed.data_ptr(), weight_stack.data_ptr(), out.data_ptr(),
      sfa.data_ptr(), sfb_stack.data_ptr(), alpha_stack.data_ptr(),
      expert_idx.data_ptr(), checked_int(m, "M"), checked_int(top_k, "top_k"),
      checked_int(n, "N"), checked_int(k, "K"),
      checked_long(weight_stack.size(1) * weight_stack.size(2), "w_stride"),
      checked_long(sfb_stack.size(1), "sfb_stride"), stream);
  TORCH_CHECK(rc == 0, "grouped_w4a4_gemv_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "grouped-moe-gemv was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("w4a16_decode_gemv_bf16(Tensor x_bf16, Tensor weight_packed, Tensor sfb, float alpha, Tensor! out) -> ()");
  ops.def("grouped_w4a16_gemv_bf16(Tensor activations, Tensor weight_stack, Tensor sfb_stack, Tensor alpha_stack, Tensor expert_idx, int w_stride, int sfb_stride, Tensor! out) -> ()");
  ops.def("quantize_activations_nvfp4_bf16(Tensor activations, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("quantize_weights_nvfp4_bf16(Tensor weights, Tensor! packed, Tensor! sfb) -> ()");
  ops.def("grouped_w4a4_gemv_bf16(Tensor activations_packed, Tensor weight_stack, Tensor sfa, Tensor sfb_stack, Tensor alpha_stack, Tensor expert_idx, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("w4a16_decode_gemv_bf16", torch::kCUDA, &w4a16_decode_gemv_bf16);
  ops.impl("grouped_w4a16_gemv_bf16", torch::kCUDA, &grouped_w4a16_gemv_bf16);
  ops.impl("quantize_activations_nvfp4_bf16", torch::kCUDA, &quantize_activations_nvfp4_bf16);
  ops.impl("quantize_weights_nvfp4_bf16", torch::kCUDA, &quantize_weights_nvfp4_bf16);
  ops.impl("grouped_w4a4_gemv_bf16", torch::kCUDA, &grouped_w4a4_gemv_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
