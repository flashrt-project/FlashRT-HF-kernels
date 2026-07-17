// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "dequantize_fp4_sfa.cuh"
#include "gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm120.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_uint8_cuda(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must have dtype torch.uint8");
}

void check_fp16_cuda(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype torch.float16");
}

void check_bf16_cuda(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

int64_t swizzled_bytes(int64_t rows, int64_t dim) {
  TORCH_CHECK(rows > 0 && dim > 0 && dim % 16 == 0,
              "rows must be positive and dim must be positive/divisible by 16");
  const int64_t n_blocks = dim / 16;
  const int64_t n_row_super = (rows + 127) / 128;
  const int64_t n_col_super = (n_blocks + 3) / 4;
  return n_row_super * n_col_super * 512;
}

void check_same_device(torch::Tensor const& a, torch::Tensor const& b,
                       const char* a_name, const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

struct GemmShape {
  int64_t m;
  int64_t n;
  int64_t k;
};

GemmShape check_fp4_gemm_inputs(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb) {
  check_uint8_cuda(a_packed, "a_packed");
  check_uint8_cuda(b_packed, "b_packed");
  check_uint8_cuda(sfa, "sfa");
  check_uint8_cuda(sfb, "sfb");
  TORCH_CHECK(a_packed.dim() == 2, "a_packed must have shape (M, K / 2)");
  TORCH_CHECK(b_packed.dim() == 2, "b_packed must have shape (N, K / 2)");
  const int64_t m = a_packed.size(0);
  const int64_t n = b_packed.size(0);
  const int64_t k_half = a_packed.size(1);
  TORCH_CHECK(m > 0 && n > 0 && k_half > 0, "M, N, and K must be positive");
  TORCH_CHECK(b_packed.size(1) == k_half,
              "a_packed and b_packed must have the same K / 2 dimension");
  const int64_t k = k_half * 2;
  TORCH_CHECK(k % 16 == 0, "K must be divisible by 16");
  TORCH_CHECK(sfa.numel() >= swizzled_bytes(m, k),
              "sfa is too small for CUTLASS SFA layout");
  TORCH_CHECK(sfb.numel() >= swizzled_bytes(n, k),
              "sfb is too small for CUTLASS SFB layout");
  check_same_device(a_packed, b_packed, "a_packed", "b_packed");
  check_same_device(a_packed, sfa, "a_packed", "sfa");
  check_same_device(a_packed, sfb, "a_packed", "sfb");
  return {m, n, k};
}

}  // namespace

void fp4_w4a16_linear_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha,
    int64_t variant) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_bf16_cuda(out, "out");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "out must have shape (M, N)");
  TORCH_CHECK(variant >= 0 && variant <= 2,
              "variant must be 0(default), 1(widen), or 2(pingpong)");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  if (variant == 1) {
    flash_rt::gemm::fp4_w4a16_gemm_sm120_bf16out_widen(
        a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
        checked_int(shape.m, "M"), checked_int(shape.n, "N"), checked_int(shape.k, "K"),
        sfa.data_ptr(), sfb.data_ptr(), static_cast<float>(alpha), stream);
  } else if (variant == 2) {
    flash_rt::gemm::fp4_w4a16_gemm_sm120_bf16out_pingpong(
        a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
        checked_int(shape.m, "M"), checked_int(shape.n, "N"), checked_int(shape.k, "K"),
        sfa.data_ptr(), sfb.data_ptr(), static_cast<float>(alpha), stream);
  } else {
    flash_rt::gemm::fp4_w4a16_gemm_sm120_bf16out(
        a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
        checked_int(shape.m, "M"), checked_int(shape.n, "N"), checked_int(shape.k, "K"),
        sfa.data_ptr(), sfb.data_ptr(), static_cast<float>(alpha), stream);
  }
#endif
}

void quantize_fp4_sfa_fp16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    bool is_sfb) {
  check_fp16_cuda(x, "x");
  check_uint8_cuda(packed, "packed");
  check_uint8_cuda(sfa, "sfa");
  TORCH_CHECK(x.dim() == 2, "x must have shape (rows, dim)");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  TORCH_CHECK(dim % 16 == 0, "x.shape[1] must be divisible by 16");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({rows, dim / 2}),
              "packed must have shape (rows, dim / 2)");
  TORCH_CHECK(sfa.numel() >= swizzled_bytes(rows, dim),
              "sfa is too small for CUTLASS SFA/SFB layout");
  check_same_device(x, packed, "x", "packed");
  check_same_device(x, sfa, "x", "sfa");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(
      x.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
      checked_int(rows, "rows"), checked_int(dim, "dim"), is_sfb, stream);
  TORCH_CHECK(rc == 0, "quantize_fp4_dynamic_sfa_fp16 failed with rc=", rc);
#endif
}

void dequantize_fp4_sfa_fp16(
    torch::Tensor const& packed,
    torch::Tensor const& sfa,
    torch::Tensor& out,
    bool is_sfb) {
  check_uint8_cuda(packed, "packed");
  check_uint8_cuda(sfa, "sfa");
  check_fp16_cuda(out, "out");
  TORCH_CHECK(out.dim() == 2, "out must have shape (rows, dim)");
  const int64_t rows = out.size(0);
  const int64_t dim = out.size(1);
  TORCH_CHECK(dim % 16 == 0, "out.shape[1] must be divisible by 16");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({rows, dim / 2}),
              "packed must have shape (rows, dim / 2)");
  TORCH_CHECK(sfa.numel() >= swizzled_bytes(rows, dim),
              "sfa is too small for CUTLASS SFA layout");
  check_same_device(packed, sfa, "packed", "sfa");
  check_same_device(packed, out, "packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed.get_device()).stream();
  flash_rt::fused_fp4::dequantize_fp4_sfa_fp16(
      reinterpret_cast<const uint8_t*>(packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
      reinterpret_cast<__half*>(out.data_ptr()),
      checked_int(rows, "rows"), checked_int(dim, "dim"), is_sfb, stream);
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("nvfp4_gemm_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int variant=0) -> ()");
  ops.def("fp4_w4a16_linear_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int variant=0) -> ()");
  ops.def("quantize_fp4_sfa_fp16(Tensor x, Tensor! packed, Tensor! sfa, bool is_sfb=False) -> ()");
  ops.def("dequantize_fp4_sfa_fp16(Tensor packed, Tensor sfa, Tensor! out, bool is_sfb=False) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("nvfp4_gemm_bf16", torch::kCUDA, &fp4_w4a16_linear_bf16);
  ops.impl("fp4_w4a16_linear_bf16", torch::kCUDA, &fp4_w4a16_linear_bf16);
  ops.impl("quantize_fp4_sfa_fp16", torch::kCUDA, &quantize_fp4_sfa_fp16);
  ops.impl("dequantize_fp4_sfa_fp16", torch::kCUDA, &dequantize_fp4_sfa_fp16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
