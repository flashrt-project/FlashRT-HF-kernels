// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "dequantize_fp4_sfa.cuh"
#if !defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
#include "gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_bf16out_sm120.cuh"
#include "gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.cuh"
#include "gemm/fp4/cutlass_nvfp4_gemm_dn_streamk_bias_sm120.cuh"
#include "gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm120.cuh"
#include "gemm/fp4/fp4_w4a4_mma_warpsplit_sm120.cuh"
#endif
#include "gemm/fp4/sm110_dispatch.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "registration.h"
#include "torch_binding.h"

flash_rt::hub::Sm110GemmDispatch flash_rt::hub::sm110_gemm_dispatch = nullptr;

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

#if defined(CUDA_KERNEL)
cudaDeviceProp const* current_device_properties(torch::Tensor const& anchor) {
  return at::cuda::getDeviceProperties(anchor.get_device());
}

void require_sm120(torch::Tensor const& anchor, const char* operation) {
  auto const* props = current_device_properties(anchor);
  TORCH_CHECK(props->major == 12 && props->minor == 0,
              operation, " is an SM120 fused epilogue; got SM",
              props->major, props->minor,
              ". On SM110 use nvfp4_gemm_bf16 with fp4-fused-ops producers.");
}
#endif

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

int select_sm110_variant(GemmShape const& shape, int64_t requested) {
  if (requested >= 0) return static_cast<int>(requested);
  if (shape.n >= 4 * shape.k) return 1;
  if (shape.n == 3 * shape.k) return 2;
  return 0;
}

}  // namespace

void fp4_w4a4_gemv_warpsplit_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha,
    int64_t warps,
    int64_t stages) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_bf16_cuda(out, "out");
  TORCH_CHECK(shape.m == 1,
              "warp-split GEMV serves the M=1 decode row only");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "out must have shape (1, N)");
  TORCH_CHECK(warps == 2 || warps == 4 || warps == 8,
              "warps must be 2, 4 or 8");
  TORCH_CHECK(stages == 3 || stages == 4 || stages == 6,
              "stages must be 3, 4 or 6");
  TORCH_CHECK(shape.n % 8 == 0, "N must be a multiple of 8");
  TORCH_CHECK(shape.k % 64 == 0 && (shape.k / 64) % warps == 0,
              "K must be a multiple of 64*warps");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 12 && props->minor == 0,
              "the warp-split GEMV is an SM120 kernel; got SM",
              props->major, props->minor);
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  TORCH_CHECK(false, "SM120 FP4 GEMM source is not present in this build");
#else
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_bf16out(
      a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
      checked_int(shape.n, "N"), checked_int(shape.k, "K"),
      sfa.data_ptr(), sfb.data_ptr(), static_cast<float>(alpha),
      static_cast<int>(warps), static_cast<int>(stages), stream);
  TORCH_CHECK(rc == 0, "fp4_w4a4_gemv_warpsplit_bf16 failed with rc=", rc);
#endif
#else
  TORCH_CHECK(false, "fp4-gemm was not built with CUDA support");
#endif
}

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
  TORCH_CHECK(variant >= -1 && variant <= 2,
              "variant must be -1(auto), 0(default), 1(widen), or 2(pingpong)");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK((props->major == 11 && props->minor == 0) ||
                  (props->major == 12 && props->minor == 0),
              "nvfp4_gemm_bf16 requires SM110 or SM120; got SM",
              props->major, props->minor);
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  if (props->major == 11) {
    variant = select_sm110_variant(shape, variant);
    TORCH_CHECK(flash_rt::hub::sm110_gemm_dispatch != nullptr,
                "SM110 FP4 GEMM source is not present in this build");
    flash_rt::hub::sm110_gemm_dispatch(
        a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
        checked_int(shape.m, "M"), checked_int(shape.n, "N"),
        checked_int(shape.k, "K"), sfa.data_ptr(), sfb.data_ptr(),
        static_cast<float>(alpha), variant, stream);
  } else {
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
    TORCH_CHECK(false, "SM120 FP4 GEMM source is not present in this build");
#else
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
#endif
}

void nvfp4_gemm_residual_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor const& residual,
    torch::Tensor& out,
    double alpha) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_bf16_cuda(residual, "residual");
  check_bf16_cuda(out, "out");
  TORCH_CHECK(residual.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "residual must have shape (M, N)");
  TORCH_CHECK(out.sizes() == residual.sizes(), "out must match residual");
  check_same_device(a_packed, residual, "a_packed", "residual");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  require_sm120(a_packed, "nvfp4_gemm_residual_bf16");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
#if !defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  flash_rt::gemm::fp4_w4a16_gemm_residual_sm120_bf16out(
      a_packed.data_ptr(), b_packed.data_ptr(), residual.data_ptr(),
      out.data_ptr(), checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), sfa.data_ptr(), sfb.data_ptr(),
      static_cast<float>(alpha), stream);
#endif
#endif
}

void nvfp4_gemm_bias_gelu_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor const& bias,
    torch::Tensor& out,
    double alpha) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_bf16_cuda(bias, "bias");
  check_bf16_cuda(out, "out");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == shape.n,
              "bias must have shape (N,)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "out must have shape (M, N)");
  check_same_device(a_packed, bias, "a_packed", "bias");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  require_sm120(a_packed, "nvfp4_gemm_bias_gelu_bf16");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
#if !defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  flash_rt::gemm::fp4_w4a16_gemm_bias_gelu_bf16out_sm120(
      a_packed.data_ptr(), b_packed.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      bias.data_ptr(), out.data_ptr(), checked_int(shape.m, "M"),
      checked_int(shape.n, "N"), checked_int(shape.k, "K"),
      static_cast<float>(alpha), stream);
#endif
#endif
}

void nvfp4_gemm_bias_gelu_nvfp4(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor const& bias,
    torch::Tensor& out_packed,
    torch::Tensor& out_sfa,
    double alpha) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_bf16_cuda(bias, "bias");
  check_uint8_cuda(out_packed, "out_packed");
  check_uint8_cuda(out_sfa, "out_sfa");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == shape.n,
              "bias must have shape (N,)");
  TORCH_CHECK(shape.n % 2 == 0 &&
                  out_packed.sizes() ==
                      torch::IntArrayRef({shape.m, shape.n / 2}),
              "out_packed must have shape (M, N / 2)");
  TORCH_CHECK(out_sfa.numel() >= swizzled_bytes(shape.m, shape.n),
              "out_sfa is too small for output scale layout");
  check_same_device(a_packed, bias, "a_packed", "bias");
  check_same_device(a_packed, out_packed, "a_packed", "out_packed");
  check_same_device(a_packed, out_sfa, "a_packed", "out_sfa");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  require_sm120(a_packed, "nvfp4_gemm_bias_gelu_nvfp4");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
#if !defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  flash_rt::gemm::fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
      a_packed.data_ptr(), b_packed.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      bias.data_ptr(), out_packed.data_ptr(), out_sfa.data_ptr(),
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), static_cast<float>(alpha), stream);
#endif
#endif
}

void nvfp4_gemm_streamk_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_bf16_cuda(out, "out");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "out must have shape (M, N)");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  require_sm120(a_packed, "nvfp4_gemm_streamk_bf16");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
#if !defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  flash_rt::gemm::fp4_w4a16_gemm_dn_streamk_bf16out_sm120(
      a_packed.data_ptr(), b_packed.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      out.data_ptr(), checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), static_cast<float>(alpha), stream);
#endif
#endif
}

void nvfp4_gemm_streamk_bias_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor const& bias,
    torch::Tensor& out,
    double alpha) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_bf16_cuda(bias, "bias");
  check_bf16_cuda(out, "out");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == shape.n,
              "bias must have shape (N,)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "out must have shape (M, N)");
  check_same_device(a_packed, bias, "a_packed", "bias");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  require_sm120(a_packed, "nvfp4_gemm_streamk_bias_bf16");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
#if !defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  flash_rt::gemm::fp4_w4a16_gemm_dn_streamk_bias_bf16out_sm120(
      a_packed.data_ptr(), b_packed.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      bias.data_ptr(), out.data_ptr(), checked_int(shape.m, "M"),
      checked_int(shape.n, "N"), checked_int(shape.k, "K"),
      static_cast<float>(alpha), stream);
#endif
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

void quantize_fp4_sfa_bf16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    bool is_sfb) {
  check_bf16_cuda(x, "x");
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
  const int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_bf16(
      x.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
      checked_int(rows, "rows"), checked_int(dim, "dim"), is_sfb, stream);
  TORCH_CHECK(rc == 0, "quantize_fp4_dynamic_sfa_bf16 failed with rc=", rc);
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
  ops.def("nvfp4_gemm_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int variant=-1) -> ()");
  ops.def("fp4_w4a16_linear_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int variant=-1) -> ()");
  ops.def("fp4_w4a4_gemv_warpsplit_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int warps=4, int stages=4) -> ()");
  ops.def("nvfp4_gemm_residual_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor residual, Tensor! out, float alpha=1.0) -> ()");
  ops.def("nvfp4_gemm_bias_gelu_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor! out, float alpha=1.0) -> ()");
  ops.def("nvfp4_gemm_bias_gelu_nvfp4(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor! out_packed, Tensor! out_sfa, float alpha=1.0) -> ()");
  ops.def("nvfp4_gemm_streamk_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0) -> ()");
  ops.def("nvfp4_gemm_streamk_bias_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor! out, float alpha=1.0) -> ()");
  ops.def("quantize_fp4_sfa_fp16(Tensor x, Tensor! packed, Tensor! sfa, bool is_sfb=False) -> ()");
  ops.def("quantize_fp4_sfa_bf16(Tensor x, Tensor! packed, Tensor! sfa, bool is_sfb=False) -> ()");
  ops.def("dequantize_fp4_sfa_fp16(Tensor packed, Tensor sfa, Tensor! out, bool is_sfb=False) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("nvfp4_gemm_bf16", torch::kCUDA, &fp4_w4a16_linear_bf16);
  ops.impl("fp4_w4a16_linear_bf16", torch::kCUDA, &fp4_w4a16_linear_bf16);
  ops.impl("fp4_w4a4_gemv_warpsplit_bf16", torch::kCUDA, &fp4_w4a4_gemv_warpsplit_bf16);
  ops.impl("nvfp4_gemm_residual_bf16", torch::kCUDA, &nvfp4_gemm_residual_bf16);
  ops.impl("nvfp4_gemm_bias_gelu_bf16", torch::kCUDA, &nvfp4_gemm_bias_gelu_bf16);
  ops.impl("nvfp4_gemm_bias_gelu_nvfp4", torch::kCUDA, &nvfp4_gemm_bias_gelu_nvfp4);
  ops.impl("nvfp4_gemm_streamk_bf16", torch::kCUDA, &nvfp4_gemm_streamk_bf16);
  ops.impl("nvfp4_gemm_streamk_bias_bf16", torch::kCUDA, &nvfp4_gemm_streamk_bias_bf16);
  ops.impl("quantize_fp4_sfa_fp16", torch::kCUDA, &quantize_fp4_sfa_fp16);
  ops.impl("quantize_fp4_sfa_bf16", torch::kCUDA, &quantize_fp4_sfa_bf16);
  ops.impl("dequantize_fp4_sfa_fp16", torch::kCUDA, &dequantize_fp4_sfa_fp16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
