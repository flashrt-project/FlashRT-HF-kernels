// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#ifndef FLASHRT_HAVE_COSMOS3_EDGE
#define FLASHRT_HAVE_COSMOS3_EDGE 1
#endif

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
#include "gemm/fp4/cutlass_nvfp4_gemm_m256_sm120.cuh"
#include "gemm/fp4/fp4_w4a4_mma_warpsplit_ilv_sm120.cuh"
#include "gemm/fp4/fp4_w4a4_mma_warpsplit_sm120.cuh"
#include "gemm/fp4/fp4_w4a4_mma_warpsplit_mrows_sm120.cuh"
#include "marlin/w4a16_marlin_sm120.cuh"
#include "marlin/gptq_marlin_repack.cuh"
#endif
#include "gemm/fp4/sm110_dispatch.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "quantize/quantize_e0m3_sfa.cuh"
#include "registration.h"
#include "torch_binding.h"

flash_rt::hub::Sm110GemmDispatch flash_rt::hub::sm110_gemm_dispatch = nullptr;
flash_rt::hub::Sm110GemmBiasDispatch
    flash_rt::hub::sm110_gemm_bias_dispatch = nullptr;
flash_rt::hub::Sm110GemmBiasResidualDispatch
    flash_rt::hub::sm110_gemm_bias_residual_dispatch = nullptr;
flash_rt::hub::Sm110GemmBiasGeluFp4Dispatch
    flash_rt::hub::sm110_gemm_bias_gelu_fp4_dispatch = nullptr;
flash_rt::hub::Sm110QuantizeBf16Dispatch
    flash_rt::hub::sm110_quantize_bf16_dispatch = nullptr;
flash_rt::hub::Sm110GemmFp16Dispatch
    flash_rt::hub::sm110_gemm_fp16_dispatch = nullptr;
flash_rt::hub::Sm110GemmBf16VariantDispatch
    flash_rt::hub::sm110_gemm_bf16_variant_dispatch = nullptr;
flash_rt::hub::Sm110GemmFp4OutDispatch
    flash_rt::hub::sm110_gemm_fp4out_dispatch = nullptr;
flash_rt::hub::Sm110GemmGegluFp4Dispatch
    flash_rt::hub::sm110_gemm_geglu_fp4_dispatch = nullptr;
flash_rt::hub::Sm110GemmBiasGeluFp4Fp16Dispatch
    flash_rt::hub::sm110_gemm_bias_gelu_fp4_fp16_dispatch = nullptr;
flash_rt::hub::Sm110GemmBiasResidualFp16Dispatch
    flash_rt::hub::sm110_gemm_bias_residual_fp16_dispatch = nullptr;
flash_rt::hub::Sm110E0m3GemmDispatch
    flash_rt::hub::sm110_e0m3_gemm_dispatch = nullptr;
flash_rt::hub::Sm110GemmRelu2Fp4Dispatch
    flash_rt::hub::sm110_gemm_relu2_fp4_dispatch = nullptr;

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

void check_float_cuda(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
}

void check_int_cuda(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
              name, " must have dtype torch.int32");
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

void fp4_w4a4_gemm_warpsplit_mrows_bf16(
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
  TORCH_CHECK(shape.m >= 1 && shape.m <= 16,
              "the multi-row warp-split tier serves 1..16 rows");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "out must have shape (M, N)");
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
              "the multi-row warp-split GEMM is an SM120 kernel; got SM",
              props->major, props->minor);
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  TORCH_CHECK(false, "SM120 FP4 GEMM source is not present in this build");
#else
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_mrows_bf16out(
      a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), sfa.data_ptr(), sfb.data_ptr(),
      static_cast<float>(alpha), static_cast<int>(warps),
      static_cast<int>(stages), stream);
  TORCH_CHECK(rc == 0,
              "fp4_w4a4_gemm_warpsplit_mrows_bf16 failed with rc=", rc);
#endif
#else
  TORCH_CHECK(false, "fp4-gemm was not built with CUDA support");
#endif
}

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

void fp4_repack_b_interleaved_sm120(
    torch::Tensor const& b_packed,
    torch::Tensor& b_interleaved) {
  check_uint8_cuda(b_packed, "b_packed");
  check_uint8_cuda(b_interleaved, "b_interleaved");
  TORCH_CHECK(b_packed.dim() == 2, "b_packed must have shape (N, K / 2)");
  TORCH_CHECK(b_interleaved.sizes() == b_packed.sizes(),
              "b_interleaved must have the same shape as b_packed");
  const int64_t n = b_packed.size(0);
  const int64_t k = b_packed.size(1) * 2;
  TORCH_CHECK(n > 0 && n % 8 == 0, "N must be positive and divisible by 8");
  TORCH_CHECK(k > 0 && k % 64 == 0, "K must be positive and divisible by 64");
  check_same_device(b_packed, b_interleaved, "b_packed", "b_interleaved");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(b_packed.device());
  require_sm120(b_packed, "fp4_repack_b_interleaved_sm120");
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  TORCH_CHECK(false, "SM120 FP4 GEMM source is not present in this build");
#else
  auto stream = at::cuda::getCurrentCUDAStream(b_packed.get_device()).stream();
  const int rc = flash_rt::gemm::fp4_w4a4_repack_b_ilv_sm120(
      b_packed.data_ptr(), b_interleaved.data_ptr(), checked_int(n, "N"),
      checked_int(k, "K"), stream);
  TORCH_CHECK(rc == 0, "fp4_repack_b_interleaved_sm120 failed with rc=", rc);
#endif
#else
  TORCH_CHECK(false, "fp4-gemm was not built with CUDA support");
#endif
}

void fp4_w4a4_gemv_warpsplit_interleaved_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_interleaved,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha,
    int64_t warps,
    int64_t stages) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_interleaved, sfa, sfb);
  check_bf16_cuda(out, "out");
  TORCH_CHECK(shape.m == 1, "interleaved warp-split GEMV serves M=1 only");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({1, shape.n}),
              "out must have shape (1, N)");
  TORCH_CHECK(warps == 2 || warps == 4 || warps == 8,
              "warps must be 2, 4 or 8");
  TORCH_CHECK(stages == 3 || stages == 4 || stages == 6,
              "stages must be 3, 4 or 6");
  TORCH_CHECK(shape.n % 8 == 0, "N must be divisible by 8");
  TORCH_CHECK(shape.k % 64 == 0 && (shape.k / 64) % warps == 0,
              "K/64 must be divisible by warps");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  require_sm120(a_packed, "fp4_w4a4_gemv_warpsplit_interleaved_bf16");
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  TORCH_CHECK(false, "SM120 FP4 GEMM source is not present in this build");
#else
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_ilv_bf16out(
      a_packed.data_ptr(), b_interleaved.data_ptr(), out.data_ptr(),
      checked_int(shape.n, "N"), checked_int(shape.k, "K"), sfa.data_ptr(),
      sfb.data_ptr(), static_cast<float>(alpha), static_cast<int>(warps),
      static_cast<int>(stages), stream);
  TORCH_CHECK(rc == 0,
              "fp4_w4a4_gemv_warpsplit_interleaved_bf16 failed with rc=", rc);
#endif
#else
  TORCH_CHECK(false, "fp4-gemm was not built with CUDA support");
#endif
}

int64_t nvfp4_gemm_m256_workspace_size(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  TORCH_CHECK(shape.m >= 512, "the M256 tier requires M >= 512");
#if defined(CUDA_KERNEL)
  require_sm120(a_packed, "nvfp4_gemm_m256_workspace_size");
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  TORCH_CHECK(false, "SM120 FP4 GEMM source is not present in this build");
#else
  return static_cast<int64_t>(flash_rt::gemm::nvfp4_gemm_m256_sm120_workspace_size(
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K")));
#endif
#else
  TORCH_CHECK(false, "fp4-gemm was not built with CUDA support");
#endif
}

void nvfp4_gemm_m256_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor const& workspace,
    torch::Tensor& out,
    double alpha) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_uint8_cuda(workspace, "workspace");
  check_bf16_cuda(out, "out");
  TORCH_CHECK(shape.m >= 512, "the M256 tier requires M >= 512");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "out must have shape (M, N)");
  check_same_device(a_packed, workspace, "a_packed", "workspace");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  require_sm120(a_packed, "nvfp4_gemm_m256_bf16");
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  TORCH_CHECK(false, "SM120 FP4 GEMM source is not present in this build");
#else
  const auto required = flash_rt::gemm::nvfp4_gemm_m256_sm120_workspace_size(
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"));
  TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= required,
              "workspace is too small: need ", required, " bytes, got ",
              workspace.numel());
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::gemm::nvfp4_gemm_m256_sm120_bf16(
      a_packed.data_ptr(), sfa.data_ptr(), b_packed.data_ptr(), sfb.data_ptr(),
      out.data_ptr(), checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), static_cast<float>(alpha),
      workspace.numel() == 0 ? nullptr : workspace.data_ptr(), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_m256_bf16 failed with rc=", rc,
              " (1=unsupported shape, 2=initialization, 3=launch)");
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
    int rc = 0;
    if (variant == 1) {
      rc = flash_rt::gemm::fp4_w4a16_gemm_sm120_bf16out_widen(
          a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
          checked_int(shape.m, "M"), checked_int(shape.n, "N"), checked_int(shape.k, "K"),
          sfa.data_ptr(), sfb.data_ptr(), static_cast<float>(alpha), stream);
    } else if (variant == 2) {
      rc = flash_rt::gemm::fp4_w4a16_gemm_sm120_bf16out_pingpong(
          a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
          checked_int(shape.m, "M"), checked_int(shape.n, "N"), checked_int(shape.k, "K"),
          sfa.data_ptr(), sfb.data_ptr(), static_cast<float>(alpha), stream);
    } else {
      rc = flash_rt::gemm::fp4_w4a16_gemm_sm120_bf16out(
          a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
          checked_int(shape.m, "M"), checked_int(shape.n, "N"), checked_int(shape.k, "K"),
          sfa.data_ptr(), sfb.data_ptr(), static_cast<float>(alpha), stream);
    }
    TORCH_CHECK(rc == 0, "nvfp4_gemm_bf16 failed CUTLASS validation/runtime with rc=", rc,
                " for M=", shape.m, " N=", shape.n, " K=", shape.k);
#endif
  }
#endif
}

void nvfp4_gemm_fp16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha,
    int64_t variant) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_fp16_cuda(out, "out");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "out must have shape (M, N)");
  TORCH_CHECK(variant >= -1 && variant <= 10,
              "variant must be -1(auto) or a native SM110 variant in [0, 10]");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "nvfp4_gemm_fp16 requires SM110; got SM",
              props->major, props->minor);
  TORCH_CHECK(flash_rt::hub::sm110_gemm_fp16_dispatch != nullptr,
              "SM110 FP16-output FP4 GEMM source is not present in this build");
  if (variant < 0) variant = 0;
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::hub::sm110_gemm_fp16_dispatch(
      static_cast<int>(variant), a_packed.data_ptr(), sfa.data_ptr(),
      b_packed.data_ptr(), sfb.data_ptr(), out.data_ptr(),
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), static_cast<float>(alpha), 0.0f, stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_fp16 failed with rc=", rc);
#endif
}

void nvfp4_gemm_variant_bf16(
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
  TORCH_CHECK(variant == 7 || variant == 10,
              "BF16 native PI0.5 GEMM supports variant 7 or 10");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "nvfp4_gemm_variant_bf16 requires SM110; got SM",
              props->major, props->minor);
  TORCH_CHECK(flash_rt::hub::sm110_gemm_bf16_variant_dispatch != nullptr,
              "SM110 BF16 native FP4 GEMM variants are absent from this build");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::hub::sm110_gemm_bf16_variant_dispatch(
      static_cast<int>(variant), a_packed.data_ptr(), sfa.data_ptr(),
      b_packed.data_ptr(), sfb.data_ptr(), out.data_ptr(),
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), static_cast<float>(alpha), 0.0f, stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_variant_bf16 failed with rc=", rc);
#endif
}

void nvfp4_gemm_nvfp4(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out_packed,
    torch::Tensor& out_sfa) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  TORCH_CHECK(shape.n % 16 == 0, "N must be divisible by 16");
  check_uint8_cuda(out_packed, "out_packed");
  check_uint8_cuda(out_sfa, "out_sfa");
  TORCH_CHECK(out_packed.sizes() ==
                  torch::IntArrayRef({shape.m, shape.n / 2}),
              "out_packed must have shape (M, N / 2)");
  TORCH_CHECK(out_sfa.numel() >= swizzled_bytes(shape.m, shape.n),
              "out_sfa is too small for the output SFA layout");
  check_same_device(a_packed, out_packed, "a_packed", "out_packed");
  check_same_device(a_packed, out_sfa, "a_packed", "out_sfa");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "nvfp4_gemm_nvfp4 requires SM110; got SM",
              props->major, props->minor);
  TORCH_CHECK(flash_rt::hub::sm110_gemm_fp4out_dispatch != nullptr,
              "SM110 FP4-output GEMM is absent from this build");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::hub::sm110_gemm_fp4out_dispatch(
      a_packed.data_ptr(), sfa.data_ptr(), b_packed.data_ptr(), sfb.data_ptr(),
      out_packed.data_ptr(), out_sfa.data_ptr(), checked_int(shape.m, "M"),
      checked_int(shape.n, "N"), checked_int(shape.k, "K"), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_nvfp4 failed with rc=", rc);
#endif
}

void nvfp4_gemm_geglu_nvfp4_fp16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_interleaved_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& scratch,
    torch::Tensor& out_packed,
    torch::Tensor& out_sfa,
    bool skinny) {
  auto shape = check_fp4_gemm_inputs(
      a_packed, b_interleaved_packed, sfa, sfb);
  TORCH_CHECK(shape.n % 32 == 0,
              "interleaved gate/up N must be divisible by 32");
  const int64_t hidden = shape.n / 2;
  check_uint8_cuda(scratch, "scratch");
  check_uint8_cuda(out_packed, "out_packed");
  check_uint8_cuda(out_sfa, "out_sfa");
  TORCH_CHECK(scratch.numel() >= shape.m * hidden,
              "scratch must contain at least M * (N / 2) bytes");
  TORCH_CHECK(out_packed.sizes() ==
                  torch::IntArrayRef({shape.m, hidden / 2}),
              "out_packed must have shape (M, N / 4)");
  TORCH_CHECK(out_sfa.numel() >= swizzled_bytes(shape.m, hidden),
              "out_sfa is too small for compact GeGLU output");
  check_same_device(a_packed, scratch, "a_packed", "scratch");
  check_same_device(a_packed, out_packed, "a_packed", "out_packed");
  check_same_device(a_packed, out_sfa, "a_packed", "out_sfa");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "nvfp4_gemm_geglu_nvfp4_fp16 requires SM110; got SM",
              props->major, props->minor);
  TORCH_CHECK(flash_rt::hub::sm110_gemm_geglu_fp4_dispatch != nullptr,
              "SM110 compact GeGLU FP4 GEMM source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::hub::sm110_gemm_geglu_fp4_dispatch(
      a_packed.data_ptr(), sfa.data_ptr(), b_interleaved_packed.data_ptr(),
      sfb.data_ptr(), scratch.data_ptr(), out_packed.data_ptr(),
      out_sfa.data_ptr(), checked_int(shape.m, "M"),
      checked_int(shape.n, "N_twice"), checked_int(shape.k, "K"), skinny,
      stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_geglu_nvfp4_fp16 failed with rc=", rc);
#endif
}

void nvfp4_gemm_bias_gelu_nvfp4_fp16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor const& bias,
    torch::Tensor& out_packed,
    torch::Tensor& out_sfa) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_fp16_cuda(bias, "bias");
  check_uint8_cuda(out_packed, "out_packed");
  check_uint8_cuda(out_sfa, "out_sfa");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == shape.n,
              "bias must have shape (N,)");
  TORCH_CHECK(shape.n % 16 == 0 && out_packed.sizes() ==
                  torch::IntArrayRef({shape.m, shape.n / 2}),
              "out_packed must have shape (M, N / 2)");
  TORCH_CHECK(out_sfa.numel() >= swizzled_bytes(shape.m, shape.n),
              "out_sfa is too small for output scale layout");
  check_same_device(a_packed, bias, "a_packed", "bias");
  check_same_device(a_packed, out_packed, "a_packed", "out_packed");
  check_same_device(a_packed, out_sfa, "a_packed", "out_sfa");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "nvfp4_gemm_bias_gelu_nvfp4_fp16 requires SM110; got SM",
              props->major, props->minor);
  TORCH_CHECK(
      flash_rt::hub::sm110_gemm_bias_gelu_fp4_fp16_dispatch != nullptr,
      "SM110 FP16-bias GELU FP4 GEMM source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::hub::sm110_gemm_bias_gelu_fp4_fp16_dispatch(
      a_packed.data_ptr(), sfa.data_ptr(), b_packed.data_ptr(), sfb.data_ptr(),
      bias.data_ptr(), out_packed.data_ptr(), out_sfa.data_ptr(),
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), stream);
  TORCH_CHECK(rc == 0,
              "nvfp4_gemm_bias_gelu_nvfp4_fp16 failed with rc=", rc);
#endif
}

void nvfp4_gemm_bias_residual_fp16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor const& bias,
    torch::Tensor const& residual,
    torch::Tensor& out) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_fp16_cuda(bias, "bias");
  check_fp16_cuda(residual, "residual");
  check_fp16_cuda(out, "out");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == shape.n,
              "bias must have shape (N,)");
  TORCH_CHECK(residual.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "residual must have shape (M, N)");
  TORCH_CHECK(out.sizes() == residual.sizes(), "out must match residual");
  check_same_device(a_packed, bias, "a_packed", "bias");
  check_same_device(a_packed, residual, "a_packed", "residual");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "nvfp4_gemm_bias_residual_fp16 requires SM110; got SM",
              props->major, props->minor);
  TORCH_CHECK(
      flash_rt::hub::sm110_gemm_bias_residual_fp16_dispatch != nullptr,
      "SM110 FP16 bias-residual GEMM source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::hub::sm110_gemm_bias_residual_fp16_dispatch(
      a_packed.data_ptr(), sfa.data_ptr(), b_packed.data_ptr(), sfb.data_ptr(),
      bias.data_ptr(), residual.data_ptr(), out.data_ptr(),
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_bias_residual_fp16 failed with rc=", rc);
#endif
}

void nvfp4_gemm_bias_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor const& bias,
    torch::Tensor& out) {
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
  auto const* props = current_device_properties(a_packed);
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  if (props->major == 11 && props->minor == 0) {
    TORCH_CHECK(flash_rt::hub::sm110_gemm_bias_dispatch != nullptr,
                "SM110 fused-bias FP4 GEMM source is not present in this build");
    const int rc = flash_rt::hub::sm110_gemm_bias_dispatch(
        a_packed.data_ptr(), sfa.data_ptr(), b_packed.data_ptr(),
        sfb.data_ptr(), bias.data_ptr(), out.data_ptr(),
        checked_int(shape.m, "M"), checked_int(shape.n, "N"),
        checked_int(shape.k, "K"), stream);
    TORCH_CHECK(rc == 0, "nvfp4_gemm_bias_bf16 failed with rc=", rc);
  } else {
    TORCH_CHECK(props->major == 12 && props->minor == 0,
                "nvfp4_gemm_bias_bf16 requires SM110 or SM120; got SM",
                props->major, props->minor);
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
    TORCH_CHECK(false, "SM120 fused-bias FP4 GEMM source is not present in this build");
#else
    const int rc = flash_rt::gemm::fp4_w4a16_gemm_dn_streamk_bias_bf16out_sm120(
        a_packed.data_ptr(), b_packed.data_ptr(), sfa.data_ptr(),
        sfb.data_ptr(), bias.data_ptr(), out.data_ptr(),
        checked_int(shape.m, "M"), checked_int(shape.n, "N"),
        checked_int(shape.k, "K"), 1.0f, stream);
    TORCH_CHECK(rc == 0, "nvfp4_gemm_bias_bf16 failed CUTLASS validation/runtime with rc=", rc,
                " for M=", shape.m, " N=", shape.n, " K=", shape.k);
#endif
  }
#endif
}

void nvfp4_gemm_bias_residual_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor const& bias,
    torch::Tensor const& residual,
    torch::Tensor& out) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_bf16_cuda(bias, "bias");
  check_bf16_cuda(residual, "residual");
  check_bf16_cuda(out, "out");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == shape.n,
              "bias must have shape (N,)");
  TORCH_CHECK(residual.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "residual must have shape (M, N)");
  TORCH_CHECK(out.sizes() == residual.sizes(), "out must match residual");
  check_same_device(a_packed, bias, "a_packed", "bias");
  check_same_device(a_packed, residual, "a_packed", "residual");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "nvfp4_gemm_bias_residual_bf16 currently requires SM110; got SM",
              props->major, props->minor);
  TORCH_CHECK(flash_rt::hub::sm110_gemm_bias_residual_dispatch != nullptr,
              "SM110 bias-residual FP4 GEMM source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::hub::sm110_gemm_bias_residual_dispatch(
      a_packed.data_ptr(), sfa.data_ptr(), b_packed.data_ptr(), sfb.data_ptr(),
      bias.data_ptr(), residual.data_ptr(), out.data_ptr(),
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_bias_residual_bf16 failed with rc=", rc);
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
  const int rc = flash_rt::gemm::fp4_w4a16_gemm_residual_sm120_bf16out(
      a_packed.data_ptr(), b_packed.data_ptr(), residual.data_ptr(),
      out.data_ptr(), checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), sfa.data_ptr(), sfb.data_ptr(),
      static_cast<float>(alpha), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_residual_bf16 failed CUTLASS validation/runtime with rc=", rc,
              " for M=", shape.m, " N=", shape.n, " K=", shape.k);
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
  const int rc = flash_rt::gemm::fp4_w4a16_gemm_bias_gelu_bf16out_sm120(
      a_packed.data_ptr(), b_packed.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      bias.data_ptr(), out.data_ptr(), checked_int(shape.m, "M"),
      checked_int(shape.n, "N"), checked_int(shape.k, "K"),
      static_cast<float>(alpha), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_bias_gelu_bf16 failed CUTLASS validation/runtime with rc=", rc,
              " for M=", shape.m, " N=", shape.n, " K=", shape.k);
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
  auto const* props = current_device_properties(a_packed);
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  if (props->major == 11 && props->minor == 0) {
    TORCH_CHECK(flash_rt::hub::sm110_gemm_bias_gelu_fp4_dispatch != nullptr,
                "SM110 bias-GELU-FP4 GEMM source is not present in this build");
    const int rc = flash_rt::hub::sm110_gemm_bias_gelu_fp4_dispatch(
        a_packed.data_ptr(), sfa.data_ptr(), b_packed.data_ptr(), sfb.data_ptr(),
        bias.data_ptr(), out_packed.data_ptr(), out_sfa.data_ptr(),
        checked_int(shape.m, "M"), checked_int(shape.n, "N"),
        checked_int(shape.k, "K"), stream);
    TORCH_CHECK(rc == 0, "nvfp4_gemm_bias_gelu_nvfp4 failed with rc=", rc);
    return;
  }
  require_sm120(a_packed, "nvfp4_gemm_bias_gelu_nvfp4");
#if !defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  const int rc = flash_rt::gemm::fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
      a_packed.data_ptr(), b_packed.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      bias.data_ptr(), out_packed.data_ptr(), out_sfa.data_ptr(),
      checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), static_cast<float>(alpha), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_bias_gelu_nvfp4 failed CUTLASS validation/runtime with rc=", rc,
              " for M=", shape.m, " N=", shape.n, " K=", shape.k);
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
  const int rc = flash_rt::gemm::fp4_w4a16_gemm_dn_streamk_bf16out_sm120(
      a_packed.data_ptr(), b_packed.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      out.data_ptr(), checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), static_cast<float>(alpha), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_streamk_bf16 failed CUTLASS validation/runtime with rc=", rc,
              " for M=", shape.m, " N=", shape.n, " K=", shape.k);
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
  const int rc = flash_rt::gemm::fp4_w4a16_gemm_dn_streamk_bias_bf16out_sm120(
      a_packed.data_ptr(), b_packed.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      bias.data_ptr(), out.data_ptr(), checked_int(shape.m, "M"),
      checked_int(shape.n, "N"), checked_int(shape.k, "K"),
      static_cast<float>(alpha), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_streamk_bias_bf16 failed CUTLASS validation/runtime with rc=", rc,
              " for M=", shape.m, " N=", shape.n, " K=", shape.k);
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

void quantize_fp4_sfa_mse_fp16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    bool is_sfb) {
  check_fp16_cuda(x, "x");
  check_uint8_cuda(packed, "packed");
  check_uint8_cuda(sfa, "sfa");
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 16 == 0,
              "x must have shape (rows, dim) with dim divisible by 16");
  TORCH_CHECK(packed.sizes() ==
                  torch::IntArrayRef({x.size(0), x.size(1) / 2}),
              "packed must have shape (rows, dim / 2)");
  TORCH_CHECK(sfa.numel() >= swizzled_bytes(x.size(0), x.size(1)),
              "sfa is too small for the scale layout");
  check_same_device(x, packed, "x", "packed");
  check_same_device(x, sfa, "x", "sfa");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_mse_fp16(
      x.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
      checked_int(x.size(0), "rows"), checked_int(x.size(1), "dim"),
      is_sfb, stream);
  TORCH_CHECK(rc == 0, "quantize_fp4_sfa_mse_fp16 failed with rc=", rc);
#endif
}

void quantize_fp4_sfa_mse_bf16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    bool is_sfb) {
  check_bf16_cuda(x, "x");
  check_uint8_cuda(packed, "packed");
  check_uint8_cuda(sfa, "sfa");
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 16 == 0,
              "x must have shape (rows, dim) with dim divisible by 16");
  TORCH_CHECK(packed.sizes() ==
                  torch::IntArrayRef({x.size(0), x.size(1) / 2}),
              "packed must have shape (rows, dim / 2)");
  TORCH_CHECK(sfa.numel() >= swizzled_bytes(x.size(0), x.size(1)),
              "sfa is too small for the scale layout");
  check_same_device(x, packed, "x", "packed");
  check_same_device(x, sfa, "x", "sfa");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_mse_bf16(
      x.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
      checked_int(x.size(0), "rows"), checked_int(x.size(1), "dim"),
      is_sfb, stream);
  TORCH_CHECK(rc == 0, "quantize_fp4_sfa_mse_bf16 failed with rc=", rc);
#endif
}

void quantize_e0m3_sfa_fp16(
    torch::Tensor const& x, torch::Tensor& packed, torch::Tensor& sfa,
    bool is_sfb) {
  check_fp16_cuda(x, "x");
  check_uint8_cuda(packed, "packed");
  check_uint8_cuda(sfa, "sfa");
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 16 == 0,
              "x must have shape (rows, dim) with dim divisible by 16");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({x.size(0), x.size(1) / 2}),
              "packed must have shape (rows, dim / 2)");
  TORCH_CHECK(sfa.numel() >= swizzled_bytes(x.size(0), x.size(1)),
              "sfa is too small for CUTLASS SFA/SFB layout");
  check_same_device(x, packed, "x", "packed");
  check_same_device(x, sfa, "x", "sfa");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::fp4::quantize_e0m3_dynamic_sfa_fp16(
      x.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
      checked_int(x.size(0), "rows"), checked_int(x.size(1), "dim"),
      is_sfb, stream);
  TORCH_CHECK(rc == 0, "quantize_e0m3_sfa_fp16 failed with rc=", rc);
#endif
}

void e0m3_weight_gemm_fp16(
    torch::Tensor const& a_packed, torch::Tensor const& b_packed,
    torch::Tensor const& sfa, torch::Tensor const& sfb, torch::Tensor& out,
    double alpha, int64_t a_format) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  check_fp16_cuda(out, "out");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({shape.m, shape.n}),
              "out must have shape (M, N)");
  TORCH_CHECK(a_format == 0 || a_format == 1,
              "a_format must be 0 (E0M3) or 1 (E2M1)");
  check_same_device(a_packed, out, "a_packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "e0m3_weight_gemm_fp16 requires SM110");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  TORCH_CHECK(flash_rt::hub::sm110_e0m3_gemm_dispatch != nullptr,
              "SM110 E0M3 GEMM source is not present in this build");
  const int rc = flash_rt::hub::sm110_e0m3_gemm_dispatch(
      a_packed.data_ptr(), sfa.data_ptr(), b_packed.data_ptr(), sfb.data_ptr(),
      out.data_ptr(), checked_int(shape.m, "M"), checked_int(shape.n, "N"),
      checked_int(shape.k, "K"), static_cast<float>(alpha), 0.0f, stream,
      static_cast<int>(a_format));
  TORCH_CHECK(rc == 0, "e0m3_weight_gemm_fp16 failed with rc=", rc);
#endif
}

void nvfp4_gemm_relu2_nvfp4(
    torch::Tensor const& a_packed, torch::Tensor const& b_packed,
    torch::Tensor const& sfa, torch::Tensor const& sfb,
    torch::Tensor& out_packed, torch::Tensor& out_sfa) {
  auto shape = check_fp4_gemm_inputs(a_packed, b_packed, sfa, sfb);
  TORCH_CHECK(shape.n % 16 == 0, "N must be divisible by 16");
  check_uint8_cuda(out_packed, "out_packed");
  check_uint8_cuda(out_sfa, "out_sfa");
  TORCH_CHECK(out_packed.sizes() == torch::IntArrayRef({shape.m, shape.n / 2}),
              "out_packed must have shape (M, N / 2)");
  TORCH_CHECK(out_sfa.numel() >= swizzled_bytes(shape.m, shape.n),
              "out_sfa is too small");
  check_same_device(a_packed, out_packed, "a_packed", "out_packed");
  check_same_device(a_packed, out_sfa, "a_packed", "out_sfa");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto const* props = current_device_properties(a_packed);
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              "nvfp4_gemm_relu2_nvfp4 requires SM110");
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  TORCH_CHECK(flash_rt::hub::sm110_gemm_relu2_fp4_dispatch != nullptr,
              "SM110 ReLU2 FP4 GEMM source is not present in this build");
  const int rc = flash_rt::hub::sm110_gemm_relu2_fp4_dispatch(
      a_packed.data_ptr(), sfa.data_ptr(), b_packed.data_ptr(), sfb.data_ptr(),
      out_packed.data_ptr(), out_sfa.data_ptr(), checked_int(shape.m, "M"),
      checked_int(shape.n, "N"), checked_int(shape.k, "K"), stream);
  TORCH_CHECK(rc == 0, "nvfp4_gemm_relu2_nvfp4 failed with rc=", rc);
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
  auto const* props = current_device_properties(x);
  int rc = 0;
  if (props->major == 11 && props->minor == 0) {
    TORCH_CHECK(flash_rt::hub::sm110_quantize_bf16_dispatch != nullptr,
                "SM110 vectorized BF16 FP4 quantizer is not present in this build");
    rc = flash_rt::hub::sm110_quantize_bf16_dispatch(
        x.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
        checked_int(rows, "rows"), checked_int(dim, "dim"), is_sfb, stream);
  } else {
    rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_bf16(
        x.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
        checked_int(rows, "rows"), checked_int(dim, "dim"), is_sfb, stream);
  }
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

void nvfp4_w4a16_marlin_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight_marlin,
    torch::Tensor const& weight_scale_marlin,
    torch::Tensor const& weight_global_scale,
    torch::Tensor& workspace,
    torch::Tensor& out) {
  check_bf16_cuda(x, "x");
  check_int_cuda(weight_marlin, "weight_marlin");
  check_cuda_contiguous(weight_scale_marlin, "weight_scale_marlin");
  TORCH_CHECK(weight_scale_marlin.scalar_type() == torch::kFloat8_e4m3fn,
              "weight_scale_marlin must have dtype torch.float8_e4m3fn");
  check_float_cuda(weight_global_scale, "weight_global_scale");
  check_int_cuda(workspace, "workspace");
  check_bf16_cuda(out, "out");
  TORCH_CHECK(x.dim() == 2, "x must have shape (M, K)");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0),
              "out must have shape (M, N)");
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = out.size(1);
  TORCH_CHECK(m >= 1 && m <= 16, "the Marlin W4A16 tier serves M in [1, 16]");
  TORCH_CHECK(k > 0 && k % 128 == 0, "K must be divisible by 128");
  TORCH_CHECK(n > 0 && n % 64 == 0, "N must be divisible by 64");
  TORCH_CHECK(weight_scale_marlin.numel() == (k / 16) * n,
              "weight_scale_marlin must contain K/16 * N entries");
  TORCH_CHECK(weight_global_scale.numel() == 1,
              "weight_global_scale must contain one value");
  TORCH_CHECK(workspace.numel() > 0, "workspace must not be empty");
  check_same_device(x, weight_marlin, "x", "weight_marlin");
  check_same_device(x, weight_scale_marlin, "x", "weight_scale_marlin");
  check_same_device(x, weight_global_scale, "x", "weight_global_scale");
  check_same_device(x, workspace, "x", "workspace");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  require_sm120(x, "nvfp4_w4a16_marlin_bf16");
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  TORCH_CHECK(false, "SM120 Marlin W4A16 source is not present in this build");
#else
  auto const* props = current_device_properties(x);
  TORCH_CHECK(workspace.numel() >= props->multiProcessorCount,
              "workspace must contain at least one int32 lock per SM");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::gemm::w4a16_marlin_sm120_bf16(
      x.data_ptr(), weight_marlin.data_ptr(), weight_scale_marlin.data_ptr(),
      weight_global_scale.data_ptr(), workspace.data_ptr(), out.data_ptr(),
      checked_int(m, "M"), checked_int(n, "N"), checked_int(k, "K"),
      checked_int(x.stride(0), "lda"), stream);
  TORCH_CHECK(rc == 0, "nvfp4_w4a16_marlin_bf16 failed with rc=", rc);
#endif
#else
  TORCH_CHECK(false, "fp4-gemm was not built with CUDA support");
#endif
}

void nvfp4_w4a16_marlin_repack(
    torch::Tensor const& qweight_kn,
    torch::Tensor& weight_marlin) {
  check_int_cuda(qweight_kn, "qweight_kn");
  check_int_cuda(weight_marlin, "weight_marlin");
  TORCH_CHECK(qweight_kn.dim() == 2, "qweight_kn must have shape (K / 8, N)");
  const int64_t k = qweight_kn.size(0) * 8;
  const int64_t n = qweight_kn.size(1);
  TORCH_CHECK(k % 128 == 0, "K must be divisible by 128");
  TORCH_CHECK(n % 64 == 0, "N must be divisible by 64");
  TORCH_CHECK(weight_marlin.sizes() == torch::IntArrayRef({k / 16, n * 2}),
              "weight_marlin must have shape (K / 16, 2 * N)");
  check_same_device(qweight_kn, weight_marlin, "qweight_kn", "weight_marlin");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(qweight_kn.device());
  require_sm120(qweight_kn, "nvfp4_w4a16_marlin_repack");
#if defined(FLASHRT_FP4_GEMM_SOURCE_SM110_ONLY)
  TORCH_CHECK(false, "SM120 Marlin repack source is not present in this build");
#else
  auto stream = at::cuda::getCurrentCUDAStream(qweight_kn.get_device()).stream();
  const int rc = flash_rt::gemm::repack_nvfp4_to_marlin_sm120(
      qweight_kn.data_ptr(), weight_marlin.data_ptr(), checked_int(k, "K"),
      checked_int(n, "N"), stream);
  TORCH_CHECK(rc == 0, "nvfp4_w4a16_marlin_repack failed with rc=", rc);
#endif
#else
  TORCH_CHECK(false, "fp4-gemm was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("nvfp4_gemm_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int variant=-1) -> ()");
  ops.def("nvfp4_gemm_fp16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int variant=-1) -> ()");
  ops.def("nvfp4_gemm_variant_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int variant=7) -> ()");
  ops.def("nvfp4_gemm_nvfp4(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out_packed, Tensor! out_sfa) -> ()");
  ops.def("nvfp4_gemm_geglu_nvfp4_fp16(Tensor a_packed, Tensor b_interleaved_packed, Tensor sfa, Tensor sfb, Tensor! scratch, Tensor! out_packed, Tensor! out_sfa, bool skinny=False) -> ()");
  ops.def("nvfp4_gemm_bias_gelu_nvfp4_fp16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor! out_packed, Tensor! out_sfa) -> ()");
  ops.def("nvfp4_gemm_bias_residual_fp16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor residual, Tensor! out) -> ()");
  ops.def("fp4_w4a16_linear_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int variant=-1) -> ()");
  ops.def("fp4_w4a4_gemv_warpsplit_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int warps=4, int stages=4) -> ()");
  ops.def("fp4_repack_b_interleaved_sm120(Tensor b_packed, Tensor! b_interleaved) -> ()");
  ops.def("fp4_w4a4_gemv_warpsplit_interleaved_bf16(Tensor a_packed, Tensor b_interleaved, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int warps=4, int stages=4) -> ()");
  ops.def("nvfp4_gemm_m256_workspace_size(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb) -> int");
  ops.def("nvfp4_gemm_m256_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor workspace, Tensor! out, float alpha=1.0) -> ()");
  ops.def("fp4_w4a4_gemm_warpsplit_mrows_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int warps=2, int stages=6) -> ()");
  ops.def("nvfp4_gemm_bias_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor! out) -> ()");
  ops.def("nvfp4_gemm_bias_residual_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor residual, Tensor! out) -> ()");
  ops.def("nvfp4_gemm_residual_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor residual, Tensor! out, float alpha=1.0) -> ()");
  ops.def("nvfp4_gemm_bias_gelu_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor! out, float alpha=1.0) -> ()");
  ops.def("nvfp4_gemm_bias_gelu_nvfp4(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor! out_packed, Tensor! out_sfa, float alpha=1.0) -> ()");
  ops.def("nvfp4_gemm_streamk_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0) -> ()");
  ops.def("nvfp4_gemm_streamk_bias_bf16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor bias, Tensor! out, float alpha=1.0) -> ()");
  ops.def("quantize_fp4_sfa_fp16(Tensor x, Tensor! packed, Tensor! sfa, bool is_sfb=False) -> ()");
  ops.def("quantize_fp4_sfa_mse_fp16(Tensor x, Tensor! packed, Tensor! sfa, bool is_sfb=False) -> ()");
  ops.def("quantize_fp4_sfa_mse_bf16(Tensor x, Tensor! packed, Tensor! sfa, bool is_sfb=False) -> ()");
  ops.def("quantize_e0m3_sfa_fp16(Tensor x, Tensor! packed, Tensor! sfa, bool is_sfb=False) -> ()");
  ops.def("e0m3_weight_gemm_fp16(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out, float alpha=1.0, int a_format=1) -> ()");
  ops.def("nvfp4_gemm_relu2_nvfp4(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, Tensor! out_packed, Tensor! out_sfa) -> ()");
  ops.def("quantize_fp4_sfa_bf16(Tensor x, Tensor! packed, Tensor! sfa, bool is_sfb=False) -> ()");
  ops.def("dequantize_fp4_sfa_fp16(Tensor packed, Tensor sfa, Tensor! out, bool is_sfb=False) -> ()");
  ops.def("nvfp4_w4a16_marlin_bf16(Tensor x, Tensor weight_marlin, Tensor weight_scale_marlin, Tensor weight_global_scale, Tensor! workspace, Tensor! out) -> ()");
  ops.def("nvfp4_w4a16_marlin_repack(Tensor qweight_kn, Tensor! weight_marlin) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("nvfp4_gemm_bf16", torch::kCUDA, &fp4_w4a16_linear_bf16);
  ops.impl("nvfp4_gemm_fp16", torch::kCUDA, &nvfp4_gemm_fp16);
  ops.impl("nvfp4_gemm_variant_bf16", torch::kCUDA, &nvfp4_gemm_variant_bf16);
  ops.impl("nvfp4_gemm_nvfp4", torch::kCUDA, &nvfp4_gemm_nvfp4);
  ops.impl("nvfp4_gemm_geglu_nvfp4_fp16", torch::kCUDA, &nvfp4_gemm_geglu_nvfp4_fp16);
  ops.impl("nvfp4_gemm_bias_gelu_nvfp4_fp16", torch::kCUDA, &nvfp4_gemm_bias_gelu_nvfp4_fp16);
  ops.impl("nvfp4_gemm_bias_residual_fp16", torch::kCUDA, &nvfp4_gemm_bias_residual_fp16);
  ops.impl("fp4_w4a16_linear_bf16", torch::kCUDA, &fp4_w4a16_linear_bf16);
  ops.impl("fp4_w4a4_gemv_warpsplit_bf16", torch::kCUDA, &fp4_w4a4_gemv_warpsplit_bf16);
  ops.impl("fp4_repack_b_interleaved_sm120", torch::kCUDA, &fp4_repack_b_interleaved_sm120);
  ops.impl("fp4_w4a4_gemv_warpsplit_interleaved_bf16", torch::kCUDA, &fp4_w4a4_gemv_warpsplit_interleaved_bf16);
  ops.impl("nvfp4_gemm_m256_workspace_size", torch::kCUDA, &nvfp4_gemm_m256_workspace_size);
  ops.impl("nvfp4_gemm_m256_bf16", torch::kCUDA, &nvfp4_gemm_m256_bf16);
  ops.impl("fp4_w4a4_gemm_warpsplit_mrows_bf16", torch::kCUDA, &fp4_w4a4_gemm_warpsplit_mrows_bf16);
  ops.impl("nvfp4_gemm_bias_bf16", torch::kCUDA, &nvfp4_gemm_bias_bf16);
  ops.impl("nvfp4_gemm_bias_residual_bf16", torch::kCUDA, &nvfp4_gemm_bias_residual_bf16);
  ops.impl("nvfp4_gemm_residual_bf16", torch::kCUDA, &nvfp4_gemm_residual_bf16);
  ops.impl("nvfp4_gemm_bias_gelu_bf16", torch::kCUDA, &nvfp4_gemm_bias_gelu_bf16);
  ops.impl("nvfp4_gemm_bias_gelu_nvfp4", torch::kCUDA, &nvfp4_gemm_bias_gelu_nvfp4);
  ops.impl("nvfp4_gemm_streamk_bf16", torch::kCUDA, &nvfp4_gemm_streamk_bf16);
  ops.impl("nvfp4_gemm_streamk_bias_bf16", torch::kCUDA, &nvfp4_gemm_streamk_bias_bf16);
  ops.impl("quantize_fp4_sfa_fp16", torch::kCUDA, &quantize_fp4_sfa_fp16);
  ops.impl("quantize_fp4_sfa_mse_fp16", torch::kCUDA, &quantize_fp4_sfa_mse_fp16);
  ops.impl("quantize_fp4_sfa_mse_bf16", torch::kCUDA, &quantize_fp4_sfa_mse_bf16);
  ops.impl("quantize_e0m3_sfa_fp16", torch::kCUDA, &quantize_e0m3_sfa_fp16);
  ops.impl("e0m3_weight_gemm_fp16", torch::kCUDA, &e0m3_weight_gemm_fp16);
  ops.impl("nvfp4_gemm_relu2_nvfp4", torch::kCUDA, &nvfp4_gemm_relu2_nvfp4);
  ops.impl("quantize_fp4_sfa_bf16", torch::kCUDA, &quantize_fp4_sfa_bf16);
  ops.impl("dequantize_fp4_sfa_fp16", torch::kCUDA, &dequantize_fp4_sfa_fp16);
  ops.impl("nvfp4_w4a16_marlin_bf16", torch::kCUDA, &nvfp4_w4a16_marlin_bf16);
  ops.impl("nvfp4_w4a16_marlin_repack", torch::kCUDA, &nvfp4_w4a16_marlin_repack);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
