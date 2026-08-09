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

#if !defined(FLASHRT_FP8_GEMM_SOURCE_SM89_ONLY) && \
    !defined(FLASHRT_FP8_GEMM_SOURCE_SM110_ONLY)
#include "fp8_gemv_m1_sm120.cuh"
#include "fp8_smallM_handtuned_ldmatrix_sm120.cuh"
#include "fp8_smallM_handtuned_sm120.cuh"
#include "cutlass_sm120_block128_fp8_gemm.cuh"
#endif
#if !defined(FLASHRT_FP8_GEMM_SOURCE_SM120_ONLY) && \
    !defined(FLASHRT_FP8_GEMM_SOURCE_SM110_ONLY)
#include "fp8_block128_gemm_mma_sm89.cuh"
#include "fp8_gemv_m1_sm89.cuh"
#endif
#if !defined(FLASHRT_FP8_GEMM_SOURCE_SM89_ONLY) && \
    !defined(FLASHRT_FP8_GEMM_SOURCE_SM120_ONLY)
#include "cutlass_sm110_fp8_gemm.cuh"
#endif
#include "cublaslt_fp8_bias_sm110.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

using KernelFn = int (*)(const void*, const void*, void*, int, int, int, float, cudaStream_t);
using Sm110KernelFn = int (*)(void*, void*, void*, int, int, int, float, float,
                              cudaStream_t);

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

void check_fp32_matrix(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
  TORCH_CHECK(tensor.dim() == 2, name, " must be rank 2");
}

void check_bf16_vector(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.dim() == 1, name, " must be rank 1");
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
  TORCH_CHECK(input.size(1) % 16 == 0,
              "K must be divisible by 16 for FP8 tensor-core kernels");
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
#if defined(CUDA_KERNEL) && !defined(FLASHRT_FP8_GEMM_SOURCE_SM89_ONLY) && \
    !defined(FLASHRT_FP8_GEMM_SOURCE_SM110_ONLY)
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

const char* sm110_tile_name_for_shape(int M, int N, int K, int variant) {
  if (variant == 1) return "sm110_sq_bf16";
  if (variant == 2) return "sm110_t1_bf16";
  if (variant == 3) return "sm110_wide_bf16";
  // Thor sweep envelope (PI0.5/GROOT/Cosmos Edge/LingBot): Wide wins
  // N>=8K expansions and larger square projections. Sq wins smaller square
  // vision projections and larger-row K>=4N contractions; T1 wins the
  // remaining projection/down paths.
  // The forced variants remain available for diagnostic tile sweeps.
  if (M >= 512 && K == 2048 && N >= 2048 && N <= 2560) {
    return "sm110_sq_bf16";
  }
  if (M >= 512 && N >= 16 * K) {
    return "sm110_t1_bf16";
  }
  if (M >= 512 && K >= 4 * N) {
    return "sm110_wide_bf16";
  }
  if (N >= 8 * K) return "sm110_wide_bf16";
  if (M >= 128 && K >= 4 * N) return "sm110_sq_bf16";
  if (N == K && M >= 512) {
    return K <= 1024 ? "sm110_sq_bf16" : "sm110_wide_bf16";
  }
  if (N == K && M >= 128) {
    return "sm110_wide_bf16";
  }
  return "sm110_t1_bf16";
}

Sm110KernelFn sm110_kernel_for_shape(int M, int N, int K, int variant) {
#if defined(CUDA_KERNEL) && !defined(FLASHRT_FP8_GEMM_SOURCE_SM89_ONLY) && \
    !defined(FLASHRT_FP8_GEMM_SOURCE_SM120_ONLY)
  const char* tile = sm110_tile_name_for_shape(M, N, K, variant);
  if (std::string(tile) == "sm110_wide_bf16") return &cutlass_fp8_wide_bf16out;
  if (std::string(tile) == "sm110_t1_bf16") return &cutlass_fp8_t1_bf16out;
  return &cutlass_fp8_sq_bf16out;
#else
  (void)M;
  (void)N;
  (void)K;
  (void)variant;
  TORCH_CHECK(false, "SM110 FP8 GEMM source is not present in this build");
#endif
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
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(input.device());
  auto* props = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK((props->major == 11 && props->minor == 0) ||
                  (props->major == 12 && props->minor == 0),
              "fp8_linear_bf16 requires SM110 or SM120; got SM",
              props->major, props->minor);
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  if (props->major == 11) {
    TORCH_CHECK(variant >= 0 && variant <= 3,
                "SM110 variant must be 0 (auto), 1 (Sq), 2 (T1), or 3 (Wide)");
    TORCH_CHECK(N % 16 == 0 && K % 16 == 0,
                "SM110 CUTLASS FP8 GEMM requires N and K divisible by 16");
#if defined(FLASHRT_FP8_GEMM_SOURCE_SM89_ONLY) || \
    defined(FLASHRT_FP8_GEMM_SOURCE_SM120_ONLY)
    TORCH_CHECK(false, "SM110 FP8 GEMM source is not present in this build");
#else
    Sm110KernelFn fn = sm110_kernel_for_shape(M, N, K, variant);
    const int rc = fn(input.data_ptr(), weight.data_ptr(), out.data_ptr(),
                      M, N, K, static_cast<float>(alpha),
                      residual ? 1.0f : 0.0f, stream);
    TORCH_CHECK(rc == 0, sm110_tile_name_for_shape(M, N, K, variant),
                " failed with rc=", rc);
#endif
  } else {
    TORCH_CHECK(K % 32 == 0,
                "SM120 FP8 GEMM requires K divisible by 32");
    TORCH_CHECK(M <= 64,
                "SM120 per-tensor FP8 path supports only M <= 64; got M=", M);
    if (residual) {
      TORCH_CHECK(M == 1, "SM120 residual path supports only M=1");
    }
    const std::string tile = tile_name_for_shape(M, N, K, variant);
#if defined(FLASHRT_FP8_GEMM_SOURCE_SM89_ONLY) || \
    defined(FLASHRT_FP8_GEMM_SOURCE_SM110_ONLY)
    TORCH_CHECK(false, "SM120 per-tensor FP8 source is not present in this build");
#else
    KernelFn fn = kernel_for_tile(tile, residual);
    const int rc = fn(input.data_ptr(), weight.data_ptr(), out.data_ptr(),
                      M, N, K, static_cast<float>(alpha), stream);
    TORCH_CHECK(rc == 0, tile, " failed with rc=", rc);
#endif
  }
#else
  TORCH_CHECK(false, "fp8-gemm was not built with CUDA support");
#endif
}

void launch_bias(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out,
    double beta,
    FlashRtFp8BiasEpilogue epilogue,
    const char* op_name) {
  check_common(input, weight, out);
  check_bf16_vector(bias, "bias");
  TORCH_CHECK(bias.size(0) == weight.size(0),
              "bias must have shape (weight.shape[0],)");
  TORCH_CHECK(input.get_device() == bias.get_device(),
              "input and bias must be on the same CUDA device");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(input.device());
  auto* props = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              op_name, " requires SM110; got SM", props->major, props->minor);
#if defined(FLASHRT_FP8_GEMM_SOURCE_SM89_ONLY) || \
    defined(FLASHRT_FP8_GEMM_SOURCE_SM120_ONLY)
  TORCH_CHECK(false, "SM110 FP8 bias GEMM source is not present in this build");
#else
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  const int M = checked_positive_int(input.size(0), "M");
  const int N = checked_positive_int(weight.size(0), "N");
  const int K = checked_positive_int(input.size(1), "K");
  int rc;
  if (M >= 512 && K >= 3 * N) {
    rc = epilogue == FlashRtFp8BiasEpilogue::kBiasGelu
             ? cutlass_fp8_wide_bias_gelu_bf16out(
                   input.data_ptr(), weight.data_ptr(), bias.data_ptr(),
                   out.data_ptr(), M, N, K, static_cast<float>(alpha), stream)
             : cutlass_fp8_wide_bias_bf16out(
                   input.data_ptr(), weight.data_ptr(), bias.data_ptr(),
                   out.data_ptr(), M, N, K, static_cast<float>(alpha),
                   static_cast<float>(beta), stream);
  } else {
    rc = fp8_linear_bias_sm110_bf16(
        input.data_ptr(), weight.data_ptr(), bias.data_ptr(), out.data_ptr(),
        M, N, K, static_cast<float>(alpha), static_cast<float>(beta),
        epilogue, stream);
  }
  TORCH_CHECK(rc == 0, op_name, " failed with rc=", rc);
#endif
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

void fp8_linear_bias_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out) {
  launch_bias(input, weight, bias, alpha, out, 0.0,
              FlashRtFp8BiasEpilogue::kBias, "fp8_linear_bias_bf16");
}

void fp8_linear_bias_residual_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& residual) {
  launch_bias(input, weight, bias, alpha, residual, 1.0,
              FlashRtFp8BiasEpilogue::kBias,
              "fp8_linear_bias_residual_bf16");
}

void fp8_linear_bias_gelu_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    double alpha,
    torch::Tensor& out) {
  launch_bias(input, weight, bias, alpha, out, 0.0,
              FlashRtFp8BiasEpilogue::kBiasGelu,
              "fp8_linear_bias_gelu_bf16");
}

void fp8_blockwise_linear_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor& out) {
  check_fp8_matrix(input, "input");
  check_fp8_matrix(weight, "weight");
  check_fp32_matrix(input_scale, "input_scale");
  check_fp32_matrix(weight_scale, "weight_scale");
  check_bf16_matrix(out, "out");
  const int64_t M = input.size(0);
  const int64_t K = input.size(1);
  const int64_t N = weight.size(0);
  TORCH_CHECK(weight.size(1) == K,
              "weight must have shape (N, input.shape[1])");
  TORCH_CHECK(K % 128 == 0 && N % 128 == 0,
              "N and K must be divisible by 128");
  TORCH_CHECK(input_scale.sizes() ==
                  torch::IntArrayRef({M, K / 128}),
              "input_scale must have shape (M, K / 128)");
  TORCH_CHECK(weight_scale.sizes() ==
                  torch::IntArrayRef({N / 128, K / 128}),
              "weight_scale must have shape (N / 128, K / 128)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({M, N}),
              "out must have shape (M, N)");
  TORCH_CHECK(input.get_device() == weight.get_device() &&
                  input.get_device() == input_scale.get_device() &&
                  input.get_device() == weight_scale.get_device() &&
                  input.get_device() == out.get_device(),
              "all tensors must be on the same CUDA device");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(input.device());
  auto* props = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK((props->major == 8 && props->minor == 9) ||
                  (props->major == 12 && props->minor == 0),
              "fp8_blockwise_linear_bf16 requires SM89 or SM120; got SM",
              props->major, props->minor);
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  if (props->major == 8) {
#if defined(FLASHRT_FP8_GEMM_SOURCE_SM120_ONLY) || \
    defined(FLASHRT_FP8_GEMM_SOURCE_SM110_ONLY)
    TORCH_CHECK(false, "SM89 blockwise kernels are not present in this source-test build");
#else
    int rc;
    if (M == 1) {
      namespace gemv89 = flash_rt::gemm::gemv_m1_sm89;
      if (N <= 2048) {
        rc = gemv89::gemv_fp8_block128_m1_w4(
            input.data_ptr(), weight.data_ptr(), out.data_ptr(), 1,
            checked_positive_int(N, "N"), checked_positive_int(K, "K"),
            input_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
            1.0f, stream);
      } else if (N <= 8192) {
        rc = gemv89::gemv_fp8_block128_m1_w8(
            input.data_ptr(), weight.data_ptr(), out.data_ptr(), 1,
            checked_positive_int(N, "N"), checked_positive_int(K, "K"),
            input_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
            1.0f, stream);
      } else {
        rc = gemv89::gemv_fp8_block128_m1_w16(
            input.data_ptr(), weight.data_ptr(), out.data_ptr(), 1,
            checked_positive_int(N, "N"), checked_positive_int(K, "K"),
            input_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
            1.0f, stream);
      }
    } else {
      rc = flash_rt::gemm::block128_sm89::
          fp8_block128_gemm_blockscaled_sm89_bf16out(
              input.data_ptr(), weight.data_ptr(), out.data_ptr(),
              checked_positive_int(M, "M"), checked_positive_int(N, "N"),
              checked_positive_int(K, "K"), input_scale.data_ptr<float>(),
              weight_scale.data_ptr<float>(), stream);
    }
    TORCH_CHECK(rc == 0, "SM89 blockwise FP8 linear failed with rc=", rc);
#endif
  } else {
#if defined(FLASHRT_FP8_GEMM_SOURCE_SM89_ONLY) || \
    defined(FLASHRT_FP8_GEMM_SOURCE_SM110_ONLY)
    TORCH_CHECK(false, "SM120 blockwise kernel is not present in this source-test build");
#else
    flash_rt::gemm::fp8_block128_gemm_cutlass_sm120_bf16out(
        input.data_ptr(), weight.data_ptr(), out.data_ptr(),
        checked_positive_int(M, "M"), checked_positive_int(N, "N"),
        checked_positive_int(K, "K"), input_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(), stream);
#endif
  }
#else
  TORCH_CHECK(false, "fp8-gemm was not built with CUDA support");
#endif
}

void fp8_blockwise_swiglu_quantize_fp8(
    torch::Tensor const& input,
    torch::Tensor const& gate_up_weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& gate_up_weight_scale,
    torch::Tensor& output,
    torch::Tensor& output_scale) {
  check_fp8_matrix(input, "input");
  check_fp8_matrix(gate_up_weight, "gate_up_weight");
  check_fp32_matrix(input_scale, "input_scale");
  check_fp32_matrix(gate_up_weight_scale, "gate_up_weight_scale");
  check_fp8_matrix(output, "output");
  check_fp32_matrix(output_scale, "output_scale");
  const int64_t M = input.size(0);
  const int64_t K = input.size(1);
  TORCH_CHECK(gate_up_weight.size(0) % 2 == 0 &&
                  gate_up_weight.size(1) == K,
              "gate_up_weight must have shape (2*N, K)");
  const int64_t N = gate_up_weight.size(0) / 2;
  TORCH_CHECK(M > 0 && M <= 256,
              "SM89 fused SwiGLU producer supports 1 <= M <= 256");
  TORCH_CHECK(N % 128 == 0 && K % 128 == 0,
              "N and K must be divisible by 128");
  TORCH_CHECK(input_scale.sizes() == torch::IntArrayRef({M, K / 128}),
              "input_scale must have shape (M, K / 128)");
  TORCH_CHECK(gate_up_weight_scale.sizes() ==
                  torch::IntArrayRef({2 * N / 128, K / 128}),
              "gate_up_weight_scale must have shape (2*N/128, K/128)");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({M, N}),
              "output must have shape (M, N)");
  TORCH_CHECK(output_scale.sizes() == torch::IntArrayRef({M, N / 128}),
              "output_scale must have shape (M, N/128)");
  TORCH_CHECK(input.get_device() == gate_up_weight.get_device() &&
                  input.get_device() == input_scale.get_device() &&
                  input.get_device() == gate_up_weight_scale.get_device() &&
                  input.get_device() == output.get_device() &&
                  input.get_device() == output_scale.get_device(),
              "all tensors must be on the same CUDA device");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(input.device());
  auto* props = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(props->major == 8 && props->minor == 9,
              "fp8_blockwise_swiglu_quantize_fp8 requires SM89; got SM",
              props->major, props->minor);
#if defined(FLASHRT_FP8_GEMM_SOURCE_SM120_ONLY) || \
    defined(FLASHRT_FP8_GEMM_SOURCE_SM110_ONLY)
  TORCH_CHECK(false, "SM89 fused producer is not present in this source-test build");
#else
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  int rc = flash_rt::gemm::block128_sm89::
      fp8_bs_geglu_silu_fold_sm89_32x128_w4_s1(
          input.data_ptr(), gate_up_weight.data_ptr(),
          checked_positive_int(M, "M"), checked_positive_int(N, "N"),
          checked_positive_int(K, "K"), input_scale.data_ptr<float>(),
          gate_up_weight_scale.data_ptr<float>(), output.data_ptr(),
          output_scale.data_ptr<float>(), stream);
  TORCH_CHECK(rc == 0, "SM89 fused SwiGLU FP8 producer failed with rc=", rc);
#endif
#else
  TORCH_CHECK(false, "fp8-gemm was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp8_linear_bf16(Tensor input, Tensor weight, float alpha, int variant, Tensor! out) -> ()");
  ops.def("fp8_linear_residual_bf16(Tensor input, Tensor weight, float alpha, int variant, Tensor! residual) -> ()");
  ops.def("fp8_linear_bias_bf16(Tensor input, Tensor weight, Tensor bias, float alpha, Tensor! out) -> ()");
  ops.def("fp8_linear_bias_residual_bf16(Tensor input, Tensor weight, Tensor bias, float alpha, Tensor! residual) -> ()");
  ops.def("fp8_linear_bias_gelu_bf16(Tensor input, Tensor weight, Tensor bias, float alpha, Tensor! out) -> ()");
  ops.def("fp8_blockwise_linear_bf16("
          "Tensor input, Tensor weight, Tensor input_scale, "
          "Tensor weight_scale, Tensor! out) -> ()");
  ops.def("fp8_blockwise_swiglu_quantize_fp8("
          "Tensor input, Tensor gate_up_weight, Tensor input_scale, "
          "Tensor gate_up_weight_scale, Tensor! output, Tensor! output_scale) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fp8_linear_bf16", torch::kCUDA, &fp8_linear_bf16);
  ops.impl("fp8_linear_residual_bf16", torch::kCUDA, &fp8_linear_residual_bf16);
  ops.impl("fp8_linear_bias_bf16", torch::kCUDA, &fp8_linear_bias_bf16);
  ops.impl("fp8_linear_bias_residual_bf16", torch::kCUDA,
           &fp8_linear_bias_residual_bf16);
  ops.impl("fp8_linear_bias_gelu_bf16", torch::kCUDA,
           &fp8_linear_bias_gelu_bf16);
  ops.impl("fp8_blockwise_linear_bf16",
           torch::kCUDA,
           &fp8_blockwise_linear_bf16);
  ops.impl("fp8_blockwise_swiglu_quantize_fp8",
           torch::kCUDA,
           &fp8_blockwise_swiglu_quantize_fp8);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
