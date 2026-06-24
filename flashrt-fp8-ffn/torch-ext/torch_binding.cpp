// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#define FLASHRT_DEVICE_GUARD(tensor) c10::cuda::CUDAGuard device_guard((tensor).device())
#define FLASHRT_CURRENT_STREAM(tensor) c10::cuda::getCurrentCUDAStream((tensor).get_device()).stream()
#elif defined(ROCM_KERNEL)
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#define FLASHRT_DEVICE_GUARD(tensor) c10::hip::HIPGuard device_guard((tensor).device())
#define FLASHRT_CURRENT_STREAM(tensor) c10::hip::getCurrentHIPStream((tensor).get_device()).stream()
#endif

#if defined(CUDA_KERNEL)
#include "fp8_ffn.cuh"
#elif defined(ROCM_KERNEL)
#include "rocm/fp8_ffn_rocm.h"
#endif
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_tensor(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_scale(torch::Tensor const& scale, const char* name, int device) {
  check_cuda_tensor(scale, name);
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
  TORCH_CHECK(scale.numel() == 1, name, " must contain exactly one value");
  TORCH_CHECK(scale.get_device() == device,
              name, " must be on the same CUDA device as input");
}

void check_fp8_matrix(torch::Tensor const& t, const char* name) {
  check_cuda_tensor(t, name);
  #if defined(ROCM_KERNEL)
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::Float8_e4m3fnuz,
              name, " must have dtype torch.float8_e4m3fnuz on ROCm");
#else
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
#endif
  TORCH_CHECK(t.dim() == 2, name, " must be a 2D tensor");
  TORCH_CHECK(t.size(0) > 0 && t.size(1) > 0,
              name, " dimensions must be non-zero");
  TORCH_CHECK(t.size(0) <= std::numeric_limits<int>::max(),
              name, ".shape[0] must fit in int");
  TORCH_CHECK(t.size(1) <= std::numeric_limits<int>::max(),
              name, ".shape[1] must fit in int");
}

void check_bf16_matrix(torch::Tensor const& t, const char* name) {
  check_cuda_tensor(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(t.dim() == 2, name, " must be a 2D tensor");
  TORCH_CHECK(t.size(0) > 0 && t.size(1) > 0,
              name, " dimensions must be non-zero");
}

void check_bf16_vector(torch::Tensor const& t, const char* name) {
  check_cuda_tensor(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(t.dim() == 1, name, " must be a 1D tensor");
  TORCH_CHECK(t.size(0) > 0, name, " length must be non-zero");
}

void check_fp8_gemm_args(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor const& out) {
  check_fp8_matrix(input, "input");
  check_fp8_matrix(weight, "weight");
  check_bf16_matrix(out, "out");
  TORCH_CHECK(input.get_device() == weight.get_device(),
              "input and weight must be on the same CUDA device");
  TORCH_CHECK(input.get_device() == out.get_device(),
              "input and out must be on the same CUDA device");
  check_scale(input_scale, "input_scale", input.get_device());
  check_scale(weight_scale, "weight_scale", input.get_device());
  TORCH_CHECK(input.size(1) == weight.size(1),
              "input.shape[1] must match weight.shape[1]");
  TORCH_CHECK(out.size(0) == input.size(0) && out.size(1) == weight.size(0),
              "out must have shape (input.shape[0], weight.shape[0])");
}

void launch_fp8_gemm_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor& out) {
  const int M = static_cast<int>(input.size(0));
  const int K = static_cast<int>(input.size(1));
  const int N = static_cast<int>(weight.size(0));

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  FLASHRT_DEVICE_GUARD(input);
  auto stream = FLASHRT_CURRENT_STREAM(input);
  flash_rt::fp8_ffn::fp8_gemm_descale_bf16out(
      input.data_ptr(),
      weight.data_ptr(),
      out.data_ptr(),
      M,
      N,
      K,
      reinterpret_cast<const float*>(input_scale.data_ptr()),
      reinterpret_cast<const float*>(weight_scale.data_ptr()),
      stream);
#else
  TORCH_CHECK(false, "flashrt-fp8-ffn was not built with CUDA/ROCm support");
#endif
}

void launch_bias_gelu_quant(
    torch::Tensor const& hidden_bf16,
    torch::Tensor const& bias,
    torch::Tensor const& output_scale,
    torch::Tensor& out_fp8) {
  const int M = static_cast<int>(hidden_bf16.size(0));
  const int N = static_cast<int>(hidden_bf16.size(1));

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  FLASHRT_DEVICE_GUARD(hidden_bf16);
  auto stream = FLASHRT_CURRENT_STREAM(hidden_bf16);
  flash_rt::fp8_ffn::bias_gelu_quantize_fp8_static_bf16(
      hidden_bf16.data_ptr(),
      bias.data_ptr(),
      out_fp8.data_ptr(),
      reinterpret_cast<const float*>(output_scale.data_ptr()),
      M,
      N,
      stream);
#else
  TORCH_CHECK(false, "flashrt-fp8-ffn was not built with CUDA/ROCm support");
#endif
}

void launch_add_bias_bf16(torch::Tensor& out, torch::Tensor const& bias) {
  const int M = static_cast<int>(out.size(0));
  const int N = static_cast<int>(out.size(1));

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  FLASHRT_DEVICE_GUARD(out);
  auto stream = FLASHRT_CURRENT_STREAM(out);
  flash_rt::fp8_ffn::add_bias_bf16(
      out.data_ptr(), bias.data_ptr(), M, N, stream);
#else
  TORCH_CHECK(false, "flashrt-fp8-ffn was not built with CUDA/ROCm support");
#endif
}

}  // namespace

void fp8_gemm_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor& out) {
  check_fp8_gemm_args(input, weight, input_scale, weight_scale, out);
  launch_fp8_gemm_bf16(input, weight, input_scale, weight_scale, out);
}

void fp8_linear_bias_gelu_quant_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    torch::Tensor const& input_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor const& output_scale,
    torch::Tensor& hidden_bf16,
    torch::Tensor& out_fp8) {
  check_fp8_gemm_args(input, weight, input_scale, weight_scale, hidden_bf16);
  check_bf16_vector(bias, "bias");
  check_cuda_tensor(out_fp8, "out_fp8");
  #if defined(ROCM_KERNEL)
  TORCH_CHECK(out_fp8.scalar_type() == c10::ScalarType::Float8_e4m3fnuz,
              "out_fp8 must have dtype torch.float8_e4m3fnuz on ROCm");
#else
  TORCH_CHECK(out_fp8.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "out_fp8 must have dtype torch.float8_e4m3fn");
#endif
  TORCH_CHECK(out_fp8.sizes() == hidden_bf16.sizes(),
              "out_fp8 must have the same shape as hidden_bf16");
  TORCH_CHECK(bias.size(0) == weight.size(0),
              "bias length must match weight.shape[0]");
  TORCH_CHECK(bias.get_device() == input.get_device(),
              "bias must be on the same CUDA device as input");
  TORCH_CHECK(out_fp8.get_device() == input.get_device(),
              "out_fp8 must be on the same CUDA device as input");
  check_scale(output_scale, "output_scale", input.get_device());

  launch_fp8_gemm_bf16(input, weight, input_scale, weight_scale, hidden_bf16);
  launch_bias_gelu_quant(hidden_bf16, bias, output_scale, out_fp8);
}

void fp8_gelu_mlp_bf16(
    torch::Tensor const& input,
    torch::Tensor const& up_weight,
    torch::Tensor const& up_bias,
    torch::Tensor const& down_weight,
    torch::Tensor const& down_bias,
    torch::Tensor const& input_scale,
    torch::Tensor const& up_weight_scale,
    torch::Tensor const& hidden_scale,
    torch::Tensor const& down_weight_scale,
    torch::Tensor& hidden_bf16,
    torch::Tensor& hidden_fp8,
    torch::Tensor& out) {
  check_fp8_gemm_args(input, up_weight, input_scale, up_weight_scale,
                      hidden_bf16);
  check_fp8_gemm_args(hidden_fp8, down_weight, hidden_scale, down_weight_scale,
                      out);
  check_bf16_vector(up_bias, "up_bias");
  check_bf16_vector(down_bias, "down_bias");
  TORCH_CHECK(up_bias.size(0) == up_weight.size(0),
              "up_bias length must match up_weight.shape[0]");
  TORCH_CHECK(down_bias.size(0) == down_weight.size(0),
              "down_bias length must match down_weight.shape[0]");
  TORCH_CHECK(hidden_fp8.sizes() == hidden_bf16.sizes(),
              "hidden_fp8 must have the same shape as hidden_bf16");
  TORCH_CHECK(up_weight.size(0) == down_weight.size(1),
              "up_weight.shape[0] must match down_weight.shape[1]");
  TORCH_CHECK(down_weight.size(0) == out.size(1),
              "down_weight.shape[0] must match out.shape[1]");

  launch_fp8_gemm_bf16(input, up_weight, input_scale, up_weight_scale,
                       hidden_bf16);
  launch_bias_gelu_quant(hidden_bf16, up_bias, hidden_scale, hidden_fp8);
  launch_fp8_gemm_bf16(hidden_fp8, down_weight, hidden_scale,
                       down_weight_scale, out);
  launch_add_bias_bf16(out, down_bias);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp8_gemm_bf16("
          "Tensor input, Tensor weight, Tensor input_scale, "
          "Tensor weight_scale, Tensor! out) -> ()");
  ops.def("fp8_linear_bias_gelu_quant_bf16("
          "Tensor input, Tensor weight, Tensor bias, Tensor input_scale, "
          "Tensor weight_scale, Tensor output_scale, Tensor! hidden_bf16, "
          "Tensor! out_fp8) -> ()");
  ops.def("fp8_gelu_mlp_bf16("
          "Tensor input, Tensor up_weight, Tensor up_bias, "
          "Tensor down_weight, Tensor down_bias, Tensor input_scale, "
          "Tensor up_weight_scale, Tensor hidden_scale, "
          "Tensor down_weight_scale, Tensor! hidden_bf16, "
          "Tensor! hidden_fp8, Tensor! out) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("fp8_gemm_bf16", torch::kCUDA, &fp8_gemm_bf16);
  ops.impl("fp8_linear_bias_gelu_quant_bf16",
           torch::kCUDA,
           &fp8_linear_bias_gelu_quant_bf16);
  ops.impl("fp8_gelu_mlp_bf16", torch::kCUDA, &fp8_gelu_mlp_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
