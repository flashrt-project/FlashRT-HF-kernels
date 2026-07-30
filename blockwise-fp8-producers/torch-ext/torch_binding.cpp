// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "fp8_per_token_block_quant.cuh"
#include "norm_act_to_fp8_block128.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == c10::ScalarType::BFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_fp8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
}

void check_f32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == c10::ScalarType::Float,
              name, " must have dtype torch.float32");
}

void same_device(torch::Tensor const& lhs, torch::Tensor const& rhs,
                 const char* lhs_name, const char* rhs_name) {
  TORCH_CHECK(lhs.get_device() == rhs.get_device(),
              lhs_name, " and ", rhs_name, " must be on the same device");
}

void check_matrix(torch::Tensor const& input) {
  check_bf16(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must have shape (rows, dim)");
  TORCH_CHECK(input.size(0) > 0, "rows must be positive");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % 128 == 0,
              "dim must be a positive multiple of 128");
}

void check_outputs(torch::Tensor const& input, torch::Tensor const& output,
                   torch::Tensor const& scale) {
  check_fp8(output, "output");
  check_f32(scale, "scale");
  TORCH_CHECK(output.sizes() == input.sizes(), "output must match input");
  TORCH_CHECK(scale.sizes() ==
                  torch::IntArrayRef({input.size(0), input.size(1) / 128}),
              "scale must have shape (rows, dim / 128)");
  same_device(input, output, "input", "output");
  same_device(input, scale, "input", "scale");
}

void check_weight(torch::Tensor const& input, torch::Tensor const& weight,
                  const char* name) {
  check_bf16(weight, name);
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({input.size(1)}),
              name, " must have shape (dim,)");
  same_device(input, weight, "input", name);
}

#if defined(CUDA_KERNEL)
cudaStream_t current_stream(torch::Tensor const& tensor) {
  return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}
#endif

}  // namespace

void quantize_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor& output, torch::Tensor& scale) {
  check_matrix(input);
  check_outputs(input, output, scale);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  flash_rt::quantize::fp8_per_token_block128_quant_bf16(
      input.data_ptr(), output.data_ptr(), scale.data_ptr<float>(),
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
      current_stream(input));
#endif
}

void layer_norm_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor const& weight,
    torch::Tensor const& bias, double eps,
    torch::Tensor& output, torch::Tensor& scale) {
  check_matrix(input);
  check_weight(input, weight, "weight");
  check_weight(input, bias, "bias");
  check_outputs(input, output, scale);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  flash_rt::kernels::layer_norm_to_fp8_block128_bf16(
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(bias.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(output.data_ptr()), scale.data_ptr<float>(),
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
      static_cast<float>(eps), current_stream(input));
#endif
}

void rms_norm_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor const& weight, double eps,
    torch::Tensor& output, torch::Tensor& scale) {
  check_matrix(input);
  check_weight(input, weight, "weight");
  check_outputs(input, output, scale);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  flash_rt::quantize::rms_norm_to_fp8_block128_bf16(
      input.data_ptr(), weight.data_ptr(), output.data_ptr(),
      scale.data_ptr<float>(), static_cast<int>(input.size(0)),
      static_cast<int>(input.size(1)), static_cast<float>(eps),
      current_stream(input));
#endif
}

void residual_add_rms_norm_fp8_block128_bf16(
    torch::Tensor const& residual, torch::Tensor const& input,
    torch::Tensor const& weight, double eps, torch::Tensor& residual_out,
    torch::Tensor& output, torch::Tensor& scale) {
  check_matrix(input);
  check_bf16(residual, "residual");
  check_bf16(residual_out, "residual_out");
  TORCH_CHECK(residual.sizes() == input.sizes() &&
                  residual_out.sizes() == input.sizes(),
              "residual and residual_out must match input");
  check_weight(input, weight, "weight");
  check_outputs(input, output, scale);
  same_device(input, residual, "input", "residual");
  same_device(input, residual_out, "input", "residual_out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  flash_rt::quantize::residual_add_rms_norm_to_fp8_block128_bf16(
      residual.data_ptr(), input.data_ptr(), residual_out.data_ptr(),
      weight.data_ptr(), output.data_ptr(), scale.data_ptr<float>(),
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
      static_cast<float>(eps), current_stream(input));
#endif
}

void gelu_tanh_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor& output, torch::Tensor& scale) {
  check_matrix(input);
  check_outputs(input, output, scale);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  flash_rt::kernels::gelu_tanh_to_fp8_block128_bf16(
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(output.data_ptr()), scale.data_ptr<float>(),
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
      current_stream(input));
#endif
}

void gelu_tanh_bias_fp8_block128_bf16(
    torch::Tensor const& input, torch::Tensor const& bias,
    torch::Tensor& output, torch::Tensor& scale) {
  check_matrix(input);
  check_weight(input, bias, "bias");
  check_outputs(input, output, scale);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  flash_rt::kernels::gelu_tanh_bias_to_fp8_block128_bf16(
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<const __nv_bfloat16*>(bias.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(output.data_ptr()), scale.data_ptr<float>(),
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
      current_stream(input));
#endif
}

void silu_mul_fp8_block128_bf16(
    torch::Tensor const& gate, torch::Tensor const& up,
    torch::Tensor& output, torch::Tensor& scale) {
  check_matrix(gate);
  check_bf16(up, "up");
  TORCH_CHECK(up.sizes() == gate.sizes(), "up must match gate");
  check_outputs(gate, output, scale);
  same_device(gate, up, "gate", "up");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(gate.device());
  flash_rt::quantize::silu_mul_to_fp8_block128_bf16(
      gate.data_ptr(), up.data_ptr(), output.data_ptr(),
      scale.data_ptr<float>(), static_cast<int>(gate.size(0)),
      static_cast<int>(gate.size(1)), current_stream(gate));
#endif
}

void silu_mul_merged_fp8_block128_bf16(
    torch::Tensor const& gate_up, torch::Tensor& output, torch::Tensor& scale) {
  check_bf16(gate_up, "gate_up");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) > 0 &&
                  gate_up.size(1) > 0 && gate_up.size(1) % 256 == 0,
              "gate_up must have shape (rows, 2 * dim), dim multiple of 128");
  TORCH_CHECK(output.sizes() ==
                  torch::IntArrayRef({gate_up.size(0), gate_up.size(1) / 2}),
              "output must have shape (rows, dim)");
  check_fp8(output, "output");
  check_f32(scale, "scale");
  TORCH_CHECK(scale.sizes() ==
                  torch::IntArrayRef({gate_up.size(0), gate_up.size(1) / 256}),
              "scale must have shape (rows, dim / 128)");
  same_device(gate_up, output, "gate_up", "output");
  same_device(gate_up, scale, "gate_up", "scale");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(gate_up.device());
  flash_rt::quantize::silu_mul_merged_to_fp8_block128_bf16(
      gate_up.data_ptr(), output.data_ptr(), scale.data_ptr<float>(),
      static_cast<int>(gate_up.size(0)), static_cast<int>(gate_up.size(1) / 2),
      current_stream(gate_up));
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("quantize_fp8_block128_bf16(Tensor input, Tensor! output, Tensor! scale) -> ()");
  ops.def("layer_norm_fp8_block128_bf16(Tensor input, Tensor weight, Tensor bias, float eps, Tensor! output, Tensor! scale) -> ()");
  ops.def("rms_norm_fp8_block128_bf16(Tensor input, Tensor weight, float eps, Tensor! output, Tensor! scale) -> ()");
  ops.def("residual_add_rms_norm_fp8_block128_bf16(Tensor residual, Tensor input, Tensor weight, float eps, Tensor! residual_out, Tensor! output, Tensor! scale) -> ()");
  ops.def("gelu_tanh_fp8_block128_bf16(Tensor input, Tensor! output, Tensor! scale) -> ()");
  ops.def("gelu_tanh_bias_fp8_block128_bf16(Tensor input, Tensor bias, Tensor! output, Tensor! scale) -> ()");
  ops.def("silu_mul_fp8_block128_bf16(Tensor gate, Tensor up, Tensor! output, Tensor! scale) -> ()");
  ops.def("silu_mul_merged_fp8_block128_bf16(Tensor gate_up, Tensor! output, Tensor! scale) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("quantize_fp8_block128_bf16", torch::kCUDA, &quantize_fp8_block128_bf16);
  ops.impl("layer_norm_fp8_block128_bf16", torch::kCUDA, &layer_norm_fp8_block128_bf16);
  ops.impl("rms_norm_fp8_block128_bf16", torch::kCUDA, &rms_norm_fp8_block128_bf16);
  ops.impl("residual_add_rms_norm_fp8_block128_bf16", torch::kCUDA, &residual_add_rms_norm_fp8_block128_bf16);
  ops.impl("gelu_tanh_fp8_block128_bf16", torch::kCUDA, &gelu_tanh_fp8_block128_bf16);
  ops.impl("gelu_tanh_bias_fp8_block128_bf16", torch::kCUDA, &gelu_tanh_bias_fp8_block128_bf16);
  ops.impl("silu_mul_fp8_block128_bf16", torch::kCUDA, &silu_mul_fp8_block128_bf16);
  ops.impl("silu_mul_merged_fp8_block128_bf16", torch::kCUDA, &silu_mul_merged_fp8_block128_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
