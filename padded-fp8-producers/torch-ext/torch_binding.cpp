// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "padded_fp8_producers.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_dtype(torch::Tensor const& tensor, c10::ScalarType dtype,
                 const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
}

void same_device(torch::Tensor const& reference, torch::Tensor const& tensor,
                 const char* name) {
  TORCH_CHECK(reference.get_device() == tensor.get_device(), name,
              " must be on the input device");
}

void check_scale(torch::Tensor const& input, torch::Tensor const& scale) {
  check_dtype(scale, c10::ScalarType::Float, "scale");
  TORCH_CHECK(scale.numel() == 1, "scale must contain one float32 value");
  same_device(input, scale, "scale");
}

void check_norm_inputs(
    torch::Tensor const& input, torch::Tensor const& weight,
    torch::Tensor const& gamma, torch::Tensor const& beta,
    torch::Tensor const& scale, torch::Tensor const& output) {
  check_dtype(input, c10::ScalarType::BFloat16, "input");
  TORCH_CHECK(input.dim() == 3 && input.size(0) > 0 && input.size(1) > 0 &&
                  input.size(2) > 0,
              "input must have shape (batch, rows, dim)");
  const auto batch = input.size(0);
  const auto rows = input.size(1);
  const auto dim = input.size(2);
  check_dtype(weight, c10::ScalarType::BFloat16, "weight");
  check_dtype(gamma, c10::ScalarType::BFloat16, "gamma");
  check_dtype(beta, c10::ScalarType::BFloat16, "beta");
  check_dtype(output, c10::ScalarType::Float8_e4m3fn, "output");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({dim}),
              "weight must have shape (dim,)");
  TORCH_CHECK(gamma.sizes() == torch::IntArrayRef({batch, dim}) &&
                  beta.sizes() == torch::IntArrayRef({batch, dim}),
              "gamma and beta must have shape (batch, dim)");
  TORCH_CHECK(output.dim() == 3 && output.size(0) == batch &&
                  output.size(1) >= rows && output.size(2) == dim,
              "output must have shape (batch, padded_rows, dim), padded_rows >= rows");
  check_scale(input, scale);
  same_device(input, weight, "weight");
  same_device(input, gamma, "gamma");
  same_device(input, beta, "beta");
  same_device(input, output, "output");
}

void check_swiglu(
    torch::Tensor const& gate, torch::Tensor const& up,
    torch::Tensor const& scale, torch::Tensor const& output,
    c10::ScalarType input_dtype, bool merged) {
  check_dtype(gate, input_dtype, merged ? "gate_up" : "gate");
  TORCH_CHECK(gate.dim() == 2 && gate.size(0) > 0 && gate.size(1) > 0,
              "input must be a non-empty matrix");
  const auto rows = gate.size(0);
  const auto dim = merged ? gate.size(1) / 2 : gate.size(1);
  TORCH_CHECK(!merged || gate.size(1) % 2 == 0,
              "merged gate_up width must be even");
  if (!merged) {
    check_dtype(up, input_dtype, "up");
    TORCH_CHECK(up.sizes() == gate.sizes(), "up must match gate");
    same_device(gate, up, "up");
  }
  check_dtype(output, c10::ScalarType::Float8_e4m3fn, "output");
  TORCH_CHECK(output.dim() == 2 && output.size(0) >= rows &&
                  output.size(1) == dim,
              "output must have shape (padded_rows, dim), padded_rows >= rows");
  check_scale(gate, scale);
  same_device(gate, output, "output");
}

#if defined(CUDA_KERNEL)
cudaStream_t current_stream(torch::Tensor const& tensor) {
  return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}
#endif

}  // namespace

void adaptive_rms_norm_quant_fp8_padded_bf16(
    torch::Tensor const& input, torch::Tensor const& weight,
    torch::Tensor const& gamma, torch::Tensor const& beta,
    torch::Tensor const& scale, double eps, torch::Tensor& output) {
  check_norm_inputs(input, weight, gamma, beta, scale, output);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  flash_rt::padded_fp8::adaptive_rms_norm_quant_fp8_padded_bf16(
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(gamma.data_ptr()),
      static_cast<const __nv_bfloat16*>(beta.data_ptr()), scale.data_ptr<float>(),
      static_cast<__nv_fp8_e4m3*>(output.data_ptr()), input.size(0),
      input.size(1), output.size(1), input.size(2), static_cast<float>(eps),
      current_stream(input));
#endif
}

void residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
    torch::Tensor const& residual, torch::Tensor const& input,
    torch::Tensor const& weight, torch::Tensor const& gamma,
    torch::Tensor const& beta, torch::Tensor const& scale, double eps,
    torch::Tensor& residual_out, torch::Tensor& output) {
  check_norm_inputs(input, weight, gamma, beta, scale, output);
  check_dtype(residual, c10::ScalarType::BFloat16, "residual");
  check_dtype(residual_out, c10::ScalarType::BFloat16, "residual_out");
  TORCH_CHECK(residual.sizes() == input.sizes() &&
                  residual_out.sizes() == input.sizes(),
              "residual and residual_out must match input");
  same_device(input, residual, "residual");
  same_device(input, residual_out, "residual_out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  flash_rt::padded_fp8::
      residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
          static_cast<const __nv_bfloat16*>(residual.data_ptr()),
          static_cast<const __nv_bfloat16*>(input.data_ptr()),
          static_cast<const __nv_bfloat16*>(weight.data_ptr()),
          static_cast<const __nv_bfloat16*>(gamma.data_ptr()),
          static_cast<const __nv_bfloat16*>(beta.data_ptr()),
          scale.data_ptr<float>(),
          static_cast<__nv_bfloat16*>(residual_out.data_ptr()),
          static_cast<__nv_fp8_e4m3*>(output.data_ptr()), input.size(0),
          input.size(1), output.size(1), input.size(2),
          static_cast<float>(eps), current_stream(input));
#endif
}

void swiglu_quant_fp8_padded_bf16(
    torch::Tensor const& gate, torch::Tensor const& up,
    torch::Tensor const& scale, torch::Tensor& output) {
  check_swiglu(gate, up, scale, output, c10::ScalarType::BFloat16, false);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(gate.device());
  flash_rt::padded_fp8::swiglu_quant_fp8_padded_bf16(
      static_cast<const __nv_bfloat16*>(gate.data_ptr()),
      static_cast<const __nv_bfloat16*>(up.data_ptr()), scale.data_ptr<float>(),
      static_cast<__nv_fp8_e4m3*>(output.data_ptr()), gate.size(0),
      output.size(0), gate.size(1), current_stream(gate));
#endif
}

void swiglu_merged_quant_fp8_padded_bf16(
    torch::Tensor const& gate_up, torch::Tensor const& scale,
    torch::Tensor& output) {
  check_swiglu(gate_up, gate_up, scale, output, c10::ScalarType::BFloat16,
               true);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(gate_up.device());
  flash_rt::padded_fp8::swiglu_merged_quant_fp8_padded_bf16(
      static_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
      scale.data_ptr<float>(), static_cast<__nv_fp8_e4m3*>(output.data_ptr()),
      gate_up.size(0), output.size(0), gate_up.size(1) / 2,
      current_stream(gate_up));
#endif
}

void swiglu_merged_quant_fp8_padded_fp16(
    torch::Tensor const& gate_up, torch::Tensor const& scale,
    torch::Tensor& output) {
  check_swiglu(gate_up, gate_up, scale, output, c10::ScalarType::Half, true);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(gate_up.device());
  flash_rt::padded_fp8::swiglu_merged_quant_fp8_padded_fp16(
      static_cast<const __half*>(gate_up.data_ptr()), scale.data_ptr<float>(),
      static_cast<__nv_fp8_e4m3*>(output.data_ptr()), gate_up.size(0),
      output.size(0), gate_up.size(1) / 2, current_stream(gate_up));
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("adaptive_rms_norm_quant_fp8_padded_bf16(Tensor input, Tensor weight, Tensor gamma, Tensor beta, Tensor scale, float eps, Tensor! output) -> ()");
  ops.def("residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(Tensor residual, Tensor input, Tensor weight, Tensor gamma, Tensor beta, Tensor scale, float eps, Tensor! residual_out, Tensor! output) -> ()");
  ops.def("swiglu_quant_fp8_padded_bf16(Tensor gate, Tensor up, Tensor scale, Tensor! output) -> ()");
  ops.def("swiglu_merged_quant_fp8_padded_bf16(Tensor gate_up, Tensor scale, Tensor! output) -> ()");
  ops.def("swiglu_merged_quant_fp8_padded_fp16(Tensor gate_up, Tensor scale, Tensor! output) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("adaptive_rms_norm_quant_fp8_padded_bf16", torch::kCUDA,
           &adaptive_rms_norm_quant_fp8_padded_bf16);
  ops.impl("residual_add_adaptive_rms_norm_quant_fp8_padded_bf16", torch::kCUDA,
           &residual_add_adaptive_rms_norm_quant_fp8_padded_bf16);
  ops.impl("swiglu_quant_fp8_padded_bf16", torch::kCUDA,
           &swiglu_quant_fp8_padded_bf16);
  ops.impl("swiglu_merged_quant_fp8_padded_bf16", torch::kCUDA,
           &swiglu_merged_quant_fp8_padded_bf16);
  ops.impl("swiglu_merged_quant_fp8_padded_fp16", torch::kCUDA,
           &swiglu_merged_quant_fp8_padded_fp16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
