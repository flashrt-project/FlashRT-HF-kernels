// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#if defined(CUDA_KERNEL)
#include "fp8_swiglu_ffn.cuh"
#elif defined(ROCM_KERNEL)
#include "rocm/fp8_swiglu_ffn_rocm.h"
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

void check_fp8_output(torch::Tensor const& t, const char* name) {
  check_cuda_tensor(t, name);
#if defined(ROCM_KERNEL)
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::Float8_e4m3fnuz,
              name, " must have dtype torch.float8_e4m3fnuz on ROCm");
#else
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
#endif
  TORCH_CHECK(t.dim() == 2, name, " must be a 2D tensor");
}

void check_same_device(torch::Tensor const& a, torch::Tensor const& b,
                       const char* a_name, const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
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
  check_same_device(input, weight, "input", "weight");
  check_same_device(input, out, "input", "out");
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
  at::cuda::CUDAGuard device_guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  flash_rt::fp8_swiglu_ffn::fp8_gemm_descale_bf16out(
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
  TORCH_CHECK(false, "flashrt-fp8-swiglu-ffn was not built with CUDA/ROCm support");
#endif
}

void launch_swiglu_quant(
    torch::Tensor const& gate_up_bf16,
    torch::Tensor const& output_scale,
    torch::Tensor& out_fp8,
    bool use_gelu) {
  const long long M = gate_up_bf16.size(0);
  const int H = static_cast<int>(out_fp8.size(1));

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  at::cuda::CUDAGuard device_guard(gate_up_bf16.device());
  auto stream =
      at::cuda::getCurrentCUDAStream(gate_up_bf16.get_device()).stream();
  if (use_gelu) {
    flash_rt::fp8_swiglu_ffn::gelu_mul_merged_quantize_fp8_static_bf16(
        gate_up_bf16.data_ptr(),
        out_fp8.data_ptr(),
        reinterpret_cast<const float*>(output_scale.data_ptr()),
        M,
        H,
        stream);
  } else {
    flash_rt::fp8_swiglu_ffn::silu_mul_merged_quantize_fp8_static_bf16(
        gate_up_bf16.data_ptr(),
        out_fp8.data_ptr(),
        reinterpret_cast<const float*>(output_scale.data_ptr()),
        M,
        H,
        stream);
  }
#else
  TORCH_CHECK(false, "flashrt-fp8-swiglu-ffn was not built with CUDA/ROCm support");
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

void silu_mul_merged_quantize_fp8_static_bf16(
    torch::Tensor const& gate_up_bf16,
    torch::Tensor const& output_scale,
    torch::Tensor& out_fp8) {
  check_bf16_matrix(gate_up_bf16, "gate_up_bf16");
  check_fp8_output(out_fp8, "out_fp8");
  check_same_device(gate_up_bf16, out_fp8, "gate_up_bf16", "out_fp8");
  check_scale(output_scale, "output_scale", gate_up_bf16.get_device());
  TORCH_CHECK(gate_up_bf16.size(1) % 2 == 0,
              "gate_up_bf16.shape[1] must be even");
  TORCH_CHECK(out_fp8.size(0) == gate_up_bf16.size(0) &&
                  out_fp8.size(1) == gate_up_bf16.size(1) / 2,
              "out_fp8 must have shape (gate_up_bf16.shape[0], "
              "gate_up_bf16.shape[1] / 2)");
  TORCH_CHECK(out_fp8.size(1) <= std::numeric_limits<int>::max(),
              "hidden dimension must fit in int");
  launch_swiglu_quant(gate_up_bf16, output_scale, out_fp8, false);
}

void gelu_mul_merged_quantize_fp8_static_bf16(
    torch::Tensor const& gate_up_bf16,
    torch::Tensor const& output_scale,
    torch::Tensor& out_fp8) {
  check_bf16_matrix(gate_up_bf16, "gate_up_bf16");
  check_fp8_output(out_fp8, "out_fp8");
  check_same_device(gate_up_bf16, out_fp8, "gate_up_bf16", "out_fp8");
  check_scale(output_scale, "output_scale", gate_up_bf16.get_device());
  TORCH_CHECK(gate_up_bf16.size(1) % 2 == 0,
              "gate_up_bf16.shape[1] must be even");
  TORCH_CHECK(out_fp8.size(0) == gate_up_bf16.size(0) &&
                  out_fp8.size(1) == gate_up_bf16.size(1) / 2,
              "out_fp8 must have shape (gate_up_bf16.shape[0], "
              "gate_up_bf16.shape[1] / 2)");
  TORCH_CHECK(out_fp8.size(1) <= std::numeric_limits<int>::max(),
              "hidden dimension must fit in int");
  launch_swiglu_quant(gate_up_bf16, output_scale, out_fp8, true);
}

void fp8_swiglu_mlp_bf16(
    torch::Tensor const& input,
    torch::Tensor const& gate_up_weight,
    torch::Tensor const& down_weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& gate_up_weight_scale,
    torch::Tensor const& hidden_scale,
    torch::Tensor const& down_weight_scale,
    torch::Tensor& gate_up_bf16,
    torch::Tensor& hidden_fp8,
    torch::Tensor& out) {
  check_fp8_gemm_args(input, gate_up_weight, input_scale, gate_up_weight_scale,
                      gate_up_bf16);
  check_fp8_gemm_args(hidden_fp8, down_weight, hidden_scale, down_weight_scale,
                      out);
  check_scale(hidden_scale, "hidden_scale", input.get_device());
  TORCH_CHECK(gate_up_weight.size(0) % 2 == 0,
              "gate_up_weight.shape[0] must be even");
  const auto hidden = gate_up_weight.size(0) / 2;
  TORCH_CHECK(hidden_fp8.size(0) == input.size(0) &&
                  hidden_fp8.size(1) == hidden,
              "hidden_fp8 must have shape "
              "(input.shape[0], gate_up_weight.shape[0] / 2)");
  TORCH_CHECK(down_weight.size(1) == hidden,
              "down_weight.shape[1] must match hidden size");

  launch_fp8_gemm_bf16(input, gate_up_weight, input_scale,
                       gate_up_weight_scale, gate_up_bf16);
  launch_swiglu_quant(gate_up_bf16, hidden_scale, hidden_fp8, false);
  launch_fp8_gemm_bf16(hidden_fp8, down_weight, hidden_scale,
                       down_weight_scale, out);
}

void fp8_geglu_mlp_bf16(
    torch::Tensor const& input,
    torch::Tensor const& gate_up_weight,
    torch::Tensor const& down_weight,
    torch::Tensor const& input_scale,
    torch::Tensor const& gate_up_weight_scale,
    torch::Tensor const& hidden_scale,
    torch::Tensor const& down_weight_scale,
    torch::Tensor& gate_up_bf16,
    torch::Tensor& hidden_fp8,
    torch::Tensor& out) {
  check_fp8_gemm_args(input, gate_up_weight, input_scale, gate_up_weight_scale,
                      gate_up_bf16);
  check_fp8_gemm_args(hidden_fp8, down_weight, hidden_scale, down_weight_scale,
                      out);
  check_scale(hidden_scale, "hidden_scale", input.get_device());
  TORCH_CHECK(gate_up_weight.size(0) % 2 == 0,
              "gate_up_weight.shape[0] must be even");
  const auto hidden = gate_up_weight.size(0) / 2;
  TORCH_CHECK(hidden_fp8.size(0) == input.size(0) &&
                  hidden_fp8.size(1) == hidden,
              "hidden_fp8 must have shape "
              "(input.shape[0], gate_up_weight.shape[0] / 2)");
  TORCH_CHECK(down_weight.size(1) == hidden,
              "down_weight.shape[1] must match hidden size");

  launch_fp8_gemm_bf16(input, gate_up_weight, input_scale,
                       gate_up_weight_scale, gate_up_bf16);
  launch_swiglu_quant(gate_up_bf16, hidden_scale, hidden_fp8, true);
  launch_fp8_gemm_bf16(hidden_fp8, down_weight, hidden_scale,
                       down_weight_scale, out);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp8_gemm_bf16("
          "Tensor input, Tensor weight, Tensor input_scale, "
          "Tensor weight_scale, Tensor! out) -> ()");
  ops.def("silu_mul_merged_quantize_fp8_static_bf16("
          "Tensor gate_up_bf16, Tensor output_scale, Tensor! out_fp8) -> ()");
  ops.def("gelu_mul_merged_quantize_fp8_static_bf16("
          "Tensor gate_up_bf16, Tensor output_scale, Tensor! out_fp8) -> ()");
  ops.def("fp8_swiglu_mlp_bf16("
          "Tensor input, Tensor gate_up_weight, Tensor down_weight, "
          "Tensor input_scale, Tensor gate_up_weight_scale, "
          "Tensor hidden_scale, Tensor down_weight_scale, "
          "Tensor! gate_up_bf16, Tensor! hidden_fp8, Tensor! out) -> ()");
  ops.def("fp8_geglu_mlp_bf16("
          "Tensor input, Tensor gate_up_weight, Tensor down_weight, "
          "Tensor input_scale, Tensor gate_up_weight_scale, "
          "Tensor hidden_scale, Tensor down_weight_scale, "
          "Tensor! gate_up_bf16, Tensor! hidden_fp8, Tensor! out) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("fp8_gemm_bf16", torch::kCUDA, &fp8_gemm_bf16);
  ops.impl("silu_mul_merged_quantize_fp8_static_bf16",
           torch::kCUDA,
           &silu_mul_merged_quantize_fp8_static_bf16);
  ops.impl("gelu_mul_merged_quantize_fp8_static_bf16",
           torch::kCUDA,
           &gelu_mul_merged_quantize_fp8_static_bf16);
  ops.impl("fp8_swiglu_mlp_bf16", torch::kCUDA, &fp8_swiglu_mlp_bf16);
  ops.impl("fp8_geglu_mlp_bf16", torch::kCUDA, &fp8_geglu_mlp_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
