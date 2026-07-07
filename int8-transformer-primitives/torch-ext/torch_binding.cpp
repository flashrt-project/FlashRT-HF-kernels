// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "int8_transformer_primitives.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be torch.bfloat16");
}

void check_i8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kInt8, name, " must be torch.int8");
}

void check_f32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be torch.float32");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

void same_device(torch::Tensor const& a, torch::Tensor const& b,
                 const char* an, const char* bn) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              an, " and ", bn, " must be on the same CUDA device");
}

void check_matrix_bf16(torch::Tensor const& tensor, const char* name) {
  check_bf16(tensor, name);
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (rows, cols)");
}

void check_matrix_i8(torch::Tensor const& tensor, const char* name) {
  check_i8(tensor, name);
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (rows, cols)");
}

}  // namespace

void quantize_int8_static_bf16(torch::Tensor const& input,
                               torch::Tensor const& scale,
                               torch::Tensor& out) {
  check_bf16(input, "input");
  check_f32(scale, "scale");
  check_i8(out, "out");
  TORCH_CHECK(out.sizes() == input.sizes(), "out shape mismatch");
  TORCH_CHECK(scale.numel() >= 1, "scale must contain at least one float32 value");
  same_device(input, scale, "input", "scale");
  same_device(input, out, "input", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  flashrt_hub::int8_transformer::quantize_int8_static_bf16(
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<int8_t*>(out.data_ptr()),
      static_cast<const float*>(scale.data_ptr()),
      checked_int(input.numel(), "input.numel"), stream);
#else
  TORCH_CHECK(false, "int8-transformer-primitives was not built with CUDA support");
#endif
}

void quantize_int8_rowwise_bf16(torch::Tensor const& input,
                                torch::Tensor& out,
                                torch::Tensor& scales) {
  check_matrix_bf16(input, "input");
  check_i8(out, "out");
  check_f32(scales, "scales");
  TORCH_CHECK(out.sizes() == input.sizes(), "out shape mismatch");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({input.size(0)}), "scales must have shape (rows,)");
  same_device(input, out, "input", "out");
  same_device(input, scales, "input", "scales");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  flashrt_hub::int8_transformer::quantize_int8_rowwise_bf16(
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<int8_t*>(out.data_ptr()),
      static_cast<float*>(scales.data_ptr()),
      checked_int(input.size(0), "rows"),
      checked_int(input.size(1), "cols"), stream);
#endif
}

void quantize_int8_rowwise_static_bf16(torch::Tensor const& input,
                                       torch::Tensor const& scales,
                                       torch::Tensor& out) {
  check_matrix_bf16(input, "input");
  check_f32(scales, "scales");
  check_i8(out, "out");
  TORCH_CHECK(out.sizes() == input.sizes(), "out shape mismatch");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({input.size(0)}), "scales must have shape (rows,)");
  same_device(input, scales, "input", "scales");
  same_device(input, out, "input", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  flashrt_hub::int8_transformer::quantize_int8_rowwise_static_bf16(
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<int8_t*>(out.data_ptr()),
      static_cast<const float*>(scales.data_ptr()),
      checked_int(input.size(0), "rows"),
      checked_int(input.size(1), "cols"), stream);
#endif
}

void rms_norm_quantize_int8_rowwise_bf16(torch::Tensor const& x,
                                         torch::Tensor const& weight,
                                         double eps,
                                         torch::Tensor& out,
                                         torch::Tensor& scales) {
  check_matrix_bf16(x, "x");
  check_bf16(weight, "weight");
  check_i8(out, "out");
  check_f32(scales, "scales");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({x.size(1)}), "weight shape mismatch");
  TORCH_CHECK(out.sizes() == x.sizes(), "out shape mismatch");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({x.size(0)}), "scales must have shape (rows,)");
  same_device(x, weight, "x", "weight");
  same_device(x, out, "x", "out");
  same_device(x, scales, "x", "scales");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flashrt_hub::int8_transformer::rms_norm_quantize_int8_rowwise_bf16(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<int8_t*>(out.data_ptr()),
      static_cast<float*>(scales.data_ptr()),
      checked_int(x.size(0), "rows"),
      checked_int(x.size(1), "cols"),
      static_cast<float>(eps), stream);
#endif
}

void residual_add_rms_norm_quantize_int8_rowwise_bf16(torch::Tensor& residual,
                                                      torch::Tensor const& x,
                                                      torch::Tensor const& weight,
                                                      double eps,
                                                      torch::Tensor& out,
                                                      torch::Tensor& scales) {
  check_matrix_bf16(residual, "residual");
  check_matrix_bf16(x, "x");
  check_bf16(weight, "weight");
  check_i8(out, "out");
  check_f32(scales, "scales");
  TORCH_CHECK(x.sizes() == residual.sizes(), "x/residual shape mismatch");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({residual.size(1)}), "weight shape mismatch");
  TORCH_CHECK(out.sizes() == residual.sizes(), "out shape mismatch");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({residual.size(0)}), "scales must have shape (rows,)");
  same_device(residual, x, "residual", "x");
  same_device(residual, weight, "residual", "weight");
  same_device(residual, out, "residual", "out");
  same_device(residual, scales, "residual", "scales");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(residual.device());
  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device()).stream();
  flashrt_hub::int8_transformer::residual_add_rms_norm_quantize_int8_rowwise_bf16(
      static_cast<__nv_bfloat16*>(residual.data_ptr()),
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<int8_t*>(out.data_ptr()),
      static_cast<float*>(scales.data_ptr()),
      checked_int(residual.size(0), "rows"),
      checked_int(residual.size(1), "cols"),
      static_cast<float>(eps), stream);
#endif
}

void int8_rowwise_linear_bf16(torch::Tensor const& input_i8,
                              torch::Tensor const& weight_i8,
                              torch::Tensor const& input_scale,
                              torch::Tensor const& weight_scale,
                              torch::Tensor& out,
                              int64_t variant) {
  check_matrix_i8(input_i8, "input_i8");
  check_matrix_i8(weight_i8, "weight_i8");
  check_f32(input_scale, "input_scale");
  check_f32(weight_scale, "weight_scale");
  check_bf16(out, "out");
  const int64_t m = input_i8.size(0);
  const int64_t k = input_i8.size(1);
  const int64_t n = weight_i8.size(0);
  TORCH_CHECK(weight_i8.size(1) == k, "weight_i8 must have shape (N, K)");
  TORCH_CHECK(input_scale.sizes() == torch::IntArrayRef({m}), "input_scale must have shape (M,)");
  TORCH_CHECK(weight_scale.sizes() == torch::IntArrayRef({n}), "weight_scale must have shape (N,)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({m, n}), "out must have shape (M, N)");
  TORCH_CHECK(k % 16 == 0 && n % 8 == 0, "K must be divisible by 16 and N by 8");
  TORCH_CHECK(variant >= 0 && variant <= 2, "variant must be 0(auto), 1(default 128x128), or 2(64x128)");
  same_device(input_i8, weight_i8, "input_i8", "weight_i8");
  same_device(input_i8, input_scale, "input_i8", "input_scale");
  same_device(input_i8, weight_scale, "input_i8", "weight_scale");
  same_device(input_i8, out, "input_i8", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input_i8.device());
  auto stream = at::cuda::getCurrentCUDAStream(input_i8.get_device()).stream();
  int rc = 0;
  if (variant == 2) {
    rc = cutlass_int8_rowwise_bf16out_t64x128(
        input_i8.data_ptr(), weight_i8.data_ptr(), input_scale.data_ptr(),
        weight_scale.data_ptr(), out.data_ptr(),
        checked_int(m, "M"), checked_int(n, "N"), checked_int(k, "K"), stream);
  } else {
    rc = cutlass_int8_rowwise_bf16out(
        input_i8.data_ptr(), weight_i8.data_ptr(), input_scale.data_ptr(),
        weight_scale.data_ptr(), out.data_ptr(),
        checked_int(m, "M"), checked_int(n, "N"), checked_int(k, "K"), stream);
  }
  TORCH_CHECK(rc == 0, "int8_rowwise_linear_bf16 failed with rc=", rc);
#endif
}

void int8_silu_gated_linear_bf16(torch::Tensor const& input_i8,
                                 torch::Tensor const& up_weight_i8,
                                 torch::Tensor const& input_scale,
                                 torch::Tensor const& weight_scale,
                                 torch::Tensor const& gate,
                                 torch::Tensor& out) {
  check_matrix_i8(input_i8, "input_i8");
  check_matrix_i8(up_weight_i8, "up_weight_i8");
  check_f32(input_scale, "input_scale");
  check_f32(weight_scale, "weight_scale");
  check_matrix_bf16(gate, "gate");
  check_bf16(out, "out");
  const int64_t m = input_i8.size(0);
  const int64_t k = input_i8.size(1);
  const int64_t n = up_weight_i8.size(0);
  TORCH_CHECK(up_weight_i8.size(1) == k, "up_weight_i8 must have shape (N, K)");
  TORCH_CHECK(input_scale.sizes() == torch::IntArrayRef({m}), "input_scale must have shape (M,)");
  TORCH_CHECK(weight_scale.sizes() == torch::IntArrayRef({n}), "weight_scale must have shape (N,)");
  TORCH_CHECK(gate.sizes() == torch::IntArrayRef({m, n}), "gate must have shape (M, N)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({m, n}), "out must have shape (M, N)");
  TORCH_CHECK(k % 16 == 0 && n % 8 == 0, "K must be divisible by 16 and N by 8");
  same_device(input_i8, up_weight_i8, "input_i8", "up_weight_i8");
  same_device(input_i8, input_scale, "input_i8", "input_scale");
  same_device(input_i8, weight_scale, "input_i8", "weight_scale");
  same_device(input_i8, gate, "input_i8", "gate");
  same_device(input_i8, out, "input_i8", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input_i8.device());
  auto stream = at::cuda::getCurrentCUDAStream(input_i8.get_device()).stream();
  const int rc = cutlass_int8_silu_gated_bf16out(
      input_i8.data_ptr(), up_weight_i8.data_ptr(), input_scale.data_ptr(),
      weight_scale.data_ptr(), gate.data_ptr(), out.data_ptr(),
      checked_int(m, "M"), checked_int(n, "N"), checked_int(k, "K"), stream);
  TORCH_CHECK(rc == 0, "int8_silu_gated_linear_bf16 failed with rc=", rc);
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("quantize_int8_static_bf16(Tensor input, Tensor scale, Tensor! out) -> ()");
  ops.def("quantize_int8_rowwise_bf16(Tensor input, Tensor! out, Tensor! scales) -> ()");
  ops.def("quantize_int8_rowwise_static_bf16(Tensor input, Tensor scales, Tensor! out) -> ()");
  ops.def("rms_norm_quantize_int8_rowwise_bf16(Tensor x, Tensor weight, float eps, Tensor! out, Tensor! scales) -> ()");
  ops.def("residual_add_rms_norm_quantize_int8_rowwise_bf16(Tensor! residual, Tensor x, Tensor weight, float eps, Tensor! out, Tensor! scales) -> ()");
  ops.def("int8_rowwise_linear_bf16(Tensor input_i8, Tensor weight_i8, Tensor input_scale, Tensor weight_scale, Tensor! out, int variant=0) -> ()");
  ops.def("int8_silu_gated_linear_bf16(Tensor input_i8, Tensor up_weight_i8, Tensor input_scale, Tensor weight_scale, Tensor gate, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("quantize_int8_static_bf16", torch::kCUDA, &quantize_int8_static_bf16);
  ops.impl("quantize_int8_rowwise_bf16", torch::kCUDA, &quantize_int8_rowwise_bf16);
  ops.impl("quantize_int8_rowwise_static_bf16", torch::kCUDA, &quantize_int8_rowwise_static_bf16);
  ops.impl("rms_norm_quantize_int8_rowwise_bf16", torch::kCUDA, &rms_norm_quantize_int8_rowwise_bf16);
  ops.impl("residual_add_rms_norm_quantize_int8_rowwise_bf16", torch::kCUDA, &residual_add_rms_norm_quantize_int8_rowwise_bf16);
  ops.impl("int8_rowwise_linear_bf16", torch::kCUDA, &int8_rowwise_linear_bf16);
  ops.impl("int8_silu_gated_linear_bf16", torch::kCUDA, &int8_silu_gated_linear_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
