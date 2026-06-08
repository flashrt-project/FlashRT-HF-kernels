// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "adaptive_norms.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_fp32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
}

void check_fp8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
}

void check_same_device(torch::Tensor const& a,
                       torch::Tensor const& b,
                       const char* a_name,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_matrix(torch::Tensor const& tensor, const char* name) {
  check_bf16(tensor, name);
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (rows, dim)");
  TORCH_CHECK(tensor.size(0) > 0 && tensor.size(1) > 0,
              name, " rows and dim must be positive");
  TORCH_CHECK((tensor.size(1) % 2) == 0, name, ".shape[1] must be even");
}

void check_common(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    torch::Tensor const& style,
    torch::Tensor const& out,
    torch::Tensor const& gate_out) {
  check_matrix(x, "x");
  check_bf16(weight, "weight");
  check_bf16(style, "style");
  check_bf16(gate_out, "gate_out");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == dim,
              "weight must have shape (dim,)");
  TORCH_CHECK(style.sizes() == torch::IntArrayRef({rows, 3 * dim}),
              "style must have shape (rows, 3 * dim)");
  TORCH_CHECK(gate_out.sizes() == x.sizes(),
              "gate_out must have the same shape as x");
  check_same_device(x, weight, "x", "weight");
  check_same_device(x, style, "x", "style");
  check_same_device(x, out, "x", "out");
  check_same_device(x, gate_out, "x", "gate_out");
}

}  // namespace

void ada_rms_norm_style_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    torch::Tensor const& style,
    double eps,
    torch::Tensor& out,
    torch::Tensor& gate_out) {
  check_bf16(out, "out");
  check_common(x, weight, style, out, gate_out);
  TORCH_CHECK(out.sizes() == x.sizes(), "out must have the same shape as x");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::adaptive_norms::ada_rms_norm_style_bf16(
      x.data_ptr(),
      weight.data_ptr(),
      style.data_ptr(),
      out.data_ptr(),
      gate_out.data_ptr(),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-adaptive-norms was not built with CUDA support");
#endif
}

void gate_residual_ada_norm_fp8_static_bf16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor const& gate,
    torch::Tensor const& weight,
    torch::Tensor const& style,
    torch::Tensor const& scale,
    double eps,
    torch::Tensor& out,
    torch::Tensor& gate_out) {
  check_matrix(residual, "residual");
  check_matrix(x, "x");
  check_bf16(gate, "gate");
  check_fp32(scale, "scale");
  check_fp8(out, "out");
  check_common(residual, weight, style, out, gate_out);
  TORCH_CHECK(x.sizes() == residual.sizes(), "x must have the same shape as residual");
  TORCH_CHECK(gate.sizes() == residual.sizes(), "gate must have the same shape as residual");
  TORCH_CHECK(out.sizes() == residual.sizes(), "out must have the same shape as residual");
  TORCH_CHECK(scale.numel() == 1, "scale must be a scalar tensor");
  check_same_device(residual, x, "residual", "x");
  check_same_device(residual, gate, "residual", "gate");
  check_same_device(residual, scale, "residual", "scale");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(residual.device());
  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device()).stream();
  flash_rt::adaptive_norms::gate_residual_ada_norm_fp8_static_bf16(
      residual.data_ptr(),
      x.data_ptr(),
      gate.data_ptr(),
      weight.data_ptr(),
      style.data_ptr(),
      out.data_ptr(),
      gate_out.data_ptr(),
      static_cast<int>(residual.size(0)),
      static_cast<int>(residual.size(1)),
      static_cast<float>(eps),
      reinterpret_cast<const float*>(scale.data_ptr()),
      stream);
#else
  TORCH_CHECK(false, "flashrt-adaptive-norms was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("ada_rms_norm_style_bf16("
          "Tensor x, Tensor weight, Tensor style, float eps, Tensor! out, Tensor! gate_out) -> ()");
  ops.def("gate_residual_ada_norm_fp8_static_bf16("
          "Tensor! residual, Tensor x, Tensor gate, Tensor weight, Tensor style, Tensor scale, "
          "float eps, Tensor! out, Tensor! gate_out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("ada_rms_norm_style_bf16",
           torch::kCUDA,
           &ada_rms_norm_style_bf16);
  ops.impl("gate_residual_ada_norm_fp8_static_bf16",
           torch::kCUDA,
           &gate_residual_ada_norm_fp8_static_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
