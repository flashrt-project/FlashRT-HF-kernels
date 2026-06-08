// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "residual_norm_quant.cuh"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16_matrix(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a 2D tensor");
  TORCH_CHECK(tensor.size(0) > 0 && tensor.size(1) > 0,
              name, " dimensions must be non-zero");
  TORCH_CHECK(tensor.size(0) <= std::numeric_limits<int>::max(),
              name, ".shape[0] must fit in int");
  TORCH_CHECK(tensor.size(1) <= std::numeric_limits<int>::max(),
              name, ".shape[1] must fit in int");
  TORCH_CHECK(tensor.size(1) % 2 == 0,
              name, ".shape[1] must be even for packed BF16 path");
}

void check_bf16_vector(torch::Tensor const& tensor,
                       const char* name,
                       int64_t dim,
                       int device) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.dim() == 1, name, " must be a 1D tensor");
  TORCH_CHECK(tensor.size(0) == dim,
              name, " must have shape (input.shape[1],)");
  TORCH_CHECK(tensor.get_device() == device,
              name, " must be on the same CUDA device as input");
}

void check_fp8_matrix(torch::Tensor const& tensor,
                      const char* name,
                      int64_t rows,
                      int64_t dim,
                      int device) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a 2D tensor");
  TORCH_CHECK(tensor.size(0) == rows && tensor.size(1) == dim,
              name, " must have same shape as input");
  TORCH_CHECK(tensor.get_device() == device,
              name, " must be on the same CUDA device as input");
}

void check_scale(torch::Tensor const& scale, const char* name, int device) {
  check_cuda_contiguous(scale, name);
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
  TORCH_CHECK(scale.numel() == 1, name, " must contain exactly one value");
  TORCH_CHECK(scale.get_device() == device,
              name, " must be on the same CUDA device as input");
}

void check_same_shape_device(torch::Tensor const& x,
                             torch::Tensor const& y,
                             const char* x_name,
                             const char* y_name) {
  TORCH_CHECK(x.sizes() == y.sizes(),
              x_name, " and ", y_name, " must have the same shape");
  TORCH_CHECK(x.get_device() == y.get_device(),
              x_name, " and ", y_name, " must be on the same CUDA device");
}

}  // namespace

void rms_norm_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    double eps,
    torch::Tensor& out) {
  check_bf16_matrix(x, "x");
  check_bf16_vector(weight, "weight", x.size(1), x.get_device());
  check_bf16_matrix(out, "out");
  check_same_shape_device(x, out, "x", "out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::residual_norm_quant::rms_norm_bf16(
      x.data_ptr(),
      weight.data_ptr(),
      out.data_ptr(),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-residual-norm-quant was not built with CUDA support");
#endif
}

void rms_norm_quant_fp8_static_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    torch::Tensor const& scale,
    double eps,
    torch::Tensor& out) {
  check_bf16_matrix(x, "x");
  check_bf16_vector(weight, "weight", x.size(1), x.get_device());
  check_scale(scale, "scale", x.get_device());
  check_fp8_matrix(out, "out", x.size(0), x.size(1), x.get_device());

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::residual_norm_quant::rms_norm_quant_fp8_static_bf16(
      x.data_ptr(),
      weight.data_ptr(),
      out.data_ptr(),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      reinterpret_cast<const float*>(scale.data_ptr()),
      stream);
#else
  TORCH_CHECK(false, "flashrt-residual-norm-quant was not built with CUDA support");
#endif
}

void residual_add_rms_norm_quant_fp8_static_bf16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor const& weight,
    torch::Tensor const& scale,
    double eps,
    torch::Tensor& out) {
  check_bf16_matrix(residual, "residual");
  check_bf16_matrix(x, "x");
  check_same_shape_device(residual, x, "residual", "x");
  check_bf16_vector(weight, "weight", x.size(1), x.get_device());
  check_scale(scale, "scale", x.get_device());
  check_fp8_matrix(out, "out", x.size(0), x.size(1), x.get_device());

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::residual_norm_quant::residual_add_rms_norm_quant_fp8_static_bf16(
      residual.data_ptr(),
      x.data_ptr(),
      weight.data_ptr(),
      out.data_ptr(),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<float>(eps),
      reinterpret_cast<const float*>(scale.data_ptr()),
      stream);
#else
  TORCH_CHECK(false, "flashrt-residual-norm-quant was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rms_norm_bf16("
          "Tensor x, Tensor weight, float eps, Tensor! out) -> ()");
  ops.def("rms_norm_quant_fp8_static_bf16("
          "Tensor x, Tensor weight, Tensor scale, float eps, Tensor! out) -> ()");
  ops.def("residual_add_rms_norm_quant_fp8_static_bf16("
          "Tensor! residual, Tensor x, Tensor weight, Tensor scale, "
          "float eps, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("rms_norm_bf16", torch::kCUDA, &rms_norm_bf16);
  ops.impl("rms_norm_quant_fp8_static_bf16",
           torch::kCUDA,
           &rms_norm_quant_fp8_static_bf16);
  ops.impl("residual_add_rms_norm_quant_fp8_static_bf16",
           torch::kCUDA,
           &residual_add_rms_norm_quant_fp8_static_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
