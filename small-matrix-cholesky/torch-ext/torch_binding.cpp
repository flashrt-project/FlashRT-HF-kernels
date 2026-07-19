// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#endif

#include "cholesky_small_fp32.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_fp32_contiguous(
    torch::Tensor const& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32,
      name,
      " must have dtype torch.float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void cholesky_small_fp32_out(
    torch::Tensor const& input,
    torch::Tensor& output) {
  check_cuda_fp32_contiguous(input, "input");
  check_cuda_fp32_contiguous(output, "output");
  TORCH_CHECK(input.dim() >= 2, "input must have at least two dimensions");
  TORCH_CHECK(
      input.sizes() == output.sizes(),
      "output must have the same shape as input");
  TORCH_CHECK(
      input.get_device() == output.get_device(),
      "input and output must be on the same CUDA device");
  TORCH_CHECK(
      input.data_ptr() != output.data_ptr(),
      "input and output must not alias");

  const int64_t n = input.size(-1);
  TORCH_CHECK(input.size(-2) == n, "the last two dimensions must be square");
  TORCH_CHECK(
      n == 32 || n == 64 || n == 128,
      "supported matrix orders are 32, 64, and 128");

  const int64_t matrix_elements = n * n;
  const int64_t batch = input.numel() / matrix_elements;
  TORCH_CHECK(batch > 0, "input batch must be non-empty");
  TORCH_CHECK(
      batch <= std::numeric_limits<int>::max(),
      "flattened batch is too large");

#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  C10_CUDA_CHECK(flashrt_hub::cholesky::cholesky_small_fp32(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(n),
      stream));
#else
  TORCH_CHECK(false, "small-matrix-cholesky was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "cholesky_small_fp32_out(Tensor input, Tensor! output) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl(
      "cholesky_small_fp32_out",
      torch::kCUDA,
      &cholesky_small_fp32_out);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
