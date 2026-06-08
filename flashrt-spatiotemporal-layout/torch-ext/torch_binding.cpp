// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "spatiotemporal_layout.cuh"
#include "torch_binding.h"

namespace {

void check_bf16(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_ncdhw(torch::Tensor const& tensor, const char* name) {
  check_bf16(tensor, name);
  TORCH_CHECK(tensor.dim() == 5, name, " must have shape (B, C, T, H, W)");
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(tensor.size(i) > 0, name, " dimensions must be positive");
  }
}

void check_same_device(torch::Tensor const& a,
                       torch::Tensor const& b,
                       const char* a_name,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

}  // namespace

void ncdhw_to_blc_bf16(torch::Tensor const& x, torch::Tensor& out) {
  check_ncdhw(x, "x");
  check_bf16(out, "out");
  const int64_t b = x.size(0);
  const int64_t c = x.size(1);
  const int64_t t = x.size(2);
  const int64_t h = x.size(3);
  const int64_t w = x.size(4);
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, t * h * w, c}),
              "out must have shape (B, T * H * W, C)");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::spatiotemporal_layout::ncdhw_to_blc_bf16(
      x.data_ptr(), out.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void time_unshuffle2_bf16(torch::Tensor const& x, torch::Tensor& out) {
  check_ncdhw(x, "x");
  check_bf16(out, "out");
  const int64_t b = x.size(0);
  const int64_t c2 = x.size(1);
  const int64_t t = x.size(2);
  const int64_t h = x.size(3);
  const int64_t w = x.size(4);
  TORCH_CHECK((c2 % 2) == 0, "x.shape[1] must be even");
  const int64_t c = c2 / 2;
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, c, 2 * t, h, w}),
              "out must have shape (B, C / 2, 2 * T, H, W)");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::spatiotemporal_layout::time_unshuffle2_bf16(
      x.data_ptr(), out.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void add_bias_ncdhw_bf16(torch::Tensor& x, torch::Tensor const& bias) {
  check_ncdhw(x, "x");
  check_bf16(bias, "bias");
  const int64_t b = x.size(0);
  const int64_t c = x.size(1);
  const int64_t t = x.size(2);
  const int64_t h = x.size(3);
  const int64_t w = x.size(4);
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c,
              "bias must have shape (C,)");
  check_same_device(x, bias, "x", "bias");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::spatiotemporal_layout::add_bias_ncdhw_bf16(
      x.data_ptr(), bias.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

void update_cache2_ncdhw_bf16(torch::Tensor const& cur, torch::Tensor const& prev, torch::Tensor& out) {
  check_ncdhw(cur, "cur");
  check_bf16(prev, "prev");
  check_bf16(out, "out");
  const int64_t b = cur.size(0);
  const int64_t c = cur.size(1);
  const int64_t t = cur.size(2);
  const int64_t h = cur.size(3);
  const int64_t w = cur.size(4);
  TORCH_CHECK(prev.sizes() == torch::IntArrayRef({b, c, 2, h, w}),
              "prev must have shape (B, C, 2, H, W)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({b, c, 2, h, w}),
              "out must have shape (B, C, 2, H, W)");
  check_same_device(cur, prev, "cur", "prev");
  check_same_device(cur, out, "cur", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(cur.device());
  auto stream = at::cuda::getCurrentCUDAStream(cur.get_device()).stream();
  flash_rt::spatiotemporal_layout::update_cache2_ncdhw_bf16(
      cur.data_ptr(), prev.data_ptr(), out.data_ptr(),
      static_cast<int>(b), static_cast<int>(c), static_cast<int>(t),
      static_cast<int>(h), static_cast<int>(w), stream);
#else
  TORCH_CHECK(false, "flashrt-spatiotemporal-layout was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("ncdhw_to_blc_bf16(Tensor x, Tensor! out) -> ()");
  ops.def("time_unshuffle2_bf16(Tensor x, Tensor! out) -> ()");
  ops.def("add_bias_ncdhw_bf16(Tensor! x, Tensor bias) -> ()");
  ops.def("update_cache2_ncdhw_bf16(Tensor cur, Tensor prev, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("ncdhw_to_blc_bf16", torch::kCUDA, &ncdhw_to_blc_bf16);
  ops.impl("time_unshuffle2_bf16", torch::kCUDA, &time_unshuffle2_bf16);
  ops.impl("add_bias_ncdhw_bf16", torch::kCUDA, &add_bias_ncdhw_bf16);
  ops.impl("update_cache2_ncdhw_bf16", torch::kCUDA, &update_cache2_ncdhw_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
