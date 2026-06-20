// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "causal_conv1d_state.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_bias(torch::Tensor const& bias, bool has_bias, int64_t c) {
  if (!has_bias) return;
  check_bf16(bias, "bias");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c,
              "bias must have shape (C,)");
}

void same_device(torch::Tensor const& a, torch::Tensor const& b,
                 const char* an, const char* bn) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              an, " and ", bn, " must be on the same CUDA device");
}

void check_weight(torch::Tensor const& w, int64_t c) {
  check_bf16(w, "w");
  TORCH_CHECK(w.dim() == 2 && w.size(0) == c, "w must have shape (C, K)");
  TORCH_CHECK(w.size(1) > 1 && w.size(1) <= 8, "K must satisfy 2 <= K <= 8");
}

const void* bias_ptr(torch::Tensor const& bias, bool has_bias) {
  return has_bias ? bias.data_ptr() : nullptr;
}

}  // namespace

void causal_conv1d_bf16(torch::Tensor const& x, torch::Tensor const& w,
                        torch::Tensor const& bias, torch::Tensor& out,
                        bool has_bias, bool apply_silu) {
  check_bf16(x, "x");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 3, "x must have shape (B, S, C)");
  const int64_t b = x.size(0);
  const int64_t s = x.size(1);
  const int64_t c = x.size(2);
  check_weight(w, c);
  check_bias(bias, has_bias, c);
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x shape");
  same_device(x, w, "x", "w");
  same_device(x, out, "x", "out");
  if (has_bias) same_device(x, bias, "x", "bias");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::kernels::causal_conv1d_qwen36_bf16(
      x.data_ptr(), w.data_ptr(), bias_ptr(bias, has_bias), out.data_ptr(),
      static_cast<int>(b), static_cast<int>(s), static_cast<int>(c),
      static_cast<int>(w.size(1)), apply_silu, stream);
#else
  TORCH_CHECK(false, "causal-conv1d-state was not built with CUDA support");
#endif
}

void causal_conv1d_update_bf16(torch::Tensor const& x_new, torch::Tensor const& w,
                               torch::Tensor const& bias, torch::Tensor& state,
                               torch::Tensor& out, bool has_bias, bool apply_silu) {
  check_bf16(x_new, "x_new");
  check_bf16(state, "state");
  check_bf16(out, "out");
  TORCH_CHECK(x_new.dim() == 2, "x_new must have shape (B, C)");
  const int64_t b = x_new.size(0);
  const int64_t c = x_new.size(1);
  check_weight(w, c);
  check_bias(bias, has_bias, c);
  const int64_t k = w.size(1);
  TORCH_CHECK(state.sizes() == torch::IntArrayRef({b, c, k - 1}),
              "state must have shape (B, C, K-1)");
  TORCH_CHECK(out.sizes() == x_new.sizes(), "out must match x_new shape");
  same_device(x_new, w, "x_new", "w");
  same_device(x_new, state, "x_new", "state");
  same_device(x_new, out, "x_new", "out");
  if (has_bias) same_device(x_new, bias, "x_new", "bias");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(x_new.device());
  auto stream = at::cuda::getCurrentCUDAStream(x_new.get_device()).stream();
  flash_rt::kernels::causal_conv1d_qwen36_update_bf16(
      x_new.data_ptr(), w.data_ptr(), bias_ptr(bias, has_bias),
      out.data_ptr(), state.data_ptr(), static_cast<int>(b), static_cast<int>(c),
      static_cast<int>(k), apply_silu, stream);
#else
  TORCH_CHECK(false, "causal-conv1d-state was not built with CUDA support");
#endif
}

void causal_conv1d_update_inout_bf16(torch::Tensor const& x_new, torch::Tensor const& w,
                                     torch::Tensor const& bias,
                                     torch::Tensor const& state_in,
                                     torch::Tensor& state_out,
                                     torch::Tensor& out,
                                     bool has_bias, bool apply_silu) {
  check_bf16(x_new, "x_new");
  check_bf16(state_in, "state_in");
  check_bf16(state_out, "state_out");
  check_bf16(out, "out");
  TORCH_CHECK(x_new.dim() == 2, "x_new must have shape (B, C)");
  const int64_t b = x_new.size(0);
  const int64_t c = x_new.size(1);
  check_weight(w, c);
  check_bias(bias, has_bias, c);
  const int64_t k = w.size(1);
  TORCH_CHECK(state_in.sizes() == torch::IntArrayRef({b, c, k - 1}),
              "state_in must have shape (B, C, K-1)");
  TORCH_CHECK(state_out.sizes() == state_in.sizes(), "state_out shape mismatch");
  TORCH_CHECK(out.sizes() == x_new.sizes(), "out must match x_new shape");
  same_device(x_new, w, "x_new", "w");
  same_device(x_new, state_in, "x_new", "state_in");
  same_device(x_new, state_out, "x_new", "state_out");
  same_device(x_new, out, "x_new", "out");
  if (has_bias) same_device(x_new, bias, "x_new", "bias");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(x_new.device());
  auto stream = at::cuda::getCurrentCUDAStream(x_new.get_device()).stream();
  flash_rt::kernels::causal_conv1d_qwen36_update_inout_bf16(
      x_new.data_ptr(), w.data_ptr(), bias_ptr(bias, has_bias), out.data_ptr(),
      state_in.data_ptr(), state_out.data_ptr(), static_cast<int>(b),
      static_cast<int>(c), static_cast<int>(k), apply_silu, stream);
#else
  TORCH_CHECK(false, "causal-conv1d-state was not built with CUDA support");
#endif
}

void causal_conv1d_update_chunk_bf16(torch::Tensor const& x, torch::Tensor const& w,
                                     torch::Tensor const& bias, torch::Tensor& state,
                                     torch::Tensor& out, bool has_bias, bool apply_silu) {
  check_bf16(x, "x");
  check_bf16(state, "state");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 3, "x must have shape (B, S, C)");
  const int64_t b = x.size(0);
  const int64_t s = x.size(1);
  const int64_t c = x.size(2);
  check_weight(w, c);
  check_bias(bias, has_bias, c);
  const int64_t k = w.size(1);
  TORCH_CHECK(state.sizes() == torch::IntArrayRef({b, c, k - 1}),
              "state must have shape (B, C, K-1)");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x shape");
  same_device(x, w, "x", "w");
  same_device(x, state, "x", "state");
  same_device(x, out, "x", "out");
  if (has_bias) same_device(x, bias, "x", "bias");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::kernels::causal_conv1d_qwen36_update_chunk_bf16(
      x.data_ptr(), w.data_ptr(), bias_ptr(bias, has_bias), out.data_ptr(),
      state.data_ptr(), static_cast<int>(b), static_cast<int>(s),
      static_cast<int>(c), static_cast<int>(k), apply_silu, stream);
#else
  TORCH_CHECK(false, "causal-conv1d-state was not built with CUDA support");
#endif
}

void causal_conv1d_update_chunk_parallel_bf16(torch::Tensor const& x, torch::Tensor const& w,
                                              torch::Tensor const& bias, torch::Tensor& state,
                                              torch::Tensor& out, bool has_bias, bool apply_silu) {
  check_bf16(x, "x");
  check_bf16(state, "state");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 3, "x must have shape (B, S, C)");
  const int64_t b = x.size(0);
  const int64_t s = x.size(1);
  const int64_t c = x.size(2);
  check_weight(w, c);
  check_bias(bias, has_bias, c);
  const int64_t k = w.size(1);
  TORCH_CHECK(state.sizes() == torch::IntArrayRef({b, c, k - 1}),
              "state must have shape (B, C, K-1)");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x shape");
  same_device(x, w, "x", "w");
  same_device(x, state, "x", "state");
  same_device(x, out, "x", "out");
  if (has_bias) same_device(x, bias, "x", "bias");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::kernels::causal_conv1d_qwen36_update_chunk_parallel_bf16(
      x.data_ptr(), w.data_ptr(), bias_ptr(bias, has_bias), out.data_ptr(),
      state.data_ptr(), static_cast<int>(b), static_cast<int>(s),
      static_cast<int>(c), static_cast<int>(k), apply_silu, stream);
#else
  TORCH_CHECK(false, "causal-conv1d-state was not built with CUDA support");
#endif
}

void causal_conv1d_update_chunk_parallel_gqa_bf16(torch::Tensor const& x, torch::Tensor const& w,
                                                  torch::Tensor const& bias, torch::Tensor& state,
                                                  torch::Tensor& q16, torch::Tensor& k16,
                                                  torch::Tensor& v48, bool has_bias,
                                                  bool apply_silu) {
  check_bf16(x, "x");
  check_bf16(state, "state");
  check_bf16(q16, "q16");
  check_bf16(k16, "k16");
  check_bf16(v48, "v48");
  TORCH_CHECK(x.dim() == 3, "x must have shape (B, S, C)");
  const int64_t b = x.size(0);
  const int64_t s = x.size(1);
  const int64_t c = x.size(2);
  TORCH_CHECK(c == 10240, "GQA split variant requires C=10240");
  check_weight(w, c);
  check_bias(bias, has_bias, c);
  const int64_t k = w.size(1);
  TORCH_CHECK(state.sizes() == torch::IntArrayRef({b, c, k - 1}),
              "state must have shape (B, C, K-1)");
  TORCH_CHECK(q16.sizes() == torch::IntArrayRef({b, s, 16, 128}), "q16 shape mismatch");
  TORCH_CHECK(k16.sizes() == q16.sizes(), "k16 shape mismatch");
  TORCH_CHECK(v48.sizes() == torch::IntArrayRef({b, s, 48, 128}), "v48 shape mismatch");
  same_device(x, w, "x", "w");
  same_device(x, state, "x", "state");
  same_device(x, q16, "x", "q16");
  same_device(x, k16, "x", "k16");
  same_device(x, v48, "x", "v48");
  if (has_bias) same_device(x, bias, "x", "bias");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::kernels::causal_conv1d_qwen36_update_chunk_parallel_gqa_bf16(
      x.data_ptr(), w.data_ptr(), bias_ptr(bias, has_bias),
      q16.data_ptr(), k16.data_ptr(), v48.data_ptr(), state.data_ptr(),
      static_cast<int>(b), static_cast<int>(s), static_cast<int>(c),
      static_cast<int>(k), apply_silu, stream);
#else
  TORCH_CHECK(false, "causal-conv1d-state was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("causal_conv1d_bf16(Tensor x, Tensor w, Tensor bias, Tensor! out, bool has_bias=True, bool apply_silu=True) -> ()");
  ops.def("causal_conv1d_update_bf16(Tensor x_new, Tensor w, Tensor bias, Tensor! state, Tensor! out, bool has_bias=True, bool apply_silu=True) -> ()");
  ops.def("causal_conv1d_update_inout_bf16(Tensor x_new, Tensor w, Tensor bias, Tensor state_in, Tensor! state_out, Tensor! out, bool has_bias=True, bool apply_silu=True) -> ()");
  ops.def("causal_conv1d_update_chunk_bf16(Tensor x, Tensor w, Tensor bias, Tensor! state, Tensor! out, bool has_bias=True, bool apply_silu=True) -> ()");
  ops.def("causal_conv1d_update_chunk_parallel_bf16(Tensor x, Tensor w, Tensor bias, Tensor! state, Tensor! out, bool has_bias=True, bool apply_silu=True) -> ()");
  ops.def("causal_conv1d_update_chunk_parallel_gqa_bf16(Tensor x, Tensor w, Tensor bias, Tensor! state, Tensor! q16, Tensor! k16, Tensor! v48, bool has_bias=True, bool apply_silu=True) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("causal_conv1d_bf16", torch::kCUDA, &causal_conv1d_bf16);
  ops.impl("causal_conv1d_update_bf16", torch::kCUDA, &causal_conv1d_update_bf16);
  ops.impl("causal_conv1d_update_inout_bf16", torch::kCUDA, &causal_conv1d_update_inout_bf16);
  ops.impl("causal_conv1d_update_chunk_bf16", torch::kCUDA, &causal_conv1d_update_chunk_bf16);
  ops.impl("causal_conv1d_update_chunk_parallel_bf16", torch::kCUDA, &causal_conv1d_update_chunk_parallel_bf16);
  ops.impl("causal_conv1d_update_chunk_parallel_gqa_bf16", torch::kCUDA, &causal_conv1d_update_chunk_parallel_gqa_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
