// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "linear_attention_primitives.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must have dtype torch.bfloat16");
}

void check_f32(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must have dtype torch.float32");
}

void same_device(torch::Tensor const& a, torch::Tensor const& b, const char* an, const char* bn) {
  TORCH_CHECK(a.get_device() == b.get_device(), an, " and ", bn, " must be on the same CUDA device");
}

void check_k_supported(int64_t k) {
  TORCH_CHECK(k > 0, "K must be positive");
  TORCH_CHECK(k <= 65536, "K is too large for this source package");
}

}  // namespace

void bf16_matvec(torch::Tensor const& x, torch::Tensor const& w, torch::Tensor& out) {
  check_bf16(x, "x");
  check_bf16(w, "w");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 1, "x must have shape (K,)");
  TORCH_CHECK(w.dim() == 2, "w must have shape (N, K)");
  const int64_t n = w.size(0);
  const int64_t k = w.size(1);
  check_k_supported(k);
  TORCH_CHECK(n >= 256, "bf16_matvec supports N >= 256; use PyTorch for launch-floor tiny-N shapes");
  TORCH_CHECK(x.size(0) == k, "x and w K mismatch");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({n}), "out must have shape (N,)");
  same_device(x, w, "x", "w");
  same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::linear_attention_primitives::bf16_matvec(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(w.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<int>(n), static_cast<int>(k), stream);
#else
  TORCH_CHECK(false, "linear-attention-primitives was not built with CUDA support");
#endif
}

void bf16_smallm_matmul(torch::Tensor const& x, torch::Tensor const& w, torch::Tensor& out) {
  check_bf16(x, "x");
  check_bf16(w, "w");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 2, "x must have shape (M, K)");
  TORCH_CHECK(w.dim() == 2, "w must have shape (N, K)");
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = w.size(0);
  check_k_supported(k);
  TORCH_CHECK(n == 96 && k == 5120,
              "bf16_smallm_matmul currently supports the tuned AB96 shape N=96,K=5120 only");
  TORCH_CHECK(m >= 2 && m <= 4,
              "bf16_smallm_matmul currently supports 2 <= M <= 4 for positive-speedup AB96 dispatch");
  TORCH_CHECK(w.size(1) == k, "x and w K mismatch");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({m, n}), "out must have shape (M, N)");
  same_device(x, w, "x", "w");
  same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::linear_attention_primitives::bf16_smallm_matmul(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(w.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream);
#else
  TORCH_CHECK(false, "linear-attention-primitives was not built with CUDA support");
#endif
}

void split_qkv_broadcast_bf16(torch::Tensor const& packed, torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
                              int64_t q_heads, int64_t kv_heads, int64_t v_heads, int64_t head_dim) {
  check_bf16(packed, "packed");
  check_bf16(q, "q");
  check_bf16(k, "k");
  check_bf16(v, "v");
  TORCH_CHECK(packed.dim() == 2, "packed must have shape (rows, (q_heads + kv_heads + v_heads)*head_dim)");
  const int64_t rows = packed.size(0);
  TORCH_CHECK(q_heads > 0 && kv_heads > 0 && v_heads > 0 && head_dim > 0, "heads/head_dim must be positive");
  TORCH_CHECK(packed.size(1) == (q_heads + kv_heads + v_heads) * head_dim, "packed width mismatch");
  TORCH_CHECK(q.sizes() == torch::IntArrayRef({rows, v_heads, head_dim}), "q shape mismatch");
  TORCH_CHECK(k.sizes() == torch::IntArrayRef({rows, v_heads, head_dim}), "k shape mismatch");
  TORCH_CHECK(v.sizes() == torch::IntArrayRef({rows, v_heads, head_dim}), "v shape mismatch");
  same_device(packed, q, "packed", "q");
  same_device(packed, k, "packed", "k");
  same_device(packed, v, "packed", "v");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed.get_device()).stream();
  flash_rt::linear_attention_primitives::split_qkv_broadcast_bf16(
      static_cast<const __nv_bfloat16*>(packed.data_ptr()),
      static_cast<__nv_bfloat16*>(q.data_ptr()),
      static_cast<__nv_bfloat16*>(k.data_ptr()),
      static_cast<__nv_bfloat16*>(v.data_ptr()),
      static_cast<int>(rows), static_cast<int>(q_heads), static_cast<int>(kv_heads),
      static_cast<int>(v_heads), static_cast<int>(head_dim), stream);
#else
  TORCH_CHECK(false, "linear-attention-primitives was not built with CUDA support");
#endif
}

void partial_rope_qk_bf16(torch::Tensor const& q_in, torch::Tensor const& k_in,
                          torch::Tensor const& cos, torch::Tensor const& sin,
                          torch::Tensor& q_out, torch::Tensor& k_out,
                          int64_t rope_dim) {
  check_bf16(q_in, "q_in");
  check_bf16(k_in, "k_in");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  TORCH_CHECK(q_in.dim() == 3 && k_in.dim() == 3, "q_in/k_in must be (rows, heads, head_dim)");
  const int64_t rows = q_in.size(0);
  const int64_t q_heads = q_in.size(1);
  const int64_t k_heads = k_in.size(1);
  const int64_t head_dim = q_in.size(2);
  TORCH_CHECK(k_in.size(0) == rows && k_in.size(2) == head_dim, "q/k shape mismatch");
  TORCH_CHECK(rope_dim > 0 && rope_dim <= head_dim && rope_dim % 2 == 0, "rope_dim must be even and <= head_dim");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({rows, rope_dim}), "cos shape mismatch");
  TORCH_CHECK(sin.sizes() == cos.sizes(), "sin shape mismatch");
  TORCH_CHECK(q_out.sizes() == q_in.sizes(), "q_out shape mismatch");
  TORCH_CHECK(k_out.sizes() == k_in.sizes(), "k_out shape mismatch");
  same_device(q_in, k_in, "q_in", "k_in");
  same_device(q_in, cos, "q_in", "cos");
  same_device(q_in, q_out, "q_in", "q_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q_in.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_in.get_device()).stream();
  flash_rt::linear_attention_primitives::partial_rope_qk_bf16(
      static_cast<const __nv_bfloat16*>(q_in.data_ptr()),
      static_cast<const __nv_bfloat16*>(k_in.data_ptr()),
      static_cast<const __nv_bfloat16*>(cos.data_ptr()),
      static_cast<const __nv_bfloat16*>(sin.data_ptr()),
      static_cast<__nv_bfloat16*>(q_out.data_ptr()),
      static_cast<__nv_bfloat16*>(k_out.data_ptr()),
      static_cast<int>(rows), static_cast<int>(q_heads), static_cast<int>(k_heads),
      static_cast<int>(head_dim), static_cast<int>(rope_dim), stream);
#else
  TORCH_CHECK(false, "linear-attention-primitives was not built with CUDA support");
#endif
}

void gated_delta_prepare_bf16(torch::Tensor const& a, torch::Tensor const& b,
                              torch::Tensor const& neg_exp_a_log, torch::Tensor const& dt_bias,
                              torch::Tensor& g_out, torch::Tensor& beta_out,
                              int64_t a_stride, int64_t b_stride) {
  check_bf16(a, "a");
  check_bf16(b, "b");
  check_f32(neg_exp_a_log, "neg_exp_a_log");
  check_f32(dt_bias, "dt_bias");
  check_bf16(g_out, "g_out");
  check_bf16(beta_out, "beta_out");
  TORCH_CHECK(g_out.dim() == 2, "g_out must have shape (rows, heads)");
  const int64_t rows = g_out.size(0);
  const int64_t heads = g_out.size(1);
  TORCH_CHECK(beta_out.sizes() == g_out.sizes(), "beta_out shape mismatch");
  TORCH_CHECK(neg_exp_a_log.sizes() == torch::IntArrayRef({heads}), "neg_exp_a_log shape mismatch");
  TORCH_CHECK(dt_bias.sizes() == torch::IntArrayRef({heads}), "dt_bias shape mismatch");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a/b must be 2D");
  TORCH_CHECK(a.size(0) >= rows && b.size(0) >= rows, "a/b rows too small");
  TORCH_CHECK(a_stride >= heads && b_stride >= heads, "strides must cover heads");
  TORCH_CHECK(a.size(1) >= a_stride && b.size(1) >= b_stride, "a/b shape does not cover stride");
  same_device(a, b, "a", "b");
  same_device(a, g_out, "a", "g_out");
  same_device(a, neg_exp_a_log, "a", "neg_exp_a_log");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  flash_rt::linear_attention_primitives::gated_delta_prepare_bf16(
      static_cast<const __nv_bfloat16*>(a.data_ptr()),
      static_cast<const __nv_bfloat16*>(b.data_ptr()),
      static_cast<const float*>(neg_exp_a_log.data_ptr()),
      static_cast<const float*>(dt_bias.data_ptr()),
      static_cast<__nv_bfloat16*>(g_out.data_ptr()),
      static_cast<__nv_bfloat16*>(beta_out.data_ptr()),
      static_cast<int>(rows), static_cast<int>(heads),
      static_cast<int>(a_stride), static_cast<int>(b_stride), stream);
#else
  TORCH_CHECK(false, "linear-attention-primitives was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("bf16_matvec(Tensor x, Tensor w, Tensor! out) -> ()");
  ops.def("bf16_smallm_matmul(Tensor x, Tensor w, Tensor! out) -> ()");
  ops.def("split_qkv_broadcast_bf16(Tensor packed, Tensor! q, Tensor! k, Tensor! v, int q_heads, int kv_heads, int v_heads, int head_dim) -> ()");
  ops.def("partial_rope_qk_bf16(Tensor q_in, Tensor k_in, Tensor cos, Tensor sin, Tensor! q_out, Tensor! k_out, int rope_dim) -> ()");
  ops.def("gated_delta_prepare_bf16(Tensor a, Tensor b, Tensor neg_exp_a_log, Tensor dt_bias, Tensor! g_out, Tensor! beta_out, int a_stride, int b_stride) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("bf16_matvec", torch::kCUDA, &bf16_matvec);
  ops.impl("bf16_smallm_matmul", torch::kCUDA, &bf16_smallm_matmul);
  ops.impl("split_qkv_broadcast_bf16", torch::kCUDA, &split_qkv_broadcast_bf16);
  ops.impl("partial_rope_qk_bf16", torch::kCUDA, &partial_rope_qk_bf16);
  ops.impl("gated_delta_prepare_bf16", torch::kCUDA, &gated_delta_prepare_bf16);
#endif
}
