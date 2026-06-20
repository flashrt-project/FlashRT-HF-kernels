// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "gated_delta_attention.cuh"
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

void check_f32(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
}

void same_device(torch::Tensor const& a, torch::Tensor const& b,
                 const char* an, const char* bn) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              an, " and ", bn, " must be on the same CUDA device");
}

void check_step_inputs(torch::Tensor const& q, torch::Tensor const& k,
                       torch::Tensor const& v, torch::Tensor const& g,
                       torch::Tensor const& beta) {
  check_bf16(q, "q");
  check_bf16(k, "k");
  check_bf16(v, "v");
  check_bf16(g, "g");
  check_bf16(beta, "beta");
  TORCH_CHECK(q.dim() == 3, "q must have shape (B,H,D)");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(),
              "k/v must match q shape");
  TORCH_CHECK(g.sizes() == torch::IntArrayRef({q.size(0), q.size(1)}),
              "g must have shape (B,H)");
  TORCH_CHECK(beta.sizes() == g.sizes(), "beta must match g shape");
  TORCH_CHECK(q.size(2) == 128, "v1 recurrent profile requires head_dim=128");
  same_device(q, k, "q", "k");
  same_device(q, v, "q", "v");
  same_device(q, g, "q", "g");
  same_device(q, beta, "q", "beta");
}

void check_state_bf16(torch::Tensor const& q, torch::Tensor const& state,
                      const char* name) {
  check_bf16(state, name);
  TORCH_CHECK(state.sizes() == torch::IntArrayRef({q.size(0), q.size(1), q.size(2), q.size(2)}),
              name, " must have shape (B,H,D,D)");
  same_device(q, state, "q", name);
}

void check_out(torch::Tensor const& q, torch::Tensor const& out) {
  check_bf16(out, "out");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
  same_device(q, out, "q", "out");
}

void check_chunk_inputs(torch::Tensor const& q, torch::Tensor const& k,
                        torch::Tensor const& v, torch::Tensor const& g,
                        torch::Tensor const& beta) {
  check_bf16(q, "q");
  check_bf16(k, "k");
  check_bf16(v, "v");
  check_bf16(g, "g");
  check_bf16(beta, "beta");
  TORCH_CHECK(q.dim() == 3, "q must have shape (S,H,D)");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(),
              "k/v must match q shape");
  TORCH_CHECK(g.sizes() == torch::IntArrayRef({q.size(0), q.size(1)}),
              "g must have shape (S,H)");
  TORCH_CHECK(beta.sizes() == g.sizes(), "beta must match g shape");
  TORCH_CHECK(q.size(2) == 128, "v1 chunk profile requires head_dim=128");
  same_device(q, k, "q", "k");
  same_device(q, v, "q", "v");
  same_device(q, g, "q", "g");
  same_device(q, beta, "q", "beta");
}

void check_conv_out(torch::Tensor const& conv_out) {
  check_bf16(conv_out, "conv_out");
  TORCH_CHECK(conv_out.dim() == 2 && conv_out.size(1) == 10240,
              "conv_out must have shape (S,10240)");
}

void check_q16(torch::Tensor const& t, const char* name, int64_t S) {
  check_bf16(t, name);
  TORCH_CHECK(t.sizes() == torch::IntArrayRef({S, 16, 128}),
              name, " must have shape (S,16,128)");
}

void check_v48(torch::Tensor const& t, const char* name, int64_t S) {
  check_bf16(t, name);
  TORCH_CHECK(t.sizes() == torch::IntArrayRef({S, 48, 128}),
              name, " must have shape (S,48,128)");
}

void check_heads48(torch::Tensor const& t, const char* name, int64_t S) {
  check_bf16(t, name);
  TORCH_CHECK(t.sizes() == torch::IntArrayRef({S, 48}),
              name, " must have shape (S,48)");
}

void check_state48(torch::Tensor const& t, const char* name) {
  check_bf16(t, name);
  TORCH_CHECK(t.sizes() == torch::IntArrayRef({48, 128, 128}),
              name, " must have shape (48,128,128)");
}

void check_wy_chunks(torch::Tensor const& t, const char* name,
                     int64_t chunks, int64_t n0, int64_t n1) {
  check_f32(t, name);
  TORCH_CHECK(t.sizes() == torch::IntArrayRef({chunks, 48, n0, n1}),
              name, " must have shape (ceil(S/64),48,", n0, ",", n1, ")");
}

void check_bf16_wy_state(torch::Tensor const& t, const char* name,
                         int64_t chunks) {
  check_bf16(t, name);
  TORCH_CHECK(t.sizes() == torch::IntArrayRef({chunks, 48, 128, 128}),
              name, " must have shape (ceil(S/64),48,128,128)");
}

}  // namespace

void gated_delta_recurrent_bf16(torch::Tensor const& q, torch::Tensor const& k,
                                torch::Tensor const& v, torch::Tensor const& g,
                                torch::Tensor const& beta, torch::Tensor& state,
                                torch::Tensor& out, bool use_qk_l2norm) {
  check_step_inputs(q, k, v, g, beta);
  check_state_bf16(q, state, "state");
  check_out(q, out);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flash_rt::kernels::gated_deltanet_recurrent_qwen36_bf16(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      state.data_ptr(), out.data_ptr(), static_cast<int>(q.size(0)),
      static_cast<int>(q.size(1)), static_cast<int>(q.size(2)),
      static_cast<int>(q.size(2)), use_qk_l2norm, stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gated_delta_recurrent_inout_bf16(torch::Tensor const& q, torch::Tensor const& k,
                                      torch::Tensor const& v, torch::Tensor const& g,
                                      torch::Tensor const& beta,
                                      torch::Tensor const& state_in,
                                      torch::Tensor& state_out,
                                      torch::Tensor& out,
                                      bool use_qk_l2norm) {
  check_step_inputs(q, k, v, g, beta);
  check_state_bf16(q, state_in, "state_in");
  check_state_bf16(q, state_out, "state_out");
  check_out(q, out);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flash_rt::kernels::gated_deltanet_recurrent_inout_qwen36_bf16(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      state_in.data_ptr(), state_out.data_ptr(), out.data_ptr(),
      static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
      static_cast<int>(q.size(2)), static_cast<int>(q.size(2)),
      use_qk_l2norm, stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gated_delta_recurrent_f32state_bf16io(torch::Tensor const& q, torch::Tensor const& k,
                                           torch::Tensor const& v, torch::Tensor const& g,
                                           torch::Tensor const& beta,
                                           torch::Tensor& state_f32,
                                           torch::Tensor& out,
                                           bool use_qk_l2norm) {
  check_step_inputs(q, k, v, g, beta);
  check_f32(state_f32, "state_f32");
  TORCH_CHECK(state_f32.sizes() == torch::IntArrayRef({q.size(0), q.size(1), q.size(2), q.size(2)}),
              "state_f32 must have shape (B,H,D,D)");
  same_device(q, state_f32, "q", "state_f32");
  check_out(q, out);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flash_rt::kernels::gated_deltanet_recurrent_qwen36_f32state_bf16io(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      state_f32.data_ptr(), out.data_ptr(), static_cast<int>(q.size(0)),
      static_cast<int>(q.size(1)), static_cast<int>(q.size(2)),
      static_cast<int>(q.size(2)), use_qk_l2norm, stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gated_delta_chunk_bf16(torch::Tensor const& q, torch::Tensor const& k,
                            torch::Tensor const& v, torch::Tensor const& g,
                            torch::Tensor const& beta, torch::Tensor& state,
                            torch::Tensor& out, bool use_qk_l2norm) {
  check_chunk_inputs(q, k, v, g, beta);
  check_bf16(state, "state");
  TORCH_CHECK(state.sizes() == torch::IntArrayRef({q.size(1), q.size(2), q.size(2)}),
              "state must have shape (H,D,D)");
  check_out(q, out);
  same_device(q, state, "q", "state");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flash_rt::kernels::gated_deltanet_chunk_qwen36_bf16(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      state.data_ptr(), out.data_ptr(), static_cast<int>(q.size(0)),
      static_cast<int>(q.size(1)), static_cast<int>(q.size(2)),
      static_cast<int>(q.size(2)), use_qk_l2norm, stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gated_delta_chunk_smem_bf16(torch::Tensor const& q, torch::Tensor const& k,
                                 torch::Tensor const& v, torch::Tensor const& g,
                                 torch::Tensor const& beta, torch::Tensor& state,
                                 torch::Tensor& out, bool use_qk_l2norm) {
  check_chunk_inputs(q, k, v, g, beta);
  check_bf16(state, "state");
  TORCH_CHECK(state.sizes() == torch::IntArrayRef({q.size(1), q.size(2), q.size(2)}),
              "state must have shape (H,D,D)");
  check_out(q, out);
  same_device(q, state, "q", "state");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flash_rt::kernels::gated_deltanet_chunk_smem_qwen36_bf16(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      state.data_ptr(), out.data_ptr(), static_cast<int>(q.size(0)),
      static_cast<int>(q.size(1)), static_cast<int>(q.size(2)),
      static_cast<int>(q.size(2)), use_qk_l2norm, stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void lin_split_qkv_broadcast_bf16(torch::Tensor const& conv_out,
                                  torch::Tensor& q48,
                                  torch::Tensor& k48,
                                  torch::Tensor& v48) {
  check_conv_out(conv_out);
  const auto S = conv_out.size(0);
  check_v48(q48, "q48", S);
  check_v48(k48, "k48", S);
  check_v48(v48, "v48", S);
  same_device(conv_out, q48, "conv_out", "q48");
  same_device(conv_out, k48, "conv_out", "k48");
  same_device(conv_out, v48, "conv_out", "v48");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(conv_out.device());
  auto stream = at::cuda::getCurrentCUDAStream(conv_out.get_device()).stream();
  flash_rt::kernels::qwen36_lin_split_qkv_broadcast_bf16(
      conv_out.data_ptr(), q48.data_ptr(), k48.data_ptr(), v48.data_ptr(),
      static_cast<int>(S), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void lin_split_qkv_gqa_bf16(torch::Tensor const& conv_out,
                            torch::Tensor& q16,
                            torch::Tensor& k16,
                            torch::Tensor& v48) {
  check_conv_out(conv_out);
  const auto S = conv_out.size(0);
  check_q16(q16, "q16", S);
  check_q16(k16, "k16", S);
  check_v48(v48, "v48", S);
  same_device(conv_out, q16, "conv_out", "q16");
  same_device(conv_out, k16, "conv_out", "k16");
  same_device(conv_out, v48, "conv_out", "v48");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(conv_out.device());
  auto stream = at::cuda::getCurrentCUDAStream(conv_out.get_device()).stream();
  flash_rt::kernels::qwen36_lin_split_qkv_gqa_bf16(
      conv_out.data_ptr(), q16.data_ptr(), k16.data_ptr(), v48.data_ptr(),
      static_cast<int>(S), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void split_q_gate_bf16(torch::Tensor const& q_proj,
                       torch::Tensor& q_pre,
                       torch::Tensor& gate) {
  check_bf16(q_proj, "q_proj");
  TORCH_CHECK(q_proj.dim() == 3 && q_proj.size(1) == 24 && q_proj.size(2) == 512,
              "q_proj must have shape (S,24,512)");
  const auto S = q_proj.size(0);
  check_bf16(q_pre, "q_pre");
  check_bf16(gate, "gate");
  TORCH_CHECK(q_pre.sizes() == torch::IntArrayRef({S, 24, 256}),
              "q_pre must have shape (S,24,256)");
  TORCH_CHECK(gate.sizes() == torch::IntArrayRef({S, 24 * 256}),
              "gate must have shape (S,6144)");
  same_device(q_proj, q_pre, "q_proj", "q_pre");
  same_device(q_proj, gate, "q_proj", "gate");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q_proj.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_proj.get_device()).stream();
  flash_rt::kernels::qwen36_split_q_gate_bf16(
      q_proj.data_ptr(), q_pre.data_ptr(), gate.data_ptr(),
      static_cast<int>(S), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gdn_gating_bf16(torch::Tensor const& a, torch::Tensor const& b,
                     torch::Tensor const& neg_exp_A_log,
                     torch::Tensor const& dt_bias,
                     torch::Tensor& g_out,
                     torch::Tensor& beta_out) {
  check_heads48(a, "a", a.size(0));
  const auto S = a.size(0);
  check_heads48(b, "b", S);
  check_heads48(g_out, "g_out", S);
  check_heads48(beta_out, "beta_out", S);
  check_f32(neg_exp_A_log, "neg_exp_A_log");
  check_f32(dt_bias, "dt_bias");
  TORCH_CHECK(neg_exp_A_log.sizes() == torch::IntArrayRef({48}),
              "neg_exp_A_log must have shape (48)");
  TORCH_CHECK(dt_bias.sizes() == torch::IntArrayRef({48}),
              "dt_bias must have shape (48)");
  same_device(a, b, "a", "b");
  same_device(a, neg_exp_A_log, "a", "neg_exp_A_log");
  same_device(a, dt_bias, "a", "dt_bias");
  same_device(a, g_out, "a", "g_out");
  same_device(a, beta_out, "a", "beta_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  flash_rt::kernels::qwen36_gdn_gating_bf16(
      a.data_ptr(), b.data_ptr(), neg_exp_A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(), g_out.data_ptr(), beta_out.data_ptr(),
      static_cast<int>(S), 48, stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gdn_gating_strided_bf16(torch::Tensor const& a, torch::Tensor const& b,
                             torch::Tensor const& neg_exp_A_log,
                             torch::Tensor const& dt_bias,
                             torch::Tensor& g_out,
                             torch::Tensor& beta_out,
                             int64_t a_stride,
                             int64_t b_stride) {
  check_heads48(g_out, "g_out", g_out.size(0));
  const auto S = g_out.size(0);
  check_heads48(beta_out, "beta_out", S);
  check_bf16(a, "a");
  check_bf16(b, "b");
  TORCH_CHECK(a.numel() >= (S - 1) * a_stride + 48,
              "a does not contain S rows at a_stride");
  TORCH_CHECK(b.numel() >= (S - 1) * b_stride + 48,
              "b does not contain S rows at b_stride");
  check_f32(neg_exp_A_log, "neg_exp_A_log");
  check_f32(dt_bias, "dt_bias");
  TORCH_CHECK(neg_exp_A_log.sizes() == torch::IntArrayRef({48}),
              "neg_exp_A_log must have shape (48)");
  TORCH_CHECK(dt_bias.sizes() == torch::IntArrayRef({48}),
              "dt_bias must have shape (48)");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(g_out.device());
  auto stream = at::cuda::getCurrentCUDAStream(g_out.get_device()).stream();
  flash_rt::kernels::qwen36_gdn_gating_strided_bf16(
      a.data_ptr(), b.data_ptr(), neg_exp_A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(), g_out.data_ptr(), beta_out.data_ptr(),
      static_cast<int>(S), 48, static_cast<int>(a_stride),
      static_cast<int>(b_stride), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gdn_chunk_from_conv_smem_bf16(torch::Tensor const& conv_out,
                                   torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& neg_exp_A_log,
                                   torch::Tensor const& dt_bias,
                                   torch::Tensor& state,
                                   torch::Tensor& out,
                                   bool use_qk_l2norm) {
  check_conv_out(conv_out);
  const auto S = conv_out.size(0);
  check_heads48(a, "a", S);
  check_heads48(b, "b", S);
  check_f32(neg_exp_A_log, "neg_exp_A_log");
  check_f32(dt_bias, "dt_bias");
  check_state48(state, "state");
  check_v48(out, "out", S);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(conv_out.device());
  auto stream = at::cuda::getCurrentCUDAStream(conv_out.get_device()).stream();
  flash_rt::kernels::qwen36_gdn_chunk_from_conv_smem_bf16(
      conv_out.data_ptr(), a.data_ptr(), b.data_ptr(),
      neg_exp_A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      state.data_ptr(), out.data_ptr(), static_cast<int>(S), 48,
      use_qk_l2norm, stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gdn_wy_norm_cumsum_pack_qk_bf16(torch::Tensor const& q16,
                                     torch::Tensor const& k16,
                                     torch::Tensor const& g,
                                     torch::Tensor& q16_l2,
                                     torch::Tensor& k16_l2,
                                     torch::Tensor& q_pack_hv,
                                     torch::Tensor& k_pack_hk,
                                     torch::Tensor& g_cumsum) {
  const auto S = q16.size(0);
  const auto chunks = (S + 63) / 64;
  check_q16(q16, "q16", S);
  check_q16(k16, "k16", S);
  check_heads48(g, "g", S);
  check_q16(q16_l2, "q16_l2", S);
  check_q16(k16_l2, "k16_l2", S);
  check_bf16(q_pack_hv, "q_pack_hv");
  check_bf16(k_pack_hk, "k_pack_hk");
  check_heads48(g_cumsum, "g_cumsum", S);
  TORCH_CHECK(q_pack_hv.sizes() == torch::IntArrayRef({chunks, 48, 64, 128}),
              "q_pack_hv must have shape (ceil(S/64),48,64,128)");
  TORCH_CHECK(k_pack_hk.sizes() == torch::IntArrayRef({chunks, 16, 64, 128}),
              "k_pack_hk must have shape (ceil(S/64),16,64,128)");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q16.device());
  auto stream = at::cuda::getCurrentCUDAStream(q16.get_device()).stream();
  flash_rt::kernels::qwen36_gdn_wy_norm_cumsum_pack_qk_bf16(
      q16.data_ptr(), k16.data_ptr(), g.data_ptr(), q16_l2.data_ptr(),
      k16_l2.data_ptr(), q_pack_hv.data_ptr(), k_pack_hk.data_ptr(),
      g_cumsum.data_ptr(), static_cast<int>(S), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gdn_wy_kkt_b64_bf16(torch::Tensor const& k16_l2,
                         torch::Tensor const& beta,
                         torch::Tensor const& g_cumsum,
                         torch::Tensor& A) {
  const auto S = k16_l2.size(0);
  const auto chunks = (S + 63) / 64;
  check_q16(k16_l2, "k16_l2", S);
  check_heads48(beta, "beta", S);
  check_heads48(g_cumsum, "g_cumsum", S);
  check_wy_chunks(A, "A", chunks, 64, 64);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(k16_l2.device());
  auto stream = at::cuda::getCurrentCUDAStream(k16_l2.get_device()).stream();
  flash_rt::kernels::qwen36_gdn_wy_kkt_b64_bf16(
      k16_l2.data_ptr(), beta.data_ptr(), g_cumsum.data_ptr(), A.data_ptr(),
      static_cast<int>(S), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gdn_wy_solve_tril_b64_f32(torch::Tensor const& A,
                               torch::Tensor& Ai,
                               int64_t S) {
  const auto chunks = (S + 63) / 64;
  check_wy_chunks(A, "A", chunks, 64, 64);
  check_wy_chunks(Ai, "Ai", chunks, 64, 64);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(A.device());
  auto stream = at::cuda::getCurrentCUDAStream(A.get_device()).stream();
  flash_rt::kernels::qwen36_gdn_wy_solve_tril_b64_f32(
      A.data_ptr(), Ai.data_ptr(), static_cast<int>(S), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gdn_wy_recompute_wu_b64_bf16(torch::Tensor const& k16_l2,
                                  torch::Tensor const& v48,
                                  torch::Tensor const& beta,
                                  torch::Tensor const& g_cumsum,
                                  torch::Tensor const& Ai,
                                  torch::Tensor& w48,
                                  torch::Tensor& u48) {
  const auto S = k16_l2.size(0);
  const auto chunks = (S + 63) / 64;
  check_q16(k16_l2, "k16_l2", S);
  check_v48(v48, "v48", S);
  check_heads48(beta, "beta", S);
  check_heads48(g_cumsum, "g_cumsum", S);
  check_wy_chunks(Ai, "Ai", chunks, 64, 64);
  check_v48(w48, "w48", S);
  check_v48(u48, "u48", S);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(k16_l2.device());
  auto stream = at::cuda::getCurrentCUDAStream(k16_l2.get_device()).stream();
  flash_rt::kernels::qwen36_gdn_wy_recompute_wu_b64_bf16(
      k16_l2.data_ptr(), v48.data_ptr(), beta.data_ptr(),
      g_cumsum.data_ptr(), Ai.data_ptr(), w48.data_ptr(), u48.data_ptr(),
      static_cast<int>(S), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gdn_wy_chunk_h_b64_bf16(torch::Tensor const& k16_l2,
                             torch::Tensor const& u48,
                             torch::Tensor const& w48,
                             torch::Tensor const& g_cumsum,
                             torch::Tensor& state,
                             torch::Tensor& h0,
                             torch::Tensor& v_new) {
  const auto S = k16_l2.size(0);
  const auto chunks = (S + 63) / 64;
  check_q16(k16_l2, "k16_l2", S);
  check_v48(u48, "u48", S);
  check_v48(w48, "w48", S);
  check_heads48(g_cumsum, "g_cumsum", S);
  check_state48(state, "state");
  check_bf16_wy_state(h0, "h0", chunks);
  check_v48(v_new, "v_new", S);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(k16_l2.device());
  auto stream = at::cuda::getCurrentCUDAStream(k16_l2.get_device()).stream();
  flash_rt::kernels::qwen36_gdn_wy_chunk_h_b64_bf16(
      k16_l2.data_ptr(), u48.data_ptr(), w48.data_ptr(),
      g_cumsum.data_ptr(), state.data_ptr(), h0.data_ptr(),
      v_new.data_ptr(), static_cast<int>(S), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

void gdn_wy_output_o_b64_bf16(torch::Tensor const& q16_l2,
                              torch::Tensor const& k16_l2,
                              torch::Tensor const& v_new,
                              torch::Tensor const& h0,
                              torch::Tensor const& g_cumsum,
                              torch::Tensor& out) {
  const auto S = q16_l2.size(0);
  const auto chunks = (S + 63) / 64;
  check_q16(q16_l2, "q16_l2", S);
  check_q16(k16_l2, "k16_l2", S);
  check_v48(v_new, "v_new", S);
  check_bf16_wy_state(h0, "h0", chunks);
  check_heads48(g_cumsum, "g_cumsum", S);
  check_v48(out, "out", S);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q16_l2.device());
  auto stream = at::cuda::getCurrentCUDAStream(q16_l2.get_device()).stream();
  flash_rt::kernels::qwen36_gdn_wy_output_o_b64_bf16(
      q16_l2.data_ptr(), k16_l2.data_ptr(), v_new.data_ptr(),
      h0.data_ptr(), g_cumsum.data_ptr(), out.data_ptr(),
      static_cast<int>(S), stream);
#else
  TORCH_CHECK(false, "gated-delta-attention was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gated_delta_recurrent_bf16(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor! state, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("gated_delta_recurrent_inout_bf16(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state_in, Tensor! state_out, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("gated_delta_recurrent_f32state_bf16io(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor! state_f32, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("gated_delta_chunk_bf16(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor! state, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("gated_delta_chunk_smem_bf16(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor! state, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("lin_split_qkv_broadcast_bf16(Tensor conv_out, Tensor! q48, Tensor! k48, Tensor! v48) -> ()");
  ops.def("lin_split_qkv_gqa_bf16(Tensor conv_out, Tensor! q16, Tensor! k16, Tensor! v48) -> ()");
  ops.def("split_q_gate_bf16(Tensor q_proj, Tensor! q_pre, Tensor! gate) -> ()");
  ops.def("gdn_gating_bf16(Tensor a, Tensor b, Tensor neg_exp_A_log, Tensor dt_bias, Tensor! g_out, Tensor! beta_out) -> ()");
  ops.def("gdn_gating_strided_bf16(Tensor a, Tensor b, Tensor neg_exp_A_log, Tensor dt_bias, Tensor! g_out, Tensor! beta_out, int a_stride, int b_stride) -> ()");
  ops.def("gdn_chunk_from_conv_smem_bf16(Tensor conv_out, Tensor a, Tensor b, Tensor neg_exp_A_log, Tensor dt_bias, Tensor! state, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("gdn_wy_norm_cumsum_pack_qk_bf16(Tensor q16, Tensor k16, Tensor g, Tensor! q16_l2, Tensor! k16_l2, Tensor! q_pack_hv, Tensor! k_pack_hk, Tensor! g_cumsum) -> ()");
  ops.def("gdn_wy_kkt_b64_bf16(Tensor k16_l2, Tensor beta, Tensor g_cumsum, Tensor! A) -> ()");
  ops.def("gdn_wy_solve_tril_b64_f32(Tensor A, Tensor! Ai, int S) -> ()");
  ops.def("gdn_wy_recompute_wu_b64_bf16(Tensor k16_l2, Tensor v48, Tensor beta, Tensor g_cumsum, Tensor Ai, Tensor! w48, Tensor! u48) -> ()");
  ops.def("gdn_wy_chunk_h_b64_bf16(Tensor k16_l2, Tensor u48, Tensor w48, Tensor g_cumsum, Tensor! state, Tensor! h0, Tensor! v_new) -> ()");
  ops.def("gdn_wy_output_o_b64_bf16(Tensor q16_l2, Tensor k16_l2, Tensor v_new, Tensor h0, Tensor g_cumsum, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gated_delta_recurrent_bf16", torch::kCUDA, &gated_delta_recurrent_bf16);
  ops.impl("gated_delta_recurrent_inout_bf16", torch::kCUDA, &gated_delta_recurrent_inout_bf16);
  ops.impl("gated_delta_recurrent_f32state_bf16io", torch::kCUDA, &gated_delta_recurrent_f32state_bf16io);
  ops.impl("gated_delta_chunk_bf16", torch::kCUDA, &gated_delta_chunk_bf16);
  ops.impl("gated_delta_chunk_smem_bf16", torch::kCUDA, &gated_delta_chunk_smem_bf16);
  ops.impl("lin_split_qkv_broadcast_bf16", torch::kCUDA, &lin_split_qkv_broadcast_bf16);
  ops.impl("lin_split_qkv_gqa_bf16", torch::kCUDA, &lin_split_qkv_gqa_bf16);
  ops.impl("split_q_gate_bf16", torch::kCUDA, &split_q_gate_bf16);
  ops.impl("gdn_gating_bf16", torch::kCUDA, &gdn_gating_bf16);
  ops.impl("gdn_gating_strided_bf16", torch::kCUDA, &gdn_gating_strided_bf16);
  ops.impl("gdn_chunk_from_conv_smem_bf16", torch::kCUDA, &gdn_chunk_from_conv_smem_bf16);
  ops.impl("gdn_wy_norm_cumsum_pack_qk_bf16", torch::kCUDA, &gdn_wy_norm_cumsum_pack_qk_bf16);
  ops.impl("gdn_wy_kkt_b64_bf16", torch::kCUDA, &gdn_wy_kkt_b64_bf16);
  ops.impl("gdn_wy_solve_tril_b64_f32", torch::kCUDA, &gdn_wy_solve_tril_b64_f32);
  ops.impl("gdn_wy_recompute_wu_b64_bf16", torch::kCUDA, &gdn_wy_recompute_wu_b64_bf16);
  ops.impl("gdn_wy_chunk_h_b64_bf16", torch::kCUDA, &gdn_wy_chunk_h_b64_bf16);
  ops.impl("gdn_wy_output_o_b64_bf16", torch::kCUDA, &gdn_wy_output_o_b64_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
