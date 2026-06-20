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

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gated_delta_recurrent_bf16(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor! state, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("gated_delta_recurrent_inout_bf16(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state_in, Tensor! state_out, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("gated_delta_recurrent_f32state_bf16io(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor! state_f32, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("gated_delta_chunk_bf16(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor! state, Tensor! out, bool use_qk_l2norm=True) -> ()");
  ops.def("gated_delta_chunk_smem_bf16(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor! state, Tensor! out, bool use_qk_l2norm=True) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gated_delta_recurrent_bf16", torch::kCUDA, &gated_delta_recurrent_bf16);
  ops.impl("gated_delta_recurrent_inout_bf16", torch::kCUDA, &gated_delta_recurrent_inout_bf16);
  ops.impl("gated_delta_recurrent_f32state_bf16io", torch::kCUDA, &gated_delta_recurrent_f32state_bf16io);
  ops.impl("gated_delta_chunk_bf16", torch::kCUDA, &gated_delta_chunk_bf16);
  ops.impl("gated_delta_chunk_smem_bf16", torch::kCUDA, &gated_delta_chunk_smem_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
