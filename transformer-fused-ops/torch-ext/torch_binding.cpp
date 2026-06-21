// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "kernels/nexn2_misc.cuh"
#include "kernels/nexn2_router_topk.cuh"
#include "kernels/qwen36_misc.cuh"
#include "kernels/rms_norm_gated_silu_qwen36.cuh"
#include "kernels/silu_mul_qwen36.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be torch.bfloat16");
}

void check_i64(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kInt64, name, " must be torch.int64");
}

void check_i32(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be torch.int32");
}

void check_f32(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be torch.float32");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

void same_device(torch::Tensor const& a, torch::Tensor const& b, const char* an, const char* bn) {
  TORCH_CHECK(a.get_device() == b.get_device(), an, " and ", bn, " must be on the same CUDA device");
}

}  // namespace

void rms_norm_gated_silu_bf16(torch::Tensor const& x, torch::Tensor const& gate,
                              torch::Tensor const& weight, double eps,
                              torch::Tensor& out) {
  check_bf16(x, "x");
  check_bf16(gate, "gate");
  check_bf16(weight, "weight");
  check_bf16(out, "out");
  TORCH_CHECK(x.dim() == 2 && gate.sizes() == x.sizes() && out.sizes() == x.sizes(),
              "x/gate/out must have shape (rows, dim)");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({x.size(1)}), "weight shape mismatch");
  TORCH_CHECK(x.size(1) == 128, "rms_norm_gated_silu_bf16 currently supports dim=128");
  same_device(x, gate, "x", "gate");
  same_device(x, weight, "x", "weight");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::kernels::rms_norm_gated_silu_qwen36_bf16(
      x.data_ptr(), gate.data_ptr(), weight.data_ptr(), out.data_ptr(),
      checked_int(x.size(0), "rows"), checked_int(x.size(1), "dim"),
      static_cast<float>(eps), stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void silu_mul_bf16(torch::Tensor const& gate, torch::Tensor const& up, torch::Tensor& out) {
  check_bf16(gate, "gate");
  check_bf16(up, "up");
  check_bf16(out, "out");
  TORCH_CHECK(gate.sizes() == up.sizes() && out.sizes() == gate.sizes(), "shape mismatch");
  same_device(gate, up, "gate", "up");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(gate.device());
  auto stream = at::cuda::getCurrentCUDAStream(gate.get_device()).stream();
  flash_rt::kernels::silu_mul_qwen36_bf16(
      static_cast<const __nv_bfloat16*>(gate.data_ptr()),
      static_cast<const __nv_bfloat16*>(up.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      checked_int(gate.numel(), "numel"), stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void sigmoid_mul_bf16(torch::Tensor const& gate, torch::Tensor const& x, torch::Tensor& out) {
  check_bf16(gate, "gate");
  check_bf16(x, "x");
  check_bf16(out, "out");
  TORCH_CHECK(gate.sizes() == x.sizes() && out.sizes() == gate.sizes(), "shape mismatch");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(gate.device());
  auto stream = at::cuda::getCurrentCUDAStream(gate.get_device()).stream();
  flash_rt::kernels::sigmoid_mul_qwen36_bf16(
      static_cast<const __nv_bfloat16*>(gate.data_ptr()),
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      checked_int(gate.numel(), "numel"), stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void embedding_lookup_bf16(torch::Tensor const& token_ids, torch::Tensor const& embed,
                           torch::Tensor& out) {
  check_i64(token_ids, "token_ids");
  check_bf16(embed, "embed");
  check_bf16(out, "out");
  TORCH_CHECK(token_ids.dim() == 1 && embed.dim() == 2, "token_ids must be (rows,), embed (vocab, hidden)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({token_ids.size(0), embed.size(1)}), "out shape mismatch");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(embed.device());
  auto stream = at::cuda::getCurrentCUDAStream(embed.get_device()).stream();
  flash_rt::kernels::qwen36_embedding_lookup_bf16(
      static_cast<const int64_t*>(token_ids.data_ptr()),
      static_cast<const __nv_bfloat16*>(embed.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      checked_int(token_ids.size(0), "rows"), checked_int(embed.size(1), "hidden"), stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void partial_rope_qk_bf16(torch::Tensor const& q_in, torch::Tensor const& k_in,
                          torch::Tensor const& cos, torch::Tensor const& sin,
                          torch::Tensor& q_out, torch::Tensor& k_out, int64_t rope_dim) {
  check_bf16(q_in, "q_in");
  check_bf16(k_in, "k_in");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  TORCH_CHECK(q_in.dim() == 3 && k_in.dim() == 3, "q/k must be (rows, heads, head_dim)");
  TORCH_CHECK(q_in.size(0) == k_in.size(0) && q_in.size(2) == k_in.size(2), "q/k shape mismatch");
  TORCH_CHECK(q_out.sizes() == q_in.sizes() && k_out.sizes() == k_in.sizes(), "output shape mismatch");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({q_in.size(0), rope_dim}) && sin.sizes() == cos.sizes(),
              "cos/sin shape mismatch");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(q_in.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_in.get_device()).stream();
  flash_rt::kernels::qwen36_partial_rope_qk_bf16(
      static_cast<const __nv_bfloat16*>(q_in.data_ptr()),
      static_cast<const __nv_bfloat16*>(k_in.data_ptr()),
      static_cast<const __nv_bfloat16*>(cos.data_ptr()),
      static_cast<const __nv_bfloat16*>(sin.data_ptr()),
      static_cast<__nv_bfloat16*>(q_out.data_ptr()),
      static_cast<__nv_bfloat16*>(k_out.data_ptr()),
      checked_int(q_in.size(0), "rows"), checked_int(q_in.size(1), "q_heads"),
      checked_int(k_in.size(1), "k_heads"), checked_int(q_in.size(2), "head_dim"),
      checked_int(rope_dim, "rope_dim"), stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void argmax_bf16(torch::Tensor const& logits, torch::Tensor& argmax_out) {
  check_bf16(logits, "logits");
  check_i64(argmax_out, "argmax_out");
  TORCH_CHECK(logits.dim() == 2, "logits must have shape (rows, vocab)");
  TORCH_CHECK(argmax_out.sizes() == torch::IntArrayRef({logits.size(0)}), "argmax_out shape mismatch");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  flash_rt::kernels::qwen36_argmax_bf16(
      static_cast<const __nv_bfloat16*>(logits.data_ptr()),
      static_cast<int64_t*>(argmax_out.data_ptr()),
      checked_int(logits.size(0), "rows"), checked_int(logits.size(1), "vocab"), stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void spec_accept_greedy_bf16(torch::Tensor const& logits, torch::Tensor const& drafts,
                             torch::Tensor& argmax_out, torch::Tensor& accept_n,
                             int64_t spec_k) {
  check_bf16(logits, "logits");
  check_i64(drafts, "drafts");
  check_i64(argmax_out, "argmax_out");
  check_i32(accept_n, "accept_n");
  TORCH_CHECK(logits.dim() == 2 && drafts.numel() >= spec_k, "invalid logits/drafts");
  TORCH_CHECK(argmax_out.sizes() == torch::IntArrayRef({logits.size(0)}), "argmax_out shape mismatch");
  TORCH_CHECK(accept_n.numel() >= 1, "accept_n must have at least one element");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  flash_rt::kernels::qwen36_spec_accept_greedy_bf16(
      static_cast<const __nv_bfloat16*>(logits.data_ptr()),
      static_cast<const int64_t*>(drafts.data_ptr()),
      static_cast<int64_t*>(argmax_out.data_ptr()),
      static_cast<int*>(accept_n.data_ptr()),
      checked_int(logits.size(0), "rows"), checked_int(logits.size(1), "vocab"),
      checked_int(spec_k, "spec_k"), stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void nexn2_lin_split_qkv_broadcast_bf16(torch::Tensor const& conv_out,
                                        torch::Tensor& q32, torch::Tensor& k32,
                                        torch::Tensor& v32) {
  check_bf16(conv_out, "conv_out");
  check_bf16(q32, "q32");
  check_bf16(k32, "k32");
  check_bf16(v32, "v32");
  TORCH_CHECK(conv_out.dim() == 2 && conv_out.size(1) == 8192, "conv_out must have shape (S,8192)");
  TORCH_CHECK(q32.sizes() == torch::IntArrayRef({conv_out.size(0), 32, 128}), "q32 shape mismatch");
  TORCH_CHECK(k32.sizes() == q32.sizes() && v32.sizes() == q32.sizes(), "k/v shape mismatch");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(conv_out.device());
  auto stream = at::cuda::getCurrentCUDAStream(conv_out.get_device()).stream();
  flash_rt::kernels::nexn2_lin_split_qkv_broadcast_bf16(
      conv_out.data_ptr(), q32.data_ptr(), k32.data_ptr(), v32.data_ptr(),
      checked_int(conv_out.size(0), "S"), stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void nexn2_split_q_gate_bf16(torch::Tensor const& q_proj,
                             torch::Tensor& q_pre, torch::Tensor& gate) {
  check_bf16(q_proj, "q_proj");
  check_bf16(q_pre, "q_pre");
  check_bf16(gate, "gate");
  TORCH_CHECK(q_proj.dim() == 3 && q_proj.size(1) == 16 && q_proj.size(2) == 512,
              "q_proj must have shape (S,16,512)");
  TORCH_CHECK(q_pre.sizes() == torch::IntArrayRef({q_proj.size(0), 16, 256}), "q_pre shape mismatch");
  TORCH_CHECK(gate.sizes() == torch::IntArrayRef({q_proj.size(0), 16 * 256}), "gate shape mismatch");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(q_proj.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_proj.get_device()).stream();
  flash_rt::kernels::nexn2_split_q_gate_bf16(
      q_proj.data_ptr(), q_pre.data_ptr(), gate.data_ptr(),
      checked_int(q_proj.size(0), "S"), stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void nexn2_router_topk_bf16(torch::Tensor const& logits, torch::Tensor& out_idx,
                            torch::Tensor& out_val, int64_t k) {
  check_bf16(logits, "logits");
  check_i32(out_idx, "out_idx");
  check_f32(out_val, "out_val");
  TORCH_CHECK(logits.dim() == 1, "logits must have shape (n_experts,)");
  TORCH_CHECK(out_idx.sizes() == torch::IntArrayRef({k}) && out_val.sizes() == torch::IntArrayRef({k}),
              "topk outputs must have shape (k,)");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  const int rc = flash_rt::kernels::nexn2_router_topk_bf16(
      logits.data_ptr(), out_idx.data_ptr(), out_val.data_ptr(),
      checked_int(logits.numel(), "n_experts"), checked_int(k, "k"), stream);
  TORCH_CHECK(rc == 0, "nexn2_router_topk_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rms_norm_gated_silu_bf16(Tensor x, Tensor gate, Tensor weight, float eps, Tensor! out) -> ()");
  ops.def("silu_mul_bf16(Tensor gate, Tensor up, Tensor! out) -> ()");
  ops.def("sigmoid_mul_bf16(Tensor gate, Tensor x, Tensor! out) -> ()");
  ops.def("embedding_lookup_bf16(Tensor token_ids, Tensor embed, Tensor! out) -> ()");
  ops.def("partial_rope_qk_bf16(Tensor q_in, Tensor k_in, Tensor cos, Tensor sin, Tensor! q_out, Tensor! k_out, int rope_dim) -> ()");
  ops.def("argmax_bf16(Tensor logits, Tensor! argmax_out) -> ()");
  ops.def("spec_accept_greedy_bf16(Tensor logits, Tensor drafts, Tensor! argmax_out, Tensor! accept_n, int spec_k) -> ()");
  ops.def("nexn2_lin_split_qkv_broadcast_bf16(Tensor conv_out, Tensor! q32, Tensor! k32, Tensor! v32) -> ()");
  ops.def("nexn2_split_q_gate_bf16(Tensor q_proj, Tensor! q_pre, Tensor! gate) -> ()");
  ops.def("nexn2_router_topk_bf16(Tensor logits, Tensor! out_idx, Tensor! out_val, int k) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("rms_norm_gated_silu_bf16", torch::kCUDA, &rms_norm_gated_silu_bf16);
  ops.impl("silu_mul_bf16", torch::kCUDA, &silu_mul_bf16);
  ops.impl("sigmoid_mul_bf16", torch::kCUDA, &sigmoid_mul_bf16);
  ops.impl("embedding_lookup_bf16", torch::kCUDA, &embedding_lookup_bf16);
  ops.impl("partial_rope_qk_bf16", torch::kCUDA, &partial_rope_qk_bf16);
  ops.impl("argmax_bf16", torch::kCUDA, &argmax_bf16);
  ops.impl("spec_accept_greedy_bf16", torch::kCUDA, &spec_accept_greedy_bf16);
  ops.impl("nexn2_lin_split_qkv_broadcast_bf16", torch::kCUDA, &nexn2_lin_split_qkv_broadcast_bf16);
  ops.impl("nexn2_split_q_gate_bf16", torch::kCUDA, &nexn2_split_q_gate_bf16);
  ops.impl("nexn2_router_topk_bf16", torch::kCUDA, &nexn2_router_topk_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
