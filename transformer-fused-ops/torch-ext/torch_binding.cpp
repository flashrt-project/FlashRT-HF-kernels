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
#include "kernels/moe_weighted_sum_sm120.cuh"
#include "kernels/qwen36_misc.cuh"
#include "kernels/relu2_quantize_fp8.cuh"
#include "kernels/rms_norm_gated_silu_qwen36.cuh"
#include "kernels/silu_mul_qwen36.cuh"
#include "kernels/vec_fp16_dispatch.cuh"
#include "registration.h"
#include "torch_binding.h"

flash_rt::hub::RmsNormFp16Dispatch flash_rt::hub::rms_norm_fp16_dispatch = nullptr;
flash_rt::hub::LayerNormFp16Dispatch flash_rt::hub::layer_norm_fp16_dispatch = nullptr;
flash_rt::hub::LayerNormFp8Fp16Dispatch flash_rt::hub::layer_norm_fp8_fp16_dispatch = nullptr;
flash_rt::hub::RopeFp16Dispatch flash_rt::hub::rope_fp16_dispatch = nullptr;
flash_rt::hub::QuantizeFp8Fp16Dispatch flash_rt::hub::quantize_fp8_fp16_dispatch = nullptr;
flash_rt::hub::ResidualAddFp16Dispatch flash_rt::hub::residual_add_fp16_dispatch = nullptr;
flash_rt::hub::RepeatHeadsFp16Dispatch flash_rt::hub::repeat_heads_fp16_dispatch = nullptr;
flash_rt::hub::QuantizeFp8Bf16Dispatch flash_rt::hub::quantize_fp8_bf16_dispatch = nullptr;
flash_rt::hub::LayerNormFp8Bf16Dispatch flash_rt::hub::layer_norm_fp8_bf16_dispatch = nullptr;
flash_rt::hub::GateGegluFp8Bf16Dispatch flash_rt::hub::gate_geglu_fp8_bf16_dispatch = nullptr;

namespace {

void check_cuda_contiguous(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be torch.bfloat16");
}

void check_fp16(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kFloat16, name, " must be torch.float16");
}

void check_fp8(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              name, " must be torch.float8_e4m3fn");
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

#if defined(CUDA_KERNEL)
void require_sm110(torch::Tensor const& tensor, const char* op) {
  auto const* props = at::cuda::getDeviceProperties(tensor.get_device());
  TORCH_CHECK(props->major == 11 && props->minor == 0,
              op, " requires SM110; got SM", props->major, props->minor);
}
#endif

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

void moe_weighted_sum_bf16_to_fp32(
    torch::Tensor const& expert_output,
    torch::Tensor const& row_indices,
    torch::Tensor const& router_weight,
    torch::Tensor& out) {
  check_bf16(expert_output, "expert_output");
  check_i32(row_indices, "row_indices");
  check_f32(router_weight, "router_weight");
  check_f32(out, "out");
  TORCH_CHECK(expert_output.dim() == 2,
              "expert_output must have shape (routed_rows, stride)");
  TORCH_CHECK(row_indices.dim() == 2 && router_weight.sizes() == row_indices.sizes(),
              "row_indices and router_weight must have shape (tokens, topk)");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == row_indices.size(0),
              "out must have shape (tokens, hidden)");
  TORCH_CHECK(expert_output.size(1) >= out.size(1),
              "expert_output stride must cover out hidden size");
  same_device(expert_output, row_indices, "expert_output", "row_indices");
  same_device(expert_output, router_weight, "expert_output", "router_weight");
  same_device(expert_output, out, "expert_output", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(expert_output.device());
  auto stream = at::cuda::getCurrentCUDAStream(expert_output.get_device()).stream();
  const int rc = flash_rt::kernels::moe_weighted_sum_sm120_bf16(
      expert_output.data_ptr(), row_indices.data_ptr(), router_weight.data_ptr(),
      out.data_ptr(), checked_int(row_indices.size(0), "tokens"),
      checked_int(row_indices.size(1), "topk"),
      checked_int(out.size(1), "hidden"),
      checked_int(expert_output.size(1), "expert_output stride"), stream);
  TORCH_CHECK(rc == 0, "moe_weighted_sum_bf16_to_fp32 failed with rc=", rc);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void relu2_quantize_fp8_static_bf16(
    torch::Tensor const& input,
    torch::Tensor const& scale,
    torch::Tensor& output) {
  check_bf16(input, "input");
  check_f32(scale, "scale");
  check_cuda_contiguous(output, "output");
  TORCH_CHECK(
      output.scalar_type() == c10::ScalarType::Float8_e4m3fn,
      "output must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(output.sizes() == input.sizes(), "output shape mismatch");
  TORCH_CHECK(input.numel() > 0 && input.numel() % 2 == 0,
              "input numel must be positive and even");
  TORCH_CHECK(scale.numel() == 1, "scale must contain one FP32 value");
  same_device(input, scale, "input", "scale");
  same_device(input, output, "input", "output");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  flash_rt::kernels::relu2_quantize_fp8_static_bf16(
      input.data_ptr(),
      output.data_ptr(),
      scale.data_ptr<float>(),
      checked_int(input.numel(), "numel"),
      stream);
#else
  TORCH_CHECK(false, "transformer-fused-ops was not built with CUDA support");
#endif
}

void rms_norm_fp16(torch::Tensor const& x, torch::Tensor const& weight,
                   double eps, torch::Tensor& out) {
  check_fp16(x, "x");
  check_fp16(weight, "weight");
  check_fp16(out, "out");
  TORCH_CHECK(x.dim() == 2 && weight.sizes() == torch::IntArrayRef({x.size(1)}),
              "x must be (rows, dim) and weight must be (dim,)");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x");
  same_device(x, weight, "x", "weight");
  same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  require_sm110(x, "rms_norm_fp16");
  TORCH_CHECK(flash_rt::hub::rms_norm_fp16_dispatch,
              "SM110 FP16 RMSNorm source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::hub::rms_norm_fp16_dispatch(
      static_cast<const __half*>(x.data_ptr()),
      static_cast<const __half*>(weight.data_ptr()),
      static_cast<__half*>(out.data_ptr()), checked_int(x.size(0), "rows"),
      checked_int(x.size(1), "dim"), static_cast<float>(eps), stream);
  TORCH_CHECK(rc == 0, "rms_norm_fp16 failed with rc=", rc);
#endif
}

void layer_norm_fp16(torch::Tensor const& x, torch::Tensor const& weight,
                     torch::Tensor const& bias, double eps,
                     torch::Tensor& out) {
  check_fp16(x, "x");
  check_fp16(weight, "weight");
  check_fp16(bias, "bias");
  check_fp16(out, "out");
  TORCH_CHECK(x.dim() == 2 && weight.sizes() == torch::IntArrayRef({x.size(1)}) &&
                  bias.sizes() == weight.sizes(),
              "x must be (rows, dim), weight and bias must be (dim,)");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x");
  same_device(x, weight, "x", "weight");
  same_device(x, bias, "x", "bias");
  same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  require_sm110(x, "layer_norm_fp16");
  TORCH_CHECK(flash_rt::hub::layer_norm_fp16_dispatch,
              "SM110 FP16 LayerNorm source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::hub::layer_norm_fp16_dispatch(
      static_cast<const __half*>(x.data_ptr()),
      static_cast<const __half*>(weight.data_ptr()),
      static_cast<const __half*>(bias.data_ptr()),
      static_cast<__half*>(out.data_ptr()), checked_int(x.size(0), "rows"),
      checked_int(x.size(1), "dim"), static_cast<float>(eps), stream);
  TORCH_CHECK(rc == 0, "layer_norm_fp16 failed with rc=", rc);
#endif
}

void layer_norm_quant_fp8_static_fp16(
    torch::Tensor const& x, torch::Tensor const& weight,
    torch::Tensor const& bias, torch::Tensor const& scale, double eps,
    torch::Tensor& out) {
  check_fp16(x, "x");
  check_fp16(weight, "weight");
  check_fp16(bias, "bias");
  check_f32(scale, "scale");
  check_fp8(out, "out");
  TORCH_CHECK(x.dim() == 2 && weight.sizes() == torch::IntArrayRef({x.size(1)}) &&
                  bias.sizes() == weight.sizes(),
              "x must be (rows, dim), weight and bias must be (dim,)");
  TORCH_CHECK(scale.numel() == 1, "scale must contain one FP32 value");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x");
  same_device(x, weight, "x", "weight");
  same_device(x, bias, "x", "bias");
  same_device(x, scale, "x", "scale");
  same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  require_sm110(x, "layer_norm_quant_fp8_static_fp16");
  TORCH_CHECK(flash_rt::hub::layer_norm_fp8_fp16_dispatch,
              "SM110 fused LayerNorm-FP8 source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::hub::layer_norm_fp8_fp16_dispatch(
      static_cast<const __half*>(x.data_ptr()),
      static_cast<const __half*>(weight.data_ptr()),
      static_cast<const __half*>(bias.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(out.data_ptr()), scale.data_ptr<float>(),
      checked_int(x.size(0), "rows"), checked_int(x.size(1), "dim"),
      static_cast<float>(eps), stream);
  TORCH_CHECK(rc == 0, "layer_norm_quant_fp8_static_fp16 failed with rc=", rc);
#endif
}

void rope_rotate_half_fp16_(torch::Tensor& x, torch::Tensor const& cos,
                            torch::Tensor const& sin) {
  check_fp16(x, "x");
  check_fp16(cos, "cos");
  check_fp16(sin, "sin");
  TORCH_CHECK(x.dim() == 3, "x must have shape (sequence, heads, head_dim)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({x.size(0), x.size(2)}) &&
                  sin.sizes() == cos.sizes(),
              "cos and sin must have shape (sequence, head_dim)");
  same_device(x, cos, "x", "cos");
  same_device(x, sin, "x", "sin");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  require_sm110(x, "rope_rotate_half_fp16_");
  TORCH_CHECK(flash_rt::hub::rope_fp16_dispatch,
              "SM110 FP16 RoPE source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::hub::rope_fp16_dispatch(
      static_cast<__half*>(x.data_ptr()),
      static_cast<const __half*>(cos.data_ptr()),
      static_cast<const __half*>(sin.data_ptr()),
      checked_int(x.size(0), "sequence"), checked_int(x.size(1), "heads"),
      checked_int(x.size(2), "head_dim"), stream);
  TORCH_CHECK(rc == 0, "rope_rotate_half_fp16_ failed with rc=", rc);
#endif
}

void quantize_fp8_static_fp16(torch::Tensor const& x,
                              torch::Tensor const& scale,
                              torch::Tensor& out) {
  check_fp16(x, "x");
  check_f32(scale, "scale");
  check_fp8(out, "out");
  TORCH_CHECK(scale.numel() == 1, "scale must contain one FP32 value");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x");
  same_device(x, scale, "x", "scale");
  same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  require_sm110(x, "quantize_fp8_static_fp16");
  TORCH_CHECK(flash_rt::hub::quantize_fp8_fp16_dispatch,
              "SM110 vectorized FP8 quantizer is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::hub::quantize_fp8_fp16_dispatch(
      static_cast<const __half*>(x.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(out.data_ptr()), scale.data_ptr<float>(),
      checked_int(x.numel(), "numel"), stream);
  TORCH_CHECK(rc == 0, "quantize_fp8_static_fp16 failed with rc=", rc);
#endif
}

void quantize_fp8_static_bf16(torch::Tensor const& x,
                              torch::Tensor const& scale,
                              torch::Tensor& out) {
  check_bf16(x, "x");
  check_f32(scale, "scale");
  check_fp8(out, "out");
  TORCH_CHECK(scale.numel() == 1, "scale must contain one FP32 value");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x");
  same_device(x, scale, "x", "scale");
  same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  require_sm110(x, "quantize_fp8_static_bf16");
  TORCH_CHECK(flash_rt::hub::quantize_fp8_bf16_dispatch,
              "SM110 BF16 FP8 quantizer source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::hub::quantize_fp8_bf16_dispatch(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(out.data_ptr()), scale.data_ptr<float>(),
      checked_int(x.numel(), "numel"), stream);
  TORCH_CHECK(rc == 0, "quantize_fp8_static_bf16 failed with rc=", rc);
#endif
}

void layer_norm_quant_fp8_static_bf16(
    torch::Tensor const& x, torch::Tensor const& weight,
    torch::Tensor const& bias, torch::Tensor const& scale, double eps,
    torch::Tensor& out) {
  check_bf16(x, "x");
  check_bf16(weight, "weight");
  check_bf16(bias, "bias");
  check_f32(scale, "scale");
  check_fp8(out, "out");
  TORCH_CHECK(x.dim() == 2 &&
                  weight.sizes() == torch::IntArrayRef({x.size(1)}) &&
                  bias.sizes() == weight.sizes(),
              "x must be (rows, dim), weight and bias must be (dim,)");
  TORCH_CHECK(scale.numel() == 1, "scale must contain one FP32 value");
  TORCH_CHECK(out.sizes() == x.sizes(), "out must match x");
  same_device(x, weight, "x", "weight");
  same_device(x, bias, "x", "bias");
  same_device(x, scale, "x", "scale");
  same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  require_sm110(x, "layer_norm_quant_fp8_static_bf16");
  TORCH_CHECK(flash_rt::hub::layer_norm_fp8_bf16_dispatch,
              "SM110 BF16 LayerNorm-FP8 source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::hub::layer_norm_fp8_bf16_dispatch(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(bias.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(out.data_ptr()), scale.data_ptr<float>(),
      checked_int(x.size(0), "rows"), checked_int(x.size(1), "dim"),
      static_cast<float>(eps), stream);
  TORCH_CHECK(rc == 0,
              "layer_norm_quant_fp8_static_bf16 failed with rc=", rc);
#endif
}

void gate_geglu_merged_quant_fp8_static_bf16(
    torch::Tensor const& merged, torch::Tensor const& scale,
    torch::Tensor& out) {
  check_bf16(merged, "merged");
  check_f32(scale, "scale");
  check_fp8(out, "out");
  TORCH_CHECK(merged.dim() == 2 && merged.size(1) % 2 == 0,
              "merged must have shape (rows, 2 * hidden)");
  TORCH_CHECK(scale.numel() == 1, "scale must contain one FP32 value");
  TORCH_CHECK(out.sizes() ==
                  torch::IntArrayRef({merged.size(0), merged.size(1) / 2}),
              "out must have shape (rows, hidden)");
  same_device(merged, scale, "merged", "scale");
  same_device(merged, out, "merged", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(merged.device());
  require_sm110(merged, "gate_geglu_merged_quant_fp8_static_bf16");
  TORCH_CHECK(flash_rt::hub::gate_geglu_fp8_bf16_dispatch,
              "SM110 BF16 GeGLU-FP8 source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(merged.get_device()).stream();
  const int rc = flash_rt::hub::gate_geglu_fp8_bf16_dispatch(
      static_cast<const __nv_bfloat16*>(merged.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(out.data_ptr()), scale.data_ptr<float>(),
      checked_int(merged.size(0), "rows"),
      checked_int(merged.size(1) / 2, "hidden"), stream);
  TORCH_CHECK(rc == 0,
              "gate_geglu_merged_quant_fp8_static_bf16 failed with rc=", rc);
#endif
}

void residual_add_fp16_(torch::Tensor& residual, torch::Tensor const& x) {
  check_fp16(residual, "residual");
  check_fp16(x, "x");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x must match");
  same_device(residual, x, "residual", "x");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(residual.device());
  require_sm110(residual, "residual_add_fp16_");
  TORCH_CHECK(flash_rt::hub::residual_add_fp16_dispatch,
              "SM110 vectorized residual-add source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device()).stream();
  const int rc = flash_rt::hub::residual_add_fp16_dispatch(
      static_cast<__half*>(residual.data_ptr()),
      static_cast<const __half*>(x.data_ptr()),
      checked_int(x.numel(), "numel"), stream);
  TORCH_CHECK(rc == 0, "residual_add_fp16_ failed with rc=", rc);
#endif
}

void repeat_interleave_heads_fp16(torch::Tensor const& x, int64_t repeat,
                                  torch::Tensor& out) {
  check_fp16(x, "x");
  check_fp16(out, "out");
  TORCH_CHECK(x.dim() == 3, "x must have shape (sequence, heads, head_dim)");
  TORCH_CHECK(repeat > 0, "repeat must be positive");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef(
                  {x.size(0), x.size(1) * repeat, x.size(2)}),
              "out must have shape (sequence, heads * repeat, head_dim)");
  same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  require_sm110(x, "repeat_interleave_heads_fp16");
  TORCH_CHECK(flash_rt::hub::repeat_heads_fp16_dispatch,
              "SM110 vectorized head-repeat source is not present in this build");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int rc = flash_rt::hub::repeat_heads_fp16_dispatch(
      static_cast<const __half*>(x.data_ptr()),
      static_cast<__half*>(out.data_ptr()), checked_int(x.size(0), "sequence"),
      checked_int(x.size(1), "heads"), checked_int(x.size(2), "head_dim"),
      checked_int(repeat, "repeat"), stream);
  TORCH_CHECK(rc == 0, "repeat_interleave_heads_fp16 failed with rc=", rc);
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
  ops.def("router_topk_bf16(Tensor logits, Tensor! out_idx, Tensor! out_val, int k) -> ()");
  ops.def("moe_weighted_sum_bf16_to_fp32(Tensor expert_output, Tensor row_indices, Tensor router_weight, Tensor! out) -> ()");
  ops.def("relu2_quantize_fp8_static_bf16(Tensor input, Tensor scale, Tensor! output) -> ()");
  ops.def("rms_norm_fp16(Tensor x, Tensor weight, float eps, Tensor! out) -> ()");
  ops.def("layer_norm_fp16(Tensor x, Tensor weight, Tensor bias, float eps, Tensor! out) -> ()");
  ops.def("layer_norm_quant_fp8_static_fp16(Tensor x, Tensor weight, Tensor bias, Tensor scale, float eps, Tensor! out) -> ()");
  ops.def("rope_rotate_half_fp16_(Tensor(a!) x, Tensor cos, Tensor sin) -> ()");
  ops.def("quantize_fp8_static_fp16(Tensor x, Tensor scale, Tensor! out) -> ()");
  ops.def("quantize_fp8_static_bf16(Tensor x, Tensor scale, Tensor! out) -> ()");
  ops.def("layer_norm_quant_fp8_static_bf16(Tensor x, Tensor weight, Tensor bias, Tensor scale, float eps, Tensor! out) -> ()");
  ops.def("gate_geglu_merged_quant_fp8_static_bf16(Tensor merged, Tensor scale, Tensor! out) -> ()");
  ops.def("residual_add_fp16_(Tensor(a!) residual, Tensor x) -> ()");
  ops.def("repeat_interleave_heads_fp16(Tensor x, int repeat, Tensor! out) -> ()");
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
  ops.impl("router_topk_bf16", torch::kCUDA, &nexn2_router_topk_bf16);
  ops.impl("moe_weighted_sum_bf16_to_fp32", torch::kCUDA,
           &moe_weighted_sum_bf16_to_fp32);
  ops.impl("relu2_quantize_fp8_static_bf16", torch::kCUDA, &relu2_quantize_fp8_static_bf16);
  ops.impl("rms_norm_fp16", torch::kCUDA, &rms_norm_fp16);
  ops.impl("layer_norm_fp16", torch::kCUDA, &layer_norm_fp16);
  ops.impl("layer_norm_quant_fp8_static_fp16", torch::kCUDA, &layer_norm_quant_fp8_static_fp16);
  ops.impl("rope_rotate_half_fp16_", torch::kCUDA, &rope_rotate_half_fp16_);
  ops.impl("quantize_fp8_static_fp16", torch::kCUDA, &quantize_fp8_static_fp16);
  ops.impl("quantize_fp8_static_bf16", torch::kCUDA, &quantize_fp8_static_bf16);
  ops.impl("layer_norm_quant_fp8_static_bf16", torch::kCUDA,
           &layer_norm_quant_fp8_static_bf16);
  ops.impl("gate_geglu_merged_quant_fp8_static_bf16", torch::kCUDA,
           &gate_geglu_merged_quant_fp8_static_bf16);
  ops.impl("residual_add_fp16_", torch::kCUDA, &residual_add_fp16_);
  ops.impl("repeat_interleave_heads_fp16", torch::kCUDA, &repeat_interleave_heads_fp16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
