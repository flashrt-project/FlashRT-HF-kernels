// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "fa2_wrapper.h"
#include "registration.h"
#include "torch_binding.h"

namespace {

struct Shape {
  int batch;
  int seqlen_q;
  int seqlen_k;
  int heads_q;
  int heads_kv;
  int head_dim;
};

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit int32");
  return static_cast<int>(value);
}

bool supported_head_dim(int64_t d) {
  return d == 64 || d == 96 || d == 128 || d == 256;
}

void check_cuda_tensor(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.layout() == torch::kStrided, name, " must be strided");
}

void check_bshd(const torch::Tensor& x, const char* name) {
  check_cuda_tensor(x, name);
  TORCH_CHECK(x.dim() == 4, name, " must have shape (B, S, H, D)");
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
              name, " dimensions must be positive");
  TORCH_CHECK(supported_head_dim(x.size(3)), name,
              " head_dim must be one of {64, 96, 128, 256}");
  TORCH_CHECK(x.stride(3) == 1, name, " last dimension must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              name, " base pointer must be 16-byte aligned");
  TORCH_CHECK(x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0 &&
                  x.stride(2) % 8 == 0,
              name, " batch, row, and head strides must preserve 16-byte alignment");
  for (int i = 0; i < 4; ++i) {
    checked_int(x.stride(i), name);
  }
}

Shape check_qkv(const torch::Tensor& q, const torch::Tensor& k,
                const torch::Tensor& v) {
  check_bshd(q, "q");
  check_bshd(k, "k");
  check_bshd(v, "v");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 ||
                  q.scalar_type() == torch::kBFloat16,
              "q must be fp16 or bf16");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type(),
              "q, k, and v must have the same dtype");
  TORCH_CHECK(q.get_device() == k.get_device() &&
                  q.get_device() == v.get_device(),
              "q, k, and v must be on the same device");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "q, k, and v batch dimensions must match");
  TORCH_CHECK(k.size(1) == v.size(1), "k and v sequence lengths must match");
  TORCH_CHECK(k.size(2) == v.size(2), "k and v head counts must match");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
              "q, k, and v head dimensions must match");
  TORCH_CHECK(q.size(2) % k.size(2) == 0,
              "query heads must be divisible by KV heads");
  return {checked_int(q.size(0), "batch"),
          checked_int(q.size(1), "seqlen_q"),
          checked_int(k.size(1), "seqlen_k"),
          checked_int(q.size(2), "heads_q"),
          checked_int(k.size(2), "heads_kv"),
          checked_int(q.size(3), "head_dim")};
}

void check_outputs(const torch::Tensor& q, const torch::Tensor& out,
                   const torch::Tensor& lse, const Shape& shape) {
  check_bshd(out, "out");
  TORCH_CHECK(out.scalar_type() == q.scalar_type(),
              "out dtype must match q");
  TORCH_CHECK(out.sizes() == q.sizes(), "out shape must match q");
  TORCH_CHECK(out.get_device() == q.get_device(),
              "out must be on the q device");
  check_cuda_tensor(lse, "softmax_lse");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.is_contiguous(),
              "softmax_lse must be contiguous fp32");
  TORCH_CHECK(lse.dim() == 3 && lse.size(0) == shape.batch &&
                  lse.size(1) == shape.heads_q &&
                  lse.size(2) == shape.seqlen_q,
              "softmax_lse must have shape (B, Hq, Sq)");
  TORCH_CHECK(lse.get_device() == q.get_device(),
              "softmax_lse must be on the q device");
}

int num_splits_heuristic(int batch_heads_mblocks, int num_sms,
                         int num_n_blocks, int max_splits) {
  if (batch_heads_mblocks >= 0.8f * num_sms) return 1;
  max_splits = std::min({max_splits, num_sms, num_n_blocks});
  float best = 0.0f;
  float efficiency[129] = {};
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  auto eligible = [&](int s) {
    return s == 1 || ceildiv(num_n_blocks, s) !=
                         ceildiv(num_n_blocks, s - 1);
  };
  for (int s = 1; s <= max_splits; ++s) {
    if (!eligible(s)) continue;
    float waves = static_cast<float>(batch_heads_mblocks * s) / num_sms;
    efficiency[s] = waves / std::ceil(waves);
    best = std::max(best, efficiency[s]);
  }
  for (int s = 1; s <= max_splits; ++s) {
    if (eligible(s) && efficiency[s] >= 0.85f * best) return s;
  }
  return 1;
}

int requested_splits(const Shape& shape, int num_sms, bool has_workspace) {
  if (!has_workspace || num_sms <= 0) return 1;
  const int block_n = shape.head_dim <= 64 ? 256 :
                      shape.head_dim <= 128 ? 128 : 64;
  const int n_blocks = (shape.seqlen_k + block_n - 1) / block_n;
  const int m_blocks = (shape.seqlen_q + 63) / 64;
  return num_splits_heuristic(shape.batch * shape.heads_q * m_blocks,
                              num_sms * 2, n_blocks, 128);
}

void check_workspace(const torch::Tensor& q, const Shape& shape,
                     const c10::optional<torch::Tensor>& lse_accum,
                     const c10::optional<torch::Tensor>& out_accum,
                     int num_sms, bool seqused) {
  TORCH_CHECK(lse_accum.has_value() == out_accum.has_value(),
              "softmax_lse_accum and out_accum must be both set or both None");
  if (!lse_accum.has_value()) {
    TORCH_CHECK(num_sms == 0,
                "num_sms must be zero when split-KV workspace is absent");
    return;
  }
  TORCH_CHECK(shape.head_dim != 64,
              "split-KV is not built for head_dim=64");
  const auto& lse = *lse_accum;
  const auto& out = *out_accum;
  check_cuda_tensor(lse, "softmax_lse_accum");
  check_cuda_tensor(out, "out_accum");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.is_contiguous(),
              "softmax_lse_accum must be contiguous fp32");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32 && out.is_contiguous(),
              "out_accum must be contiguous fp32");
  TORCH_CHECK(lse.get_device() == q.get_device() &&
                  out.get_device() == q.get_device(),
              "workspace tensors must be on the q device");
  TORCH_CHECK(num_sms > 0, "num_sms must be positive with split-KV workspace");
  const int splits = requested_splits(shape, num_sms, true);
  const int64_t lse_elems = static_cast<int64_t>(splits) * shape.batch *
                            shape.heads_q * shape.seqlen_q;
  const int64_t d_rounded = (shape.head_dim + 31) & ~31;
  const int64_t out_elems = lse_elems * d_rounded;
  TORCH_CHECK(lse.numel() >= lse_elems,
              "softmax_lse_accum is too small; need ", lse_elems,
              " fp32 elements");
  TORCH_CHECK(out.numel() >= out_elems,
              "out_accum is too small; need ", out_elems,
              " fp32 elements");
  (void)seqused;
}

int device_sm_count(const torch::Tensor& q, int64_t requested) {
  if (requested > 0) return checked_int(requested, "num_sms");
  return 0;
}

void launch_forward(const torch::Tensor& q, const torch::Tensor& k,
                    const torch::Tensor& v, torch::Tensor& out,
                    torch::Tensor& lse,
                    const c10::optional<torch::Tensor>& lse_accum,
                    const c10::optional<torch::Tensor>& out_accum,
                    double scale, bool causal, int num_sms,
                    const Shape& shape) {
  void* lse_accum_ptr = lse_accum ? lse_accum->data_ptr() : nullptr;
  void* out_accum_ptr = out_accum ? out_accum->data_ptr() : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  auto fn = q.scalar_type() == torch::kFloat16
                ? &fvk_attention_fa2_fwd_fp16
                : (causal ? &fvk_attention_fa2_fwd_bf16_causal
                          : &fvk_attention_fa2_fwd_bf16);
  fn(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), lse.data_ptr(),
     lse_accum_ptr, out_accum_ptr, shape.batch, shape.seqlen_q,
     shape.seqlen_k, shape.heads_q, shape.heads_kv, shape.head_dim,
     checked_int(q.stride(0), "q batch stride"),
     checked_int(q.stride(1), "q row stride"),
     checked_int(q.stride(2), "q head stride"),
     checked_int(k.stride(0), "k batch stride"),
     checked_int(k.stride(1), "k row stride"),
     checked_int(k.stride(2), "k head stride"),
     checked_int(v.stride(0), "v batch stride"),
     checked_int(v.stride(1), "v row stride"),
     checked_int(v.stride(2), "v head stride"),
     checked_int(out.stride(0), "out batch stride"),
     checked_int(out.stride(1), "out row stride"),
     checked_int(out.stride(2), "out head stride"),
     static_cast<float>(scale), num_sms, stream);
}

}  // namespace

void fa2_forward_static(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    torch::Tensor& out, torch::Tensor& softmax_lse,
    const c10::optional<torch::Tensor>& softmax_lse_accum,
    const c10::optional<torch::Tensor>& out_accum, double softmax_scale,
    bool causal, int64_t num_sms) {
  const Shape shape = check_qkv(q, k, v);
  check_outputs(q, out, softmax_lse, shape);
  TORCH_CHECK(softmax_scale > 0.0 && std::isfinite(softmax_scale),
              "softmax_scale must be finite and positive");
  TORCH_CHECK(!causal || q.scalar_type() == torch::kBFloat16,
              "causal v1 supports bf16 only");
  TORCH_CHECK(!causal || shape.head_dim == 128 || shape.head_dim == 256,
              "causal v1 supports head_dim 128 or 256 only");
  const int sms = device_sm_count(q, num_sms);
  check_workspace(q, shape, softmax_lse_accum, out_accum, sms, false);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q.device());
  launch_forward(q, k, v, out, softmax_lse, softmax_lse_accum,
                 out_accum, softmax_scale, causal, sms, shape);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "fa2-seqused-runtime was not built with CUDA support");
#endif
}

void fa2_forward_seqused_static(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& seqused_k, torch::Tensor& out,
    torch::Tensor& softmax_lse,
    const c10::optional<torch::Tensor>& softmax_lse_accum,
    const c10::optional<torch::Tensor>& out_accum, double softmax_scale,
    int64_t num_sms) {
  const Shape shape = check_qkv(q, k, v);
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16,
              "seqused v1 supports bf16 only");
  check_outputs(q, out, softmax_lse, shape);
  check_cuda_tensor(seqused_k, "seqused_k");
  TORCH_CHECK(seqused_k.scalar_type() == torch::kInt32 &&
                  seqused_k.is_contiguous() &&
                  seqused_k.numel() == shape.batch,
              "seqused_k must be contiguous int32 with B elements");
  TORCH_CHECK(seqused_k.get_device() == q.get_device(),
              "seqused_k must be on the q device");
  TORCH_CHECK(softmax_scale > 0.0 && std::isfinite(softmax_scale),
              "softmax_scale must be finite and positive");
  const int sms = device_sm_count(q, num_sms);
  check_workspace(q, shape, softmax_lse_accum, out_accum, sms, true);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  if (softmax_lse_accum.has_value()) {
    fvk_attention_fa2_fwd_bf16_seqused_splitkv(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        softmax_lse.data_ptr(), seqused_k.data_ptr(),
        softmax_lse_accum->data_ptr(), out_accum->data_ptr(), shape.batch,
        shape.seqlen_q, shape.seqlen_k, shape.heads_q, shape.heads_kv,
        shape.head_dim, checked_int(q.stride(0), "q batch stride"),
        checked_int(q.stride(1), "q row stride"),
        checked_int(q.stride(2), "q head stride"),
        checked_int(k.stride(0), "k batch stride"),
        checked_int(k.stride(1), "k row stride"),
        checked_int(k.stride(2), "k head stride"),
        checked_int(v.stride(0), "v batch stride"),
        checked_int(v.stride(1), "v row stride"),
        checked_int(v.stride(2), "v head stride"),
        checked_int(out.stride(0), "out batch stride"),
        checked_int(out.stride(1), "out row stride"),
        checked_int(out.stride(2), "out head stride"),
        static_cast<float>(softmax_scale), sms, stream);
  } else {
    fvk_attention_fa2_fwd_bf16_seqused(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        softmax_lse.data_ptr(), seqused_k.data_ptr(), shape.batch,
        shape.seqlen_q, shape.seqlen_k, shape.heads_q, shape.heads_kv,
        shape.head_dim, checked_int(q.stride(0), "q batch stride"),
        checked_int(q.stride(1), "q row stride"),
        checked_int(q.stride(2), "q head stride"),
        checked_int(k.stride(0), "k batch stride"),
        checked_int(k.stride(1), "k row stride"),
        checked_int(k.stride(2), "k head stride"),
        checked_int(v.stride(0), "v batch stride"),
        checked_int(v.stride(1), "v row stride"),
        checked_int(v.stride(2), "v head stride"),
        checked_int(out.stride(0), "out batch stride"),
        checked_int(out.stride(1), "out row stride"),
        checked_int(out.stride(2), "out head stride"),
        static_cast<float>(softmax_scale), 0, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "fa2-seqused-runtime was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("forward_static(Tensor q, Tensor k, Tensor v, Tensor! out, Tensor! softmax_lse, Tensor(a!)? softmax_lse_accum, Tensor(b!)? out_accum, float softmax_scale, bool causal=False, int num_sms=0) -> ()");
  ops.def("forward_seqused_static(Tensor q, Tensor k, Tensor v, Tensor seqused_k, Tensor! out, Tensor! softmax_lse, Tensor(a!)? softmax_lse_accum, Tensor(b!)? out_accum, float softmax_scale, int num_sms=0) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("forward_static", torch::kCUDA, &fa2_forward_static);
  ops.impl("forward_seqused_static", torch::kCUDA,
           &fa2_forward_seqused_static);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
