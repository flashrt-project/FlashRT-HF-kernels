// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "qkv_cache_rope.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_fp32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
}

void check_int32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
              name, " must have dtype torch.int32");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

void check_same_device(torch::Tensor const& a,
                       torch::Tensor const& b,
                       const char* a_name,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_qkv_common(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    int64_t heads,
    int64_t head_dim,
    const char* packed_name) {
  TORCH_CHECK(packed_qkv.dim() == 3,
              packed_name, " must have shape (batch, seq_len, 3 * heads * head_dim)");
  TORCH_CHECK(packed_qkv.size(0) > 0 && packed_qkv.size(1) > 0,
              packed_name, " batch and seq_len must be positive");
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
  const int64_t dim = heads * head_dim;
  TORCH_CHECK(dim > 0 && dim % 2 == 0, "heads * head_dim must be positive and even");
  TORCH_CHECK(packed_qkv.size(2) == 3 * dim,
              packed_name, ".shape[2] must be 3 * heads * head_dim");
  TORCH_CHECK(norm_q_weight.dim() == 1 && norm_q_weight.size(0) == dim,
              "norm_q_weight must have shape (heads * head_dim,)");
  TORCH_CHECK(norm_k_weight.dim() == 1 && norm_k_weight.size(0) == dim,
              "norm_k_weight must have shape (heads * head_dim,)");
}

void check_freqs(
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    int64_t head_dim,
    int64_t rope_seq_len) {
  check_fp32(freqs_re, "freqs_re");
  check_fp32(freqs_im, "freqs_im");
  TORCH_CHECK(freqs_re.dim() == 2 && freqs_re.size(1) == head_dim / 2,
              "freqs_re must have shape (rope_seq_len, head_dim / 2)");
  TORCH_CHECK(freqs_im.dim() == 2 && freqs_im.sizes() == freqs_re.sizes(),
              "freqs_im must have the same shape as freqs_re");
  TORCH_CHECK(rope_seq_len >= 0,
              "rope_seq_len must be non-negative");
  TORCH_CHECK(freqs_re.size(0) >= rope_seq_len,
              "freqs_re must have at least rope_seq_len rows");
}

void check_cat_out(
    torch::Tensor const& out,
    int64_t batch,
    int64_t total_seq_len,
    int64_t heads,
    int64_t head_dim,
    const char* name) {
  check_bf16(out, name);
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({batch, total_seq_len, heads, head_dim}),
              name, " must have shape (batch, total_seq_len, heads, head_dim)");
}

}  // namespace

void decode_q_norm_rope_stage_bf16(
    torch::Tensor const& q_pre,
    torch::Tensor const& q_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    double eps,
    torch::Tensor& q_out) {
  check_bf16(q_pre, "q_pre");
  check_bf16(q_norm_weight, "q_norm_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(q_out, "q_out");
  TORCH_CHECK(q_pre.dim() == 2 && q_pre.size(1) == 128,
              "q_pre must have shape (n_q_heads, 128)");
  const int64_t n_q_heads = q_pre.size(0);
  TORCH_CHECK(n_q_heads > 0, "n_q_heads must be positive");
  TORCH_CHECK(q_norm_weight.sizes() == torch::IntArrayRef({128}),
              "q_norm_weight must have shape (128,)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({64}) &&
              sin.sizes() == torch::IntArrayRef({64}),
              "cos and sin must have shape (64,)");
  TORCH_CHECK(q_out.sizes() == q_pre.sizes(),
              "q_out must have shape (n_q_heads, 128)");
  check_same_device(q_pre, q_norm_weight, "q_pre", "q_norm_weight");
  check_same_device(q_pre, cos, "q_pre", "cos");
  check_same_device(q_pre, sin, "q_pre", "sin");
  check_same_device(q_pre, q_out, "q_pre", "q_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q_pre.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_pre.get_device()).stream();
  flash_rt::qkv_cache_rope::decode_q_norm_rope_stage_bf16(
      q_pre.data_ptr(),
      q_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      q_out.data_ptr(),
      checked_int(n_q_heads, "n_q_heads"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void decode_k_norm_rope_kvwrite_bf16(
    torch::Tensor const& k_pre,
    torch::Tensor const& v_pre,
    torch::Tensor const& k_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    double eps,
    torch::Tensor& k_cache_dst,
    torch::Tensor& v_cache_dst) {
  check_bf16(k_pre, "k_pre");
  check_bf16(v_pre, "v_pre");
  check_bf16(k_norm_weight, "k_norm_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(k_cache_dst, "k_cache_dst");
  check_bf16(v_cache_dst, "v_cache_dst");
  TORCH_CHECK(k_pre.dim() == 2 && k_pre.size(1) == 128,
              "k_pre must have shape (n_kv_heads, 128)");
  const int64_t n_kv_heads = k_pre.size(0);
  TORCH_CHECK(n_kv_heads > 0, "n_kv_heads must be positive");
  TORCH_CHECK(v_pre.sizes() == k_pre.sizes(),
              "v_pre must have shape (n_kv_heads, 128)");
  TORCH_CHECK(k_norm_weight.sizes() == torch::IntArrayRef({128}),
              "k_norm_weight must have shape (128,)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({64}) &&
              sin.sizes() == torch::IntArrayRef({64}),
              "cos and sin must have shape (64,)");
  TORCH_CHECK(k_cache_dst.sizes() == k_pre.sizes() &&
              v_cache_dst.sizes() == k_pre.sizes(),
              "k_cache_dst and v_cache_dst must have shape (n_kv_heads, 128)");
  check_same_device(k_pre, v_pre, "k_pre", "v_pre");
  check_same_device(k_pre, k_norm_weight, "k_pre", "k_norm_weight");
  check_same_device(k_pre, cos, "k_pre", "cos");
  check_same_device(k_pre, sin, "k_pre", "sin");
  check_same_device(k_pre, k_cache_dst, "k_pre", "k_cache_dst");
  check_same_device(k_pre, v_cache_dst, "k_pre", "v_cache_dst");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k_pre.device());
  auto stream = at::cuda::getCurrentCUDAStream(k_pre.get_device()).stream();
  flash_rt::qkv_cache_rope::decode_k_norm_rope_kvwrite_bf16(
      k_pre.data_ptr(),
      v_pre.data_ptr(),
      k_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      k_cache_dst.data_ptr(),
      v_cache_dst.data_ptr(),
      checked_int(n_kv_heads, "n_kv_heads"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void decode_k_norm_rope_kvwrite_devpos_bf16(
    torch::Tensor const& k_pre,
    torch::Tensor const& v_pre,
    torch::Tensor const& k_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    torch::Tensor const& cur_pos,
    double eps,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache) {
  check_bf16(k_pre, "k_pre");
  check_bf16(v_pre, "v_pre");
  check_bf16(k_norm_weight, "k_norm_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_int32(cur_pos, "cur_pos");
  check_bf16(k_cache, "k_cache");
  check_bf16(v_cache, "v_cache");
  TORCH_CHECK(k_pre.dim() == 2 && k_pre.size(1) == 128,
              "k_pre must have shape (n_kv_heads, 128)");
  const int64_t n_kv_heads = k_pre.size(0);
  TORCH_CHECK(n_kv_heads > 0, "n_kv_heads must be positive");
  TORCH_CHECK(v_pre.sizes() == k_pre.sizes(),
              "v_pre must have shape (n_kv_heads, 128)");
  TORCH_CHECK(k_norm_weight.sizes() == torch::IntArrayRef({128}),
              "k_norm_weight must have shape (128,)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({64}) &&
              sin.sizes() == torch::IntArrayRef({64}),
              "cos and sin must have shape (64,)");
  TORCH_CHECK(cur_pos.numel() == 1, "cur_pos must have one int32 element");
  TORCH_CHECK(k_cache.dim() == 3 &&
              k_cache.size(1) == n_kv_heads &&
              k_cache.size(2) == 128,
              "k_cache must have shape (max_seq_len, n_kv_heads, 128)");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(),
              "v_cache must have the same shape as k_cache");
  check_same_device(k_pre, v_pre, "k_pre", "v_pre");
  check_same_device(k_pre, k_norm_weight, "k_pre", "k_norm_weight");
  check_same_device(k_pre, cos, "k_pre", "cos");
  check_same_device(k_pre, sin, "k_pre", "sin");
  check_same_device(k_pre, cur_pos, "k_pre", "cur_pos");
  check_same_device(k_pre, k_cache, "k_pre", "k_cache");
  check_same_device(k_pre, v_cache, "k_pre", "v_cache");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k_pre.device());
  auto stream = at::cuda::getCurrentCUDAStream(k_pre.get_device()).stream();
  flash_rt::qkv_cache_rope::decode_k_norm_rope_kvwrite_devpos_bf16(
      k_pre.data_ptr(),
      v_pre.data_ptr(),
      k_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      k_cache.data_ptr(),
      v_cache.data_ptr(),
      cur_pos.data_ptr(),
      checked_int(n_kv_heads * 128, "row_elems"),
      checked_int(n_kv_heads, "n_kv_heads"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_norm_rope_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    int64_t heads64,
    int64_t head_dim64,
    int64_t rope_seq_len64,
    double eps,
    torch::Tensor& q_out,
    torch::Tensor& k_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(norm_q_weight, "norm_q_weight");
  check_bf16(norm_k_weight, "norm_k_weight");
  check_fp32(freqs_re, "freqs_re");
  check_fp32(freqs_im, "freqs_im");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  TORCH_CHECK(packed_qkv.dim() == 3,
              "packed_qkv must have shape (batch, seq_len, 3 * heads * head_dim)");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int heads = checked_int(heads64, "heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  const int64_t dim = static_cast<int64_t>(heads) * head_dim;
  TORCH_CHECK(batch > 0 && seq_len > 0, "batch and seq_len must be positive");
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
  TORCH_CHECK(dim % 2 == 0, "heads * head_dim must be even");
  TORCH_CHECK(packed_qkv.size(2) == 3 * dim,
              "packed_qkv.shape[2] must be 3 * heads * head_dim");
  TORCH_CHECK(norm_q_weight.dim() == 1 && norm_q_weight.size(0) == dim,
              "norm_q_weight must have shape (heads * head_dim,)");
  TORCH_CHECK(norm_k_weight.dim() == 1 && norm_k_weight.size(0) == dim,
              "norm_k_weight must have shape (heads * head_dim,)");
  TORCH_CHECK(freqs_re.dim() == 2 && freqs_re.size(1) == head_dim / 2,
              "freqs_re must have shape (rope_seq_len, head_dim / 2)");
  TORCH_CHECK(freqs_im.dim() == 2 && freqs_im.sizes() == freqs_re.sizes(),
              "freqs_im must have the same shape as freqs_re");
  TORCH_CHECK(rope_seq_len64 >= 0 && rope_seq_len64 <= seq_len,
              "rope_seq_len must be in [0, packed_qkv.shape[1]]");
  TORCH_CHECK(freqs_re.size(0) >= rope_seq_len64,
              "freqs_re must have at least rope_seq_len rows");
  TORCH_CHECK(q_out.sizes() == torch::IntArrayRef({batch, seq_len, heads, head_dim}),
              "q_out must have shape (batch, seq_len, heads, head_dim)");
  TORCH_CHECK(k_out.sizes() == q_out.sizes(),
              "k_out must have the same shape as q_out");

  check_same_device(packed_qkv, norm_q_weight, "packed_qkv", "norm_q_weight");
  check_same_device(packed_qkv, norm_k_weight, "packed_qkv", "norm_k_weight");
  check_same_device(packed_qkv, freqs_re, "packed_qkv", "freqs_re");
  check_same_device(packed_qkv, freqs_im, "packed_qkv", "freqs_im");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_out, "packed_qkv", "k_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_norm_rope_bf16(
      packed_qkv.data_ptr(),
      norm_q_weight.data_ptr(),
      norm_k_weight.data_ptr(),
      reinterpret_cast<const float*>(freqs_re.data_ptr()),
      reinterpret_cast<const float*>(freqs_im.data_ptr()),
      q_out.data_ptr(),
      k_out.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(seq_len, "seq_len"),
      heads,
      head_dim,
      checked_int(rope_seq_len64, "rope_seq_len"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_bias_norm_rope_v_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& qkv_bias,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    int64_t heads64,
    int64_t head_dim64,
    int64_t rope_seq_len64,
    double eps,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(qkv_bias, "qkv_bias");
  check_bf16(norm_q_weight, "norm_q_weight");
  check_bf16(norm_k_weight, "norm_k_weight");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  check_bf16(v_out, "v_out");
  const int heads = checked_int(heads64, "heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  check_qkv_common(packed_qkv, norm_q_weight, norm_k_weight, heads, head_dim, "packed_qkv");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int64_t dim = static_cast<int64_t>(heads) * head_dim;
  TORCH_CHECK(qkv_bias.dim() == 1 && qkv_bias.size(0) == 3 * dim,
              "qkv_bias must have shape (3 * heads * head_dim,)");
  TORCH_CHECK(rope_seq_len64 >= 0 && rope_seq_len64 <= seq_len,
              "rope_seq_len must be in [0, packed_qkv.shape[1]]");
  check_freqs(freqs_re, freqs_im, head_dim, rope_seq_len64);
  TORCH_CHECK(q_out.sizes() == torch::IntArrayRef({batch, seq_len, heads, head_dim}),
              "q_out must have shape (batch, seq_len, heads, head_dim)");
  TORCH_CHECK(k_out.sizes() == q_out.sizes() && v_out.sizes() == q_out.sizes(),
              "k_out and v_out must have the same shape as q_out");

  check_same_device(packed_qkv, qkv_bias, "packed_qkv", "qkv_bias");
  check_same_device(packed_qkv, norm_q_weight, "packed_qkv", "norm_q_weight");
  check_same_device(packed_qkv, norm_k_weight, "packed_qkv", "norm_k_weight");
  check_same_device(packed_qkv, freqs_re, "packed_qkv", "freqs_re");
  check_same_device(packed_qkv, freqs_im, "packed_qkv", "freqs_im");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_out, "packed_qkv", "k_out");
  check_same_device(packed_qkv, v_out, "packed_qkv", "v_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_bias_norm_rope_v_bf16(
      packed_qkv.data_ptr(),
      qkv_bias.data_ptr(),
      norm_q_weight.data_ptr(),
      norm_k_weight.data_ptr(),
      reinterpret_cast<const float*>(freqs_re.data_ptr()),
      reinterpret_cast<const float*>(freqs_im.data_ptr()),
      q_out.data_ptr(),
      k_out.data_ptr(),
      v_out.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(seq_len, "seq_len"),
      heads,
      head_dim,
      checked_int(rope_seq_len64, "rope_seq_len"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_bias_norm_rope_v_cat_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& qkv_bias,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    int64_t heads64,
    int64_t head_dim64,
    int64_t video_offset64,
    int64_t rope_seq_len64,
    double eps,
    torch::Tensor& q_cat_out,
    torch::Tensor& k_cat_out,
    torch::Tensor& v_cat_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(qkv_bias, "qkv_bias");
  check_bf16(norm_q_weight, "norm_q_weight");
  check_bf16(norm_k_weight, "norm_k_weight");
  const int heads = checked_int(heads64, "heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  check_qkv_common(packed_qkv, norm_q_weight, norm_k_weight, heads, head_dim, "packed_qkv");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int64_t dim = static_cast<int64_t>(heads) * head_dim;
  TORCH_CHECK(qkv_bias.dim() == 1 && qkv_bias.size(0) == 3 * dim,
              "qkv_bias must have shape (3 * heads * head_dim,)");
  TORCH_CHECK(q_cat_out.dim() == 4,
              "q_cat_out must have shape (batch, total_seq_len, heads, head_dim)");
  const int64_t total_seq_len = q_cat_out.size(1);
  TORCH_CHECK(video_offset64 >= 0 && video_offset64 + seq_len <= total_seq_len,
              "video_offset + packed_qkv.shape[1] must be within q_cat_out.shape[1]");
  TORCH_CHECK(rope_seq_len64 >= 0 && rope_seq_len64 <= seq_len,
              "rope_seq_len must be in [0, packed_qkv.shape[1]]");
  check_freqs(freqs_re, freqs_im, head_dim, rope_seq_len64);
  check_cat_out(q_cat_out, batch, total_seq_len, heads, head_dim, "q_cat_out");
  check_cat_out(k_cat_out, batch, total_seq_len, heads, head_dim, "k_cat_out");
  check_cat_out(v_cat_out, batch, total_seq_len, heads, head_dim, "v_cat_out");

  check_same_device(packed_qkv, qkv_bias, "packed_qkv", "qkv_bias");
  check_same_device(packed_qkv, norm_q_weight, "packed_qkv", "norm_q_weight");
  check_same_device(packed_qkv, norm_k_weight, "packed_qkv", "norm_k_weight");
  check_same_device(packed_qkv, freqs_re, "packed_qkv", "freqs_re");
  check_same_device(packed_qkv, freqs_im, "packed_qkv", "freqs_im");
  check_same_device(packed_qkv, q_cat_out, "packed_qkv", "q_cat_out");
  check_same_device(packed_qkv, k_cat_out, "packed_qkv", "k_cat_out");
  check_same_device(packed_qkv, v_cat_out, "packed_qkv", "v_cat_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_bias_norm_rope_v_cat_bf16(
      packed_qkv.data_ptr(),
      qkv_bias.data_ptr(),
      norm_q_weight.data_ptr(),
      norm_k_weight.data_ptr(),
      reinterpret_cast<const float*>(freqs_re.data_ptr()),
      reinterpret_cast<const float*>(freqs_im.data_ptr()),
      q_cat_out.data_ptr(),
      k_cat_out.data_ptr(),
      v_cat_out.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(total_seq_len, "total_seq_len"),
      checked_int(video_offset64, "video_offset"),
      checked_int(seq_len, "video_seq_len"),
      heads,
      head_dim,
      checked_int(rope_seq_len64, "rope_seq_len"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_joint3_cat_bf16(
    torch::Tensor const& packed_v,
    torch::Tensor const& qkv_v_bias,
    torch::Tensor const& norm_v_q_weight,
    torch::Tensor const& norm_v_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    torch::Tensor const& packed_a,
    torch::Tensor const& norm_a_q_weight,
    torch::Tensor const& norm_a_k_weight,
    torch::Tensor const& packed_u,
    torch::Tensor const& norm_u_q_weight,
    torch::Tensor const& norm_u_k_weight,
    int64_t heads64,
    int64_t head_dim64,
    int64_t rope_seq_len64,
    double eps_v,
    double eps_a,
    double eps_u,
    torch::Tensor& q_cat_out,
    torch::Tensor& k_cat_out,
    torch::Tensor& v_cat_out) {
  check_bf16(packed_v, "packed_v");
  check_bf16(qkv_v_bias, "qkv_v_bias");
  check_bf16(norm_v_q_weight, "norm_v_q_weight");
  check_bf16(norm_v_k_weight, "norm_v_k_weight");
  check_bf16(packed_a, "packed_a");
  check_bf16(norm_a_q_weight, "norm_a_q_weight");
  check_bf16(norm_a_k_weight, "norm_a_k_weight");
  check_bf16(packed_u, "packed_u");
  check_bf16(norm_u_q_weight, "norm_u_q_weight");
  check_bf16(norm_u_k_weight, "norm_u_k_weight");
  const int heads = checked_int(heads64, "heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  check_qkv_common(packed_v, norm_v_q_weight, norm_v_k_weight, heads, head_dim, "packed_v");
  check_qkv_common(packed_a, norm_a_q_weight, norm_a_k_weight, heads, head_dim, "packed_a");
  check_qkv_common(packed_u, norm_u_q_weight, norm_u_k_weight, heads, head_dim, "packed_u");
  const int64_t batch = packed_v.size(0);
  TORCH_CHECK(batch == 1, "qkv_split_joint3_cat_bf16 currently supports batch == 1");
  TORCH_CHECK(packed_a.size(0) == batch && packed_u.size(0) == batch,
              "packed_v, packed_a, and packed_u must have the same batch");
  const int64_t video_seq_len = packed_v.size(1);
  const int64_t action_seq_len = packed_a.size(1);
  const int64_t und_seq_len = packed_u.size(1);
  const int64_t total_seq_len = video_seq_len + action_seq_len + und_seq_len;
  const int64_t dim = static_cast<int64_t>(heads) * head_dim;
  TORCH_CHECK(qkv_v_bias.dim() == 1 && qkv_v_bias.size(0) == 3 * dim,
              "qkv_v_bias must have shape (3 * heads * head_dim,)");
  TORCH_CHECK(rope_seq_len64 >= 0 && rope_seq_len64 <= video_seq_len,
              "rope_seq_len must be in [0, packed_v.shape[1]]");
  check_freqs(freqs_re, freqs_im, head_dim, rope_seq_len64);
  check_cat_out(q_cat_out, batch, total_seq_len, heads, head_dim, "q_cat_out");
  check_cat_out(k_cat_out, batch, total_seq_len, heads, head_dim, "k_cat_out");
  check_cat_out(v_cat_out, batch, total_seq_len, heads, head_dim, "v_cat_out");

  check_same_device(packed_v, qkv_v_bias, "packed_v", "qkv_v_bias");
  check_same_device(packed_v, norm_v_q_weight, "packed_v", "norm_v_q_weight");
  check_same_device(packed_v, norm_v_k_weight, "packed_v", "norm_v_k_weight");
  check_same_device(packed_v, freqs_re, "packed_v", "freqs_re");
  check_same_device(packed_v, freqs_im, "packed_v", "freqs_im");
  check_same_device(packed_v, packed_a, "packed_v", "packed_a");
  check_same_device(packed_v, packed_u, "packed_v", "packed_u");
  check_same_device(packed_v, q_cat_out, "packed_v", "q_cat_out");
  check_same_device(packed_v, k_cat_out, "packed_v", "k_cat_out");
  check_same_device(packed_v, v_cat_out, "packed_v", "v_cat_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_v.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_v.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_joint3_cat_bf16(
      packed_v.data_ptr(),
      qkv_v_bias.data_ptr(),
      norm_v_q_weight.data_ptr(),
      norm_v_k_weight.data_ptr(),
      reinterpret_cast<const float*>(freqs_re.data_ptr()),
      reinterpret_cast<const float*>(freqs_im.data_ptr()),
      packed_a.data_ptr(),
      norm_a_q_weight.data_ptr(),
      norm_a_k_weight.data_ptr(),
      packed_u.data_ptr(),
      norm_u_q_weight.data_ptr(),
      norm_u_k_weight.data_ptr(),
      q_cat_out.data_ptr(),
      k_cat_out.data_ptr(),
      v_cat_out.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(total_seq_len, "total_seq_len"),
      checked_int(video_seq_len, "video_seq_len"),
      checked_int(action_seq_len, "action_seq_len"),
      checked_int(und_seq_len, "und_seq_len"),
      heads,
      head_dim,
      checked_int(rope_seq_len64, "rope_seq_len"),
      static_cast<float>(eps_v),
      static_cast<float>(eps_a),
      static_cast<float>(eps_u),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("decode_q_norm_rope_stage_bf16("
          "Tensor q_pre, Tensor q_norm_weight, Tensor cos, Tensor sin, "
          "float eps, Tensor! q_out) -> ()");
  ops.def("decode_k_norm_rope_kvwrite_bf16("
          "Tensor k_pre, Tensor v_pre, Tensor k_norm_weight, Tensor cos, Tensor sin, "
          "float eps, Tensor! k_cache_dst, Tensor! v_cache_dst) -> ()");
  ops.def("decode_k_norm_rope_kvwrite_devpos_bf16("
          "Tensor k_pre, Tensor v_pre, Tensor k_norm_weight, Tensor cos, Tensor sin, "
          "Tensor cur_pos, float eps, Tensor! k_cache, Tensor! v_cache) -> ()");
  ops.def("qkv_split_norm_rope_bf16("
          "Tensor packed_qkv, Tensor norm_q_weight, Tensor norm_k_weight, "
          "Tensor freqs_re, Tensor freqs_im, int heads, int head_dim, "
          "int rope_seq_len, float eps, Tensor! q_out, Tensor! k_out) -> ()");
  ops.def("qkv_split_bias_norm_rope_v_bf16("
          "Tensor packed_qkv, Tensor qkv_bias, Tensor norm_q_weight, Tensor norm_k_weight, "
          "Tensor freqs_re, Tensor freqs_im, int heads, int head_dim, "
          "int rope_seq_len, float eps, Tensor! q_out, Tensor! k_out, Tensor! v_out) -> ()");
  ops.def("qkv_split_bias_norm_rope_v_cat_bf16("
          "Tensor packed_qkv, Tensor qkv_bias, Tensor norm_q_weight, Tensor norm_k_weight, "
          "Tensor freqs_re, Tensor freqs_im, int heads, int head_dim, int video_offset, "
          "int rope_seq_len, float eps, Tensor! q_cat_out, Tensor! k_cat_out, Tensor! v_cat_out) -> ()");
  ops.def("qkv_split_joint3_cat_bf16("
          "Tensor packed_v, Tensor qkv_v_bias, Tensor norm_v_q_weight, Tensor norm_v_k_weight, "
          "Tensor freqs_re, Tensor freqs_im, Tensor packed_a, Tensor norm_a_q_weight, "
          "Tensor norm_a_k_weight, Tensor packed_u, Tensor norm_u_q_weight, Tensor norm_u_k_weight, "
          "int heads, int head_dim, int rope_seq_len, float eps_v, float eps_a, float eps_u, "
          "Tensor! q_cat_out, Tensor! k_cat_out, Tensor! v_cat_out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("decode_q_norm_rope_stage_bf16",
           torch::kCUDA,
           &decode_q_norm_rope_stage_bf16);
  ops.impl("decode_k_norm_rope_kvwrite_bf16",
           torch::kCUDA,
           &decode_k_norm_rope_kvwrite_bf16);
  ops.impl("decode_k_norm_rope_kvwrite_devpos_bf16",
           torch::kCUDA,
           &decode_k_norm_rope_kvwrite_devpos_bf16);
  ops.impl("qkv_split_norm_rope_bf16",
           torch::kCUDA,
           &qkv_split_norm_rope_bf16);
  ops.impl("qkv_split_bias_norm_rope_v_bf16",
           torch::kCUDA,
           &qkv_split_bias_norm_rope_v_bf16);
  ops.impl("qkv_split_bias_norm_rope_v_cat_bf16",
           torch::kCUDA,
           &qkv_split_bias_norm_rope_v_cat_bf16);
  ops.impl("qkv_split_joint3_cat_bf16",
           torch::kCUDA,
           &qkv_split_joint3_cat_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
