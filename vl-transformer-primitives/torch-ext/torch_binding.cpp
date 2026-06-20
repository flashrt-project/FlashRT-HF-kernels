// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "avg_pool_vision_tokens.cuh"
#include "qwen3_qkv_post_proc.cuh"
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

void check_int32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
              name, " must have dtype torch.int32");
}

int checked_positive_int(int64_t value, const char* name) {
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

void check_decode_rope(torch::Tensor const& x,
                       torch::Tensor const& weight,
                       torch::Tensor const& cos,
                       torch::Tensor const& sin,
                       torch::Tensor const& out,
                       const char* x_name,
                       const char* out_name) {
  check_bf16(x, x_name);
  check_bf16(weight, "norm_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(out, out_name);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 128,
              x_name, " must have shape (heads, 128)");
  TORCH_CHECK(x.size(0) > 0, "heads must be positive");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({128}),
              "norm_weight must have shape (128,)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({64}) &&
              sin.sizes() == torch::IntArrayRef({64}),
              "cos and sin must have shape (64,)");
  TORCH_CHECK(out.sizes() == x.sizes(),
              out_name, " must have the same shape as ", x_name);
  check_same_device(x, weight, x_name, "norm_weight");
  check_same_device(x, cos, x_name, "cos");
  check_same_device(x, sin, x_name, "sin");
  check_same_device(x, out, x_name, out_name);
}

}  // namespace

void qwen3_q_norm_rope_qstage_bf16(
    torch::Tensor const& q_pre,
    torch::Tensor const& q_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    double eps,
    torch::Tensor& q_out) {
  check_decode_rope(q_pre, q_norm_weight, cos, sin, q_out, "q_pre", "q_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q_pre.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_pre.get_device()).stream();
  const int rc = flash_rt::kernels::qwen3_q_norm_rope_qstage_bf16(
      q_pre.data_ptr(),
      q_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      q_out.data_ptr(),
      checked_positive_int(q_pre.size(0), "n_q_heads"),
      static_cast<float>(eps),
      stream);
  TORCH_CHECK(rc == 0, "qwen3_q_norm_rope_qstage_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "vl-transformer-primitives was not built with CUDA support");
#endif
}

void qwen3_k_norm_rope_kvwrite_bf16(
    torch::Tensor const& k_pre,
    torch::Tensor const& v_pre,
    torch::Tensor const& k_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    double eps,
    torch::Tensor& k_cache_dst,
    torch::Tensor& v_cache_dst) {
  check_decode_rope(k_pre, k_norm_weight, cos, sin, k_cache_dst,
                    "k_pre", "k_cache_dst");
  check_bf16(v_pre, "v_pre");
  check_bf16(v_cache_dst, "v_cache_dst");
  TORCH_CHECK(v_pre.sizes() == k_pre.sizes(),
              "v_pre must have the same shape as k_pre");
  TORCH_CHECK(v_cache_dst.sizes() == k_pre.sizes(),
              "v_cache_dst must have the same shape as k_pre");
  check_same_device(k_pre, v_pre, "k_pre", "v_pre");
  check_same_device(k_pre, v_cache_dst, "k_pre", "v_cache_dst");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k_pre.device());
  auto stream = at::cuda::getCurrentCUDAStream(k_pre.get_device()).stream();
  const int rc = flash_rt::kernels::qwen3_k_norm_rope_kvwrite_bf16(
      k_pre.data_ptr(),
      v_pre.data_ptr(),
      k_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      k_cache_dst.data_ptr(),
      v_cache_dst.data_ptr(),
      checked_positive_int(k_pre.size(0), "n_kv_heads"),
      static_cast<float>(eps),
      stream);
  TORCH_CHECK(rc == 0, "qwen3_k_norm_rope_kvwrite_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "vl-transformer-primitives was not built with CUDA support");
#endif
}

void qwen3_k_norm_rope_kvwrite_devpos_bf16(
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
              "v_pre must have the same shape as k_pre");
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
  const int rc = flash_rt::kernels::qwen3_k_norm_rope_kvwrite_devpos_bf16(
      k_pre.data_ptr(),
      v_pre.data_ptr(),
      k_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      k_cache.data_ptr(),
      v_cache.data_ptr(),
      cur_pos.data_ptr(),
      checked_positive_int(n_kv_heads * 128, "row_elems"),
      checked_positive_int(n_kv_heads, "n_kv_heads"),
      static_cast<float>(eps),
      stream);
  TORCH_CHECK(rc == 0, "qwen3_k_norm_rope_kvwrite_devpos_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "vl-transformer-primitives was not built with CUDA support");
#endif
}

void avg_pool_vision_tokens_bf16(
    torch::Tensor const& x,
    int64_t nv64,
    int64_t h64,
    int64_t w64,
    int64_t pool_factor64,
    torch::Tensor& out) {
  check_bf16(x, "x");
  check_bf16(out, "out");
  const int nv = checked_positive_int(nv64, "nv");
  const int h = checked_positive_int(h64, "h");
  const int w = checked_positive_int(w64, "w");
  const int pool_factor = checked_positive_int(pool_factor64, "pool_factor");
  TORCH_CHECK(pool_factor <= 8, "pool_factor must be <= 8");
  TORCH_CHECK(h % pool_factor == 0 && w % pool_factor == 0,
              "h and w must be divisible by pool_factor");
  TORCH_CHECK(x.dim() == 2, "x must have shape (nv * h * w, dim)");
  const int64_t dim64 = x.size(1);
  const int dim = checked_positive_int(dim64, "dim");
  TORCH_CHECK(x.size(0) == static_cast<int64_t>(nv) * h * w,
              "x.shape[0] must equal nv * h * w");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({
                  static_cast<int64_t>(nv) * (h / pool_factor) * (w / pool_factor),
                  dim64}),
              "out must have shape (nv * h/pool_factor * w/pool_factor, dim)");
  check_same_device(x, out, "x", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::vl_transformer_primitives::avg_pool_vision_tokens_bf16(
      x.data_ptr(),
      out.data_ptr(),
      nv,
      h,
      w,
      dim,
      pool_factor,
      stream);
#else
  TORCH_CHECK(false, "vl-transformer-primitives was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("qwen3_q_norm_rope_qstage_bf16("
          "Tensor q_pre, Tensor q_norm_weight, Tensor cos, Tensor sin, "
          "float eps, Tensor! q_out) -> ()");
  ops.def("qwen3_k_norm_rope_kvwrite_bf16("
          "Tensor k_pre, Tensor v_pre, Tensor k_norm_weight, Tensor cos, Tensor sin, "
          "float eps, Tensor! k_cache_dst, Tensor! v_cache_dst) -> ()");
  ops.def("qwen3_k_norm_rope_kvwrite_devpos_bf16("
          "Tensor k_pre, Tensor v_pre, Tensor k_norm_weight, Tensor cos, Tensor sin, "
          "Tensor cur_pos, float eps, Tensor! k_cache, Tensor! v_cache) -> ()");
  ops.def("avg_pool_vision_tokens_bf16("
          "Tensor x, int nv, int h, int w, int pool_factor, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("qwen3_q_norm_rope_qstage_bf16",
           torch::kCUDA,
           &qwen3_q_norm_rope_qstage_bf16);
  ops.impl("qwen3_k_norm_rope_kvwrite_bf16",
           torch::kCUDA,
           &qwen3_k_norm_rope_kvwrite_bf16);
  ops.impl("qwen3_k_norm_rope_kvwrite_devpos_bf16",
           torch::kCUDA,
           &qwen3_k_norm_rope_kvwrite_devpos_bf16);
  ops.impl("avg_pool_vision_tokens_bf16",
           torch::kCUDA,
           &avg_pool_vision_tokens_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
