// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "torch_binding.h"
#include "transformer_layout_primitives.cuh"

namespace {

void check_cuda_contiguous(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be torch.bfloat16");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(), name, " must fit in int");
  return static_cast<int>(value);
}

void same_device(torch::Tensor const& a, torch::Tensor const& b, const char* an, const char* bn) {
  TORCH_CHECK(a.get_device() == b.get_device(), an, " and ", bn, " must be on the same device");
}

}  // namespace

void fill_neginf_bf16(torch::Tensor& dst) {
  check_bf16(dst, "dst");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(dst.device());
  auto stream = at::cuda::getCurrentCUDAStream(dst.get_device()).stream();
  flashrt_hub::transformer_layout::fill_neginf_bf16(
      static_cast<__nv_bfloat16*>(dst.data_ptr()), checked_int(dst.numel(), "dst.numel"), stream);
#endif
}

void add_bias_bf16_(torch::Tensor& data, torch::Tensor const& bias) {
  check_bf16(data, "data");
  check_bf16(bias, "bias");
  TORCH_CHECK(data.dim() == 2 && bias.sizes() == torch::IntArrayRef({data.size(1)}),
              "data must be (rows, cols), bias (cols,)");
  same_device(data, bias, "data", "bias");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(data.device());
  auto stream = at::cuda::getCurrentCUDAStream(data.get_device()).stream();
  flashrt_hub::transformer_layout::add_bias_bf16(
      static_cast<__nv_bfloat16*>(data.data_ptr()),
      static_cast<const __nv_bfloat16*>(bias.data_ptr()),
      checked_int(data.size(0), "rows"), checked_int(data.size(1), "cols"), stream);
#endif
}

void repeat_interleave_heads_bf16(torch::Tensor const& src, int64_t repeat, torch::Tensor& dst) {
  check_bf16(src, "src");
  check_bf16(dst, "dst");
  TORCH_CHECK(src.dim() == 3 && repeat > 0, "src must be (seq, heads, head_dim), repeat > 0");
  TORCH_CHECK(dst.sizes() == torch::IntArrayRef({src.size(0), src.size(1) * repeat, src.size(2)}),
              "dst shape mismatch");
  same_device(src, dst, "src", "dst");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream(src.get_device()).stream();
  flashrt_hub::transformer_layout::repeat_interleave_heads_bf16(
      static_cast<const __nv_bfloat16*>(src.data_ptr()),
      static_cast<__nv_bfloat16*>(dst.data_ptr()),
      checked_int(src.size(0), "seq"), checked_int(src.size(1), "heads"),
      checked_int(src.size(2), "head_dim"), checked_int(repeat, "repeat"), stream);
#endif
}

void text_gather_bf16(torch::Tensor const& src, int64_t batch, int64_t seq, torch::Tensor& dst) {
  check_bf16(src, "src");
  check_bf16(dst, "dst");
  TORCH_CHECK(src.dim() == 2 && src.size(0) == batch * seq, "src must be (batch * seq, dim)");
  TORCH_CHECK(dst.sizes() == torch::IntArrayRef({2 * batch, src.size(1)}), "dst must be (2 * batch, dim)");
  same_device(src, dst, "src", "dst");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream(src.get_device()).stream();
  flashrt_hub::transformer_layout::text_gather_bf16(
      static_cast<const __nv_bfloat16*>(src.data_ptr()),
      static_cast<__nv_bfloat16*>(dst.data_ptr()),
      checked_int(batch, "batch"), checked_int(seq, "seq"),
      checked_int(src.size(1), "dim"), stream);
#endif
}

void text_scatter_bf16(torch::Tensor& dst, torch::Tensor const& src, int64_t batch, int64_t seq) {
  check_bf16(dst, "dst");
  check_bf16(src, "src");
  TORCH_CHECK(dst.dim() == 2 && dst.size(0) == batch * seq, "dst must be (batch * seq, dim)");
  TORCH_CHECK(src.sizes() == torch::IntArrayRef({2 * batch, dst.size(1)}), "src must be (2 * batch, dim)");
  same_device(dst, src, "dst", "src");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(dst.device());
  auto stream = at::cuda::getCurrentCUDAStream(dst.get_device()).stream();
  flashrt_hub::transformer_layout::text_scatter_bf16(
      static_cast<__nv_bfloat16*>(dst.data_ptr()),
      static_cast<const __nv_bfloat16*>(src.data_ptr()),
      checked_int(batch, "batch"), checked_int(seq, "seq"),
      checked_int(dst.size(1), "dim"), stream);
#endif
}

void rope_rotate_half_bf16_(torch::Tensor& x, torch::Tensor const& cos, torch::Tensor const& sin) {
  check_bf16(x, "x");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  TORCH_CHECK(x.dim() == 3, "x must be (seq, heads, head_dim)");
  TORCH_CHECK(x.size(2) % 2 == 0, "head_dim must be even");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({x.size(0), x.size(2)}) && sin.sizes() == cos.sizes(),
              "cos/sin must be (seq, head_dim)");
  same_device(x, cos, "x", "cos");
  same_device(x, sin, "x", "sin");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flashrt_hub::transformer_layout::rope_rotate_half_bf16(
      static_cast<__nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(cos.data_ptr()),
      static_cast<const __nv_bfloat16*>(sin.data_ptr()),
      checked_int(x.size(0), "seq"), checked_int(x.size(1), "heads"),
      checked_int(x.size(2), "head_dim"), stream);
#endif
}

void qk_rmsnorm_rope_bf16_(torch::Tensor& qk, torch::Tensor const& weight,
                           torch::Tensor const& cos, torch::Tensor const& sin,
                           double eps) {
  check_bf16(qk, "qk");
  check_bf16(weight, "weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  TORCH_CHECK(qk.dim() == 3 && qk.size(2) % 2 == 0, "qk must be (rows, heads, head_dim)");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({qk.size(2)}), "weight shape mismatch");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({qk.size(0), qk.size(2)}) && sin.sizes() == cos.sizes(),
              "cos/sin must be (rows, head_dim)");
  same_device(qk, weight, "qk", "weight");
  same_device(qk, cos, "qk", "cos");
  same_device(qk, sin, "qk", "sin");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(qk.device());
  auto stream = at::cuda::getCurrentCUDAStream(qk.get_device()).stream();
  flashrt_hub::transformer_layout::qk_rmsnorm_rope_bf16(
      static_cast<__nv_bfloat16*>(qk.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(cos.data_ptr()),
      static_cast<const __nv_bfloat16*>(sin.data_ptr()),
      checked_int(qk.size(0), "rows"), checked_int(qk.size(1), "heads"),
      checked_int(qk.size(2), "head_dim"), static_cast<float>(eps), stream);
#endif
}

void qk_pair_rmsnorm_rope_bf16(
    torch::Tensor const& q, torch::Tensor const& k,
    torch::Tensor const& q_weight, torch::Tensor const& k_weight,
    torch::Tensor const& cos, torch::Tensor const& sin,
    double eps, torch::Tensor& q_out, torch::Tensor& k_out) {
  check_bf16(q, "q");
  check_bf16(k, "k");
  check_bf16(q_weight, "q_weight");
  check_bf16(k_weight, "k_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3,
              "q and k must be (rows, heads, head_dim)");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2),
              "q and k must have matching rows and head_dim");
  TORCH_CHECK(q.size(2) >= 8 && q.size(2) <= 256 && q.size(2) % 2 == 0,
              "head_dim must be even and in [8, 256]");
  TORCH_CHECK(q_weight.sizes() == torch::IntArrayRef({q.size(2)}) &&
                  k_weight.sizes() == q_weight.sizes(),
              "q_weight and k_weight must have shape (head_dim,)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({q.size(0), q.size(2)}) &&
                  sin.sizes() == cos.sizes(),
              "cos and sin must have shape (rows, head_dim)");
  TORCH_CHECK(q_out.sizes() == q.sizes() && k_out.sizes() == k.sizes(),
              "q_out and k_out must match q and k");
  same_device(q, k, "q", "k");
  same_device(q, q_weight, "q", "q_weight");
  same_device(q, k_weight, "q", "k_weight");
  same_device(q, cos, "q", "cos");
  same_device(q, sin, "q", "sin");
  same_device(q, q_out, "q", "q_out");
  same_device(q, k_out, "q", "k_out");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flashrt_hub::transformer_layout::qk_pair_rmsnorm_rope_bf16(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const __nv_bfloat16*>(k.data_ptr()),
      static_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(k_weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(cos.data_ptr()),
      static_cast<const __nv_bfloat16*>(sin.data_ptr()),
      static_cast<__nv_bfloat16*>(q_out.data_ptr()),
      static_cast<__nv_bfloat16*>(k_out.data_ptr()),
      checked_int(q.size(0), "rows"), checked_int(q.size(1), "q_heads"),
      checked_int(k.size(1), "k_heads"), checked_int(q.size(2), "head_dim"),
      static_cast<float>(eps), stream);
#endif
}

void gather_rows_bf16(
    torch::Tensor const& src,
    torch::Tensor const& row_indices,
    torch::Tensor& dst) {
  check_bf16(src, "src");
  check_bf16(dst, "dst");
  check_cuda_contiguous(row_indices, "row_indices");
  TORCH_CHECK(row_indices.scalar_type() == torch::kInt64,
              "row_indices must be torch.int64");
  TORCH_CHECK(src.dim() == 2 && dst.dim() == 2 && row_indices.dim() == 1,
              "src/dst must be matrices and row_indices a vector");
  TORCH_CHECK(dst.size(0) == row_indices.numel() && dst.size(1) == src.size(1),
              "dst must have shape (row_indices.numel(), src.size(1))");
  same_device(src, row_indices, "src", "row_indices");
  same_device(src, dst, "src", "dst");
  if (row_indices.numel() == 0) return;
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream(src.get_device()).stream();
  flashrt_hub::transformer_layout::gather_rows_bf16(
      static_cast<const __nv_bfloat16*>(src.data_ptr()),
      row_indices.data_ptr<int64_t>(),
      static_cast<__nv_bfloat16*>(dst.data_ptr()),
      checked_int(dst.size(0), "rows"), checked_int(src.size(1), "hidden"), stream);
#endif
}

void scatter_rows_bf16(
    torch::Tensor const& src,
    torch::Tensor const& row_indices,
    torch::Tensor& dst) {
  check_bf16(src, "src");
  check_bf16(dst, "dst");
  check_cuda_contiguous(row_indices, "row_indices");
  TORCH_CHECK(row_indices.scalar_type() == torch::kInt64,
              "row_indices must be torch.int64");
  TORCH_CHECK(src.dim() == 2 && dst.dim() == 2 && row_indices.dim() == 1,
              "src/dst must be matrices and row_indices a vector");
  TORCH_CHECK(src.size(0) == row_indices.numel() && src.size(1) == dst.size(1),
              "src must have shape (row_indices.numel(), dst.size(1))");
  same_device(src, row_indices, "src", "row_indices");
  same_device(src, dst, "src", "dst");
  if (row_indices.numel() == 0) return;
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream(src.get_device()).stream();
  flashrt_hub::transformer_layout::scatter_rows_bf16(
      static_cast<const __nv_bfloat16*>(src.data_ptr()),
      row_indices.data_ptr<int64_t>(),
      static_cast<__nv_bfloat16*>(dst.data_ptr()),
      checked_int(src.size(0), "rows"), checked_int(src.size(1), "hidden"), stream);
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fill_neginf_bf16(Tensor! dst) -> ()");
  ops.def("add_bias_bf16_(Tensor! data, Tensor bias) -> ()");
  ops.def("repeat_interleave_heads_bf16(Tensor src, int repeat, Tensor! dst) -> ()");
  ops.def("text_gather_bf16(Tensor src, int batch, int seq, Tensor! dst) -> ()");
  ops.def("text_scatter_bf16(Tensor! dst, Tensor src, int batch, int seq) -> ()");
  ops.def("rope_rotate_half_bf16_(Tensor! x, Tensor cos, Tensor sin) -> ()");
  ops.def("qk_rmsnorm_rope_bf16_(Tensor! qk, Tensor weight, Tensor cos, Tensor sin, float eps=1e-6) -> ()");
  ops.def("qk_pair_rmsnorm_rope_bf16(Tensor q, Tensor k, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, float eps, Tensor! q_out, Tensor! k_out) -> ()");
  ops.def("gather_rows_bf16(Tensor src, Tensor row_indices, Tensor! dst) -> ()");
  ops.def("scatter_rows_bf16(Tensor src, Tensor row_indices, Tensor! dst) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fill_neginf_bf16", torch::kCUDA, &fill_neginf_bf16);
  ops.impl("add_bias_bf16_", torch::kCUDA, &add_bias_bf16_);
  ops.impl("repeat_interleave_heads_bf16", torch::kCUDA, &repeat_interleave_heads_bf16);
  ops.impl("text_gather_bf16", torch::kCUDA, &text_gather_bf16);
  ops.impl("text_scatter_bf16", torch::kCUDA, &text_scatter_bf16);
  ops.impl("rope_rotate_half_bf16_", torch::kCUDA, &rope_rotate_half_bf16_);
  ops.impl("qk_rmsnorm_rope_bf16_", torch::kCUDA, &qk_rmsnorm_rope_bf16_);
  ops.impl("qk_pair_rmsnorm_rope_bf16", torch::kCUDA, &qk_pair_rmsnorm_rope_bf16);
  ops.impl("gather_rows_bf16", torch::kCUDA, &gather_rows_bf16);
  ops.impl("scatter_rows_bf16", torch::kCUDA, &scatter_rows_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
