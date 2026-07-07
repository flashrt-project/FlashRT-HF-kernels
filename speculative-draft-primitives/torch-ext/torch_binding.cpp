// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "speculative_draft_primitives.cuh"
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

void check_logits_outputs(torch::Tensor const& logits, torch::Tensor const& argmax_out) {
  check_bf16(logits, "logits");
  check_i64(argmax_out, "argmax_out");
  TORCH_CHECK(logits.dim() == 2, "logits must have shape (rows, vocab)");
  TORCH_CHECK(argmax_out.sizes() == torch::IntArrayRef({logits.size(0)}),
              "argmax_out must have shape (rows,)");
}

void check_accept_common(torch::Tensor const& logits, torch::Tensor const& drafts,
                         torch::Tensor const& argmax_out, torch::Tensor const& accept_n,
                         int64_t spec_k) {
  check_logits_outputs(logits, argmax_out);
  check_i64(drafts, "drafts");
  check_i32(accept_n, "accept_n");
  TORCH_CHECK(drafts.dim() == 1, "drafts must have shape (spec_k,) or larger");
  TORCH_CHECK(spec_k > 0 && spec_k <= logits.size(0), "spec_k must be in (0, rows]");
  TORCH_CHECK(drafts.numel() >= spec_k, "drafts must contain at least spec_k entries");
  TORCH_CHECK(accept_n.numel() >= 1, "accept_n must contain at least one int32");
}

}  // namespace

void argmax_bf16(torch::Tensor const& logits, torch::Tensor& argmax_out) {
  check_logits_outputs(logits, argmax_out);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  flashrt_hub::speculative::argmax_bf16(
      static_cast<const __nv_bfloat16*>(logits.data_ptr()),
      static_cast<int64_t*>(argmax_out.data_ptr()),
      checked_int(logits.size(0), "rows"),
      checked_int(logits.size(1), "vocab"),
      stream);
#else
  TORCH_CHECK(false, "speculative-draft-primitives was not built with CUDA support");
#endif
}

void accept_greedy_bf16(torch::Tensor const& logits, torch::Tensor const& drafts,
                        torch::Tensor& argmax_out, torch::Tensor& accept_n,
                        int64_t spec_k) {
  check_accept_common(logits, drafts, argmax_out, accept_n, spec_k);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  flashrt_hub::speculative::accept_greedy_bf16(
      static_cast<const __nv_bfloat16*>(logits.data_ptr()),
      static_cast<const int64_t*>(drafts.data_ptr()),
      static_cast<int64_t*>(argmax_out.data_ptr()),
      static_cast<int*>(accept_n.data_ptr()),
      checked_int(logits.size(0), "rows"),
      checked_int(logits.size(1), "vocab"),
      checked_int(spec_k, "spec_k"),
      stream);
#else
  TORCH_CHECK(false, "speculative-draft-primitives was not built with CUDA support");
#endif
}

void accept_partitioned_bf16(torch::Tensor const& logits, torch::Tensor const& drafts,
                             torch::Tensor& argmax_out, torch::Tensor& accept_n,
                             torch::Tensor& partial_vals, torch::Tensor& partial_idx,
                             int64_t spec_k, int64_t parts) {
  check_accept_common(logits, drafts, argmax_out, accept_n, spec_k);
  check_f32(partial_vals, "partial_vals");
  check_i32(partial_idx, "partial_idx");
  TORCH_CHECK(parts > 0 && parts <= 128, "parts must be in [1, 128]");
  TORCH_CHECK(partial_vals.sizes() == torch::IntArrayRef({logits.size(0), parts}),
              "partial_vals must have shape (rows, parts)");
  TORCH_CHECK(partial_idx.sizes() == torch::IntArrayRef({logits.size(0), parts}),
              "partial_idx must have shape (rows, parts)");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  flashrt_hub::speculative::accept_partitioned_bf16(
      static_cast<const __nv_bfloat16*>(logits.data_ptr()),
      static_cast<const int64_t*>(drafts.data_ptr()),
      static_cast<int64_t*>(argmax_out.data_ptr()),
      static_cast<int*>(accept_n.data_ptr()),
      static_cast<float*>(partial_vals.data_ptr()),
      static_cast<int*>(partial_idx.data_ptr()),
      checked_int(logits.size(0), "rows"),
      checked_int(logits.size(1), "vocab"),
      checked_int(spec_k, "spec_k"),
      checked_int(parts, "parts"),
      stream);
#else
  TORCH_CHECK(false, "speculative-draft-primitives was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("argmax_bf16(Tensor logits, Tensor! argmax_out) -> ()");
  ops.def("accept_greedy_bf16(Tensor logits, Tensor drafts, Tensor! argmax_out, Tensor! accept_n, int spec_k) -> ()");
  ops.def("accept_partitioned_bf16(Tensor logits, Tensor drafts, Tensor! argmax_out, Tensor! accept_n, Tensor! partial_vals, Tensor! partial_idx, int spec_k, int parts) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("argmax_bf16", torch::kCUDA, &argmax_bf16);
  ops.impl("accept_greedy_bf16", torch::kCUDA, &accept_greedy_bf16);
  ops.impl("accept_partitioned_bf16", torch::kCUDA, &accept_partitioned_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
