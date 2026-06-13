// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "msa_topk_from_scores.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
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

}  // namespace

void msa_topk_from_scores(torch::Tensor const& score,
                          torch::Tensor const& seq_lens,
                          int64_t block_size,
                          int64_t topk,
                          torch::Tensor& topk_idx) {
  check_cuda_contiguous(score, "score");
  check_int32(seq_lens, "seq_lens");
  check_int32(topk_idx, "topk_idx");
  TORCH_CHECK(score.scalar_type() == torch::kFloat32,
              "score must have dtype torch.float32");
  TORCH_CHECK(score.dim() == 3, "score must have shape (heads, batch, max_blocks)");
  const int64_t heads = score.size(0);
  const int64_t batch = score.size(1);
  const int64_t max_blocks = score.size(2);
  TORCH_CHECK(heads > 0 && batch > 0 && max_blocks > 0,
              "score dimensions must be positive");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == batch,
              "seq_lens must have shape (batch,)");
  TORCH_CHECK(topk > 0 && topk <= 64,
              "topk must be in [1, 64]");
  TORCH_CHECK(topk_idx.sizes() == torch::IntArrayRef({heads, batch, topk}),
              "topk_idx must have shape (heads, batch, topk)");
  TORCH_CHECK(score.get_device() == seq_lens.get_device() &&
              score.get_device() == topk_idx.get_device(),
              "score, seq_lens, and topk_idx must be on the same CUDA device");

#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard device_guard(score.device());
  auto stream = at::cuda::getCurrentCUDAStream(score.get_device()).stream();
  flashrt_minimax_msa::msa_topk_from_scores_cuda(
      score.data_ptr<float>(),
      seq_lens.data_ptr<int>(),
      topk_idx.data_ptr<int>(),
      checked_positive_int(heads, "heads"),
      checked_positive_int(batch, "batch"),
      checked_positive_int(max_blocks, "max_blocks"),
      checked_positive_int(block_size, "block_size"),
      checked_positive_int(topk, "topk"),
      stream);
#else
  TORCH_CHECK(false, "minimaxai-msa-sm121 was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("msa_topk_from_scores(Tensor score, Tensor seq_lens, int block_size, "
          "int topk, Tensor! topk_idx) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("msa_topk_from_scores", torch::kCUDA, &msa_topk_from_scores);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
