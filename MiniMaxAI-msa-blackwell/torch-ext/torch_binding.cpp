// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "msa_decode_attn.cuh"
#include "msa_decode_attn_mma.cuh"
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
  TORCH_CHECK(false, "minimaxai-msa-blackwell was not built with CUDA support");
#endif
}

void msa_decode_sparse_attn(torch::Tensor const& q,
                            torch::Tensor const& kv_cache,
                            torch::Tensor const& seq_lens,
                            torch::Tensor const& slot_ids,
                            torch::Tensor const& topk_idx,
                            int64_t block_size,
                            double sm_scale,
                            torch::Tensor& out) {
  check_cuda_contiguous(q, "q");
  check_cuda_contiguous(kv_cache, "kv_cache");
  check_int32(seq_lens, "seq_lens");
  check_int32(topk_idx, "topk_idx");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
              kv_cache.scalar_type() == torch::kBFloat16 &&
              out.scalar_type() == torch::kBFloat16,
              "q, kv_cache, out must be bf16");
  TORCH_CHECK(slot_ids.scalar_type() == torch::kInt64, "slot_ids int64");
  TORCH_CHECK(q.dim() == 3, "q must be (B, Hq, D)");
  TORCH_CHECK(kv_cache.dim() == 5,
              "kv_cache must be (max_slots, 2, max_len, Hkv, D)");
  const int64_t B = q.size(0), Hq = q.size(1), D = q.size(2);
  const int64_t max_slots = kv_cache.size(0), max_len = kv_cache.size(2);
  const int64_t Hkv = kv_cache.size(3);
  TORCH_CHECK(kv_cache.size(1) == 2 && kv_cache.size(4) == D,
              "kv_cache dims mismatch");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(D % 32 == 0 && D <= 256, "D must be a multiple of 32, <= 256");
  TORCH_CHECK(topk_idx.dim() == 3 && topk_idx.size(0) == Hkv &&
              topk_idx.size(1) == B, "topk_idx must be (Hkv, B, topk)");
  const int64_t topk = topk_idx.size(2);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flashrt_minimax_msa::msa_decode_sparse_attn_cuda(
      q.data_ptr(), kv_cache.data_ptr(),
      seq_lens.data_ptr<int>(), slot_ids.data_ptr<int64_t>(),
      topk_idx.data_ptr<int>(), out.data_ptr(),
      checked_positive_int(B, "B"), checked_positive_int(Hq, "Hq"),
      checked_positive_int(Hkv, "Hkv"), checked_positive_int(D, "D"),
      checked_positive_int(max_slots, "max_slots"),
      checked_positive_int(max_len, "max_len"),
      checked_positive_int(block_size, "block_size"),
      checked_positive_int(topk, "topk"),
      static_cast<float>(sm_scale), stream);
#else
  TORCH_CHECK(false, "minimaxai-msa-blackwell was not built with CUDA support");
#endif
}

void msa_decode_sparse_attn_mma(torch::Tensor const& q,
                                torch::Tensor const& kv_cache,
                                torch::Tensor const& seq_lens,
                                torch::Tensor const& slot_ids,
                                torch::Tensor const& topk_idx,
                                int64_t block_size,
                                double sm_scale,
                                torch::Tensor& out) {
  check_cuda_contiguous(q, "q");
  check_cuda_contiguous(kv_cache, "kv_cache");
  check_int32(seq_lens, "seq_lens");
  check_int32(topk_idx, "topk_idx");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
              kv_cache.scalar_type() == torch::kBFloat16 &&
              out.scalar_type() == torch::kBFloat16,
              "q, kv_cache, out must be bf16");
  TORCH_CHECK(slot_ids.scalar_type() == torch::kInt64, "slot_ids int64");
  TORCH_CHECK(q.dim() == 3, "q must be (B, Hq, D)");
  TORCH_CHECK(kv_cache.dim() == 5,
              "kv_cache must be (max_slots, 2, max_len, Hkv, D)");
  const int64_t B = q.size(0), Hq = q.size(1), D = q.size(2);
  const int64_t max_slots = kv_cache.size(0), max_len = kv_cache.size(2);
  const int64_t Hkv = kv_cache.size(3);
  TORCH_CHECK(kv_cache.size(1) == 2 && kv_cache.size(4) == D,
              "kv_cache dims mismatch");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  // The tensor-core variant is specialized for the M3 MSA shape.
  TORCH_CHECK(D == 128, "mma variant requires head_dim == 128");
  TORCH_CHECK(Hq / Hkv == 16, "mma variant requires GQA group (Hq/Hkv) == 16");
  TORCH_CHECK(block_size % 64 == 0, "mma variant requires block_size % 64 == 0");
  TORCH_CHECK(topk_idx.dim() == 3 && topk_idx.size(0) == Hkv &&
              topk_idx.size(1) == B, "topk_idx must be (Hkv, B, topk)");
  const int64_t topk = topk_idx.size(2);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flashrt_minimax_msa::msa_decode_sparse_attn_mma_cuda(
      q.data_ptr(), kv_cache.data_ptr(),
      seq_lens.data_ptr<int>(), slot_ids.data_ptr<int64_t>(),
      topk_idx.data_ptr<int>(), out.data_ptr(),
      checked_positive_int(B, "B"), checked_positive_int(Hq, "Hq"),
      checked_positive_int(Hkv, "Hkv"), checked_positive_int(D, "D"),
      checked_positive_int(max_slots, "max_slots"),
      checked_positive_int(max_len, "max_len"),
      checked_positive_int(block_size, "block_size"),
      checked_positive_int(topk, "topk"),
      static_cast<float>(sm_scale), stream);
#else
  TORCH_CHECK(false, "minimaxai-msa-blackwell was not built with CUDA support");
#endif
}

void msa_decode_sparse_attn_mma_paged(torch::Tensor const& q,
                                      torch::Tensor const& k_cache,
                                      torch::Tensor const& v_cache,
                                      torch::Tensor const& req_to_token,
                                      torch::Tensor const& seq_lens,
                                      torch::Tensor const& slot_ids,
                                      torch::Tensor const& topk_idx,
                                      int64_t block_size,
                                      double sm_scale,
                                      torch::Tensor& out) {
  check_cuda_contiguous(q, "q");
  check_cuda_contiguous(k_cache, "k_cache");
  check_cuda_contiguous(v_cache, "v_cache");
  check_int32(req_to_token, "req_to_token");
  check_int32(seq_lens, "seq_lens");
  check_int32(topk_idx, "topk_idx");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
              k_cache.scalar_type() == torch::kBFloat16 &&
              v_cache.scalar_type() == torch::kBFloat16 &&
              out.scalar_type() == torch::kBFloat16,
              "q, k_cache, v_cache, out must be bf16");
  TORCH_CHECK(slot_ids.scalar_type() == torch::kInt64, "slot_ids int64");
  TORCH_CHECK(q.dim() == 3, "q must be (B, Hq, D)");
  TORCH_CHECK(k_cache.dim() == 3 && v_cache.dim() == 3,
              "k_cache/v_cache must be (max_slots, Hkv, D)");
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be (max_reqs, max_kv_len)");
  const int64_t B = q.size(0), Hq = q.size(1), D = q.size(2);
  const int64_t max_slots = k_cache.size(0), Hkv = k_cache.size(1);
  const int64_t max_kv_len = req_to_token.size(1);
  TORCH_CHECK(k_cache.size(2) == D && v_cache.sizes() == k_cache.sizes(),
              "k/v cache dims mismatch");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(D == 128, "mma variant requires head_dim == 128");
  TORCH_CHECK(Hq / Hkv == 16, "mma variant requires GQA group (Hq/Hkv) == 16");
  TORCH_CHECK(block_size % 64 == 0, "mma variant requires block_size % 64 == 0");
  TORCH_CHECK(topk_idx.dim() == 3 && topk_idx.size(0) == Hkv &&
              topk_idx.size(1) == B, "topk_idx must be (Hkv, B, topk)");
  const int64_t topk = topk_idx.size(2);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flashrt_minimax_msa::msa_decode_sparse_attn_mma_paged_cuda(
      q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
      req_to_token.data_ptr<int>(), seq_lens.data_ptr<int>(),
      slot_ids.data_ptr<int64_t>(), topk_idx.data_ptr<int>(), out.data_ptr(),
      checked_positive_int(B, "B"), checked_positive_int(Hq, "Hq"),
      checked_positive_int(Hkv, "Hkv"), checked_positive_int(D, "D"),
      checked_positive_int(max_slots, "max_slots"),
      checked_positive_int(max_kv_len, "max_kv_len"),
      checked_positive_int(block_size, "block_size"),
      checked_positive_int(topk, "topk"),
      static_cast<float>(sm_scale), stream);
#else
  TORCH_CHECK(false, "minimaxai-msa-blackwell was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("msa_topk_from_scores(Tensor score, Tensor seq_lens, int block_size, "
          "int topk, Tensor! topk_idx) -> ()");
  ops.def("msa_decode_sparse_attn(Tensor q, Tensor kv_cache, Tensor seq_lens, "
          "Tensor slot_ids, Tensor topk_idx, int block_size, float sm_scale, "
          "Tensor! out) -> ()");
  ops.def("msa_decode_sparse_attn_mma(Tensor q, Tensor kv_cache, Tensor "
          "seq_lens, Tensor slot_ids, Tensor topk_idx, int block_size, "
          "float sm_scale, Tensor! out) -> ()");
  ops.def("msa_decode_sparse_attn_mma_paged(Tensor q, Tensor k_cache, Tensor "
          "v_cache, Tensor req_to_token, Tensor seq_lens, Tensor slot_ids, "
          "Tensor topk_idx, int block_size, float sm_scale, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("msa_topk_from_scores", torch::kCUDA, &msa_topk_from_scores);
  ops.impl("msa_decode_sparse_attn", torch::kCUDA, &msa_decode_sparse_attn);
  ops.impl("msa_decode_sparse_attn_mma", torch::kCUDA,
           &msa_decode_sparse_attn_mma);
  ops.impl("msa_decode_sparse_attn_mma_paged", torch::kCUDA,
           &msa_decode_sparse_attn_mma_paged);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
