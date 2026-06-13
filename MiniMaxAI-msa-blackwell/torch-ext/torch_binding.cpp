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
#include "msa_indexer_block_scores.cuh"
#include "msa_nvfp4_dequant.cuh"
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

void msa_indexer_block_scores(torch::Tensor const& q,
                              torch::Tensor const& k_pages,
                              torch::Tensor const& batch_of_q,
                              torch::Tensor const& cu_q,
                              torch::Tensor const& cu_k,
                              torch::Tensor const& cu_pages,
                              torch::Tensor const& kv_indices,
                              int64_t causal,
                              torch::Tensor& scores) {
  check_cuda_contiguous(q, "q");
  check_cuda_contiguous(k_pages, "k_pages");
  check_int32(batch_of_q, "batch_of_q");
  check_int32(cu_q, "cu_q");
  check_int32(cu_k, "cu_k");
  check_int32(cu_pages, "cu_pages");
  check_int32(kv_indices, "kv_indices");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
              k_pages.scalar_type() == torch::kBFloat16,
              "q and k_pages must be bf16");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && scores.is_cuda() &&
              scores.is_contiguous(), "scores must be contiguous f32 CUDA");
  TORCH_CHECK(q.dim() == 3, "q must be (total_q, Hq, D)");
  TORCH_CHECK(k_pages.dim() == 4, "k_pages must be (num_pages, Hkv, page, D)");
  const int64_t total_q = q.size(0), Hq = q.size(1), D = q.size(2);
  const int64_t num_pages = k_pages.size(0), Hkv = k_pages.size(1);
  const int64_t page_size = k_pages.size(2);
  TORCH_CHECK(k_pages.size(3) == D, "q/k head_dim mismatch");
  TORCH_CHECK(D % 32 == 0 && D <= 256, "D must be a multiple of 32, <= 256");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(scores.dim() == 3 && scores.size(0) == Hq &&
              scores.size(2) == total_q, "scores must be (Hq, max_blocks, total_q)");
  const int64_t max_blocks = scores.size(1);
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flashrt_minimax_msa::msa_indexer_block_scores_cuda(
      q.data_ptr(), k_pages.data_ptr(), batch_of_q.data_ptr<int>(),
      cu_q.data_ptr<int>(), cu_k.data_ptr<int>(), cu_pages.data_ptr<int>(),
      kv_indices.data_ptr<int>(), scores.data_ptr<float>(),
      checked_positive_int(total_q, "total_q"), checked_positive_int(Hq, "Hq"),
      checked_positive_int(Hkv, "Hkv"), checked_positive_int(D, "D"),
      checked_positive_int(num_pages, "num_pages"),
      checked_positive_int(max_blocks, "max_blocks"),
      checked_positive_int(page_size, "page_size"), causal != 0, stream);
#else
  TORCH_CHECK(false, "minimaxai-msa-blackwell was not built with CUDA support");
#endif
}

void msa_nvfp4_dequant_swizzled_to_bf16(torch::Tensor const& packed,
                                        torch::Tensor const& scale_128x4,
                                        double global_scale,
                                        torch::Tensor& out) {
  check_cuda_contiguous(packed, "packed");
  check_cuda_contiguous(scale_128x4, "scale_128x4");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8,
              "packed must have dtype torch.uint8");
  TORCH_CHECK(scale_128x4.scalar_type() == torch::kUInt8,
              "scale_128x4 must have dtype torch.uint8");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
              "out must have dtype torch.bfloat16");
  TORCH_CHECK(packed.dim() >= 2, "packed must have at least 2 dimensions");
  TORCH_CHECK(out.dim() == packed.dim(), "out rank must match packed rank");
  const int64_t packed_last = packed.size(packed.dim() - 1);
  const int64_t cols = out.size(out.dim() - 1);
  TORCH_CHECK(cols == packed_last * 2,
              "out last dim must be packed last dim * 2");
  TORCH_CHECK(cols % 16 == 0, "logical columns must be a multiple of 16");
  int64_t rows = 1;
  for (int64_t dim = 0; dim < packed.dim() - 1; ++dim) {
    TORCH_CHECK(out.size(dim) == packed.size(dim),
                "out shape must match packed shape except last dim");
    rows *= packed.size(dim);
  }
  TORCH_CHECK(rows > 0 && rows <= std::numeric_limits<int>::max(),
              "flattened rows must fit int");
  TORCH_CHECK(cols > 0 && cols <= std::numeric_limits<int>::max(),
              "logical columns must fit int");
  const int64_t scale_cols = cols / 16;
  const int64_t padded_rows = ((rows + 127) / 128) * 128;
  const int64_t padded_scale_cols = ((scale_cols + 3) / 4) * 4;
  TORCH_CHECK(scale_128x4.numel() >= padded_rows * padded_scale_cols,
              "scale_128x4 is too small for the requested logical shape");
  TORCH_CHECK(packed.get_device() == scale_128x4.get_device() &&
              packed.get_device() == out.get_device(),
              "packed, scale_128x4, and out must be on the same CUDA device");

#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard device_guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed.get_device()).stream();
  flashrt_minimax_msa::nvfp4_dequant_swizzled_to_bf16_cuda(
      packed.data_ptr<uint8_t>(),
      scale_128x4.data_ptr<uint8_t>(),
      out.data_ptr(),
      checked_positive_int(rows, "rows"),
      checked_positive_int(cols, "cols"),
      static_cast<float>(global_scale),
      stream);
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
  ops.def("msa_indexer_block_scores(Tensor q, Tensor k_pages, Tensor "
          "batch_of_q, Tensor cu_q, Tensor cu_k, Tensor cu_pages, Tensor "
          "kv_indices, int causal, Tensor! scores) -> ()");
  ops.def("msa_nvfp4_dequant_swizzled_to_bf16(Tensor packed, Tensor "
          "scale_128x4, float global_scale, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("msa_topk_from_scores", torch::kCUDA, &msa_topk_from_scores);
  ops.impl("msa_decode_sparse_attn", torch::kCUDA, &msa_decode_sparse_attn);
  ops.impl("msa_decode_sparse_attn_mma", torch::kCUDA,
           &msa_decode_sparse_attn_mma);
  ops.impl("msa_decode_sparse_attn_mma_paged", torch::kCUDA,
           &msa_decode_sparse_attn_mma_paged);
  ops.impl("msa_indexer_block_scores", torch::kCUDA,
           &msa_indexer_block_scores);
  ops.impl("msa_nvfp4_dequant_swizzled_to_bf16", torch::kCUDA,
           &msa_nvfp4_dequant_swizzled_to_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
