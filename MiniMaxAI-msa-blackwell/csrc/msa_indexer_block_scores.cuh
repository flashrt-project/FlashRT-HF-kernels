// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace flashrt_minimax_msa {

// Block-max QK scoring for the MSA lightning indexer. For every
// (q_head, kv block, query) it computes the maximum over the block's visible
// keys of the raw dot product q . k (no softmax / scale), matching the public
// fp4_indexer_block_scores reference numerics. K/V live in 128-token pages.
//   q          [total_q, Hq, D]            bf16 (already dequantized)
//   k_pages    [num_pages, Hkv, 128, D]    bf16 (already dequantized)
//   batch_of_q [total_q]                   int32  query -> batch id
//   cu_q       [batch+1] int32, cu_k [batch+1] int32, cu_pages [batch+1] int32
//   kv_indices [total_pages]               int32  logical page -> physical page
//   scores     [Hq, max_blocks, total_q]   float32, pre-filled with -inf
void msa_indexer_block_scores_cuda(
    const void* q, const void* k_pages, const int* batch_of_q, const int* cu_q,
    const int* cu_k, const int* cu_pages, const int* kv_indices, float* scores,
    int total_q, int Hq, int Hkv, int D, int num_pages, int max_blocks,
    int page_size, bool causal, cudaStream_t stream);

}  // namespace flashrt_minimax_msa
