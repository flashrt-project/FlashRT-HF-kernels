// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace flashrt_minimax_msa {

// Block-sparse GQA decode attention (one query token per request).
//   q         [B, Hq, D]                      bf16
//   kv_cache  [max_slots, 2, max_len, Hkv, D] bf16  (k=0, v=1)
//   seq_lens  [B] int32, slot_ids [B] int64
//   topk_idx  [Hkv, B, topk] int32            (block ids, -1 padded)
//   out       [B, Hq, D]                      bf16
// Each GQA group (Hq/Hkv q-heads) shares its kv-head's selected blocks.
void msa_decode_sparse_attn_cuda(const void* q, const void* kv_cache,
                                 const int* seq_lens, const int64_t* slot_ids,
                                 const int* topk_idx, void* out,
                                 int B, int Hq, int Hkv, int D,
                                 int max_slots, int max_len,
                                 int block_size, int topk,
                                 float sm_scale, cudaStream_t stream);

}  // namespace flashrt_minimax_msa
