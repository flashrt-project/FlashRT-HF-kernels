// SPDX-License-Identifier: Apache-2.0

#include "msa_topk_from_scores.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace flashrt_minimax_msa {
namespace {

constexpr int kMaxTopK = 64;

__global__ void msa_topk_from_scores_kernel(const float* __restrict__ score,
                                            const int* __restrict__ seq_lens,
                                            int* __restrict__ topk_idx,
                                            int heads,
                                            int batch,
                                            int max_blocks,
                                            int block_size,
                                            int topk) {
  const int row = blockIdx.x;
  const int h = row / batch;
  const int b = row - h * batch;
  if (h >= heads) {
    return;
  }

  const int seq_len = max(seq_lens[b], 0);
  int valid_blocks = (seq_len + block_size - 1) / block_size;
  valid_blocks = min(valid_blocks, max_blocks);

  float vals[kMaxTopK];
  int idx[kMaxTopK];
  #pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    vals[i] = -CUDART_INF_F;
    idx[i] = -1;
  }

  const int row_base = (h * batch + b) * max_blocks;
  for (int k = 0; k < valid_blocks; ++k) {
    const float v = score[row_base + k];
    int insert = topk;
    #pragma unroll
    for (int t = 0; t < kMaxTopK; ++t) {
      if (t >= topk) {
        break;
      }
      if (v > vals[t] || (v == vals[t] && k < idx[t])) {
        insert = t;
        break;
      }
    }
    if (insert < topk) {
      for (int t = topk - 1; t > insert; --t) {
        vals[t] = vals[t - 1];
        idx[t] = idx[t - 1];
      }
      vals[insert] = v;
      idx[insert] = k;
    }
  }

  const int out_base = (h * batch + b) * topk;
  for (int t = 0; t < topk; ++t) {
    topk_idx[out_base + t] = idx[t];
  }
}

}  // namespace

void msa_topk_from_scores_cuda(const float* score,
                               const int* seq_lens,
                               int* topk_idx,
                               int heads,
                               int batch,
                               int max_blocks,
                               int block_size,
                               int topk,
                               cudaStream_t stream) {
  const int rows = heads * batch;
  if (rows <= 0) {
    return;
  }
  msa_topk_from_scores_kernel<<<rows, 1, 0, stream>>>(
      score, seq_lens, topk_idx, heads, batch, max_blocks, block_size, topk);
}

}  // namespace flashrt_minimax_msa
