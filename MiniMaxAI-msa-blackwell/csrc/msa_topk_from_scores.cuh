// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flashrt_minimax_msa {

void msa_topk_from_scores_cuda(const float* score,
                               const int* seq_lens,
                               int* topk_idx,
                               int heads,
                               int batch,
                               int max_blocks,
                               int block_size,
                               int topk,
                               cudaStream_t stream);

}  // namespace flashrt_minimax_msa
