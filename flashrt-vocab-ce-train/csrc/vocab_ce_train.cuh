// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flashrt_hub {
namespace vocab_ce_train {

// Streaming forward for small-N, huge-vocab linear-CE: one pass over the
// fp32 head computing logits, per-vocab-tile online-softmax partials and
// the label logits. rows must be <= kMaxRows; V % kVTile == 0.
constexpr int kVTile = 32;
constexpr int kMaxRows = 128;

void vocab_ce_fwd_launch(const float* hidden,  // (rows, H) fp32 contiguous
                         const float* weight,  // (V, H) fp32 contiguous
                         const long* labels,   // (rows,)
                         float* logits,        // (rows, V)
                         float* partial_max,   // (rows, V/kVTile)
                         float* partial_sum,   // (rows, V/kVTile)
                         float* label_logit,   // (rows,) pre-filled 0
                         int rows, int v, int h, cudaStream_t stream);

}  // namespace vocab_ce_train
}  // namespace flashrt_hub
