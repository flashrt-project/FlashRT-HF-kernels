// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flashrt_hub::speculative {

void argmax_bf16(
    const __nv_bfloat16* logits,
    int64_t* argmax_out,
    int rows,
    int vocab,
    cudaStream_t stream);

void accept_greedy_bf16(
    const __nv_bfloat16* logits,
    const int64_t* drafts,
    int64_t* argmax_out,
    int* accept_n,
    int rows,
    int vocab,
    int spec_k,
    cudaStream_t stream);

void accept_partitioned_bf16(
    const __nv_bfloat16* logits,
    const int64_t* drafts,
    int64_t* argmax_out,
    int* accept_n,
    float* partial_vals,
    int* partial_idx,
    int rows,
    int vocab,
    int spec_k,
    int parts,
    cudaStream_t stream);

}  // namespace flashrt_hub::speculative
