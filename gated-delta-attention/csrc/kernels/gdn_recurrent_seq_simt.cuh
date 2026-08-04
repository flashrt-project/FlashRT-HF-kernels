// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

// Portable SIMT recurrent sequence scan. The sm_120a `gdn_recurrent_seq`
// kernel is unavailable on pre-sm120 devices; this reference performs the
// same per-token Gated DeltaNet update as a one-launch scan with the state
// held in FP32 registers across the whole sequence (matches the sm_120
// kernel's documented semantics: state stays FP32 during the scan and is
// written to BF16 once at the end). sm_120 keeps the MMA path.

namespace flash_rt {
namespace kernels {

int gdn_recurrent_seq_bf16_simt(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, void* state, void* out, int S, int num_v_heads,
    int head_dim, bool use_qk_l2norm, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
