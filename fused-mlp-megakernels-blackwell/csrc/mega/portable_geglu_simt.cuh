// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

// Portable SIMT implementation of the fused FP16 GeGLU megakernel. The
// SM100-family CUTLASS 4.0 megakernel (flashrt_megakernel_geglu_fp16) uses
// tcgen05/TMA descriptor paths that assert on sm_110a (Thor); this reference
// computes the same fusion in pure SIMT FMA so the op stays usable there:
//
//   gate[m, n]         = sum_k  X[m,k] * W_gate[n,k]          (fp32 acc)
//   gate_scratch[m, n] = fp16( gelu_tanh(gate[m, n]) )
//   hidden[m, n]       = fp16( gate_scratch[m, n] * sum_k X[m,k]*W_up[n,k] )
//
// sm_100/sm_103 keep the CUTLASS megakernel path; this is a compatibility
// path only (matches the fp8-gemm / world-model-conv portable fallbacks).

namespace flashrt {
namespace megakernel {

// hidden[M,N] fp16, gate_scratch[M,N] fp16 =
//   gelu_tanh(X@W_gate^T) * (X@W_up^T), with gate_scratch = gelu_tanh(gate).
// Returns 0 on success, non-zero on invalid dimensions.
int geglu_fused_fp16_simt(
    const void* X, const void* W_gate, const void* W_up,
    void* gate_scratch, void* hidden,
    int M, int N, int K, cudaStream_t stream);

}  // namespace megakernel
}  // namespace flashrt
