// ============================================================================
//  FlashRT native E0M3/INT4 + UE4M3 block-scale GEMM for SM100-class GPUs.
//
//  Minimal C-style entry points used by the pybind layer. Kernel bodies live
//  in int4_tcgen05_gemm.cu and are based on CUTLASS example 72a.
//
//  This header is intentionally free of any CUTLASS includes so it can be
//  pulled into bindings.cpp and standalone test harnesses without paying
//  the CUTLASS compile cost there.
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flashrt_int4 {

// Invariants (enforced at call sites and / or static_assert'd):
//   - block_size == 16 (fixed; matches NVFP4 spec and hardware TCGEN05_MXF4_MMA)
//   - M, N, K multiples of 128 in the current validated public binding
//   - Row-major A [M, K], Column-major B [N, K] (natural for GEMM)
//   - Output D bf16 [M, N], row-major
//
//  Packing:
//   - A_fp4_packed: uint8 [M, K/2]  — each byte = 2 int4 elements (low nibble
//     holds element 2i, high nibble holds 2i+1)
//   - SFA / SFB: fp8 (UE4M3 bitpattern via torch.float8_e4m3fn positive range)
//     physical CUTLASS block-16 layouts. The public binding conservatively
//     requires at least M*K and N*K bytes of backing storage.
//
// Returns 0 on success, nonzero CUTLASS/CUDA error code otherwise.

#if CUDART_VERSION >= 13000
int tcgen05_int4_gemm_bf16(
    void const* A_fp4_packed,   // device ptr, uint8 [M, K/2]
    void const* SFA,            // device ptr, fp8  [M, K/16]
    void const* B_fp4_packed,   // device ptr, uint8 [N, K/2]
    void const* SFB,            // device ptr, fp8  [N, K/16]
    void* D_bf16,               // device ptr, bf16 [M, N]
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream);

// Runtime capability check — cheap, no-arg.
bool has_tcgen05_int4();
#else
inline int tcgen05_int4_gemm_bf16(
    void const*, void const*, void const*, void const*, void*, int, int, int,
    float, float, cudaStream_t) {
  return -2;
}

inline bool has_tcgen05_int4() { return false; }
#endif

} // namespace flashrt_int4
