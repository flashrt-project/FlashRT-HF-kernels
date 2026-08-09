// SPDX-License-Identifier: Apache-2.0

#include "kernels/nexn2_moe_grouped_w4a16.cuh"
#include "kernels/nexn2_w4a16_gemv.cuh"
#include "kernels/grouped_w4a4_gemv_sm120.cuh"
#include "kernels/w4a16_edge_sm120.cuh"

namespace flash_rt {
namespace kernels {

int nexn2_w4a16_matvec_bf16(
    const void* x_bf16, const void* W_packed, const void* SFB, void* out,
    int N, int K, float alpha, cudaStream_t stream) {
  return w4a16_matvec_edge_sm120_bf16(
      x_bf16, W_packed, SFB, out, N, K, alpha, stream);
}

int nexn2_moe_grouped_w4a16_bf16(
    const void* A_stack, const void* W_stack, const void* SFB_stack,
    const void* alpha_stack, const void* expert_idx, void* D, int slots,
    int N, int K, long a_stride, long w_stride, long sfb_stride,
    cudaStream_t stream) {
  return moe_grouped_w4a16_edge_sm120_bf16(
      A_stack, W_stack, SFB_stack, alpha_stack, expert_idx, D, slots, N, K,
      a_stride, w_stride, sfb_stride, stream);
}

}  // namespace kernels

namespace gemm {

int grouped_w4a4_gemv_sm120_bf16(
    const void*, const void*, void*, const void*, const void*, const void*,
    const void*, int, int, int, int, long, long, cudaStream_t) {
  // SM110 does not expose the block-scaled MMA used by this SM120 W4A4 path.
  return 110;
}

}  // namespace gemm
}  // namespace flash_rt
