// SPDX-License-Identifier: Apache-2.0

#include "kernels/nexn2_moe_grouped_w4a16.cuh"
#include "kernels/nexn2_w4a16_gemv.cuh"
#include "kernels/grouped_w4a4_gemv_sm120.cuh"
#include "kernels/w4a16_edge_sm120.cuh"

namespace flash_rt {
namespace kernels {

int nexn2_w4a16_matvec_bf16_sm120(
    const void*, const void*, const void*, void*, int, int, float,
    cudaStream_t);

int nexn2_moe_grouped_w4a16_bf16_sm120(
    const void*, const void*, const void*, const void*, const void*, void*,
    int, int, int, long, long, long, cudaStream_t);

namespace {

int current_device_major() {
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess) return -1;
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return -1;
  return properties.major;
}

}  // namespace

int nexn2_w4a16_matvec_bf16(
    const void* x_bf16, const void* W_packed, const void* SFB, void* out,
    int N, int K, float alpha, cudaStream_t stream) {
  const int major = current_device_major();
  if (major == 11) {
    return w4a16_matvec_edge_sm120_bf16(
        x_bf16, W_packed, SFB, out, N, K, alpha, stream);
  }
  if (major == 12) {
    return nexn2_w4a16_matvec_bf16_sm120(
        x_bf16, W_packed, SFB, out, N, K, alpha, stream);
  }
  return 100;
}

int nexn2_moe_grouped_w4a16_bf16(
    const void* A_stack, const void* W_stack, const void* SFB_stack,
    const void* alpha_stack, const void* expert_idx, void* D, int slots,
    int N, int K, long a_stride, long w_stride, long sfb_stride,
    cudaStream_t stream) {
  const int major = current_device_major();
  if (major == 11) {
    return moe_grouped_w4a16_edge_sm120_bf16(
        A_stack, W_stack, SFB_stack, alpha_stack, expert_idx, D, slots, N, K,
        a_stride, w_stride, sfb_stride, stream);
  }
  if (major == 12) {
    return nexn2_moe_grouped_w4a16_bf16_sm120(
        A_stack, W_stack, SFB_stack, alpha_stack, expert_idx, D, slots, N, K,
        a_stride, w_stride, sfb_stride, stream);
  }
  return 100;
}

}  // namespace kernels

namespace gemm {

int grouped_w4a4_gemv_sm120_bf16_impl(
    const void*, const void*, void*, const void*, const void*, const void*,
    const void*, int, int, int, int, long, long, cudaStream_t);

int grouped_w4a4_gemv_sm120_bf16(
    const void* A_packed, const void* B_stack, void* D, const void* SFA,
    const void* SFB_stack, const void* alpha_stack, const void* expert_idx,
    int M, int top_k, int N, int K, long w_stride, long sfb_stride,
    cudaStream_t stream) {
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return 100;
  }
  // SM110 does not expose the block-scaled MMA used by this SM120 W4A4 path.
  if (properties.major == 11) return 110;
  if (properties.major != 12) return 100;
  return grouped_w4a4_gemv_sm120_bf16_impl(
      A_packed, B_stack, D, SFA, SFB_stack, alpha_stack, expert_idx, M,
      top_k, N, K, w_stride, sfb_stride, stream);
}

}  // namespace gemm
}  // namespace flash_rt
