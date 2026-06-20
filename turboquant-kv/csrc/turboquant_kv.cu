// SPDX-License-Identifier: Apache-2.0

#include "turboquant_kv.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace turboquant_kv {
namespace {

constexpr int D = 256;
constexpr int IDX_PACKED_BYTES = D / 2;
constexpr int QJL_PACKED_BYTES = D / 8;
constexpr int CB_MAX = 16;

template <int B_MSE_K, int B_V>
__global__ __launch_bounds__(D, 4)
void unpack_bf16_kernel(
    const uint8_t* __restrict__ k_idx_packed,
    const uint8_t* __restrict__ k_qjl_packed,
    const uint8_t* __restrict__ v_idx_packed,
    const float* __restrict__ cb_k_mse,
    const float* __restrict__ cb_v,
    __nv_bfloat16* __restrict__ y_k,
    __nv_bfloat16* __restrict__ qjl_bf,
    __nv_bfloat16* __restrict__ y_v,
    int m) {
  constexpr int K_MASK = (1 << B_MSE_K) - 1;
  constexpr int V_MASK = (1 << B_V) - 1;
  constexpr int CB_K_LEN = 1 << B_MSE_K;
  constexpr int CB_V_LEN = 1 << B_V;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= m) return;

  __shared__ float scb_k[CB_MAX];
  __shared__ float scb_v[CB_MAX];
  if (tid < CB_K_LEN) scb_k[tid] = cb_k_mse[tid];
  if (tid < CB_V_LEN) scb_v[tid] = cb_v[tid];
  __syncthreads();

  const int byte_idx = tid >> 1;
  const bool high = (tid & 1) != 0;
  const uint8_t kb = k_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
  const uint8_t vb = v_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
  const int knib = high ? ((kb >> 4) & 0xF) : (kb & 0xF);
  const int vnib = high ? ((vb >> 4) & 0xF) : (vb & 0xF);
  y_k[row * D + tid] = __float2bfloat16(scb_k[knib & K_MASK]);
  y_v[row * D + tid] = __float2bfloat16(scb_v[vnib & V_MASK]);

  const uint8_t qb = k_qjl_packed[row * QJL_PACKED_BYTES + (tid >> 3)];
  const bool bit = ((qb >> (tid & 7)) & 1u) != 0;
  qjl_bf[row * D + tid] = bit ? __float2bfloat16(1.0f) : __float2bfloat16(-1.0f);
}

template <int B_MSE_K, int B_V>
__global__ __launch_bounds__(D, 4)
void unpack_mixed_kernel(
    const uint8_t* __restrict__ k_idx_packed,
    const uint8_t* __restrict__ k_qjl_packed,
    const uint8_t* __restrict__ v_idx_packed,
    const float* __restrict__ cb_k_mse,
    const float* __restrict__ cb_v,
    __nv_bfloat16* __restrict__ y_k,
    float* __restrict__ qjl_f,
    __nv_bfloat16* __restrict__ y_v,
    int m) {
  constexpr int K_MASK = (1 << B_MSE_K) - 1;
  constexpr int V_MASK = (1 << B_V) - 1;
  constexpr int CB_K_LEN = 1 << B_MSE_K;
  constexpr int CB_V_LEN = 1 << B_V;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= m) return;

  __shared__ float scb_k[CB_MAX];
  __shared__ float scb_v[CB_MAX];
  if (tid < CB_K_LEN) scb_k[tid] = cb_k_mse[tid];
  if (tid < CB_V_LEN) scb_v[tid] = cb_v[tid];
  __syncthreads();

  const int byte_idx = tid >> 1;
  const bool high = (tid & 1) != 0;
  const uint8_t kb = k_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
  const uint8_t vb = v_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
  const int knib = high ? ((kb >> 4) & 0xF) : (kb & 0xF);
  const int vnib = high ? ((vb >> 4) & 0xF) : (vb & 0xF);
  y_k[row * D + tid] = __float2bfloat16(scb_k[knib & K_MASK]);
  y_v[row * D + tid] = __float2bfloat16(scb_v[vnib & V_MASK]);

  const uint8_t qb = k_qjl_packed[row * QJL_PACKED_BYTES + (tid >> 3)];
  const bool bit = ((qb >> (tid & 7)) & 1u) != 0;
  qjl_f[row * D + tid] = bit ? 1.0f : -1.0f;
}

__global__ __launch_bounds__(D, 4)
void combine_kernel(
    const __nv_bfloat16* __restrict__ k_mse,
    const __nv_bfloat16* __restrict__ k_qjl,
    const __nv_bfloat16* __restrict__ v_unit,
    const __half* __restrict__ k_norm,
    const __half* __restrict__ k_rnorm,
    const __half* __restrict__ v_norm,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int m,
    float coef) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= m) return;
  const int off = row * D + tid;
  const float kn = __half2float(k_norm[row]);
  const float krn = __half2float(k_rnorm[row]);
  const float vn = __half2float(v_norm[row]);
  const float kval = kn * (__bfloat162float(k_mse[off]) + coef * krn * __bfloat162float(k_qjl[off]));
  const float vval = vn * __bfloat162float(v_unit[off]);
  k_out[off] = __float2bfloat16(kval);
  v_out[off] = __float2bfloat16(vval);
}

template <typename Launcher>
void dispatch_bits(int b_k_mse, int b_v, Launcher&& launch) {
  if (b_k_mse == 3 && b_v == 4) {
    launch(std::integral_constant<int, 3>{}, std::integral_constant<int, 4>{});
  } else if (b_k_mse == 2 && b_v == 3) {
    launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 3>{});
  } else if (b_k_mse == 3 && b_v == 3) {
    launch(std::integral_constant<int, 3>{}, std::integral_constant<int, 3>{});
  } else {
    launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  }
}

}  // namespace

void unpack_packed_bf16(
    const void* k_idx_packed,
    const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse,
    const void* cb_v,
    void* y_k,
    void* qjl_bf,
    void* y_v,
    int m,
    int b_k_mse,
    int b_v,
    cudaStream_t stream) {
  dispatch_bits(b_k_mse, b_v, [&](auto bk, auto bv) {
    unpack_bf16_kernel<decltype(bk)::value, decltype(bv)::value><<<m, D, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(k_idx_packed),
        reinterpret_cast<const uint8_t*>(k_qjl_packed),
        reinterpret_cast<const uint8_t*>(v_idx_packed),
        reinterpret_cast<const float*>(cb_k_mse),
        reinterpret_cast<const float*>(cb_v),
        reinterpret_cast<__nv_bfloat16*>(y_k),
        reinterpret_cast<__nv_bfloat16*>(qjl_bf),
        reinterpret_cast<__nv_bfloat16*>(y_v),
        m);
  });
}

void unpack_packed_mixed(
    const void* k_idx_packed,
    const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse,
    const void* cb_v,
    void* y_k_bf16,
    void* qjl_fp32,
    void* y_v_bf16,
    int m,
    int b_k_mse,
    int b_v,
    cudaStream_t stream) {
  dispatch_bits(b_k_mse, b_v, [&](auto bk, auto bv) {
    unpack_mixed_kernel<decltype(bk)::value, decltype(bv)::value><<<m, D, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(k_idx_packed),
        reinterpret_cast<const uint8_t*>(k_qjl_packed),
        reinterpret_cast<const uint8_t*>(v_idx_packed),
        reinterpret_cast<const float*>(cb_k_mse),
        reinterpret_cast<const float*>(cb_v),
        reinterpret_cast<__nv_bfloat16*>(y_k_bf16),
        reinterpret_cast<float*>(qjl_fp32),
        reinterpret_cast<__nv_bfloat16*>(y_v_bf16),
        m);
  });
}

void combine_kv_bf16(
    const void* k_mse,
    const void* k_qjl,
    const void* v_unit,
    const void* k_norm,
    const void* k_rnorm,
    const void* v_norm,
    void* k_out,
    void* v_out,
    int m,
    float coef,
    cudaStream_t stream) {
  combine_kernel<<<m, D, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_mse),
      reinterpret_cast<const __nv_bfloat16*>(k_qjl),
      reinterpret_cast<const __nv_bfloat16*>(v_unit),
      reinterpret_cast<const __half*>(k_norm),
      reinterpret_cast<const __half*>(k_rnorm),
      reinterpret_cast<const __half*>(v_norm),
      reinterpret_cast<__nv_bfloat16*>(k_out),
      reinterpret_cast<__nv_bfloat16*>(v_out),
      m,
      coef);
}

}  // namespace turboquant_kv
}  // namespace flash_rt
