// ============================================================================
//  Fused FP4 quantize + CUTLASS SFA/SFB tile-interleaved scale write.
//
//  Implementation = kernel_quantize_fp4 (quantize_fp4_dynamic.cu) with the
//  scale-store address replaced by the CUTLASS layout functor. Packed fp4
//  elements layout is UNCHANGED (still linear [N, D/2]), only the scale
//  byte goes to a different location.
// ============================================================================
#include "quantize_fp4_sfa.cuh"
#include "sfa_layout.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace flash_rt {
namespace fp4 {

// ── Device helpers (duplicated locally to stay additive — not linking against
//    quantize_fp4_dynamic.cu so we don't risk ODR issues). Identical logic. ──
__device__ __forceinline__ uint8_t fp32_to_e2m1_sfa(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

__device__ __forceinline__ __nv_fp8_e4m3 quantize_ue4m3_sfa(float x) {
    float v = fmaxf(x, 0.f);
    return __nv_fp8_e4m3(v);
}

__device__ __forceinline__ float dequantize_ue4m3_sfa(__nv_fp8_e4m3 s) {
    return static_cast<float>(s);
}

__device__ __forceinline__ float input_to_float(__half value) {
    return __half2float(value);
}

__device__ __forceinline__ float input_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

// ── Fused kernel ──
// One thread per (row, 16-element block). Scale byte goes to
// dst_sfa[layout(row, block_idx*16, 0)].
template <typename Input>
__global__ void kernel_quantize_fp4_sfa(
    const Input* __restrict__ src,
    uint8_t* __restrict__ dst_packed,
    uint8_t* __restrict__ dst_sfa,   // raw byte view of the CUTLASS SFA/SFB buffer
    int N, int D) {
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row       = blockIdx.y;
  const int n_blocks  = D / 16;
  if (row >= N || block_idx >= n_blocks) return;

  const int base = row * D + block_idx * 16;
  float vals[16];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    vals[i] = input_to_float(src[base + i]);
    float a = fabsf(vals[i]);
    if (a > amax) amax = a;
  }

  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  __nv_fp8_e4m3 bs_q = quantize_ue4m3_sfa(desired);
  float bs_dq        = dequantize_ue4m3_sfa(bs_q);

  // ── CORE FUSION: direct SFA tile-layout write ──
  // The shared integer helper is the exact CUTLASS SFA/SFB mapping. K is the
  // full coordinate, with one scale byte per 16-element block.
  const int sfa_off = sfa_offset_128x64(row, block_idx * 16, D);
  dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

  // Packed fp4 elements: layout unchanged.
  const int out_base = row * (D / 2) + block_idx * 8;
  const float inv_bs = 1.f / bs_dq;
  #pragma unroll
  for (int p = 0; p < 8; ++p) {
    float v_lo = vals[2 * p    ] * inv_bs;
    float v_hi = vals[2 * p + 1] * inv_bs;
    uint8_t lo = fp32_to_e2m1_sfa(v_lo);
    uint8_t hi = fp32_to_e2m1_sfa(v_hi);
    dst_packed[out_base + p] = lo | (hi << 4);
  }
}

int quantize_fp4_dynamic_sfa_fp16(
    const void* src_fp16, void* dst_packed, void* dst_sfa,
    int N, int D, bool is_sfb, cudaStream_t stream) {
  if (D % 16 != 0) return -1;
  const int n_blocks = D / 16;
  const int threads = 128;
  dim3 grid((n_blocks + threads - 1) / threads, N);
  dim3 block(threads);

  if (is_sfb) {
    kernel_quantize_fp4_sfa<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(src_fp16),
        reinterpret_cast<uint8_t*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa),
        N, D);
  } else {
    kernel_quantize_fp4_sfa<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(src_fp16),
        reinterpret_cast<uint8_t*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa),
        N, D);
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int quantize_fp4_dynamic_sfa_bf16(
    const void* src_bf16, void* dst_packed, void* dst_sfa,
    int N, int D, bool is_sfb, cudaStream_t stream) {
  if (D % 16 != 0) return -1;
  const int n_blocks = D / 16;
  const int threads = 128;
  dim3 grid((n_blocks + threads - 1) / threads, N);
  dim3 block(threads);
  if (is_sfb) {
    kernel_quantize_fp4_sfa<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_bf16),
        reinterpret_cast<uint8_t*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa), N, D);
  } else {
    kernel_quantize_fp4_sfa<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_bf16),
        reinterpret_cast<uint8_t*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa), N, D);
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt
