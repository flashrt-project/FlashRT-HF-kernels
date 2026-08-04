// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT reference for the FP4 block-scaled GEMM fused epilogues.
//
// The sm_120a CUTLASS kernels use block-scaled MMA paths that are not
// available on pre-sm120 devices. These kernels compute the same
//   D[m,n] = epilogue( alpha * sum_k fp4(A[m,k])*ue4m3(SFA[m,k])
//                                     * fp4(B[n,k])*ue4m3(SFB[n,k]) )
// in pure SIMT FMA so the fused epilogue ops remain usable (slowly) on
// sm_110 Thor. sm_120 keeps the CUTLASS path.
//
// Packed layout: byte k/2 holds element (k&1) in the high nibble, element
// (k&1)==0 in the low nibble, e2m1 (NVFP4) code points.
// Scale layout: NVFP4 128-row super-block swizzle used by the SFA/SFB
// producers (Sm1xxBlockScaledConfig<16>); identical for SFA and SFB.

#include "portable_fp4_gemm_simt.cuh"

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace flash_rt {
namespace gemm {

namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float e2m1_to_float(uint8_t v) {
  static constexpr float mags[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float mag = mags[v & 0x7];
  return (v & 0x8) ? -mag : mag;
}

__device__ __forceinline__ float sf_read(const uint8_t* sf, int off) {
  __nv_fp8_e4m3 scale_q;
  *reinterpret_cast<uint8_t*>(&scale_q) = sf[off];
  return static_cast<float>(scale_q);
}

// NVFP4 super-block swizzle byte offset for (row, k) within a flat SFA/SFB buf.
__device__ __forceinline__ int sf_off(int row, int k, int n_col_super) {
  int rb = row >> 7;
  int ri = row & 127;
  int kt = k >> 6;
  int cb = (k >> 4) & 3;
  return (rb * n_col_super + kt) * 512 + (ri & 31) * 16 + ((ri >> 5) & 3) * 4 + cb;
}

__device__ __forceinline__ float read_val(
    const uint8_t* __restrict__ packed, int row, int k, int K,
    const uint8_t* __restrict__ sf, int n_col_super) {
  uint8_t byte = packed[(size_t)row * (K / 2) + (k >> 1)];
  uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
  return e2m1_to_float(nib) * sf_read(sf, sf_off(row, k, n_col_super));
}

__device__ __forceinline__ float gelu_taylor(float x) {
  const float k0 = 0.7978845608028654f;
  return 0.5f * x * (1.f + tanhf(k0 * x * (1.f + 0.044715f * x * x)));
}

__device__ __forceinline__ uint8_t fp32_to_e2m1(float x) {
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

enum class Ep : int { Linear, Residual, Bias, BiasGelu };

template <Ep EPI>
__global__ void gemm_epilogue_simt_kernel(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ sfa, const uint8_t* __restrict__ sfb,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ out,
    int m, int n, int k, float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m * n) return;
  int mi = idx / n, ni = idx - mi * n;
  const int n_col_super = (k / 16 + 3) / 4;
  float acc = 0.f;
  for (int kk = 0; kk < k; ++kk) {
    float av = read_val(a, mi, kk, k, sfa, n_col_super);
    float bv = read_val(b, ni, kk, k, sfb, n_col_super);
    acc += av * bv;
  }
  float r = acc * alpha;
  if (EPI == Ep::Residual) r += __bfloat162float(residual[idx]);
  if (EPI == Ep::Bias || EPI == Ep::BiasGelu) r += __bfloat162float(bias[ni]);
  if (EPI == Ep::BiasGelu) r = gelu_taylor(r);
  out[idx] = __float2bfloat16(r);
}

// One thread per (row, 16-element column block): computes the 16 gelu values
// then quantizes them to NVFP4 (e2m1 packed + per-16-block scale), matching
// the fp4 producer's quantization.
__global__ void gemm_bias_gelu_fp4_simt_kernel(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ sfa, const uint8_t* __restrict__ sfb,
    const __nv_bfloat16* __restrict__ bias,
    uint8_t* __restrict__ out_packed, uint8_t* __restrict__ out_sfa,
    int m, int n, int k, float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_blocks = n / 16;
  if (idx >= m * n_blocks) return;
  const int mi = idx / n_blocks, bi = idx - mi * n_blocks;
  const int n_col_super = (k / 16 + 3) / 4;
  float vals[16];
#pragma unroll
  for (int cc = 0; cc < 16; ++cc) {
    const int ni = bi * 16 + cc;
    float acc = 0.f;
    for (int kk = 0; kk < k; ++kk) {
      float av = read_val(a, mi, kk, k, sfa, n_col_super);
      float bv = read_val(b, ni, kk, k, sfb, n_col_super);
      acc += av * bv;
    }
    vals[cc] = gelu_taylor(acc * alpha + __bfloat162float(bias[ni]));
  }
  float amax = 0.f;
#pragma unroll
  for (int cc = 0; cc < 16; ++cc) amax = fmaxf(amax, fabsf(vals[cc]));
  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  __nv_fp8_e4m3 bs = __nv_fp8_e4m3(desired);
  const float bs_dq = static_cast<float>(bs);
  const int out_n_col_super = (n / 16 + 3) / 4;
  out_sfa[sf_off(mi, bi * 16, out_n_col_super)] =
      *reinterpret_cast<uint8_t*>(&bs);
  const float inv = 1.f / bs_dq;
  uint8_t* op = out_packed + (size_t)mi * (n / 2) + bi * 8;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    uint8_t lo = fp32_to_e2m1(vals[2 * p] * inv);
    uint8_t hi = fp32_to_e2m1(vals[2 * p + 1] * inv);
    op[p] = lo | (hi << 4);
  }
}

int launch_epilogue(const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
                    const uint8_t* sfb, const void* bias, const void* residual,
                    void* out, int m, int n, int k, float alpha, int mode,
                    cudaStream_t stream) {
  const int total = m * n;
  const int blocks = (total + THREADS - 1) / THREADS;
  const auto* bias_b = static_cast<const __nv_bfloat16*>(bias);
  const auto* res_b = static_cast<const __nv_bfloat16*>(residual);
  auto* out_b = static_cast<__nv_bfloat16*>(out);
  if (mode == 0) {
    gemm_epilogue_simt_kernel<Ep::Linear><<<blocks, THREADS, 0, stream>>>(
        a, b, sfa, sfb, bias_b, res_b, out_b, m, n, k, alpha);
  } else if (mode == 1) {
    gemm_epilogue_simt_kernel<Ep::Residual><<<blocks, THREADS, 0, stream>>>(
        a, b, sfa, sfb, bias_b, res_b, out_b, m, n, k, alpha);
  } else if (mode == 2) {
    gemm_epilogue_simt_kernel<Ep::Bias><<<blocks, THREADS, 0, stream>>>(
        a, b, sfa, sfb, bias_b, res_b, out_b, m, n, k, alpha);
  } else {
    gemm_epilogue_simt_kernel<Ep::BiasGelu><<<blocks, THREADS, 0, stream>>>(
        a, b, sfa, sfb, bias_b, res_b, out_b, m, n, k, alpha);
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}

}  // namespace

int nvfp4_gemm_linear_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, void* out_bf16, int m, int n, int k, float alpha,
    cudaStream_t stream) {
  return launch_epilogue(
      static_cast<const uint8_t*>(a_packed), static_cast<const uint8_t*>(b_packed),
      static_cast<const uint8_t*>(sfa), static_cast<const uint8_t*>(sfb),
      nullptr, nullptr, out_bf16, m, n, k, alpha, 0, stream);
}

int nvfp4_gemm_residual_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, const void* residual_bf16, void* out_bf16,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  return launch_epilogue(
      static_cast<const uint8_t*>(a_packed), static_cast<const uint8_t*>(b_packed),
      static_cast<const uint8_t*>(sfa), static_cast<const uint8_t*>(sfb),
      nullptr, residual_bf16, out_bf16, m, n, k, alpha, 1, stream);
}

int nvfp4_gemm_bias_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, const void* bias_bf16, void* out_bf16,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  return launch_epilogue(
      static_cast<const uint8_t*>(a_packed), static_cast<const uint8_t*>(b_packed),
      static_cast<const uint8_t*>(sfa), static_cast<const uint8_t*>(sfb),
      bias_bf16, nullptr, out_bf16, m, n, k, alpha, 2, stream);
}

int nvfp4_gemm_bias_gelu_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, const void* bias_bf16, void* out_bf16,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  return launch_epilogue(
      static_cast<const uint8_t*>(a_packed), static_cast<const uint8_t*>(b_packed),
      static_cast<const uint8_t*>(sfa), static_cast<const uint8_t*>(sfb),
      bias_bf16, nullptr, out_bf16, m, n, k, alpha, 3, stream);
}

int nvfp4_gemm_bias_gelu_fp4_simt(
    const void* a_packed, const void* b_packed, const void* sfa,
    const void* sfb, const void* bias_bf16, void* out_packed, void* out_sfa,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || n % 16 != 0) return 1;
  const int n_blocks = n / 16;
  const int total = m * n_blocks;
  gemm_bias_gelu_fp4_simt_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0,
                                   stream>>>(
      static_cast<const uint8_t*>(a_packed), static_cast<const uint8_t*>(b_packed),
      static_cast<const uint8_t*>(sfa), static_cast<const uint8_t*>(sfb),
      static_cast<const __nv_bfloat16*>(bias_bf16),
      static_cast<uint8_t*>(out_packed), static_cast<uint8_t*>(out_sfa),
      m, n, k, alpha);
  return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}

}  // namespace gemm
}  // namespace flash_rt
