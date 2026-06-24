#include "rocm/gemm_epilogues_rocm.h"

#include <cmath>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include "rocm/hipblaslt_simple.h"

namespace {

constexpr int kBlock = 256;
constexpr float kFp8FnuzMax = 240.0f;

__device__ __forceinline__ float gelu_tanh(float x) {
  return 0.5f * x *
         (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_inplace_bf16_kernel(hip_bfloat16* out, long long total) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
  if (idx >= total) {
    return;
  }
  out[idx] = hip_bfloat16(gelu_tanh(static_cast<float>(out[idx])));
}

__global__ void bias_gelu_quantize_fp8_static_bf16_kernel(
    const hip_bfloat16* __restrict__ in,
    const hip_bfloat16* __restrict__ bias,
    __hip_fp8_e4m3_fnuz* __restrict__ out,
    const float* __restrict__ scale,
    long long total,
    int N,
    int has_bias) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % N);
  float v = static_cast<float>(in[idx]);
  if (has_bias) {
    v += static_cast<float>(bias[col]);
  }
  float q = gelu_tanh(v) / scale[0];
  q = fminf(fmaxf(q, -kFp8FnuzMax), kFp8FnuzMax);
  out[idx] = __hip_fp8_e4m3_fnuz(q);
}

__global__ void channel_scale_quantize_fp8_static_bf16_kernel(
    const hip_bfloat16* __restrict__ in,
    const hip_bfloat16* __restrict__ channel_scale,
    __hip_fp8_e4m3_fnuz* __restrict__ out,
    const float* __restrict__ scale,
    long long total,
    int K) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % K);
  float q = static_cast<float>(in[idx]) *
            static_cast<float>(channel_scale[col]) / scale[0];
  q = fminf(fmaxf(q, -kFp8FnuzMax), kFp8FnuzMax);
  out[idx] = __hip_fp8_e4m3_fnuz(q);
}

}  // namespace

namespace flash_rt {
namespace gemm {

void bf16_gemm(
    const void* a,
    const void* b,
    void* out,
    int M,
    int N,
    int K,
    hipStream_t stream) {
  rocm::hipblaslt_linear_bf16(a, b, nullptr, out, M, N, K, stream);
}

void bf16_gemm_bias(
    const void* a,
    const void* b,
    const void* bias,
    void* out,
    int M,
    int N,
    int K,
    hipStream_t stream) {
  rocm::hipblaslt_linear_bf16(a, b, bias, out, M, N, K, stream);
}

void bf16_gemm_bias_gelu(
    const void* a,
    const void* b,
    const void* bias,
    void* out,
    int M,
    int N,
    int K,
    hipStream_t stream) {
  rocm::hipblaslt_linear_bf16(a, b, bias, out, M, N, K, stream);
  const long long total = static_cast<long long>(M) * N;
  const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(gelu_inplace_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<hip_bfloat16*>(out), total);
}

}  // namespace gemm

namespace quantize {

void bias_gelu_quantize_fp8_static_bf16(
    const void* in_bf16,
    const void* bias_bf16,
    void* out_fp8,
    const float* act_scale,
    long long M,
    int N,
    hipStream_t stream) {
  const long long total = M * static_cast<long long>(N);
  if (total <= 0) {
    return;
  }
  const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(bias_gelu_quantize_fp8_static_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(in_bf16),
                     static_cast<const hip_bfloat16*>(bias_bf16),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out_fp8),
                     act_scale,
                     total,
                     N,
                     bias_bf16 != nullptr ? 1 : 0);
}

void channel_scale_quantize_fp8_static_bf16(
    const void* in_bf16,
    const void* channel_scale_bf16,
    void* out_fp8,
    const float* act_scale,
    long long M,
    int K,
    hipStream_t stream) {
  const long long total = M * static_cast<long long>(K);
  if (total <= 0) {
    return;
  }
  const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(channel_scale_quantize_fp8_static_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(in_bf16),
                     static_cast<const hip_bfloat16*>(channel_scale_bf16),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out_fp8),
                     act_scale,
                     total,
                     K);
}

}  // namespace quantize
}  // namespace flash_rt
