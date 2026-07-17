#include "rocm/fp8_swiglu_ffn_rocm.h"

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

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

template <bool UseGelu>
__global__ void mul_merged_quantize_fp8_static_bf16_kernel(
    const hip_bfloat16* __restrict__ gate_up,
    __hip_fp8_e4m3_fnuz* __restrict__ out,
    const float* __restrict__ scale,
    long long total,
    int H) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % H);
  const long long row = idx / H;
  const long long base = row * static_cast<long long>(2 * H);
  const float gate = static_cast<float>(gate_up[base + col]);
  const float up = static_cast<float>(gate_up[base + H + col]);
  const float act = UseGelu ? gelu_tanh(gate) : silu(gate);
  float q = (act * up) / scale[0];
  q = fminf(fmaxf(q, -kFp8FnuzMax), kFp8FnuzMax);
  out[idx] = __hip_fp8_e4m3_fnuz(q);
}

__global__ void add_bias_bf16_kernel(
    hip_bfloat16* __restrict__ input,
    const hip_bfloat16* __restrict__ bias,
    long long total,
    int N) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % N);
  input[idx] = hip_bfloat16(static_cast<float>(input[idx]) +
                            static_cast<float>(bias[col]));
}

__global__ void quantize_fp8_static_bf16_mpad_kernel(
    const hip_bfloat16* __restrict__ input,
    __hip_fp8_e4m3_fnuz* __restrict__ output,
    const float* __restrict__ scale,
    int logical_m,
    int K) {
  const int row = blockIdx.x;
  const float inv_scale = 1.0f / scale[0];
  const long long base = static_cast<long long>(row) * K;
  for (int col = threadIdx.x; col < K; col += blockDim.x) {
    float q = row < logical_m ? static_cast<float>(input[base + col]) * inv_scale
                              : 0.0f;
    q = fminf(fmaxf(q, -kFp8FnuzMax), kFp8FnuzMax);
    output[base + col] = __hip_fp8_e4m3_fnuz(q);
  }
}

}  // namespace

namespace flash_rt {
namespace fp8_swiglu_ffn {

void fp8_gemm_descale_bf16out(
    const void* input_fp8,
    const void* weight_fp8,
    void* out_bf16,
    int M,
    int N,
    int K,
    const float* input_scale,
    const float* weight_scale,
    hipStream_t stream) {
  rocm::hipblaslt_matmul_fp8_e4m3fnuz_bf16(
      input_fp8, weight_fp8, input_scale, weight_scale, out_bf16,
      M, N, K, stream);
}

void quantize_fp8_static_bf16_mpad(
    const void* input_bf16,
    void* out_fp8,
    const float* scale,
    int logical_m,
    int padded_m,
    int K,
    hipStream_t stream) {
  if (logical_m <= 0 || padded_m < logical_m || K <= 0) return;
  hipLaunchKernelGGL(quantize_fp8_static_bf16_mpad_kernel,
                     dim3(padded_m), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(input_bf16),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out_fp8), scale,
                     logical_m, K);
}

void silu_mul_merged_quantize_fp8_static_bf16(
    const void* gate_up_bf16,
    void* out_fp8,
    const float* scale,
    long long M,
    int H,
    hipStream_t stream) {
  const long long total = M * static_cast<long long>(H);
  if (total <= 0) {
    return;
  }
  const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(mul_merged_quantize_fp8_static_bf16_kernel<false>,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(gate_up_bf16),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out_fp8),
                     scale,
                     total,
                     H);
}

void gelu_mul_merged_quantize_fp8_static_bf16(
    const void* gate_up_bf16,
    void* out_fp8,
    const float* scale,
    long long M,
    int H,
    hipStream_t stream) {
  const long long total = M * static_cast<long long>(H);
  if (total <= 0) {
    return;
  }
  const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(mul_merged_quantize_fp8_static_bf16_kernel<true>,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(gate_up_bf16),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out_fp8),
                     scale,
                     total,
                     H);
}

void add_bias_bf16(
    void* input_bf16,
    const void* bias_bf16,
    long long M,
    int N,
    hipStream_t stream) {
  const long long total = M * static_cast<long long>(N);
  if (total <= 0) {
    return;
  }
  const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(add_bias_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<hip_bfloat16*>(input_bf16),
                     static_cast<const hip_bfloat16*>(bias_bf16),
                     total,
                     N);
}

}  // namespace fp8_swiglu_ffn
}  // namespace flash_rt
