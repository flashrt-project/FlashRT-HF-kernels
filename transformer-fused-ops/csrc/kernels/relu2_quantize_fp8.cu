// SPDX-License-Identifier: Apache-2.0

#include "kernels/relu2_quantize_fp8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash_rt::kernels {
namespace {

__global__ void relu2_quantize_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    const float* __restrict__ scale,
    int pairs) {
  const float inverse_scale = 1.0f / scale[0];
  const auto* input2 =
      reinterpret_cast<const __nv_bfloat162*>(input);
  for (int pair = blockIdx.x * blockDim.x + threadIdx.x; pair < pairs;
       pair += gridDim.x * blockDim.x) {
    const __nv_bfloat162 value = input2[pair];
    float first = fmaxf(__bfloat162float(value.x), 0.0f);
    float second = fmaxf(__bfloat162float(value.y), 0.0f);
    first = fminf(first * first * inverse_scale, 448.0f);
    second = fminf(second * second * inverse_scale, 448.0f);
    output[2 * pair] = __nv_fp8_e4m3(first);
    output[2 * pair + 1] = __nv_fp8_e4m3(second);
  }
}

}  // namespace

void relu2_quantize_fp8_static_bf16(
    const void* input,
    void* output,
    const float* scale,
    int numel,
    cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int pairs = numel / 2;
  const int blocks = min((pairs + kThreads - 1) / kThreads, 65535);
  relu2_quantize_fp8_static_bf16_kernel<<<
      blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input),
      reinterpret_cast<__nv_fp8_e4m3*>(output),
      scale,
      pairs);
}

}  // namespace flash_rt::kernels
