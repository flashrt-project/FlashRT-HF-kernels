#include "weight_only_ffn.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flashrt_weight_only {
namespace {

__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float k = 0.7978845608028654f;
  return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

__global__ void gated_activation_kernel(
    const __nv_bfloat16* merged, const __nv_bfloat16* bias,
    __nv_bfloat16* hidden, int rows, int width, bool gelu) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * width;
  if (idx >= total) return;
  const int row = idx / width;
  const int col = idx - row * width;
  float gate = __bfloat162float(merged[static_cast<size_t>(row) * 2 * width + col]);
  float up = __bfloat162float(merged[static_cast<size_t>(row) * 2 * width + width + col]);
  if (bias) {
    gate += __bfloat162float(bias[col]);
    up += __bfloat162float(bias[width + col]);
  }
  const float activation = gelu ? gelu_tanh(gate) : gate / (1.0f + expf(-gate));
  hidden[idx] = __float2bfloat16(activation * up);
}

__global__ void gelu_activation_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_bfloat16* output, int total, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float value = __bfloat162float(input[idx]);
  if (bias) value += __bfloat162float(bias[idx % width]);
  output[idx] = __float2bfloat16(gelu_tanh(value));
}

__global__ void add_bias_kernel(__nv_bfloat16* output,
                                const __nv_bfloat16* bias,
                                int total, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const float value = __bfloat162float(output[idx]) + __bfloat162float(bias[idx % width]);
  output[idx] = __float2bfloat16(value);
}

}  // namespace

void gated_activation_bf16(const void* merged, const void* bias, void* hidden,
                           int rows, int hidden_size, bool gelu,
                           cudaStream_t stream) {
  const int total = rows * hidden_size;
  gated_activation_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(merged),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(hidden), rows, hidden_size, gelu);
}

void gelu_activation_bf16(const void* input, const void* bias, void* output,
                          int rows, int hidden_size, cudaStream_t stream) {
  const int total = rows * hidden_size;
  gelu_activation_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), total, hidden_size);
}

void add_bias_bf16(void* output, const void* bias, int rows, int cols,
                   cudaStream_t stream) {
  if (!bias) return;
  const int total = rows * cols;
  add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(output),
      static_cast<const __nv_bfloat16*>(bias), total, cols);
}

}  // namespace flashrt_weight_only
