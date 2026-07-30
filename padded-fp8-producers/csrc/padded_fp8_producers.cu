// SPDX-License-Identifier: Apache-2.0
#include "padded_fp8_producers.cuh"

#include <type_traits>

namespace flash_rt::padded_fp8 {
namespace {

__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  return shared[0];
}

__device__ __forceinline__ __nv_fp8_e4m3 quantize_fp8(float value,
                                                       float inv_scale) {
  value = fmaxf(-448.0f, fminf(448.0f, value * inv_scale));
  return __nv_fp8_e4m3(value);
}

template <bool Residual>
__global__ void adaptive_rms_norm_quant_fp8_padded_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ residual_out,
    __nv_fp8_e4m3* __restrict__ output, int rows, int padded_rows, int dim,
    float eps) {
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  const int output_offset = (batch * padded_rows + row) * dim;
  if (row >= rows) {
    for (int col = tid; col < dim; col += blockDim.x) {
      output[output_offset + col] = __nv_fp8_e4m3(0.0f);
    }
    return;
  }

  const int input_offset = (batch * rows + row) * dim;
  const int modulation_offset = batch * dim;
  extern __shared__ float shared[];
  float sum_sq = 0.0f;
  for (int col = tid; col < dim; col += blockDim.x) {
    float value = __bfloat162float(input[input_offset + col]);
    if constexpr (Residual) {
      value += __bfloat162float(residual[input_offset + col]);
      const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
      residual_out[input_offset + col] = rounded;
      value = __bfloat162float(rounded);
    }
    sum_sq += value * value;
  }
  const float inv_rms =
      rsqrtf(block_reduce_sum(sum_sq, shared) / static_cast<float>(dim) + eps);
  const float inv_scale = 1.0f / __ldg(scale);

  for (int col = tid; col < dim; col += blockDim.x) {
    float value = Residual
                      ? __bfloat162float(residual_out[input_offset + col])
                      : __bfloat162float(input[input_offset + col]);
    const float normalized =
        value * inv_rms * __bfloat162float(weight[col]);
    const float modulated =
        (1.0f + __bfloat162float(gamma[modulation_offset + col])) *
            normalized +
        __bfloat162float(beta[modulation_offset + col]);
    // Match the production BF16 activation seam before static FP8 quantization.
    const float rounded =
        __bfloat162float(__float2bfloat16_rn(modulated));
    output[output_offset + col] = quantize_fp8(rounded, inv_scale);
  }
}

template <typename Input, bool Merged>
__global__ void swiglu_quant_fp8_padded_kernel(
    const Input* __restrict__ gate, const Input* __restrict__ up,
    const float* __restrict__ scale, __nv_fp8_e4m3* __restrict__ output,
    int rows, int padded_rows, int dim) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int output_offset = row * dim;
  if (row >= rows) {
    for (int col = tid; col < dim; col += blockDim.x) {
      output[output_offset + col] = __nv_fp8_e4m3(0.0f);
    }
    return;
  }

  const int gate_offset = Merged ? row * 2 * dim : row * dim;
  const int up_offset = Merged ? gate_offset + dim : row * dim;
  const float inv_scale = 1.0f / __ldg(scale);
  for (int col = tid; col < dim; col += blockDim.x) {
    float gate_value;
    float up_value;
    if constexpr (std::is_same_v<Input, __nv_bfloat16>) {
      gate_value = __bfloat162float(gate[gate_offset + col]);
      up_value = __bfloat162float(up[up_offset + col]);
    } else {
      gate_value = __half2float(gate[gate_offset + col]);
      up_value = __half2float(up[up_offset + col]);
    }
    const float activated = gate_value / (1.0f + __expf(-gate_value));
    const float rounded =
        __bfloat162float(__float2bfloat16_rn(activated * up_value));
    output[output_offset + col] = quantize_fp8(rounded, inv_scale);
  }
}

constexpr int kThreads = 256;

}  // namespace

void adaptive_rms_norm_quant_fp8_padded_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
    const float* scale, __nv_fp8_e4m3* output, int batch, int rows,
    int padded_rows, int dim, float eps, cudaStream_t stream) {
  adaptive_rms_norm_quant_fp8_padded_kernel<false>
      <<<dim3(padded_rows, batch), kThreads, kThreads * sizeof(float), stream>>>(
          nullptr, input, weight, gamma, beta, scale, nullptr, output, rows,
          padded_rows, dim, eps);
}

void residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
    const __nv_bfloat16* residual, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, const float* scale,
    __nv_bfloat16* residual_out, __nv_fp8_e4m3* output, int batch, int rows,
    int padded_rows, int dim, float eps, cudaStream_t stream) {
  adaptive_rms_norm_quant_fp8_padded_kernel<true>
      <<<dim3(padded_rows, batch), kThreads, kThreads * sizeof(float), stream>>>(
          residual, input, weight, gamma, beta, scale, residual_out, output,
          rows, padded_rows, dim, eps);
}

void swiglu_quant_fp8_padded_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, const float* scale,
    __nv_fp8_e4m3* output, int rows, int padded_rows, int dim,
    cudaStream_t stream) {
  swiglu_quant_fp8_padded_kernel<__nv_bfloat16, false>
      <<<padded_rows, kThreads, 0, stream>>>(gate, up, scale, output, rows,
                                            padded_rows, dim);
}

void swiglu_merged_quant_fp8_padded_bf16(
    const __nv_bfloat16* gate_up, const float* scale, __nv_fp8_e4m3* output,
    int rows, int padded_rows, int dim, cudaStream_t stream) {
  swiglu_quant_fp8_padded_kernel<__nv_bfloat16, true>
      <<<padded_rows, kThreads, 0, stream>>>(gate_up, gate_up, scale, output,
                                            rows, padded_rows, dim);
}

void swiglu_merged_quant_fp8_padded_fp16(
    const __half* gate_up, const float* scale, __nv_fp8_e4m3* output,
    int rows, int padded_rows, int dim, cudaStream_t stream) {
  swiglu_quant_fp8_padded_kernel<__half, true>
      <<<padded_rows, kThreads, 0, stream>>>(gate_up, gate_up, scale, output,
                                            rows, padded_rows, dim);
}

}  // namespace flash_rt::padded_fp8
