// SPDX-License-Identifier: Apache-2.0
// BF16 producer twins for the PI0.5 Thor FP4 pipeline.

#include "fused_fp4/pi05_bf16_fp4_producers.cuh"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace flash_rt::fused_fp4 {
namespace {

using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ float block_sum(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < (blockDim.x >> 5) ? shared[lane] : 0.f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  const float result = shared[0];
  // Prevent a following reduction from overwriting shared[0] before every
  // thread has consumed the current result.
  __syncthreads();
  return result;
}

template <class Layout>
__device__ __forceinline__ void quantize_block(
    const __nv_bfloat16* values, uint8_t* packed_row, uint8_t* sfa,
    Layout layout, int row, int block_index, int value_offset) {
  float local[16];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    local[i] = __bfloat162float(values[value_offset + i]);
    amax = fmaxf(amax, fabsf(local[i]));
  }
  const __nv_fp8_e4m3 scale_q = __nv_fp8_e4m3(fmaxf(amax / 6.f, 1e-12f));
  const float inverse = 1.f / static_cast<float>(scale_q);
  sfa[layout(row, block_index * 16, 0)] =
      *reinterpret_cast<const uint8_t*>(&scale_q);
  uint2 packed_values;
  uint8_t* packed_bytes = reinterpret_cast<uint8_t*>(&packed_values);
  #pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    packed_bytes[pair] = static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(
            make_float2(local[2 * pair] * inverse,
                        local[2 * pair + 1] * inverse),
            __NV_E2M1, cudaRoundNearest));
  }
  reinterpret_cast<uint2*>(packed_row)[block_index] = packed_values;
}

template <class Layout>
__global__ void gelu_mul_nvfp4_bf16_kernel(
    const __nv_bfloat16* __restrict__ merged,
    const __nv_bfloat16* __restrict__ inv_s,
    uint8_t* __restrict__ packed, uint8_t* __restrict__ sfa,
    Layout layout, int hidden) {
  const int row = blockIdx.x;
  const int block_index = blockIdx.y * blockDim.x + threadIdx.x;
  const int blocks = hidden / 16;
  if (block_index >= blocks) return;
  const int base = block_index * 16;
  const __nv_bfloat16* row_ptr = merged +
      static_cast<long long>(row) * 2 * hidden;
  const int4* gate_ptr = reinterpret_cast<const int4*>(row_ptr + base);
  const int4* up_ptr = reinterpret_cast<const int4*>(
      row_ptr + hidden + base);
  const int4 gate_raw[2] = {gate_ptr[0], gate_ptr[1]};
  const int4 up_raw[2] = {up_ptr[0], up_ptr[1]};
  const __nv_bfloat16* gate_values =
      reinterpret_cast<const __nv_bfloat16*>(gate_raw);
  const __nv_bfloat16* up_values =
      reinterpret_cast<const __nv_bfloat16*>(up_raw);
  int4 inv_raw[2];
  const __nv_bfloat16* inv_values = nullptr;
  if (inv_s != nullptr) {
    const int4* inv_ptr = reinterpret_cast<const int4*>(inv_s + base);
    inv_raw[0] = inv_ptr[0];
    inv_raw[1] = inv_ptr[1];
    inv_values = reinterpret_cast<const __nv_bfloat16*>(inv_raw);
  }
  __nv_bfloat16 rounded[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float gate = __bfloat162float(gate_values[i]);
    const float up = __bfloat162float(up_values[i]);
    float value = gate /
        (1.f + expf(-1.5957691216057308f * gate *
                   (1.f + 0.044715f * gate * gate))) * up;
    if (inv_values != nullptr) value *= __bfloat162float(inv_values[i]);
    rounded[i] = __float2bfloat16_rn(value);
  }
  quantize_block(rounded, packed + static_cast<long long>(row) * hidden / 2,
                 sfa, layout, row, block_index, 0);
}

template <bool AddResidual, bool Multiply, class Layout>
__global__ void rms_norm_nvfp4_bf16_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ inv_s,
    uint8_t* __restrict__ packed, uint8_t* __restrict__ sfa,
    Layout layout, int dim, float eps) {
  const int row = blockIdx.x;
  constexpr int kValuesPerThread = 8;
  float values[kValuesPerThread];
  int columns[kValuesPerThread];
  float sum_sq = 0.f;
  #pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int column = threadIdx.x + item * blockDim.x;
    columns[item] = column;
    float value = 0.f;
    if (column < dim) {
      const long long index = static_cast<long long>(row) * dim + column;
      value = __bfloat162float(x[index]);
      if constexpr (AddResidual) {
        const float sum = __bfloat162float(residual[index]) + value;
        residual[index] = __float2bfloat16_rn(sum);
        value = __bfloat162float(residual[index]);
        sum_sq += sum * sum;
      } else {
        sum_sq += value * value;
      }
    }
    values[item] = value;
  }
  __shared__ float reduction[8];
  const float rstd = rsqrtf(block_sum(sum_sq, reduction) / dim + eps);
  extern __shared__ __nv_bfloat16 normalized[];
  #pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int column = columns[item];
    if (column < dim) {
      float value = values[item] * rstd;
      if constexpr (Multiply) value *= __bfloat162float(inv_s[column]);
      normalized[column] = __float2bfloat16_rn(value);
    }
  }
  __syncthreads();
  uint8_t* packed_row = packed + static_cast<long long>(row) * dim / 2;
  for (int block_index = threadIdx.x; block_index < dim / 16;
       block_index += blockDim.x) {
    quantize_block(normalized, packed_row, sfa, layout, row, block_index,
                   block_index * 16);
  }
}

constexpr int kLayerNormThreads = 128;

__device__ __forceinline__ uint8_t fp32_to_e2m1(float value) {
  const uint8_t sign = value < 0.f ? 0x8u : 0x0u;
  const float magnitude = fabsf(value);
  uint8_t mantissa;
  if (magnitude <= 0.25f) mantissa = 0u;
  else if (magnitude <= 0.75f) mantissa = 1u;
  else if (magnitude <= 1.25f) mantissa = 2u;
  else if (magnitude <= 1.75f) mantissa = 3u;
  else if (magnitude <= 2.5f) mantissa = 4u;
  else if (magnitude <= 3.5f) mantissa = 5u;
  else if (magnitude <= 5.f) mantissa = 6u;
  else mantissa = 7u;
  return sign | mantissa;
}

__device__ __forceinline__ float load_bf16_block16(
    const __nv_bfloat16* row, int block_index, int blocks,
    float values[16]) {
  float sum = 0.f;
  if (block_index < blocks) {
    const int4* source = reinterpret_cast<const int4*>(row) + 2 * block_index;
    const int4 raw[2] = {source[0], source[1]};
    const __nv_bfloat16* loaded =
        reinterpret_cast<const __nv_bfloat16*>(raw);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      values[i] = __bfloat162float(loaded[i]);
      sum += values[i];
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = 0.f;
  }
  return sum;
}

template <bool ToFp4, class Layout>
__global__ void layer_norm_bf16_producer_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ inv_s,
    uint8_t* __restrict__ out, uint8_t* __restrict__ sfa,
    Layout layout, int dim, float eps) {
  const int row = blockIdx.x;
  const int block_index = threadIdx.x;
  const int blocks = dim / 16;
  float values[16] = {};
  const __nv_bfloat16* row_ptr = x + static_cast<long long>(row) * dim;
  const float sum = load_bf16_block16(row_ptr, block_index, blocks, values);
  __shared__ float reduction[32];
  const float mean = block_sum(sum, reduction) / dim;
  float variance = 0.f;
  if (block_index < blocks) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float delta = values[i] - mean;
      variance += delta * delta;
    }
  }
  const float rstd = rsqrtf(block_sum(variance, reduction) / dim + eps);
  if (block_index >= blocks) return;

  const int4* gamma_ptr =
      reinterpret_cast<const int4*>(gamma) + 2 * block_index;
  const int4* beta_ptr =
      reinterpret_cast<const int4*>(beta) + 2 * block_index;
  const int4 gamma_raw[2] = {gamma_ptr[0], gamma_ptr[1]};
  const int4 beta_raw[2] = {beta_ptr[0], beta_ptr[1]};
  const __nv_bfloat16* gamma_values =
      reinterpret_cast<const __nv_bfloat16*>(gamma_raw);
  const __nv_bfloat16* beta_values =
      reinterpret_cast<const __nv_bfloat16*>(beta_raw);
  int4 inv_raw[2];
  const __nv_bfloat16* inv_values = nullptr;
  if (inv_s != nullptr) {
    const int4* inv_ptr =
        reinterpret_cast<const int4*>(inv_s) + 2 * block_index;
    inv_raw[0] = inv_ptr[0];
    inv_raw[1] = inv_ptr[1];
    inv_values = reinterpret_cast<const __nv_bfloat16*>(inv_raw);
  }

  if constexpr (!ToFp4) {
    uint4 packed_out;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed_out);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float value = (values[i] - mean) * rstd *
          __bfloat162float(gamma_values[i]) + __bfloat162float(beta_values[i]);
      // This is a fused LayerNorm-to-FP8 contract: quantize the FP32
      // normalization result directly, matching the established FP16 native
      // kernel rather than materializing an intermediate BF16 tensor.
      const __nv_fp8_e4m3 quantized = __nv_fp8_e4m3(value);
      bytes[i] = *reinterpret_cast<const uint8_t*>(&quantized);
    }
    reinterpret_cast<uint4*>(out + static_cast<long long>(row) * dim)
        [block_index] = packed_out;
  } else {
    float rounded[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float value = (values[i] - mean) * rstd *
          __bfloat162float(gamma_values[i]) + __bfloat162float(beta_values[i]);
      if (inv_values != nullptr) value *= __bfloat162float(inv_values[i]);
      rounded[i] = __bfloat162float(__float2bfloat16_rn(value));
    }
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(rounded[i]));
    const __nv_fp8_e4m3 scale_q =
        __nv_fp8_e4m3(fmaxf(amax / 6.f, 1e-12f));
    const float inverse = 1.f / static_cast<float>(scale_q);
    sfa[layout(row, block_index * 16, 0)] =
        *reinterpret_cast<const uint8_t*>(&scale_q);
    uint2 packed_out;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed_out);
    #pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
      const uint8_t low = fp32_to_e2m1(rounded[2 * pair] * inverse);
      const uint8_t high = fp32_to_e2m1(rounded[2 * pair + 1] * inverse);
      bytes[pair] = static_cast<uint8_t>(low | (high << 4));
    }
    reinterpret_cast<uint2*>(out + static_cast<long long>(row) * dim / 2)
        [block_index] = packed_out;
  }
}

int launch_status() {
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

}  // namespace

int gelu_mul_nvfp4_bf16(
    const __nv_bfloat16* merged, const __nv_bfloat16* inv_s,
    uint8_t* packed, uint8_t* sfa, int rows, int hidden,
    cudaStream_t stream) {
  if (hidden % 16 != 0) return -1;
  auto layout = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(rows, 1, hidden, 1));
  constexpr int threads = 128;
  dim3 grid(rows, (hidden / 16 + threads - 1) / threads);
  gelu_mul_nvfp4_bf16_kernel<<<grid, threads, 0, stream>>>(
      merged, inv_s, packed, sfa, layout, hidden);
  return launch_status();
}

int rms_norm_mul_nvfp4_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* inv_s,
    uint8_t* packed, uint8_t* sfa, int rows, int dim, float eps,
    cudaStream_t stream) {
  if (dim % 16 != 0 || dim > 2048) return -1;
  auto layout = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(rows, 1, dim, 1));
  rms_norm_nvfp4_bf16_kernel<false, true><<<
      rows, 256, dim * sizeof(__nv_bfloat16), stream>>>(
      nullptr, x, inv_s, packed, sfa, layout, dim, eps);
  return launch_status();
}

int residual_add_rms_norm_nvfp4_bf16(
    __nv_bfloat16* residual, const __nv_bfloat16* x,
    const __nv_bfloat16* inv_s, uint8_t* packed, uint8_t* sfa,
    int rows, int dim, float eps, cudaStream_t stream) {
  if (dim % 16 != 0 || dim > 2048) return -1;
  auto layout = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(rows, 1, dim, 1));
  if (inv_s == nullptr) {
    rms_norm_nvfp4_bf16_kernel<true, false><<<
        rows, 256, dim * sizeof(__nv_bfloat16), stream>>>(
        residual, x, nullptr, packed, sfa, layout, dim, eps);
  } else {
    rms_norm_nvfp4_bf16_kernel<true, true><<<
        rows, 256, dim * sizeof(__nv_bfloat16), stream>>>(
        residual, x, inv_s, packed, sfa, layout, dim, eps);
  }
  return launch_status();
}

int layer_norm_fp8_vec_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, void* out_fp8,
    int rows, int dim, float eps, cudaStream_t stream) {
  if (dim % 16 != 0 || dim / 16 > kLayerNormThreads ||
      (reinterpret_cast<uintptr_t>(x) & 15) ||
      (reinterpret_cast<uintptr_t>(gamma) & 15) ||
      (reinterpret_cast<uintptr_t>(beta) & 15) ||
      (reinterpret_cast<uintptr_t>(out_fp8) & 15)) return -1;
  auto layout = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(rows, 1, dim, 1));
  const int threads = dim / 16 <= 96 ? 96 : kLayerNormThreads;
  layer_norm_bf16_producer_kernel<false><<<
      rows, threads, 0, stream>>>(
      x, gamma, beta, nullptr, reinterpret_cast<uint8_t*>(out_fp8), nullptr,
      layout, dim, eps);
  return launch_status();
}

int layer_norm_mul_nvfp4_vec_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, const __nv_bfloat16* inv_s,
    uint8_t* packed, uint8_t* sfa,
    int rows, int dim, float eps, cudaStream_t stream) {
  if (dim % 16 != 0 || dim / 16 > kLayerNormThreads ||
      (reinterpret_cast<uintptr_t>(x) & 15) ||
      (reinterpret_cast<uintptr_t>(gamma) & 15) ||
      (reinterpret_cast<uintptr_t>(beta) & 15) ||
      (inv_s != nullptr && (reinterpret_cast<uintptr_t>(inv_s) & 15)) ||
      (reinterpret_cast<uintptr_t>(packed) & 7)) return -1;
  auto layout = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(rows, 1, dim, 1));
  const int threads = dim / 16 <= 96 ? 96 : kLayerNormThreads;
  layer_norm_bf16_producer_kernel<true><<<
      rows, threads, 0, stream>>>(
      x, gamma, beta, inv_s, packed, sfa, layout, dim, eps);
  return launch_status();
}

}  // namespace flash_rt::fused_fp4
