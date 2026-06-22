// SPDX-License-Identifier: Apache-2.0
//
// BF16 no-affine LayerNorm -> static FP8 producer.
// Ported from FlashRT csrc/kernels/dit_bf16.cu for Hub Tensor APIs.

#include "dit_layer_norm_fp8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace adaln_producers {
namespace {

__global__ void layer_norm_no_affine_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ d_scale,
    int dim,
    float eps) {
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + static_cast<long long>(row) * dim);
  __nv_fp8_e4m3* out_row = out + static_cast<long long>(row) * dim;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 v = x2[i];
    local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
  }
  float value = local_sum;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) {
    shared[wid] = value;
  }
  __syncthreads();
  if (wid == 0) {
    value = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  const float mean = shared[0] / static_cast<float>(dim);

  float local_var = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 v = x2[i];
    const float d0 = __bfloat162float(v.x) - mean;
    const float d1 = __bfloat162float(v.y) - mean;
    local_var += d0 * d0 + d1 * d1;
  }
  value = local_var;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) {
    shared[wid] = value;
  }
  __syncthreads();
  if (wid == 0) {
    value = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  const float inv_std = rsqrtf(shared[0] / static_cast<float>(dim) + eps);
  const float inv_scale = 1.0f / fmaxf(*d_scale, 1e-12f);

  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 xv = x2[i];
    float v0 = (__bfloat162float(xv.x) - mean) * inv_std;
    float v1 = (__bfloat162float(xv.y) - mean) * inv_std;
    v0 = __bfloat162float(__float2bfloat16(v0)) * inv_scale;
    v1 = __bfloat162float(__float2bfloat16(v1)) * inv_scale;
    __nv_fp8_e4m3 pair[2];
    pair[0] = __nv_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
    pair[1] = __nv_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    *reinterpret_cast<uint16_t*>(out_row + 2 * i) =
        *reinterpret_cast<uint16_t*>(pair);
  }
}

}  // namespace

void layer_norm_no_affine_fp8_static_bf16(
    const void* x_bf16,
    void* out_fp8,
    const float* scale,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) {
    return;
  }
  layer_norm_no_affine_fp8_static_bf16_kernel
      <<<rows, 256, 256 * sizeof(float), stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x_bf16),
          reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
          scale,
          dim,
          eps);
}

}  // namespace adaln_producers
}  // namespace flash_rt
