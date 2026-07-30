// SPDX-License-Identifier: Apache-2.0

#include "adaln_modulation6.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace adaln_producers {
namespace {

__global__ void adaln_modulation6_bf16_kernel(
    const float* __restrict__ adaln_params,
    const float* __restrict__ layer_modulation,
    __nv_bfloat16* __restrict__ out0,
    __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2,
    __nv_bfloat16* __restrict__ out3,
    __nv_bfloat16* __restrict__ out4,
    __nv_bfloat16* __restrict__ out5,
    int dim,
    long long elements) {
  const long long index =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int d = static_cast<int>(index % dim);
  const long long base = (index / dim) * 6LL * dim + d;
  out0[index] = __float2bfloat16(
      adaln_params[base] + layer_modulation[d]);
  out1[index] = __float2bfloat16(
      adaln_params[base + dim] + layer_modulation[dim + d]);
  out2[index] = __float2bfloat16(
      adaln_params[base + 2LL * dim] + layer_modulation[2LL * dim + d]);
  out3[index] = __float2bfloat16(
      adaln_params[base + 3LL * dim] + layer_modulation[3LL * dim + d]);
  out4[index] = __float2bfloat16(
      adaln_params[base + 4LL * dim] + layer_modulation[4LL * dim + d]);
  out5[index] = __float2bfloat16(
      adaln_params[base + 5LL * dim] + layer_modulation[5LL * dim + d]);
}

}  // namespace

void adaln_modulation6_bf16(
    const float* adaln_params,
    const float* layer_modulation,
    void* out0,
    void* out1,
    void* out2,
    void* out3,
    void* out4,
    void* out5,
    int batch,
    int sequence,
    int dim,
    cudaStream_t stream) {
  const long long elements =
      static_cast<long long>(batch) * sequence * dim;
  constexpr int threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((elements + threads - 1) / threads);
  adaln_modulation6_bf16_kernel<<<blocks, threads, 0, stream>>>(
      adaln_params,
      layer_modulation,
      static_cast<__nv_bfloat16*>(out0),
      static_cast<__nv_bfloat16*>(out1),
      static_cast<__nv_bfloat16*>(out2),
      static_cast<__nv_bfloat16*>(out3),
      static_cast<__nv_bfloat16*>(out4),
      static_cast<__nv_bfloat16*>(out5),
      dim,
      elements);
}

}  // namespace adaln_producers
}  // namespace flash_rt
