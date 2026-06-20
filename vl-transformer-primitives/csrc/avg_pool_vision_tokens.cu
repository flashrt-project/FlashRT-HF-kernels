// SPDX-License-Identifier: Apache-2.0

#include "avg_pool_vision_tokens.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace vl_transformer_primitives {
namespace {

__global__ void avg_pool_vision_tokens_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int h,
    int w,
    int dim,
    int pool_factor) {
  const int out_tok = blockIdx.x;
  const int h_out = h / pool_factor;
  const int w_out = w / pool_factor;
  const int spv_out = h_out * w_out;

  const int view = out_tok / spv_out;
  const int spatial = out_tok % spv_out;
  const int r_out = spatial / w_out;
  const int c_out = spatial % w_out;
  const float inv = 1.0f / static_cast<float>(pool_factor * pool_factor);

  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d < dim) {
    float sum = 0.0f;
    #pragma unroll
    for (int dr = 0; dr < 8; ++dr) {
      if (dr >= pool_factor) break;
      #pragma unroll
      for (int dc = 0; dc < 8; ++dc) {
        if (dc >= pool_factor) break;
        const int r_in = r_out * pool_factor + dr;
        const int c_in = c_out * pool_factor + dc;
        const int in_tok = view * h * w + r_in * w + c_in;
        sum += __bfloat162float(x[in_tok * dim + d]);
      }
    }
    out[out_tok * dim + d] = __float2bfloat16(sum * inv);
  }
}

}  // namespace

void avg_pool_vision_tokens_bf16(
    const void* x,
    void* out,
    int nv,
    int h,
    int w,
    int dim,
    int pool_factor,
    cudaStream_t stream) {
  const int out_tokens = nv * (h / pool_factor) * (w / pool_factor);
  const dim3 grid(out_tokens, (dim + 255) / 256);
  avg_pool_vision_tokens_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<__nv_bfloat16*>(out),
      h,
      w,
      dim,
      pool_factor);
}

}  // namespace vl_transformer_primitives
}  // namespace flash_rt
