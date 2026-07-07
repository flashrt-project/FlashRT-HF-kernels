// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flashrt_hub {
namespace transformer_layout {

void fill_neginf_bf16(__nv_bfloat16* dst, int n, cudaStream_t stream);
void add_bias_bf16(__nv_bfloat16* data, const __nv_bfloat16* bias,
                   int rows, int cols, cudaStream_t stream);
void repeat_interleave_heads_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                  int seq, int src_heads, int head_dim, int repeat,
                                  cudaStream_t stream);
void text_gather_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst,
                      int batch, int seq, int dim, cudaStream_t stream);
void text_scatter_bf16(__nv_bfloat16* dst, const __nv_bfloat16* src,
                       int batch, int seq, int dim, cudaStream_t stream);
void rope_rotate_half_bf16(__nv_bfloat16* x, const __nv_bfloat16* cos,
                           const __nv_bfloat16* sin, int seq, int heads,
                           int head_dim, cudaStream_t stream);
void qk_rmsnorm_rope_bf16(__nv_bfloat16* qk, const __nv_bfloat16* weight,
                          const __nv_bfloat16* cos, const __nv_bfloat16* sin,
                          int rows, int heads, int head_dim, float eps,
                          cudaStream_t stream);

}  // namespace transformer_layout
}  // namespace flashrt_hub
