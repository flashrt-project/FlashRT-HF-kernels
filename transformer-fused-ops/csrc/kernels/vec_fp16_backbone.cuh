#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" {

int rms_norm_fp16_vec(const __half* x, const __half* w, __half* out,
                      int rows, int dim, float eps, cudaStream_t stream);
int layer_norm_fp16_vec(const __half* x, const __half* w, const __half* b,
                        __half* out, int rows, int dim, float eps,
                        cudaStream_t stream);
int layer_norm_fp8_static_fp16_vec(const __half* x, const __half* w,
                                   const __half* b, __nv_fp8_e4m3* out,
                                   const float* scale, int rows, int dim,
                                   float eps, cudaStream_t stream);
int rope_rotate_half_fp16_vec(__half* x, const __half* cos_t,
                              const __half* sin_t, int sequence,
                              int heads, int head_dim,
                              cudaStream_t stream);
int quantize_fp8_static_fp16_vec(const __half* input, __nv_fp8_e4m3* output,
                                 const float* scale, int numel,
                                 cudaStream_t stream);
int residual_add_fp16_vec(__half* residual, const __half* x, int numel,
                          cudaStream_t stream);
int gpu_repeat_interleave_heads_vec(const __half* src, __half* dst,
                                    int sequence, int source_heads,
                                    int head_dim, int repeat,
                                    cudaStream_t stream);

}
