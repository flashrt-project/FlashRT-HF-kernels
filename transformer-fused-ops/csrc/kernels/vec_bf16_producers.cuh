#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" {

int quantize_fp8_static_bf16_vec(const __nv_bfloat16* input,
                                 __nv_fp8_e4m3* output,
                                 const float* scale,
                                 int numel,
                                 cudaStream_t stream);
int layer_norm_fp8_static_bf16_vec(const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   const __nv_bfloat16* bias,
                                   __nv_fp8_e4m3* out,
                                   const float* scale,
                                   int rows,
                                   int dim,
                                   float eps,
                                   cudaStream_t stream);
int gate_geglu_merged_fp8_static_bf16_vec(const __nv_bfloat16* merged,
                                          __nv_fp8_e4m3* out,
                                          const float* scale,
                                          int rows,
                                          int hidden,
                                          cudaStream_t stream);

}
