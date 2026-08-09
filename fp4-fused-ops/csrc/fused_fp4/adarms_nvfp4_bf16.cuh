#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace fused_fp4 {

void adarms_nvfp4_native_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* style,
    uint8_t* packed, uint8_t* sfa, __nv_bfloat16* gate,
    int rows, int dim, int style_rows, cudaStream_t stream);

void gate_res_adarms_nvfp4_native_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* previous_gate,
    __nv_bfloat16* residual, const __nv_bfloat16* style,
    uint8_t* packed, uint8_t* sfa, __nv_bfloat16* gate,
    int rows, int dim, int style_rows, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
