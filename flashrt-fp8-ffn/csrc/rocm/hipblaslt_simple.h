#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace flash_rt {
namespace rocm {

void hipblaslt_matmul_fp8_e4m3fnuz_bf16(
    const void* a,
    const void* b,
    const float* a_scale,
    const float* b_scale,
    void* out,
    int64_t m,
    int64_t n,
    int64_t k,
    hipStream_t stream);

void hipblaslt_linear_bf16(
    const void* x,
    const void* weight,
    const void* bias,
    void* out,
    int64_t m,
    int64_t n,
    int64_t k,
    hipStream_t stream);

}  // namespace rocm
}  // namespace flash_rt
