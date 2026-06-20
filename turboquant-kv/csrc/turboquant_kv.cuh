// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace turboquant_kv {

void unpack_packed_bf16(
    const void* k_idx_packed,
    const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse,
    const void* cb_v,
    void* y_k,
    void* qjl_bf,
    void* y_v,
    int m,
    int b_k_mse,
    int b_v,
    cudaStream_t stream);

void unpack_packed_mixed(
    const void* k_idx_packed,
    const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse,
    const void* cb_v,
    void* y_k_bf16,
    void* qjl_fp32,
    void* y_v_bf16,
    int m,
    int b_k_mse,
    int b_v,
    cudaStream_t stream);

void combine_kv_bf16(
    const void* k_mse,
    const void* k_qjl,
    const void* v_unit,
    const void* k_norm,
    const void* k_rnorm,
    const void* v_norm,
    void* k_out,
    void* v_out,
    int m,
    float coef,
    cudaStream_t stream);

}  // namespace turboquant_kv
}  // namespace flash_rt
