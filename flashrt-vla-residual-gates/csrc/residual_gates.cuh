// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace vla_residual_gates {

void joint3_bias_gate_residual_bf16(
    const void* v_residual,
    const void* v_x,
    const void* v_bias,
    const void* v_gate,
    void* v_out,
    int v_n,
    int v_dim,
    const void* a_residual,
    const void* a_x,
    const void* a_bias,
    const void* a_gate,
    void* a_out,
    int a_n,
    int a_dim,
    const void* u_residual,
    const void* u_x,
    void* u_out,
    int u_n,
    int u_dim,
    cudaStream_t stream);

void joint3_bias_gate_residual_action_nobias_bf16(
    const void* v_residual,
    const void* v_x,
    const void* v_bias,
    const void* v_gate,
    void* v_out,
    int v_n,
    int v_dim,
    const void* a_residual,
    const void* a_x,
    const void* a_gate,
    void* a_out,
    int a_n,
    int a_dim,
    const void* u_residual,
    const void* u_x,
    void* u_out,
    int u_n,
    int u_dim,
    cudaStream_t stream);

}  // namespace vla_residual_gates
}  // namespace flash_rt
