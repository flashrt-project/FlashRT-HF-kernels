// SPDX-License-Identifier: Apache-2.0
//
// Tensor-facing VLA joint residual/gate kernels. The math follows
// official/FlashRT csrc/kernels/elementwise.cu motus_joint_residual3_*.

#include "residual_gates.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace vla_residual_gates {
namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
  return __float2bfloat16(x);
}

__global__ void joint3_bias_gate_residual_kernel(
    const __nv_bfloat16* __restrict__ v_residual,
    const __nv_bfloat16* __restrict__ v_x,
    const __nv_bfloat16* __restrict__ v_bias,
    const __nv_bfloat16* __restrict__ v_gate,
    __nv_bfloat16* __restrict__ v_out,
    int v_rows,
    int v_dim2,
    const __nv_bfloat16* __restrict__ a_residual,
    const __nv_bfloat16* __restrict__ a_x,
    const __nv_bfloat16* __restrict__ a_bias,
    const __nv_bfloat16* __restrict__ a_gate,
    __nv_bfloat16* __restrict__ a_out,
    int a_rows,
    int a_dim2,
    const __nv_bfloat16* __restrict__ u_residual,
    const __nv_bfloat16* __restrict__ u_x,
    __nv_bfloat16* __restrict__ u_out,
    int u_rows,
    int u_dim2) {
  const int row = blockIdx.x;
  if (row < v_rows) {
    const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(v_residual);
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(v_x);
    const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(v_gate);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(v_bias);
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(v_out);
    const int base = row * v_dim2;
    for (int col = threadIdx.x; col < v_dim2; col += blockDim.x) {
      const int idx = base + col;
      const __nv_bfloat162 rv = r[idx];
      const __nv_bfloat162 xv = x[idx];
      const __nv_bfloat162 gv = g[idx];
      const __nv_bfloat162 bv = b[col];
      const float xb0 = __fadd_rn(bf16_to_f32(xv.x), bf16_to_f32(bv.x));
      const float xb1 = __fadd_rn(bf16_to_f32(xv.y), bf16_to_f32(bv.y));
      const float o0 = __fadd_rn(bf16_to_f32(rv.x), __fmul_rn(xb0, bf16_to_f32(gv.x)));
      const float o1 = __fadd_rn(bf16_to_f32(rv.y), __fmul_rn(xb1, bf16_to_f32(gv.y)));
      o[idx] = __halves2bfloat162(f32_to_bf16(o0), f32_to_bf16(o1));
    }
  } else if (row < v_rows + a_rows) {
    const int a_row = row - v_rows;
    const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(a_residual);
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(a_x);
    const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(a_gate);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(a_bias);
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(a_out);
    const int base = a_row * a_dim2;
    for (int col = threadIdx.x; col < a_dim2; col += blockDim.x) {
      const int idx = base + col;
      const __nv_bfloat162 rv = r[idx];
      const __nv_bfloat162 xv = x[idx];
      const __nv_bfloat162 gv = g[idx];
      const __nv_bfloat162 bv = b[col];
      const float xb0 = __fadd_rn(bf16_to_f32(xv.x), bf16_to_f32(bv.x));
      const float xb1 = __fadd_rn(bf16_to_f32(xv.y), bf16_to_f32(bv.y));
      const float o0 = __fadd_rn(bf16_to_f32(rv.x), __fmul_rn(xb0, bf16_to_f32(gv.x)));
      const float o1 = __fadd_rn(bf16_to_f32(rv.y), __fmul_rn(xb1, bf16_to_f32(gv.y)));
      o[idx] = __halves2bfloat162(f32_to_bf16(o0), f32_to_bf16(o1));
    }
  } else if (row < v_rows + a_rows + u_rows) {
    const int u_row = row - v_rows - a_rows;
    const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(u_residual);
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(u_x);
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(u_out);
    const int base = u_row * u_dim2;
    for (int col = threadIdx.x; col < u_dim2; col += blockDim.x) {
      const int idx = base + col;
      const __nv_bfloat162 rv = r[idx];
      const __nv_bfloat162 xv = x[idx];
      const float o0 = __fadd_rn(bf16_to_f32(rv.x), bf16_to_f32(xv.x));
      const float o1 = __fadd_rn(bf16_to_f32(rv.y), bf16_to_f32(xv.y));
      o[idx] = __halves2bfloat162(f32_to_bf16(o0), f32_to_bf16(o1));
    }
  }
}

__global__ void joint3_bias_gate_residual_action_nobias_kernel(
    const __nv_bfloat16* __restrict__ v_residual,
    const __nv_bfloat16* __restrict__ v_x,
    const __nv_bfloat16* __restrict__ v_bias,
    const __nv_bfloat16* __restrict__ v_gate,
    __nv_bfloat16* __restrict__ v_out,
    int v_rows,
    int v_dim2,
    const __nv_bfloat16* __restrict__ a_residual,
    const __nv_bfloat16* __restrict__ a_x,
    const __nv_bfloat16* __restrict__ a_gate,
    __nv_bfloat16* __restrict__ a_out,
    int a_rows,
    int a_dim2,
    const __nv_bfloat16* __restrict__ u_residual,
    const __nv_bfloat16* __restrict__ u_x,
    __nv_bfloat16* __restrict__ u_out,
    int u_rows,
    int u_dim2) {
  const int row = blockIdx.x;
  if (row < v_rows) {
    const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(v_residual);
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(v_x);
    const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(v_gate);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(v_bias);
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(v_out);
    const int base = row * v_dim2;
    for (int col = threadIdx.x; col < v_dim2; col += blockDim.x) {
      const int idx = base + col;
      const __nv_bfloat162 rv = r[idx];
      const __nv_bfloat162 xv = x[idx];
      const __nv_bfloat162 gv = g[idx];
      const __nv_bfloat162 bv = b[col];
      const float xb0 = __fadd_rn(bf16_to_f32(xv.x), bf16_to_f32(bv.x));
      const float xb1 = __fadd_rn(bf16_to_f32(xv.y), bf16_to_f32(bv.y));
      const float o0 = __fadd_rn(bf16_to_f32(rv.x), __fmul_rn(xb0, bf16_to_f32(gv.x)));
      const float o1 = __fadd_rn(bf16_to_f32(rv.y), __fmul_rn(xb1, bf16_to_f32(gv.y)));
      o[idx] = __halves2bfloat162(f32_to_bf16(o0), f32_to_bf16(o1));
    }
  } else if (row < v_rows + a_rows) {
    const int a_row = row - v_rows;
    const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(a_residual);
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(a_x);
    const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(a_gate);
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(a_out);
    const int base = a_row * a_dim2;
    for (int col = threadIdx.x; col < a_dim2; col += blockDim.x) {
      const int idx = base + col;
      const __nv_bfloat162 rv = r[idx];
      const __nv_bfloat162 xv = x[idx];
      const __nv_bfloat162 gv = g[idx];
      const float o0 = __fadd_rn(bf16_to_f32(rv.x), __fmul_rn(bf16_to_f32(xv.x), bf16_to_f32(gv.x)));
      const float o1 = __fadd_rn(bf16_to_f32(rv.y), __fmul_rn(bf16_to_f32(xv.y), bf16_to_f32(gv.y)));
      o[idx] = __halves2bfloat162(f32_to_bf16(o0), f32_to_bf16(o1));
    }
  } else if (row < v_rows + a_rows + u_rows) {
    const int u_row = row - v_rows - a_rows;
    const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(u_residual);
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(u_x);
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(u_out);
    const int base = u_row * u_dim2;
    for (int col = threadIdx.x; col < u_dim2; col += blockDim.x) {
      const int idx = base + col;
      const __nv_bfloat162 rv = r[idx];
      const __nv_bfloat162 xv = x[idx];
      const float o0 = __fadd_rn(bf16_to_f32(rv.x), bf16_to_f32(xv.x));
      const float o1 = __fadd_rn(bf16_to_f32(rv.y), bf16_to_f32(xv.y));
      o[idx] = __halves2bfloat162(f32_to_bf16(o0), f32_to_bf16(o1));
    }
  }
}

}  // namespace

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
    cudaStream_t stream) {
  if (v_n <= 0 || a_n <= 0 || u_n <= 0 || v_dim <= 0 || a_dim <= 0 || u_dim <= 0) return;
  const int v_rows = v_n / v_dim;
  const int a_rows = a_n / a_dim;
  const int u_rows = u_n / u_dim;
  joint3_bias_gate_residual_kernel<<<v_rows + a_rows + u_rows, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(v_residual),
      reinterpret_cast<const __nv_bfloat16*>(v_x),
      reinterpret_cast<const __nv_bfloat16*>(v_bias),
      reinterpret_cast<const __nv_bfloat16*>(v_gate),
      reinterpret_cast<__nv_bfloat16*>(v_out),
      v_rows, v_dim >> 1,
      reinterpret_cast<const __nv_bfloat16*>(a_residual),
      reinterpret_cast<const __nv_bfloat16*>(a_x),
      reinterpret_cast<const __nv_bfloat16*>(a_bias),
      reinterpret_cast<const __nv_bfloat16*>(a_gate),
      reinterpret_cast<__nv_bfloat16*>(a_out),
      a_rows, a_dim >> 1,
      reinterpret_cast<const __nv_bfloat16*>(u_residual),
      reinterpret_cast<const __nv_bfloat16*>(u_x),
      reinterpret_cast<__nv_bfloat16*>(u_out),
      u_rows, u_dim >> 1);
}

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
    cudaStream_t stream) {
  if (v_n <= 0 || a_n <= 0 || u_n <= 0 || v_dim <= 0 || a_dim <= 0 || u_dim <= 0) return;
  const int v_rows = v_n / v_dim;
  const int a_rows = a_n / a_dim;
  const int u_rows = u_n / u_dim;
  joint3_bias_gate_residual_action_nobias_kernel<<<v_rows + a_rows + u_rows, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(v_residual),
      reinterpret_cast<const __nv_bfloat16*>(v_x),
      reinterpret_cast<const __nv_bfloat16*>(v_bias),
      reinterpret_cast<const __nv_bfloat16*>(v_gate),
      reinterpret_cast<__nv_bfloat16*>(v_out),
      v_rows, v_dim >> 1,
      reinterpret_cast<const __nv_bfloat16*>(a_residual),
      reinterpret_cast<const __nv_bfloat16*>(a_x),
      reinterpret_cast<const __nv_bfloat16*>(a_gate),
      reinterpret_cast<__nv_bfloat16*>(a_out),
      a_rows, a_dim >> 1,
      reinterpret_cast<const __nv_bfloat16*>(u_residual),
      reinterpret_cast<const __nv_bfloat16*>(u_x),
      reinterpret_cast<__nv_bfloat16*>(u_out),
      u_rows, u_dim >> 1);
}

}  // namespace vla_residual_gates
}  // namespace flash_rt
