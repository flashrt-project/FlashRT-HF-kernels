// SPDX-License-Identifier: Apache-2.0
//
// Tensor-facing adaptive norm kernels. The math follows FlashRT
// csrc/kernels/norm.cu::ada_rms_norm_style and
// csrc/kernels/fusion.cu::gate_residual_ada_norm_fp8.

#include "adaptive_norms.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace adaptive_norms {
namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
  return __float2bfloat16(x);
}

__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) {
    shared[wid] = value;
  }
  __syncthreads();
  if (wid == 0) {
    value = (lane < ((blockDim.x + 31) >> 5)) ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

__global__ void ada_rms_norm_style_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ style,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ gate_out,
    int dim,
    float eps) {
  const int row = blockIdx.x;
  const int dim2 = dim >> 1;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + static_cast<long long>(row) * dim);
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight);
  const __nv_bfloat16* style_row = style + static_cast<long long>(row) * 3 * dim;
  const __nv_bfloat162* scale2 = reinterpret_cast<const __nv_bfloat162*>(style_row);
  const __nv_bfloat162* shift2 = reinterpret_cast<const __nv_bfloat162*>(style_row + dim);
  const __nv_bfloat162* gate2 = reinterpret_cast<const __nv_bfloat162*>(style_row + 2 * dim);
  __nv_bfloat162* out2 =
      reinterpret_cast<__nv_bfloat162*>(out + static_cast<long long>(row) * dim);
  __nv_bfloat162* gate_out2 =
      reinterpret_cast<__nv_bfloat162*>(gate_out + static_cast<long long>(row) * dim);

  extern __shared__ float shared[];
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 xv = x2[i];
    const float x0 = bf16_to_f32(xv.x);
    const float x1 = bf16_to_f32(xv.y);
    local_sum = __fadd_rn(local_sum, __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1)));
  }
  const float rms = rsqrtf(block_reduce_sum(local_sum, shared) / static_cast<float>(dim) + eps);

  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 xv = x2[i];
    const __nv_bfloat162 wv = w2[i];
    const __nv_bfloat162 sv = scale2[i];
    const __nv_bfloat162 hv = shift2[i];
    const __nv_bfloat162 gv = gate2[i];
    const float n0 = __fmul_rn(__fmul_rn(bf16_to_f32(xv.x), rms), bf16_to_f32(wv.x));
    const float n1 = __fmul_rn(__fmul_rn(bf16_to_f32(xv.y), rms), bf16_to_f32(wv.y));
    const float o0 = __fadd_rn(__fmul_rn(n0, __fadd_rn(1.0f, bf16_to_f32(sv.x))), bf16_to_f32(hv.x));
    const float o1 = __fadd_rn(__fmul_rn(n1, __fadd_rn(1.0f, bf16_to_f32(sv.y))), bf16_to_f32(hv.y));
    out2[i] = __halves2bfloat162(f32_to_bf16(o0), f32_to_bf16(o1));
    gate_out2[i] = gv;
  }
}

__global__ void gate_residual_ada_norm_fp8_static_bf16_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ style,
    __nv_fp8_e4m3* __restrict__ out,
    __nv_bfloat16* __restrict__ gate_out,
    int dim,
    float eps,
    const float* __restrict__ scale) {
  const int row = blockIdx.x;
  const int dim2 = dim >> 1;
  __nv_bfloat162* residual2 =
      reinterpret_cast<__nv_bfloat162*>(residual + static_cast<long long>(row) * dim);
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + static_cast<long long>(row) * dim);
  const __nv_bfloat162* input_gate2 =
      reinterpret_cast<const __nv_bfloat162*>(gate + static_cast<long long>(row) * dim);
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight);
  const __nv_bfloat16* style_row = style + static_cast<long long>(row) * 3 * dim;
  const __nv_bfloat162* style_scale2 = reinterpret_cast<const __nv_bfloat162*>(style_row);
  const __nv_bfloat162* shift2 = reinterpret_cast<const __nv_bfloat162*>(style_row + dim);
  const __nv_bfloat162* style_gate2 = reinterpret_cast<const __nv_bfloat162*>(style_row + 2 * dim);
  __nv_fp8_e4m3* out_row = out + static_cast<long long>(row) * dim;
  __nv_bfloat162* gate_out2 =
      reinterpret_cast<__nv_bfloat162*>(gate_out + static_cast<long long>(row) * dim);

  extern __shared__ float shared[];
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 rv = residual2[i];
    const __nv_bfloat162 xv = x2[i];
    const __nv_bfloat162 gv = input_gate2[i];
    const float r0 = __fadd_rn(bf16_to_f32(rv.x), __fmul_rn(bf16_to_f32(xv.x), bf16_to_f32(gv.x)));
    const float r1 = __fadd_rn(bf16_to_f32(rv.y), __fmul_rn(bf16_to_f32(xv.y), bf16_to_f32(gv.y)));
    const __nv_bfloat16 r0_b = f32_to_bf16(r0);
    const __nv_bfloat16 r1_b = f32_to_bf16(r1);
    residual2[i] = __halves2bfloat162(r0_b, r1_b);
    const float rr0 = bf16_to_f32(r0_b);
    const float rr1 = bf16_to_f32(r1_b);
    local_sum = __fadd_rn(local_sum, __fadd_rn(__fmul_rn(rr0, rr0), __fmul_rn(rr1, rr1)));
  }
  const float rms = rsqrtf(block_reduce_sum(local_sum, shared) / static_cast<float>(dim) + eps);
  const float inv_scale = 1.0f / (*scale);

  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 rv = residual2[i];
    const __nv_bfloat162 wv = w2[i];
    const __nv_bfloat162 sv = style_scale2[i];
    const __nv_bfloat162 hv = shift2[i];
    const __nv_bfloat162 gv = style_gate2[i];
    const float n0 = __fmul_rn(__fmul_rn(bf16_to_f32(rv.x), rms), bf16_to_f32(wv.x));
    const float n1 = __fmul_rn(__fmul_rn(bf16_to_f32(rv.y), rms), bf16_to_f32(wv.y));
    const float v0 = __fmul_rn(__fadd_rn(__fmul_rn(n0, __fadd_rn(1.0f, bf16_to_f32(sv.x))), bf16_to_f32(hv.x)), inv_scale);
    const float v1 = __fmul_rn(__fadd_rn(__fmul_rn(n1, __fadd_rn(1.0f, bf16_to_f32(sv.y))), bf16_to_f32(hv.y)), inv_scale);
    out_row[2 * i] = __nv_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
    out_row[2 * i + 1] = __nv_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    gate_out2[i] = gv;
  }
}

}  // namespace

void ada_rms_norm_style_bf16(
    const void* x,
    const void* weight,
    const void* style,
    void* out,
    void* gate_out,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  ada_rms_norm_style_bf16_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<const __nv_bfloat16*>(style),
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<__nv_bfloat16*>(gate_out),
      dim,
      eps);
}

void gate_residual_ada_norm_fp8_static_bf16(
    void* residual,
    const void* x,
    const void* gate,
    const void* weight,
    const void* style,
    void* out,
    void* gate_out,
    int rows,
    int dim,
    float eps,
    const float* scale,
    cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  gate_residual_ada_norm_fp8_static_bf16_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
      reinterpret_cast<__nv_bfloat16*>(residual),
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<const __nv_bfloat16*>(style),
      reinterpret_cast<__nv_fp8_e4m3*>(out),
      reinterpret_cast<__nv_bfloat16*>(gate_out),
      dim,
      eps,
      scale);
}

}  // namespace adaptive_norms
}  // namespace flash_rt
