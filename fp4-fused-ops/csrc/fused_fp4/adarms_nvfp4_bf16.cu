#include "fused_fp4/adarms_nvfp4_bf16.cuh"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <stdexcept>
#include <string>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FLASHRT_ADARMS_BF16_HAVE_CUTLASS 1
#else
#  define FLASHRT_ADARMS_BF16_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fused_fp4 {

#if FLASHRT_ADARMS_BF16_HAVE_CUTLASS

using AdaRmsBf16Config = cutlass::detail::Sm1xxBlockScaledConfig<16>;

template <class LayoutSF>
__device__ __forceinline__ void quantize_native_nvfp4_bf16(
    float value, uint8_t* packed_row, uint8_t* sfa, LayoutSF layout,
    int row, int block_idx, int lane_in_block) {
  float amax = fabsf(value);
  #pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, 16));
  }

  float desired = fmaxf(amax / 6.f, 1e-12f);
  __nv_fp8_e4m3 scale = __nv_fp8_e4m3(desired);
  const float inv_scale = 1.f / static_cast<float>(scale);
  if (lane_in_block == 0) {
    sfa[layout(row, block_idx * 16, 0)] =
        *reinterpret_cast<uint8_t*>(&scale);
  }

  const float next = __shfl_down_sync(0xffffffff, value, 1, 16);
  if ((lane_in_block & 1) == 0) {
    packed_row[block_idx * 8 + lane_in_block / 2] =
        static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(
            make_float2(value * inv_scale, next * inv_scale),
            __NV_E2M1, cudaRoundNearest));
  }
}

template <bool GatedResidual, class LayoutSF>
__global__ void adarms_nvfp4_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ previous_gate,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ style,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    __nv_bfloat16* __restrict__ gate,
    LayoutSF layout,
    int dim) {
  const int row = blockIdx.x;
  const __nv_bfloat16* scale = style + static_cast<long long>(row) * 3 * dim;
  const __nv_bfloat16* shift = scale + dim;
  const __nv_bfloat16* next_gate = shift + dim;
  uint8_t* packed_row = packed + static_cast<long long>(row) * (dim / 2);

  float values[4];
  float sum_sq = 0.f;
  #pragma unroll
  for (int segment = 0; segment < 4; ++segment) {
    const int col = threadIdx.x + segment * blockDim.x;
    const long long index = static_cast<long long>(row) * dim + col;
    float value = __bfloat162float(x[index]);
    if constexpr (GatedResidual) {
      value = __bfloat162float(residual[index]) +
          value * __bfloat162float(previous_gate[index]);
      residual[index] = __float2bfloat16_rn(value);
      values[segment] = __bfloat162float(residual[index]);
    } else {
      values[segment] = value;
    }
    // Match the production FP16 contract: reduction uses the FP32 operation
    // result while normalization consumes the stored low-precision value.
    sum_sq += value * value;
  }

  __shared__ float reduction[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
  }
  if (lane == 0) reduction[warp] = sum_sq;
  __syncthreads();
  if (warp == 0) {
    sum_sq = lane < 8 ? reduction[lane] : 0.f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) reduction[0] = sum_sq;
  __syncthreads();

  const float rstd = rsqrtf(reduction[0] / dim + 1e-6f);
  const int lane_in_block = threadIdx.x & 15;
  const int block_group = threadIdx.x >> 4;
  #pragma unroll
  for (int segment = 0; segment < 4; ++segment) {
    const int col = threadIdx.x + segment * blockDim.x;
    const long long index = static_cast<long long>(row) * dim + col;
    const float normed = values[segment] * rstd *
        (1.f + __bfloat162float(scale[col])) + __bfloat162float(shift[col]);
    const __nv_bfloat16 rounded = __float2bfloat16_rn(normed);
    gate[index] = next_gate[col];
    quantize_native_nvfp4_bf16(
        __bfloat162float(rounded), packed_row, sfa, layout, row,
        segment * 16 + block_group, lane_in_block);
  }
}

inline void check_adarms_bf16_launch(const char* name) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string(name) + " launch failed: " + cudaGetErrorString(error));
  }
}

#endif

void adarms_nvfp4_native_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* style,
    uint8_t* packed, uint8_t* sfa, __nv_bfloat16* gate,
    int rows, int dim, cudaStream_t stream) {
#if FLASHRT_ADARMS_BF16_HAVE_CUTLASS
  auto shape = cute::make_shape(rows, 1, dim, 1);
  auto layout = AdaRmsBf16Config::tile_atom_to_shape_SFA(shape);
  adarms_nvfp4_bf16_kernel<false><<<rows, 256, 0, stream>>>(
      x, nullptr, nullptr, style, packed, sfa, gate, layout, dim);
  check_adarms_bf16_launch("adarms_nvfp4_native_bf16");
#else
  (void)x; (void)style; (void)packed; (void)sfa; (void)gate;
  (void)rows; (void)dim; (void)stream;
#endif
}

void gate_res_adarms_nvfp4_native_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* previous_gate,
    __nv_bfloat16* residual, const __nv_bfloat16* style,
    uint8_t* packed, uint8_t* sfa, __nv_bfloat16* gate,
    int rows, int dim, cudaStream_t stream) {
#if FLASHRT_ADARMS_BF16_HAVE_CUTLASS
  auto shape = cute::make_shape(rows, 1, dim, 1);
  auto layout = AdaRmsBf16Config::tile_atom_to_shape_SFA(shape);
  adarms_nvfp4_bf16_kernel<true><<<rows, 256, 0, stream>>>(
      x, previous_gate, residual, style, packed, sfa, gate, layout, dim);
  check_adarms_bf16_launch("gate_res_adarms_nvfp4_native_bf16");
#else
  (void)x; (void)previous_gate; (void)residual; (void)style;
  (void)packed; (void)sfa; (void)gate; (void)rows; (void)dim; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
