// SPDX-License-Identifier: Apache-2.0
#include "fused_fp4/dequantize_fp4_sfa.cuh"

#include <cuda_fp8.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fused_fp4 {

#if FV_HAVE_CUTLASS

using CfgDequant = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ float e2m1_to_fp32_dequant(uint8_t value) {
  static constexpr float mags[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float mag = mags[value & 0x7];
  return (value & 0x8) ? -mag : mag;
}

template <class LayoutSF>
__global__ void dequantize_fp4_sfa_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ sfa,
    __half* __restrict__ out,
    LayoutSF layout,
    int dim) {
  int row = blockIdx.y;
  int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n_blocks = dim / 16;
  if (block_idx >= n_blocks) return;

  int col_base = block_idx * 16;
  int sfa_off = layout(row, col_base, 0);
  __nv_fp8_e4m3 scale_q;
  *reinterpret_cast<uint8_t*>(&scale_q) = sfa[sfa_off];
  float scale = static_cast<float>(scale_q);

  const uint8_t* packed_block = packed + row * (dim / 2) + block_idx * 8;
  __half* out_block = out + row * dim + col_base;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    uint8_t byte = packed_block[p];
    out_block[2 * p] = __float2half(e2m1_to_fp32_dequant(byte & 0xF) * scale);
    out_block[2 * p + 1] = __float2half(e2m1_to_fp32_dequant(byte >> 4) * scale);
  }
}

#endif

void dequantize_fp4_sfa_fp16(
    const uint8_t* packed,
    const uint8_t* sfa,
    __half* out,
    int rows,
    int dim,
    cudaStream_t stream) {
#if FV_HAVE_CUTLASS
  auto shape = cute::make_shape(rows, 1, dim, 1);
  auto layout = CfgDequant::tile_atom_to_shape_SFA(shape);
  int n_blocks = dim / 16;
  dim3 block(256);
  dim3 grid((n_blocks + block.x - 1) / block.x, rows);
  dequantize_fp4_sfa_kernel<<<grid, block, 0, stream>>>(
      packed, sfa, out, layout, dim);
#else
  (void)packed; (void)sfa; (void)out; (void)rows; (void)dim; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
