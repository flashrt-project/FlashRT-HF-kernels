#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// Integer form of CUTLASS Sm1xxBlockScaledConfig<16>'s SFA/SFB layout.
// The physical atom covers 128 rows by 64 K elements and stores one byte
// per 16-element K group. Both SFA(M,K) and SFB(N,K) use this mapping.
__host__ __device__ __forceinline__ int sfa_offset_128x64(
    int row, int k, int dim) {
  const int row_block = row >> 7;
  const int row_in_block = row & 127;
  const int k_block = k >> 6;
  const int k_in_block = k & 63;
  const int k_blocks = (dim + 63) >> 6;
  return row_block * k_blocks * 512 +
      k_block * 512 +
      (row_in_block & 31) * 16 +
      (row_in_block >> 5) * 4 +
      (k_in_block >> 4);
}

}  // namespace fp4
}  // namespace flash_rt
