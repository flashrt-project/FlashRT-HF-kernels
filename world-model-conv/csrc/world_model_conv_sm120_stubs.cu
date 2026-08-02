// SPDX-License-Identifier: Apache-2.0
// Link-time stub for the SM110-only BF16 probe in SM120 artifacts.

#include "world_model_conv.cuh"

namespace flash_rt {
namespace conv {

extern "C" int bf16_conv3d_v0_ndhwc_bf16out(
    const void*, const void*, const void*, void*, const void*, int, int, int,
    int, int, int, int, float, cudaStream_t) {
  return -120;
}

}  // namespace conv
}  // namespace flash_rt
