// SPDX-License-Identifier: Apache-2.0
// Link-time stubs for SM120-only entry points in the SM110 artifact.

#include "world_model_conv.cuh"

namespace flash_rt {
namespace conv {

#define FLASHRT_WEAK_STUB extern "C" __attribute__((weak)) int

FLASHRT_WEAK_STUB fp8_conv3d_v18_ncdhw_res_bf16out(
    const void*, const void*, const void*, void*, const void*, const void*,
    int, int, int, int, int, int, int, float, cudaStream_t) { return -110; }
FLASHRT_WEAK_STUB fp8_conv3d_v17_ndhwc_bf16out(
    const void*, const void*, const void*, void*, const void*, int, int, int,
    int, int, int, int, float, cudaStream_t) { return -110; }
FLASHRT_WEAK_STUB fp8_conv3d_v17_anyco_ndhwc_bf16out(
    const void*, const void*, const void*, void*, const void*, int, int, int,
    int, int, int, int, float, cudaStream_t) { return -110; }
FLASHRT_WEAK_STUB fp8_conv2d_3x3_v2_nhwc_bf16out(
    const void*, const void*, void*, const void*, int, int, int, int, int,
    float, cudaStream_t) { return -110; }
FLASHRT_WEAK_STUB fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out(
    const void*, const void*, void*, const void*, int, int, int, int, int,
    int, float, cudaStream_t) { return -110; }
FLASHRT_WEAK_STUB motus_fp4_conv3d_v19sf_ndhwc_bf16out(
    const void*, const void*, const void*, const void*, const void*,
    const void*, void*, const void*, int, int, int, int, int, int, int,
    float, cudaStream_t) { return -110; }
FLASHRT_WEAK_STUB motus_fp4_conv3d_v19sf_ndhwc_bf16out_v2(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, void*, const void*, int, int, int, int, int,
    int, int, float, cudaStream_t) { return -110; }
FLASHRT_WEAK_STUB motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out(
    const void*, const void*, const void*, const void*, const void*,
    const void*, void*, const void*, const void*, int, int, int, int, int,
    int, int, float, cudaStream_t) { return -110; }
FLASHRT_WEAK_STUB motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out_v2(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, void*, const void*, const void*, int, int, int,
    int, int, int, int, float, cudaStream_t) { return -110; }
FLASHRT_WEAK_STUB motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out(
    const void*, const void*, const void*, const void*, const void*,
    const void*, void*, const void*, const void*, int, int, int, int, int,
    int, int, float, cudaStream_t) { return -110; }

#undef FLASHRT_WEAK_STUB

}  // namespace conv
}  // namespace flash_rt
