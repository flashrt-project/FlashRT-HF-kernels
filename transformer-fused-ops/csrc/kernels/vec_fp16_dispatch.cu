#include "kernels/vec_fp16_dispatch.cuh"

namespace flash_rt::hub {
namespace {

struct VecFp16Registration {
  VecFp16Registration() {
    rms_norm_fp16_dispatch = &rms_norm_fp16_vec;
    layer_norm_fp16_dispatch = &layer_norm_fp16_vec;
    layer_norm_fp8_fp16_dispatch = &layer_norm_fp8_static_fp16_vec;
    rope_fp16_dispatch = &rope_rotate_half_fp16_vec;
    quantize_fp8_fp16_dispatch = &quantize_fp8_static_fp16_vec;
    residual_add_fp16_dispatch = &residual_add_fp16_vec;
    repeat_heads_fp16_dispatch = &gpu_repeat_interleave_heads_vec;
  }
};

VecFp16Registration registration;

}  // namespace
}  // namespace flash_rt::hub
