#pragma once

#include "kernels/vec_fp16_backbone.cuh"

namespace flash_rt::hub {

using RmsNormFp16Dispatch = decltype(&rms_norm_fp16_vec);
using LayerNormFp16Dispatch = decltype(&layer_norm_fp16_vec);
using LayerNormFp8Fp16Dispatch = decltype(&layer_norm_fp8_static_fp16_vec);
using RopeFp16Dispatch = decltype(&rope_rotate_half_fp16_vec);
using QuantizeFp8Fp16Dispatch = decltype(&quantize_fp8_static_fp16_vec);
using ResidualAddFp16Dispatch = decltype(&residual_add_fp16_vec);
using RepeatHeadsFp16Dispatch = decltype(&gpu_repeat_interleave_heads_vec);

extern RmsNormFp16Dispatch rms_norm_fp16_dispatch;
extern LayerNormFp16Dispatch layer_norm_fp16_dispatch;
extern LayerNormFp8Fp16Dispatch layer_norm_fp8_fp16_dispatch;
extern RopeFp16Dispatch rope_fp16_dispatch;
extern QuantizeFp8Fp16Dispatch quantize_fp8_fp16_dispatch;
extern ResidualAddFp16Dispatch residual_add_fp16_dispatch;
extern RepeatHeadsFp16Dispatch repeat_heads_fp16_dispatch;

}  // namespace flash_rt::hub
