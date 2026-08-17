---
library_name: kernels
license: apache-2.0
tags:
  - cuda
  - native-cuda
  - flashrt
  - transformer
  - fused-ops
---

# transformer-fused-ops

Native CUDA transformer helper kernels from FlashRT for fused activation,
layout, RoPE, argmax/spec-accept, and router top-k work.

See `README.md` for the public function list.

The public surface includes `relu2_quantize_fp8_static_bf16`, a fused BF16
ReLU-squared to FP8 producer.

`rms_norm_gated_silu_quant_fp4_bf16` preserves the BF16 gated-RMSNorm output
and emits its flattened NVFP4 packed tensor and 128x64-atom scale-factor
buffer in the same launch. It supports contiguous `(rows, 128)` BF16 inputs
and caller-provided static output buffers for CUDA Graph capture.

The SM110 BF16 producer surface also includes `quantize_fp8_static_bf16`,
`layer_norm_quant_fp8_static_bf16`, and
`gate_geglu_merged_quant_fp8_static_bf16` for PI0.5 prefill and SigLIP MLP
pipelines.

The SM110 surface also includes `rms_norm_fp16`, `layer_norm_fp16`,
`layer_norm_quant_fp8_static_fp16`, `rope_rotate_half_fp16_`,
the explicit GROOT vec aliases (`rms_norm_fp16_vec`,
`layer_norm_fp16_vec`, `layer_norm_fp8_static_fp16_vec`,
`rope_rotate_half_fp16_vec`, `quantize_fp8_static_fp16_vec`,
`residual_add_fp16_vec`, `gpu_repeat_interleave_heads_vec`),
`quantize_fp8_static_fp16`, `residual_add_fp16_`, and
`repeat_interleave_heads_fp16` for GROOT N1.7 Thor static-buffer pipelines.

The MoE surface includes model-neutral `router_topk_bf16` and
`moe_weighted_sum_bf16_to_fp32`; the older `nexn2_router_topk_bf16` name remains
an exact compatibility alias.
