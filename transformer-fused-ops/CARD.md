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

The MoE surface includes model-neutral `router_topk_bf16` and
`moe_weighted_sum_bf16_to_fp32`; the older `nexn2_router_topk_bf16` name remains
an exact compatibility alias.
