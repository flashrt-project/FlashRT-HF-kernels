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
