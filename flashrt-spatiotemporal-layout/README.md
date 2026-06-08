# flashrt-spatiotemporal-layout

Tensor-facing FlashRT spatiotemporal layout kernels for Hugging Face
`kernels`.

This package targets small but frequent VLA, video, diffusion, and world-model
layout operations that sit between attention, cache update, and block glue:

```text
NCDHW latent -> BLC token matrix
2C,T,H,W latent -> C,2T,H,W temporal unshuffle
NCDHW latent + channel bias
current latent + previous latent -> 2-frame cache
```

These kernels are not the headline FP8/GEMM speedup path. They are pipeline
glue kernels used to keep model-demo hot paths on CUDA with predictable Tensor
APIs and no Python tensor-layout chains.

## Exported APIs

- `ncdhw_to_blc_bf16(x, out=None)`
- `time_unshuffle2_bf16(x, out=None)`
- `add_bias_ncdhw_bf16(x, bias)`
- `update_cache2_ncdhw_bf16(cur, prev, out=None)`

## Tensor Conventions

- Inputs are contiguous CUDA BF16 tensors.
- `ncdhw_to_blc_bf16` converts `(B, C, T, H, W)` to `(B, T * H * W, C)`.
- `time_unshuffle2_bf16` converts `(B, 2 * C, T, H, W)` to
  `(B, C, 2 * T, H, W)`.
- `add_bias_ncdhw_bf16` adds a contiguous BF16 channel vector of shape `(C,)`
  in place to `(B, C, T, H, W)`.
- `update_cache2_ncdhw_bf16` writes a two-frame cache with shape
  `(B, C, 2, H, W)`. `prev` is the previous two-frame cache. If `cur` has one
  frame, `prev[:, :, 1]` supplies the first output cache frame and `cur`
  supplies the second. If `cur` has two or more frames, the last two frames of
  `cur` are copied.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel(
    "flashrt/flashrt-spatiotemporal-layout",
    version=1,
    trust_remote_code=True,
)

x = torch.randn((1, 64, 4, 32, 32), device="cuda", dtype=torch.bfloat16)
tokens = ops.ncdhw_to_blc_bf16(x)

x2 = torch.randn((1, 128, 4, 32, 32), device="cuda", dtype=torch.bfloat16)
expanded = ops.time_unshuffle2_bf16(x2)

bias = torch.randn((64,), device="cuda", dtype=torch.bfloat16)
ops.add_bias_ncdhw_bf16(x, bias)

prev = torch.randn((1, 64, 2, 32, 32), device="cuda", dtype=torch.bfloat16)
cache = ops.update_cache2_ncdhw_bf16(x[:, :, -1:, :, :].contiguous(), prev)
```

## Validation

```bash
python flashrt-spatiotemporal-layout/tests/test_spatiotemporal_layout.py --backend source --mode full
python flashrt-spatiotemporal-layout/benchmarks/benchmark.py --backend source --shapes all
```

Current RTX 5090 source-extension validation is bit-level exact against the
PyTorch eager reference across the source shape grid. Source benchmarks show
roughly `1.2x-6.4x` vs PyTorch eager layout/reference chains. Built-artifact
and multi-hardware validation are pending.
