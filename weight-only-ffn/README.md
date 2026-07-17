# weight-only-ffn

Native CUDA small-M FFN regions with BF16 activations and static W4 or W8
weights. The activation remains BF16; this package does not insert activation
quantization or change the mathematical region into A4W4/A8W8.

The production auto dispatch is intentionally limited to qualified shapes in
`M=1..4`. It also checks weight dimensions because W4 does not win every narrow
projection. Unsupported shapes raise an error instead of silently selecting a
slower compatibility kernel.

## Load from Kernel Hub

```python
import torch
from kernels import get_kernel

ops = get_kernel(
    "flashrt/weight-only-ffn",
    version=1,
    trust_remote_code=True,
)

x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
gate_up = torch.randn(22016, 4096, device="cuda", dtype=torch.bfloat16)
down = torch.randn(4096, 11008, device="cuda", dtype=torch.bfloat16)

gate_up_w4, gate_up_sfb = ops.quantize_w4_weight_bf16(gate_up)
down_w4, down_sfb = ops.quantize_w4_weight_bf16(down)

y = ops.w4a16_swiglu_ffn_bf16(
    x,
    gate_up_w4,
    gate_up_sfb,
    down_w4,
    down_sfb,
)
```

Quantize weights once during model loading. Do not quantize static weights in
the inference hot path.

## Public API

Weight preparation:

- `quantize_w4_weight_bf16(weight)`
- `dequantize_w4_weight_bf16(weight_packed, weight_sfb, cols=K)`
- `quantize_w8_weight_bf16(weight)`
- `dequantize_w8_weight_bf16(weight_int8, weight_scale)`

Linear primitives:

- `w4a16_linear_bf16`
- `w8a16_linear_bf16`

Complete FFN regions:

- `w4a16_swiglu_ffn_bf16`
- `w4a16_geglu_ffn_bf16`
- `w4a16_gelu_ffn_bf16`
- `w8a16_swiglu_ffn_bf16`
- `w8a16_geglu_ffn_bf16`
- `w8a16_gelu_ffn_bf16`

All compute APIs return BF16 tensors. W4 uses packed E2M1 data with swizzled
UE4M3 block-16 scales. W8 uses signed INT8 data with one FP32 scale per output
channel.

## Static Buffers And CUDA Graphs

Pass reusable buffers to avoid Python-side allocation in a captured region:

```python
gate_up_tmp = torch.empty(1, 22016, device="cuda", dtype=torch.bfloat16)
hidden_tmp = torch.empty(1, 11008, device="cuda", dtype=torch.bfloat16)
out = torch.empty(1, 4096, device="cuda", dtype=torch.bfloat16)

ops.w4a16_swiglu_ffn_bf16(
    x,
    gate_up_w4,
    gate_up_sfb,
    down_w4,
    down_sfb,
    workspace=(gate_up_tmp, hidden_tmp),
    out=out,
)
```

Run one eager warmup before CUDA Graph capture. The warmup initializes the W4
scale lookup table and all library state.

## Supported Domain

- Hardware: Blackwell SM120/SM121
- CUDA: 12.8 or newer; the exact Hub artifact depends on CUDA and Torch ABI
- Activations/output: contiguous BF16 matrices
- Static weights: W4 or W8 formats described above
- `K`: divisible by 64
- W8 production auto dispatch: qualified `M=1..4` shapes; narrow-output,
  large-`K` linear geometries are rejected
- W4 linear production auto dispatch: `M=1` with `N>=4096`, or `M=2`
  with `N>=2048`, `2048<=K<=8192`, and at least 8M weight elements
- W4 gated FFN production auto dispatch: `M<=3`, with a measured minimum
  static-weight footprint of 12M/32M/64M elements for `M=1/2/3`
- W4 GELU FFN production auto dispatch: `M<=3` and at least 64M static
  weight elements

`variant=1..3` exists for reproducible tile diagnostics. Applications should
use the default `variant=0`; diagnostic variants are not a compatibility
fallback.

See [VALIDATION.md](VALIDATION.md) and
[benchmarks/RESULTS.md](benchmarks/RESULTS.md) for the exact correctness and
performance contracts.

The complete model runtime and serving pipeline live in
[FlashRT](https://github.com/flashrt-project/FlashRT).
