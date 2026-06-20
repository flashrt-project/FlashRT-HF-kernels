# FlashRT Linear Attention Primitives

CUDA kernels for small-M BF16 linear projections and linear-attention helper
ops used by transformer decode, verify, and Gated DeltaNet-style paths.

This package is intentionally not a full model runtime. It exposes generic
Tensor APIs suitable for Hugging Face Kernel Hub loading and for integration
inside static-buffer runtime demos.

## Functions

- `bf16_matvec(x, w, out=None)`: BF16 `(K,) @ (N,K).T -> (N,)`.
- `bf16_smallm_matmul(x, w, out=None)`: tuned BF16 AB96 projection,
  currently `2 <= M <= 4`, `N=96`, `K=5120`.
- `split_qkv_broadcast_bf16(packed, q_heads, kv_heads, v_heads, head_dim)`.
- `partial_rope_qk_bf16(q, k, cos, sin, rope_dim)`.
- `gated_delta_prepare_bf16(a, b, neg_exp_a_log, dt_bias, ...)`.

The BF16 matvec kernel includes tuned paths for `K=4096` and `K=5120`, plus a
generic fallback for other `K` values with `N >= 256`. The small-M matmul API is
deliberately restricted to the AB96 shape where this kernel is faster than
PyTorch eager. The QKV/gating helpers are shape-generic over rows and heads,
with Qwen3.6/Gated DeltaNet-style dimensions covered by the validation matrix.

## Example

```python
from kernels import get_kernel

lap = get_kernel("flashrt/linear-attention-primitives")

out = lap.bf16_smallm_matmul(x_bf16, weight_bf16)
q, k, v = lap.split_qkv_broadcast_bf16(packed, 16, 16, 48, 128)
g, beta = lap.gated_delta_prepare_bf16(a, b, neg_exp_a_log, dt_bias, heads=48)
```

For static-buffer/CUDA Graph runtimes, pass output tensors explicitly:

```python
lap.bf16_smallm_matmul(x_bf16, weight_bf16, out=linear_out)
```

## Validation

See `VALIDATION.md` and `benchmarks/RESULTS.md`.
