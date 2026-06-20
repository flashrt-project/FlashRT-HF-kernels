# TurboQuant KV

FlashRT CUDA kernels for TurboQuant-style KV cache unpack and K/V combine.

This package exposes the core tensor APIs needed by a `transformers` or
serving-style KV compression demo. It intentionally does not expose placeholder
write-side or GEMM orchestration APIs in v1.

## Available Functions

- `unpack_packed_bf16`: packed K/V indices and QJL signs to BF16 `(M, 256)` tensors.
- `unpack_packed_mixed`: packed K/V indices to BF16 plus QJL signs to FP32.
- `combine_kv_bf16`: combine GEMM outputs and per-row norms into BF16 K/V tensors.

## Usage

```python
from kernels import get_kernel

tq = get_kernel("flashrt/turboquant-kv")

y_k, qjl_bf, y_v = tq.unpack_packed_bf16(
    k_idx_packed, k_qjl_packed, v_idx_packed, cb_k_mse, cb_v, b_k_mse=3, b_v=4
)
k_out, v_out = tq.combine_kv_bf16(
    k_mse, k_qjl, v_unit, k_norm, k_rnorm, v_norm, coef=0.125
)
```

Shape contracts:

- packed K/V indices: `(M, 128)` uint8
- packed QJL signs: `(M, 32)` uint8
- codebooks: FP32 tensors with at least `2**bits` entries
- unpack outputs: `(M, 256)`
- combine inputs/outputs: `(M, 256)` BF16 plus `(M,)` FP16 norms
