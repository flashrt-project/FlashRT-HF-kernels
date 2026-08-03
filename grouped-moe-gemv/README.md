# grouped-moe-gemv

FlashRT native CUDA grouped expert projection kernels for Blackwell decode and
small verify batches. Version 2 adds W4A4 with device-side top-k routing while
preserving the version 1 W4A16 APIs.

## Functions

- `w4a16_decode_gemv_bf16(x_bf16, weight_packed, sfb, alpha=1.0, out=None)`
- `grouped_w4a16_gemv_bf16(activations, weight_stack, sfb_stack, alpha_stack, expert_idx, n, w_stride=None, sfb_stride=None, out=None)`
- `quantize_activations_nvfp4_bf16(activations, packed=None, sfa=None)`
- `quantize_weights_nvfp4_bf16(weights, packed=None, sfb=None)`
- `grouped_w4a4_gemv_bf16(activations_packed, weight_stack, sfa, sfb_stack, alpha_stack, expert_idx, out=None)`
- `grouped_w4a4_gemv_from_bf16(activations, weight_stack, sfb_stack, alpha_stack, expert_idx, packed=None, sfa=None, out=None)`

The grouped API runs one BF16-activation x NVFP4-weight GEMV per routed slot.
It is intended for static routed expert batches where the caller already owns
packed weights and swizzled scale-factor buffers.

The W4A4 API accepts packed activations `[M,K/2]`, expert weights
`[E,N,K/2]`, and a contiguous device routing tensor `[M,top_k]`. It emits
`[M,top_k,N]` in one grouped compute launch. For down projections with a
different activation per routed pair, flatten to `M=routed_pairs, top_k=1`.

`K` must be divisible by 16 and `N` by 8. Target `K%64==0` shapes use tuned
SM120 paths; the remaining `K%16` shapes use a fixed-order SIMT contract path.
No atomics, host synchronization, or dynamic workspace are used by the native
ops. Pass `packed`, `sfa`, and `out` buffers to the composed helper for
allocation-free CUDA Graph capture.

## Example

```python
from kernels import get_kernel
import torch

moe = get_kernel("flashrt/grouped-moe-gemv", version=2)

M, TOP_K, E, N, K = 7, 8, 8, 1024, 2048
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
expert_idx = torch.randint(E, (M, TOP_K), device="cuda", dtype=torch.int32)

def sf_bytes(rows, dim):
    return ((rows + 127) // 128) * (((dim // 16) + 3) // 4) * 512

# Do this once while loading the checkpoint, not in the inference hot path.
weights_bf16 = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
weights_packed = torch.empty(E, N, K // 2, device="cuda", dtype=torch.uint8)
weight_sfs = torch.empty(E, sf_bytes(N, K), device="cuda", dtype=torch.uint8)
for expert in range(E):
    moe.quantize_weights_nvfp4_bf16(
        weights_bf16[expert],
        packed=weights_packed[expert],
        sfb=weight_sfs[expert],
    )
weight_alpha = torch.ones(E, device="cuda", dtype=torch.float32)

packed = torch.empty(M, K // 2, device="cuda", dtype=torch.uint8)
sfa = torch.empty(sf_bytes(M, K), device="cuda", dtype=torch.uint8)
out = torch.empty(M, TOP_K, N, device="cuda", dtype=torch.bfloat16)
y = moe.grouped_w4a4_gemv_from_bf16(
    x, weights_packed, weight_sfs, weight_alpha, expert_idx,
    packed=packed, sfa=sfa, out=out,
)
```

The example buffers may be larger than the minimum; wrappers validate storage.
For production code, derive SF sizes from the checkpoint packer metadata.

## Dispatch guidance

Use the built-artifact benchmark to dispatch rather than selecting only by
dtype. On the tested cu128 artifact W4A16 wins gate-up, W4A4 wins down verify,
and down decode is effectively tied kernel-only. A fused/upstream FP4 producer
removes the standalone quantization charge, but callers still should not assume
lower precision is automatically faster.

## Validation

```bash
python grouped-moe-gemv/tests/test_grouped_moe_gemv.py --backend source --mode full
python grouped-moe-gemv/benchmarks/benchmark.py --backend source
```
