# grouped-moe-gemv

FlashRT native CUDA W4A16 GEMV kernels for MoE decode and routed-slot prefill.

## Functions

- `w4a16_decode_gemv_bf16(x_bf16, weight_packed, sfb, alpha=1.0, out=None)`
- `grouped_w4a16_gemv_bf16(activations, weight_stack, sfb_stack, alpha_stack, expert_idx, n, w_stride=None, sfb_stride=None, out=None)`

The grouped API runs one BF16-activation x NVFP4-weight GEMV per routed slot.
It is intended for static routed expert batches where the caller already owns
packed weights and swizzled scale-factor buffers.

## Validation

```bash
python grouped-moe-gemv/tests/test_grouped_moe_gemv.py --backend source --mode full
```
