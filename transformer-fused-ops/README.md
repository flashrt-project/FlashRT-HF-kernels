# transformer-fused-ops

FlashRT native CUDA fused helper kernels for transformer hot paths.

## Functions

- `rms_norm_gated_silu_bf16`
- `silu_mul_bf16`
- `sigmoid_mul_bf16`
- `embedding_lookup_bf16`
- `partial_rope_qk_bf16`
- `argmax_bf16`
- `spec_accept_greedy_bf16`
- `nexn2_lin_split_qkv_broadcast_bf16`
- `nexn2_split_q_gate_bf16`
- `nexn2_router_topk_bf16`
- `router_topk_bf16`
- `moe_weighted_sum_bf16_to_fp32`
- `relu2_quantize_fp8_static_bf16(input, scale, out=None)`

These are Tensor APIs meant for static-buffer runtimes and CUDA Graph friendly
model demos. Unsupported shapes fail explicitly.

`relu2_quantize_fp8_static_bf16` fuses ReLU-squared and static-scale FP8 E4M3
quantization. Pass `out=` for static-buffer and CUDA Graph runtimes.

`router_topk_bf16` is the model-neutral alias for the existing deterministic
router contract. `moe_weighted_sum_bf16_to_fp32` gathers routed expert rows and
accumulates router-weighted BF16 expert outputs into an FP32 token output.

## Validation

```bash
python transformer-fused-ops/tests/test_transformer_fused_ops.py --backend source --mode full
```
