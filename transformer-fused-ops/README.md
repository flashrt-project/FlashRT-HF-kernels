# transformer-fused-ops

FlashRT native CUDA fused helper kernels for transformer hot paths.

## Functions

- `rms_norm_gated_silu_bf16`
- `rms_norm_gated_silu_quant_fp4_bf16`
- `silu_mul_bf16`
- `sigmoid_mul_bf16`
- `per_head_sigmoid_gate_bf16` (`NHD * 2*sigmoid([B,T,H])`, caller-owned `out=`)
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
- `rms_norm_fp16(x, weight, eps=1e-6, out=None)`
- `rms_norm_fp16_vec(x, weight, eps=1e-6, out=None)`
- `layer_norm_fp16(x, weight, bias, eps=1e-6, out=None)`
- `layer_norm_fp16_vec(x, weight, bias, eps=1e-6, out=None)`
- `layer_norm_quant_fp8_static_fp16(x, weight, bias, scale, eps=1e-6, out=None)`
- `layer_norm_fp8_static_fp16_vec(x, weight, bias, scale, eps=1e-6, out=None)`
- `rope_rotate_half_fp16_(x, cos, sin)`
- `rope_rotate_half_fp16_vec(x, cos, sin)`
- `quantize_fp8_static_fp16(x, scale, out=None)`
- `quantize_fp8_static_fp16_vec(x, scale, out=None)`
- `quantize_fp8_static_bf16(x, scale, out=None)`
- `layer_norm_quant_fp8_static_bf16(x, weight, bias, scale, eps=1e-6, out=None)`
- `gate_geglu_merged_quant_fp8_static_bf16(merged, scale, out=None)`
- `residual_add_fp16_(residual, x)`
- `residual_add_fp16_vec(residual, x)`
- `repeat_interleave_heads_fp16(x, repeat, out=None)`
- `gpu_repeat_interleave_heads_vec(x, repeat, out=None)`

These are Tensor APIs meant for static-buffer runtimes and CUDA Graph friendly
model demos. Unsupported shapes fail explicitly.

`relu2_quantize_fp8_static_bf16` fuses ReLU-squared and static-scale FP8 E4M3
quantization. Pass `out=` for static-buffer and CUDA Graph runtimes.

`router_topk_bf16` is the model-neutral alias for the existing deterministic
router contract. `moe_weighted_sum_bf16_to_fp32` gathers routed expert rows and
accumulates router-weighted BF16 expert outputs into an FP32 token output.

The FP16 vector family is the native GROOT N1.7 Thor hot path. It covers
ViT/LLM normalization, split-half RoPE, FP8 production, residual update, and
GQA head expansion. These entries require SM110 and CUDA 13; unsupported
architectures fail before launch. Static `out=` buffers and the in-place
entries are suitable for CUDA Graph replay.

The explicit `_vec` names are additive aliases for runtime capability
discovery. They dispatch the exact same native CUDA implementations as the
established model-neutral names; callers can bind either surface without a
performance or numerical change.

The additive SM110 BF16 producer family covers PI0.5 prefill and SigLIP hot
paths. `quantize_fp8_static_bf16` computes
`clamp(x / scale, -448, 448)` into FP8 E4M3. The LayerNorm entry performs FP32
statistics, rounds the normalized value to BF16, and emits FP8 without an
intermediate tensor. The merged GeGLU entry splits `(M,2H)` into gate/up,
applies the native tanh-GELU approximation, multiplies by up, and emits static
FP8 `(M,H)`. All three require SM110 and support static `out=` buffers.

## Validation

```bash
python transformer-fused-ops/tests/test_transformer_fused_ops.py --backend source --mode full
```

The full SM110 grid includes non-vector-aligned tails and PI0.5/SigLIP shapes,
plus `torch.compile(fullgraph=True)` and CUDA Graph replay gates.
