# sageattention2-blackwell

FlashRT SageAttention2-style native CUDA prefill attention kernels for
Blackwell GPUs.

This package covers prefill/self-attention workloads, not M=1 decode
attention. For decode over FP8 K/V cache, use `flashrt/fp8-kv-attention`.

The package exposes Tensor APIs for:

- Q/K BF16 -> int8 per-warp/per-block quantization.
- V BF16 -> FP16 contiguous layout.
- V BF16 -> FP8 transposed/padded Sage layout.
- Sage2 attention over already-quantized Q/K and FP16 or FP8 V.
- Convenience BF16 wrapper APIs that quantize and run attention in one call.
- Non-causal Wan/video self-attention and causal Qwen-style prefill.
- GQA shapes where `q_heads % kv_heads == 0`, including Qwen3-style `32/8`.

Source basis:

- FlashRT `official/FlashRT/csrc/attention/sage2/`
- SageAttention2 QK-int8/SV-fp8 and QK-int8/SV-fp16 core headers, Apache-2.0

The complete FlashRT runtime and serving pipeline live upstream at
[LiangSu8899/FlashRT](https://github.com/LiangSu8899/FlashRT).

## Available Functions

- `padded_k64(seqlen_k)`
- `q_scale_elems(batch, seqlen_q, q_heads)`
- `k_scale_elems(batch, seqlen_k, kv_heads)`
- `v_scale_elems(batch, kv_heads)`
- `quantize_q_bf16_d128(q, q_i8=None, q_scale=None)`
- `quantize_k_bf16_d128(k, k_i8=None, k_scale=None)`
- `quantize_v_fp16_bf16_d128(v, v_half=None)`
- `quantize_v_fp8_bf16_d128(v, v_fp8_tpp=None, v_scale=None)`
- `sage2_qk_int8_sv_f16_bf16_d128(q_i8, k_i8, v_half, q_scale, k_scale, softmax_scale=None, causal=False, out=None)`
- `sage2_qk_int8_sv_f8_bf16_d128(q_i8, k_i8, v_fp8_tpp, q_scale, k_scale, v_scale, softmax_scale=None, causal=False, out=None)`
- `sage2_prefill_f16_bf16_d128(q, k, v, softmax_scale=None, causal=False, out=None)`
- `sage2_prefill_fp8v_bf16_d128(q, k, v, softmax_scale=None, causal=False, out=None)`

## Tensor Contract

- `q`: contiguous CUDA BF16, shape `(batch, seqlen_q, q_heads, 128)`.
- `k`, `v`: contiguous CUDA BF16, shape `(batch, seqlen_k, kv_heads, 128)`.
- `q_heads % kv_heads == 0`.
- Output: contiguous CUDA BF16, shape `(batch, seqlen_q, q_heads, 128)`.
- `causal=False` for Wan/video non-causal self-attention.
- `causal=True` for Qwen-style prefill self-attention.
- Build target: CUDA 12.8+ Blackwell (`sm_120`, `sm_120a`, `sm_121`).

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/sageattention2-blackwell", version=1, trust_remote_code=True)

q = torch.randn((1, 4096, 32, 128), device="cuda", dtype=torch.bfloat16)
k = torch.randn((1, 4096, 8, 128), device="cuda", dtype=torch.bfloat16)
v = torch.randn((1, 4096, 8, 128), device="cuda", dtype=torch.bfloat16)

out = ops.sage2_prefill_f16_bf16_d128(q, k, v, causal=True)
```

Static-buffer/core usage:

```python
q_i8, q_scale = ops.quantize_q_bf16_d128(q)
k_i8, k_scale = ops.quantize_k_bf16_d128(k)
v_half = ops.quantize_v_fp16_bf16_d128(v)

out = torch.empty_like(q)
ops.sage2_qk_int8_sv_f16_bf16_d128(
    q_i8, k_i8, v_half, q_scale, k_scale, causal=True, out=out
)
```

## Validation

```bash
python sageattention2-blackwell/tests/test_sageattention2_blackwell.py --backend source --mode full
python sageattention2-blackwell/benchmarks/benchmark.py --backend source --mode full
```

Correctness is measured against PyTorch SDPA with the same BF16 Q/K/V inputs.
This is a quantized attention kernel, so validation uses numerical gates
instead of bit-exact equality.

Current RTX 5090 source-build benchmark rows are in `benchmarks/RESULTS.md`.
