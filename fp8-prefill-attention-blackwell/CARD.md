# Kernel card

## Callable API

`fp8_causal_gqa_attention_bf16(query, key, value, *, softmax_scale=None, out=None)`

- Query: contiguous CUDA FP8 E4M3FN `[S, 32, 128]`.
- Key/value: contiguous CUDA FP8 E4M3FN `[S, 8, 128]`.
- Output: contiguous CUDA BF16 `[S, 32, 128]`.
- Semantics: causal self-attention from position zero, GQA ratio 4:1.
- Supported sequence lengths: multiples of 128 from 256 upward. The native
  implementation is numerically valid at 128, but v1 rejects that regime
  because it did not beat the BF16 SDPA eager baseline on RTX 5090.
- Architecture: SM120/SM121 Blackwell; CUDA 12.8 or newer.

The package does not claim arbitrary head counts, head dimensions, ragged
sequences, cross-attention, backward, or pre-SM120 support in v1.

## Provenance

Packaged from FlashRT's native CUDA `fmha_fp8_causal_gqa_sm120` implementation.
The public Tensor API replaces FlashRT's serving-internal pointer/stream ABI.
