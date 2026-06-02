# Model Block Note: Packed-QKV Post-Processing

This note describes the v1 `flashrt-vla-video` showcase path in model-block
terms. The public API remains model-agnostic.

## Replaced Operation Sequence

Many VLA, vision-language, and video-decoder blocks produce a packed QKV tensor
and then run several small post-processing operations before attention:

```python
q = packed_qkv[..., :dim].reshape(batch, tokens, heads, head_dim)
k = packed_qkv[..., dim : 2 * dim].reshape(batch, tokens, heads, head_dim)

q = rmsnorm(q, norm_q_weight)
k = rmsnorm(k, norm_k_weight)

q = apply_interleaved_rope(q, freqs_re, freqs_im)
k = apply_interleaved_rope(k, freqs_re, freqs_im)
```

`qkv_split_norm_rope_bf16` replaces the split, Q/K RMSNorm, and interleaved
RoPE steps with one fused Tensor API:

```python
q, k = ops.qkv_split_norm_rope_bf16(
    packed_qkv,
    norm_q_weight,
    norm_k_weight,
    freqs_re,
    freqs_im,
    heads=heads,
    head_dim=128,
)
```

## V1 Shape Focus

The first v1 grid focuses on:

- batch `1`;
- token counts `1,4,16,64,256,1024,2520,4096`;
- heads `8,16,24,32,48`;
- `head_dim=128`;
- BF16 packed QKV input.

These shapes cover decode, short-context, image-token, and video-token
post-processing regimes.

## Current Local Evidence

Local RTX 5090 source-extension benchmarks show:

- about `39x` for token counts `1,4,16,64` with `heads=24`;
- about `19-29x` for long video-token counts `1024,2520,4096`;
- about `26-40x` across the `heads=8,16,32,48` sweep at `tokens=64`.

The release table must be refreshed with built-artifact HF benchmark runner
results before public upload.

## Claim Boundary

This is a launch-bound fusion win over a PyTorch eager sequence. It should be
presented as a post-processing block speedup, not as an end-to-end model
throughput claim until an actual model-level benchmark is recorded.
