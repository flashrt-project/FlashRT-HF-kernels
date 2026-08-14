# flashrt/sageattention3-blackwell

Speed-first FP4 self-attention for SM120a Blackwell GPUs, packaged from the
Apache-2.0 SageAttention3 implementation in `thu-ml/SageAttention`.

Use this package as an explicitly gated lower-precision tier for long video or
audio sequences. Its preferred `sage3_prefill_fp4_bf16` entry accepts raw BF16
NHD Q/K/V and owns Sage3 centering, padding, FP4 quantization, and correction
preparation behind a caller-owned CUDA Graph-safe workspace. Head dimension 64
is available in CUDA 12.8+ artifacts; head dimension 128 requires a CUDA 13.0+
artifact. Query `capabilities()` rather than assuming a head-dimension set. It
does not support GQA/MQA in v1 and does not replace
the higher-fidelity
`flashrt/sageattention2-blackwell` default.

## Published and experimental variants

The v1 artifacts currently publish one executable quantization tier:

- **Sage3 NVFP4/E2M1**: FP4 Q/K/V preparation plus block-scaled FP4
  attention, exposed through `sage3_prefill_fp4_bf16` and the lower-level
  `prepare_qkv_fp4_nhd` / `blockscaled_fp4_attention_static` APIs.

The **INT4RHT** tier is not present in the v1 binaries and is not callable yet.
It is a design-stage fallback for deployments that need an accuracy point
between SageAttention2 and Sage3 NVFP4. Its proposed contract is symmetric
INT4 Q/K with per-16-element floating-point scales, a shared Hadamard-128
rotation after RoPE and centering, and the SageAttention2 FP8 P/V path:

```python
out = sage3.sage3_prefill_int4rht_bf16(
    q, k, v, out=out_buffer, workspace=workspace,
)
```

That symbol will only be published after it passes all three gates: cosine at
least 0.9999 against the INT4RHT reference on random and real activations,
model-capture cosine at least 0.997, and all-in latency below SageAttention2
FP8V. Until then, callers must not probe or advertise it as a supported
variant.

See `README.md` for preprocessing, API usage, accuracy expectations, and the
complete capability contract.
