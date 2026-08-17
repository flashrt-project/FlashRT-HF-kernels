# Validation

Required source gate:

```bash
python transformer-fused-ops/tests/test_transformer_fused_ops.py --backend source --mode full
```

References are deterministic PyTorch eager equivalents. Current source gate
covers RMS-gated-SiLU, SiLU/sigmoid multiply, embedding lookup, partial RoPE,
argmax/spec accept, NexN2 split helpers, router top-k, and MoE weighted gather.

On SM110 the same full gate additionally covers the GROOT N1.7 vector family:
FP16 RMSNorm/LayerNorm, fused LayerNorm-to-FP8 staged equality, split-half
RoPE, static FP8 quantization, in-place residual add, and GQA head expansion.
The production rows include `M=41`, `S=277`, hidden dimensions `128/1536`,
and a CUDA-Graph-safe static-buffer contract.

The public GROOT names `rms_norm_fp16_vec`, `layer_norm_fp16_vec`,
`layer_norm_fp8_static_fp16_vec`, `rope_rotate_half_fp16_vec`,
`quantize_fp8_static_fp16_vec`, `residual_add_fp16_vec`, and
`gpu_repeat_interleave_heads_vec` are additive aliases over the established
native dispatch. Release gating requires bitwise alias parity and no more than
5% latency regression under preallocated CUDA Graph replay.

The additive PI0.5/SigLIP BF16 producer gate covers:

- static FP8 quantization, including a non-vector-aligned `1x127` tail;
- fused BF16 LayerNorm to FP8 at `512x1152`, `712x2048`, and `768x4304`;
- merged tanh-GeGLU to FP8 at hidden dimensions `4096/4304/3456`;
- `torch.compile(fullgraph=True)` and CUDA Graph replay for all three API
  families.

The fused `rms_norm_gated_silu_quant_fp4_bf16` gate covers rows
`1/8/64/257`. Its BF16 output must exactly equal the established package
RMSNorm-gated-SiLU entry. Packed FP4 values and scale-factor bytes must exactly
equal the production `fp4-gemm` quantizer, compiled into the test as an
independent native oracle. The full installed-artifact gate also requires
bit-identical CUDA Graph replay and `torch.compile(fullgraph=True)` execution.

Cold Hub artifacts passed `28/28` on both
`torch211-cxx11-cu128-x86_64-linux` and
`torch213-cxx11-cu130-x86_64-linux`. The latter is the previously missing ABI
variant. The older kernels 0.10.5 client also loaded and launched the new entry
through the unversioned `main` alias.

The August 8 Thor source and clean installed-artifact gates passed `67/67`;
the representative producer benchmark reported FP8 p99 error zero and
`5.97x..12.20x` over PyTorch eager. Explicit vector-entry latency was
`0.917x..1.021x` of the corresponding established package entry.

MoE weighted-sum rows cover `(tokens, topk, hidden, stride)` values
`(1,1,128,128)`, `(3,4,320,384)`, and `(17,8,2048,2112)`, including padded
expert-row strides. The reference gathers the same routed rows and accumulates
in FP32; max/p99/mean error, cosine, dtype, and shape are gated.

ReLU2-to-FP8 is checked for bitwise FP8 equality over rows
`1/51/277/1024`, fullgraph compile compatibility and raw native parity.
