# Validation

Required source gate:

```bash
python transformer-fused-ops/tests/test_transformer_fused_ops.py --backend source --mode full
```

References are deterministic PyTorch eager equivalents. Current source gate
covers RMS-gated-SiLU, SiLU/sigmoid multiply, embedding lookup, partial RoPE,
argmax/spec accept, NexN2 split helpers, router top-k, and MoE weighted gather.

MoE weighted-sum rows cover `(tokens, topk, hidden, stride)` values
`(1,1,128,128)`, `(3,4,320,384)`, and `(17,8,2048,2112)`, including padded
expert-row strides. The reference gathers the same routed rows and accumulates
in FP32; max/p99/mean error, cosine, dtype, and shape are gated.

ReLU2-to-FP8 is checked for bitwise FP8 equality over rows
`1/51/277/1024`, fullgraph compile compatibility and raw native parity.
