# Validation

Required source gate:

```bash
python transformer-fused-ops/tests/test_transformer_fused_ops.py --backend source --mode full
```

References are deterministic PyTorch eager equivalents. Current source gate
covers RMS-gated-SiLU, SiLU/sigmoid multiply, embedding lookup, partial RoPE,
argmax/spec accept, NexN2 split helpers, and router top-k.

ReLU2-to-FP8 is checked for bitwise FP8 equality over rows
`1/51/277/1024`, fullgraph compile compatibility and raw native parity.
