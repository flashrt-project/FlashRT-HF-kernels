# Validation

Required source gate:

```bash
python transformer-fused-ops/tests/test_transformer_fused_ops.py --backend source --mode full
```

References are deterministic PyTorch eager equivalents. Current source gate
covers RMS-gated-SiLU, SiLU/sigmoid multiply, embedding lookup, partial RoPE,
argmax/spec accept, NexN2 split helpers, and router top-k.
