# Validation

Local validation covers:

- BF16 matvec and small-M matmul for `K=128,4096,5120`.
- Rows `M=2,3,4` for the tuned AB96 matmul path and representative matvec
  `N=512,1024`.
- Q/K broadcast layouts for decode/prefill rows.
- Partial RoPE for `head_dim=128/256` and `rope_dim=64/128`.
- Gated DeltaNet prepare for contiguous and strided `a/b` tensors.

Correctness records max error, mean error, and cosine similarity. Layout,
RoPE, and gating helpers are checked against PyTorch references with exact
BF16 output matching. BF16 linear projections are checked against an FP32
accumulation reference converted to BF16, with tight BF16-production tolerance.

Run:

```bash
python tests/test_linear_attention_primitives.py --backend source
python benchmarks/benchmark.py --backend source --warmup 100 --iters 1000
kernel-builder check-config .
```
