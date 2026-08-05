# Validation

Required on NVIDIA Thor with PyTorch 2.11, CUDA 13, and CUTLASS DSL 4.5.0:

```bash
python tests/test_fa4_cute_runtime.py --backend source --mode full
python tests/test_fa4_cute_runtime.py --backend installed --mode full \
  --artifact build/torch-cuda
```

The gate covers non-causal and causal GQA, head dimensions 48/72/128, real
GROOT sequence families, max/p99/mean/cosine metrics, and bitwise CUDA Graph
replay after JIT warmup.

Source qualification passed on SM110 with both CUTLASS DSL 4.4.2 and the
builder's 4.5.0 dependency. Four of four real-shape cases passed; worst cosine
was `0.99999982` and worst p99 absolute error was `0.000244`. The
`forward_static` output matched the public wrapper exactly, and repeated CUDA
Graph replay was bitwise deterministic.

Installed-artifact execution and performance must still be rerun after every
Hub rebuild. Source qualification alone is not a release artifact claim.
