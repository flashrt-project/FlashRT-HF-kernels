# Validation

Required on NVIDIA Thor with CUDA 13 and CUTLASS DSL 4.4.x or 4.5.x:

```bash
python tests/test_fa4_cute_runtime.py --backend source --mode full
python tests/test_fa4_cute_runtime.py --backend installed --mode full \
  --artifact build/torch-cuda
```

The gate covers non-causal and causal GQA, head dimensions 48/72/128, and the
PI0.5 D256 encoder family. D256 cases include dense lengths 320/968 and padded
`seqused_k` lengths 456/712 in a 968-token buffer. Every case reports
max/p99/mean/cosine metrics and requires bitwise CUDA Graph replay after JIT
warmup.

On 2026-08-06, the current source snapshot passed `8/8` rows on NVIDIA Thor
(SM110), PyTorch `2.13.0+cu130`, CUDA 13.0, and CUTLASS DSL 4.4.2. D48/D72/D128
and all four D256 rows passed; worst p99 absolute error was `0.000244` and
worst cosine was `0.99999982`. CUDA Graph replay was bitwise deterministic.
The torch-noarch final-layout import probe also passed from the isolated Thor
environment.

Installed-artifact execution and performance must still be rerun after every
Hub rebuild. Source qualification alone is not a release artifact claim.
