# Validation

Required checks before publishing:

```bash
python tests/test_transformer_layout_primitives.py --backend source --mode full
python benchmarks/benchmark.py --backend source --mode headline
python scripts/prebuild_check.py --package transformer-layout-primitives --check-config
```

After HF Jobs publishing:

```python
from kernels import get_kernel
ops = get_kernel("flashrt/transformer-layout-primitives", version=1)
```

Run a smoke call on `repeat_interleave_heads_bf16` and
`qk_pair_rmsnorm_rope_bf16`.

The full source gate covers:

- head dimensions 64, 72, 80 and 128;
- unequal Q/K head counts;
- rows 1, 17, 49, 51, 65, 277, 512, 2520 and 5070;
- exact pair-vs-staged parity;
- valid RoPE tables against the PyTorch reference;
- mismatched rows, invalid head dimensions and non-contiguous rejection.
- indexed gather/scatter rows `51/105` from sources `277/2520`, bitwise raw
  native parity, fullgraph compile comparison and invalid-index rejection.
