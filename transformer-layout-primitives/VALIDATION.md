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
`qk_rmsnorm_rope_bf16_`.
