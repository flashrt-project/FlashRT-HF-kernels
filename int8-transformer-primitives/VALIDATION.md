# Validation

Required checks before publishing:

```bash
python tests/test_int8_transformer_primitives.py --backend source --mode full
python benchmarks/benchmark.py --backend source --mode headline
python scripts/prebuild_check.py --package int8-transformer-primitives --check-config
```

After HF Jobs publishing:

```python
from kernels import get_kernel
ops = get_kernel("flashrt/int8-transformer-primitives", version=1)
```

Run a smoke call on `quantize_int8_rowwise_bf16` and
`int8_rowwise_linear_bf16`.
