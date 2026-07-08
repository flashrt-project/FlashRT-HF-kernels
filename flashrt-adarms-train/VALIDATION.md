# Validation

Required checks before publishing:

```bash
python flashrt-adarms-train/tests/test_flashrt_adarms_train.py --backend installed --mode full
python scripts/prebuild_check.py --package flashrt-adarms-train --check-config
```

After HF Jobs publishing:

```python
from kernels import get_kernel
ops = get_kernel("flashrt/flashrt-adarms-train", revision="v1")
```

Run the package test with the installed artifact path. Correctness Gate 1 must
remain zero-fail. Gate 2 is reported in `benchmarks/RESULTS.md`; do not convert
conditional rows into public performance claims.
