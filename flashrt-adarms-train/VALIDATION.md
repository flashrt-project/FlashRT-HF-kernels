# Validation

Required checks before publishing:

```bash
python flashrt-adarms-train/tests/test_flashrt_adarms_train.py --backend installed --mode full
python scripts/prebuild_check.py --package flashrt-adarms-train --check-config
```

After HF Jobs publishing:

```python
from kernels import get_kernel
ops = get_kernel("flashrt/flashrt-adarms-train", version=1)
```

Run the package test with the installed artifact path. Performance acceptance is not claimed until the CUDA optimized implementation clears `kernel_acceptance_specs.md`.
