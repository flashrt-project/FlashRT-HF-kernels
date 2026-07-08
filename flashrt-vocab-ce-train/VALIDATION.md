# Validation

Required checks before publishing:

```bash
python flashrt-vocab-ce-train/tests/test_flashrt_vocab_ce_train.py --backend installed --mode full
python scripts/prebuild_check.py --package flashrt-vocab-ce-train --check-config
```

After HF Jobs publishing:

```python
from kernels import get_kernel
ops = get_kernel("flashrt/flashrt-vocab-ce-train", version=1)
```

Run the package test with the installed artifact path. Performance acceptance is not claimed until the CUDA optimized implementation clears `kernel_acceptance_specs.md`.
