# Validation

Required checks before publishing:

```bash
python flashrt-qkv-epilogue-train/tests/test_flashrt_qkv_epilogue_train.py --backend installed --mode full
python scripts/prebuild_check.py --package flashrt-qkv-epilogue-train --check-config
```

After HF Jobs publishing:

```python
from kernels import get_kernel
ops = get_kernel("flashrt/flashrt-qkv-epilogue-train", version=1)
```

Run the package test with the installed artifact path. Performance acceptance is not claimed until the CUDA optimized implementation clears `kernel_acceptance_specs.md`.
