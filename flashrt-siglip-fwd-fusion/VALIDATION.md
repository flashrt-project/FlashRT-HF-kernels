# Validation

Required checks before publishing:

```bash
python flashrt-siglip-fwd-fusion/tests/test_flashrt_siglip_fwd_fusion.py --backend installed --mode full
python scripts/prebuild_check.py --package flashrt-siglip-fwd-fusion --check-config
```

After HF Jobs publishing:

```python
from kernels import get_kernel
ops = get_kernel("flashrt/flashrt-siglip-fwd-fusion", version=1)
```

Run the package test with the installed artifact path. Performance acceptance is not claimed until the CUDA optimized implementation clears `kernel_acceptance_specs.md`.
