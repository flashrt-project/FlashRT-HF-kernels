# Validation

Local source validation on NVIDIA GeForce RTX 5090:

```bash
python fp4-gemm/tests/test_fp4_gemm.py \
  --backend source \
  --mode full \
  --json-out internal-tests/fp4-gemm-source-full.json
```

Result:

- `9/9` checks passed.
- Variants `0`, `1`, and `2` were checked.
- Correctness reference dequantizes the same FP4/SFA and FP4/SFB inputs used
  by the kernel, then computes PyTorch GEMM on those dequantized values.

| Shape | Variant | Max abs | Mean abs | P99 abs | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| M=16, N=128, K=128 | 0 | 0.0 | 0.0 | 0.0 | 1.0 |
| M=16, N=128, K=128 | 1 | 0.0 | 0.0 | 0.0 | 1.0 |
| M=16, N=128, K=128 | 2 | 0.0 | 0.0 | 0.0 | 1.0 |
| M=32, N=256, K=256 | 0 | 0.0 | 0.0 | 0.0 | 1.0 |
| M=32, N=256, K=256 | 1 | 0.0 | 0.0 | 0.0 | 1.0 |
| M=32, N=256, K=256 | 2 | 0.0 | 0.0 | 0.0 | 1.0 |
| M=64, N=512, K=512 | 0 | 0.0 | 0.0 | 0.0 | 1.0 |
| M=64, N=512, K=512 | 1 | 0.0 | 0.0 | 0.0 | 1.0 |
| M=64, N=512, K=512 | 2 | 0.0 | 0.0 | 0.0 | 1.0 |

## HF Jobs Publish Status

`flashrt/fp4-gemm` v1 was built and uploaded through the repository HF Jobs
workflow after the package flake was updated to the current
`genKernelFlakeOutputs` builder API.

- Hub revision checked on June 20, 2026: `4fb53e52`
- Uploaded variants:
  - `torch211-cxx11-cu128-x86_64-linux`
  - `torch211-cxx11-cu130-x86_64-linux`
  - `torch212-cxx11-cu130-x86_64-linux`
  - `torch212-cxx11-cu132-x86_64-linux`

Installed-artifact correctness through `get_kernel("flashrt/fp4-gemm")`
should be rerun in a torch211 or torch212 CUDA environment. The local
development environment used for the source tests is PyTorch 2.9.1+cu128,
which intentionally does not match the uploaded torch211/torch212 variants.
