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
- `nvfp4_gemm_bf16` is the canonical public API.
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

## Installed Artifact Validation

The local kernel-builder release candidate produced and passed ABI, manylinux,
layout, and builder `get_kernel` checks for:

- `torch211-cxx11-cu128-x86_64-linux`
- `torch211-cxx11-cu130-x86_64-linux`
- `torch212-cxx11-cu130-x86_64-linux`
- `torch212-cxx11-cu132-x86_64-linux`

The cu128/Torch 2.11 artifact passed `10/10` runtime gates: all nine
shape/variant correctness rows were exact against the staged reference, and
the public `nvfp4_gemm_bf16` wrapper was exact under
`torch.compile(fullgraph=True)`.

The release flake pins upstream kernel-builder commit
`19aaa6421e674e9fecc352bbae6eab81d19a6bf4`, adding the eligible Torch 2.13
cu130/cu132 variants. HF Jobs and a cold Hub load must pass before the rebuilt
Hub release is considered complete.
