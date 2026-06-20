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

Installed-artifact validation is required after HF Jobs publishes the package.
First publish uses the repository workflow's GitHub-runner
`create_repo(..., repo_type="kernel", exist_ok=True)` step before launching
the HF Jobs builder.
