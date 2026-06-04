# Built Artifact Results

Validated on June 2-3, 2026 on NVIDIA GeForce RTX 5090.

This document records tests against copied `kernel-builder` artifacts, not
local source-extension builds.

## Release-Candidate Variant

Variant:

```text
torch211-cxx11-cu128-x86_64-linux
```

Artifact source:

- Built with `kernel-builder-docker build --variant torch211-cxx11-cu128-x86_64-linux`.
- Copied from the Docker container Nix store into each package's `build/`
  directory with `scripts/copy_docker_variant_artifacts.py`.
- The selected runtime-correctness artifact set was first validated from clean
  commit `21417e6`.
- The full v1 matrix was later validated from commit
  `d0a125d9d6eb88bf55c63712a78eeb7e12ab97e7` using a local
  `kernel-builder` Triton hash workaround. Those full-matrix artifacts are
  local validation artifacts and have a `_d0a125d_dirty` suffix.

Runtime validation environment:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Torch inside HF testshell: 2.11.0+cu128
- CUDA runtime inside Torch: 12.8
- `LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}` was required inside the
  GPU-enabled Docker container so Torch could find `libcuda.so`.
- The FP8 FFN validation used a privileged GPU testshell container because the
  original GPU container exposed `/dev/nvidia*` but CUDA driver init returned
  no device. The working container was launched with `--privileged --gpus all`.
- `torch.compile` inside the minimal testshell also required `/sbin/ldconfig`;
  the validation container provided it via a symlink to the Nix glibc
  `ldconfig`.

Command:

```bash
PYTHONPATH=flashrt-gemm-epilogues/build/torch211-cxx11-cu128-x86_64-linux:flashrt-fp8-ffn/build/torch211-cxx11-cu128-x86_64-linux:flashrt-vla-video/build/torch211-cxx11-cu128-x86_64-linux:flashrt-nvfp4/build/torch211-cxx11-cu128-x86_64-linux:flashrt-smallm-gemm/build/torch211-cxx11-cu128-x86_64-linux:flashrt-fused-quant/build/torch211-cxx11-cu128-x86_64-linux \
  python scripts/accuracy_sweep.py --backend installed --mode full --package all --smallm-max-ulp 5 --quiet
```

Result:

```text
accuracy sweep passed: 345 checks
```

## Accuracy Notes

- `flashrt-gemm-epilogues`: FP8 quant epilogue byte/exact parity passed over
  the v1 grid.
- `flashrt-vla-video`: BF16 Q/K tolerance and V byte parity passed over the v1
  grid.
- `flashrt-nvfp4`: scale-factor swizzle byte parity passed.
- `flashrt-smallm-gemm`: release artifact BF16 matvec gate is
  `max_ulp <= 5`. The observed worst case is a near-cancellation random output
  at `K=4096,N=12288`: expected `2.953125`, got `3.03125`.
- `flashrt-fused-quant`: split and merged NVFP4 packed bytes and scale bytes
  passed byte parity over the v1 grid.
- `flashrt-fp8-ffn`: unified installed-artifact sweep and package correctness
  test passed against the copied `torch211-cxx11-cu128-x86_64-linux` artifact
  over PI0.5/GROOT model-shaped FP8 GEMM, fused GELU quant, and full MLP cases.

## Package Tests

Package tests were run separately per package to avoid `tests` package-name
collisions:

| Package | Result |
| --- | --- |
| `flashrt-gemm-epilogues` | 15 passed |
| `flashrt-fp8-ffn` | built-artifact package test passed |
| `flashrt-vla-video` | 8 passed |
| `flashrt-nvfp4` | 18 passed |
| `flashrt-smallm-gemm` | 4 passed |
| `flashrt-fused-quant` | 5 passed |

## Examples

All package examples ran against the copied built artifacts. The HF-style
examples prefer `kernels.get_kernel` when available and fall back to local
artifact imports for release-candidate validation.

Representative output:

| Example | Result |
| --- | --- |
| `flashrt-gemm-epilogues/examples/fp8_quant_epilogue_block.py` | `M=64,N=4096`: 5.047 us vs 24.888 us, 4.93x |
| `flashrt-vla-video/examples/qkv_postprocess_block.py` | `T=256,H=24,D=128`: 7.710 us vs 143.796 us, 18.65x |
| `flashrt-nvfp4/examples/nvfp4_scale_factor_layout.py` | `rows=128,D=4096`: 3.212 us |
| `flashrt-smallm-gemm/examples/nvfp4_w4a4_decode_matvec.py` | output `(1024,)`, BF16 |
| `flashrt-fused-quant/examples/swiglu_nvfp4_quant_block.py` | split and merged packed/scales outputs produced |

## Benchmark Summary

Benchmarks used the local release-candidate runner:

```bash
python scripts/run_built_artifact_benchmarks.py \
  --package all --warmup 10 --iterations 50
```

The runner executes the public `kernels.benchmark.Benchmark` scripts against
the copied built artifacts. It does not replace the official Hub
`kernels benchmark` run after upload.

| Package | Built-artifact benchmark result |
| --- | --- |
| `flashrt-gemm-epilogues` | FP8 quant epilogues verified, 2.58x-4.44x vs PyTorch eager references; default BF16 GEMM public rows are verified-only |
| `flashrt-fp8-ffn` | Full FP8 GELU MLP verified over the PI0.5/GROOT shape grid; headline rows are 6.5x-7.2x vs eager and 5.9x-6.7x vs compile-stable `torch.compile` reference |
| `flashrt-vla-video` | Q/K/QKV post-processing verified, 9.79x-29.30x vs PyTorch eager references |
| `flashrt-nvfp4` | scale-factor layout helper byte-verified, 70.68x-18031.32x vs Python layout reference |
| `flashrt-smallm-gemm` | W4A4 decode matvec verified, 5.86x-16.12x vs random/dequant PyTorch readability baseline |
| `flashrt-fused-quant` | split and merged fused quant latency grid completed; multi-output byte parity remains covered by accuracy sweep |

The strict benchmark runner completed 170 rows with `verified=False` count 0
and `mean_ms=nan` count 0. Diagnostic benchmark failures are no longer recorded
by default; they require explicit `--allow-diagnostic-failures`.

## Full Matrix Status

The full local `kernel-builder build-and-copy` matrix completed on June 3,
2026 from commit `d0a125d9d6eb88bf55c63712a78eeb7e12ab97e7` after applying a
builder-side Triton fixed-output hash workaround. This produced 28 `.so`
artifacts across the six v1 packages.

Every produced artifact passed:

- manylinux/Python ABI compatibility check;
- kernel layout check;
- `get_kernel` load check.

Actual local artifact matrix:

| Package | Variants built |
| --- | --- |
| `flashrt-gemm-epilogues` | `torch211-cxx11-cu126`, `torch211-cxx11-cu128`, `torch211-cxx11-cu130`, `torch212-cxx11-cu126`, `torch212-cxx11-cu130`, `torch212-cxx11-cu132` |
| `flashrt-fp8-ffn` | `torch211-cxx11-cu128`, `torch211-cxx11-cu130`, `torch212-cxx11-cu130`, `torch212-cxx11-cu132` |
| `flashrt-vla-video` | `torch211-cxx11-cu126`, `torch211-cxx11-cu128`, `torch211-cxx11-cu130`, `torch212-cxx11-cu126`, `torch212-cxx11-cu130`, `torch212-cxx11-cu132` |
| `flashrt-nvfp4` | `torch211-cxx11-cu128`, `torch211-cxx11-cu130`, `torch212-cxx11-cu130`, `torch212-cxx11-cu132` |
| `flashrt-smallm-gemm` | `torch211-cxx11-cu128`, `torch211-cxx11-cu130`, `torch212-cxx11-cu130`, `torch212-cxx11-cu132` |
| `flashrt-fused-quant` | `torch211-cxx11-cu128`, `torch211-cxx11-cu130`, `torch212-cxx11-cu130`, `torch212-cxx11-cu132` |

Installed-artifact correctness was rerun against the full-matrix
`torch211-cxx11-cu128` artifacts on RTX 5090:

```text
accuracy sweep passed: 345 checks
```

The remaining release distinction is cleanliness, not package functionality:
the local full-matrix artifacts have a `_d0a125d_dirty` suffix because the
disposable validation clone pointed to a local patched `kernel-builder`.
Before public upload, regenerate the same matrix from a clean upstream builder
revision after HF confirms or fixes the Triton hash mismatch.

Hardware validation on other target machines is still required before widening
public hardware claims. SM120 packages should stay labeled CUDA 12.8+ SM120
until a non-SM120 source path is added.
