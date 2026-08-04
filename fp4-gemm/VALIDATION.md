# Validation

Local source validation covers NVIDIA GeForce RTX 5090 (SM120) and NVIDIA
Jetson AGX Thor (SM110).

```bash
python fp4-gemm/tests/test_fp4_gemm.py \
  --backend source \
  --mode full \
  --json-out internal-tests/fp4-gemm-source-full.json
```

Result:

- SM120 full gate: `25/25` checks passed, including all fused epilogues and
  the aggregate BF16 direct-producer layout gate.
- SM110 model-shape gate: `24/24` checks passed across PI0.5, GROOT, Cosmos
  Edge, and LingBot VLA projection shapes.
- Variants `0`, `1`, and `2` were checked.
- SM110 additionally checks production auto-dispatch (`variant=-1`).
- `nvfp4_gemm_bf16` is the canonical public API.
- Correctness reference dequantizes the same FP4/SFA and FP4/SFB inputs used
  by the kernel, then computes PyTorch GEMM on those dequantized values.
- The direct BF16 producer is byte-exact against the established
  BF16-to-FP16 plus FP16-producer contract for packed E2M1, mapped SFA/SFB
  bytes, and dequantized output. Covered activation shapes are `(1,5120)`,
  `(1,6144)`, `(1,17408)`, `(16,2048)`, and `(128,512)`; SFB coverage uses
  `(64,1024)`.

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

The SM110 release flake pins kernel-builder commit
`d720fa90fb9cd92d1bc60a9dc5c55bef2aafabb8`, which includes CUTLASS 4.5
support and the corrected CUTLASS 4.5.2 fixed-output hash. HF Jobs, the
SM110 aarch64 artifact build, and cold Hub loads must pass before the rebuilt
Hub release is considered complete.

## BF16 Direct Producer

RTX 5090, 100 warmup iterations and 1000 measured iterations:

| Shape | BF16 direct | BF16 cast + FP16 producer | Speedup | Native BF16 producer | Hub/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| M=1, K=5120 | 4.098 us | 6.404 us | 1.563x | 6.150 us | 0.666x |
| M=1, K=6144 | 4.098 us | 6.403 us | 1.562x | 8.190 us | 0.500x |
| M=1, K=17408 | 4.096 us | 6.413 us | 1.566x | 18.442 us | 0.222x |

The native BF16 producer is included as a latency comparison but uses a
different FlashRT quantization strategy. Correctness acceptance is therefore
against this package's established FP16 producer contract, where all tested
packed and mapped scale bytes are exact.

## Thor Native Parity

The Tensor wrapper was compared against the same native FlashRT launchers on
Thor with 20 warmup and 100 measured iterations. For production auto-dispatch
across the six model shapes, wrapper/native latency ratio had median `1.019`
and maximum `1.086`. Correctness was exact (`max_abs=mean_abs=p99_abs=0`).

## SM110 portable SIMT fallback (Thor real hardware)

On NVIDIA Thor (`sm_110a`) the SM120-only fused-epilogue GEMMs and the
warp-split GEMV route to `portable_fp4_gemm_simt` (compiled for `sm_110a`):

- `fp4_w4a4_gemv_warpsplit_bf16`
- `nvfp4_gemm_residual_bf16`
- `nvfp4_gemm_bias_gelu_bf16`
- `nvfp4_gemm_bias_gelu_nvfp4`
- `nvfp4_gemm_streamk_bf16`
- `nvfp4_gemm_streamk_bias_bf16`

Validated on Thor with PyTorch 2.9.1+cu130 / CUDA 13.2: all six ops launch,
produce finite BF16/FP4 output, and match the FP32 reference within the BF16
contract. SM120 keeps the CUTLASS fused-epilogue path.
