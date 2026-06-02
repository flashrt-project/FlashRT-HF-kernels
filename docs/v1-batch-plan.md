# V1 Batch Plan

The first FlashRT HF kernel release is a four-block batch. Do not run full
Nix/kernel-builder packaging for every small source edit. Finish the source
sync, Tensor bindings, correctness tests, benchmark grids, and examples first;
then run full builder validation in one release window.

## V1 Blocks

| Block | Packages | Role |
| --- | --- | --- |
| FP8/GEMM epilogues | `flashrt-gemm-epilogues` | FP8 quant epilogue and selected BF16 GEMM epilogue wrappers |
| VLA/video post-processing | `flashrt-vla-video` | Launch-bound VLA/video/decoder QKV, RMSNorm, RoPE, cache/stage kernels |
| Blackwell NVFP4/FP4 low-bit | `flashrt-nvfp4`, `flashrt-smallm-gemm` | NVFP4 layout helpers, fused FP4/NVFP4 GEMM epilogues, small-M decode kernels |
| Fused quantization | `flashrt-fused-quant` | Activation, residual, RMSNorm, and low-bit quantization fusion |

## Pre-Build Development Order

This order is about dependency and risk, not priority. All four blocks are part
of v1.

1. Finish public API surfaces.
2. Finish package-local Tensor bindings.
3. Finish correctness tests against PyTorch or deterministic dequant
   references.
4. Finish shape grids and tile policies.
5. Finish benchmark scripts and public `RESULTS.md` summaries.
6. Finish HF-style examples or model-block examples.
7. Run source-extension compile and smoke tests.
8. Only then run full `kernel-builder` builds for all promoted packages.

## Package Checklist

### `flashrt-gemm-epilogues`

Current status: buildable, benchmarked, example added.

Before v1 build window:

- Remove or regenerate stale local `result` symlink before using artifacts.
- Run the FP8 benchmark scripts against a built package artifact.
- Run `examples/fp8_quant_epilogue_block.py` against a built or Hub package.
- Keep BF16 GEMM epilogue claims shape-specific.

### `flashrt-vla-video`

Current status: buildable by config, local source-extension and benchmark
evidence recorded, example added.

Before v1 build window:

- Complete full builder build and `check-builds`.
- Run `benchmarks/benchmark_q_norm_rope.py` against a built package artifact.
- Run `examples/qkv_postprocess_block.py` against a built or Hub package.
- Add one model-block note showing the PyTorch/HF op sequence replaced by the
  fused QKV post-processing API.

### `flashrt-nvfp4`

Current status: buildable layout helper, benchmark and example paths added.

Before v1 build window:

- Complete full builder build and `check-builds`.
- Run `benchmarks/benchmark_nvfp4_sf_reshape.py` against a built package
  artifact.
- Decide whether v1 includes only layout helpers or also one fused NVFP4 GEMM
  epilogue surface.
- If a fused GEMM epilogue is included, add fair CUTLASS/cuBLAS or unfused
  baseline reporting.

### `flashrt-smallm-gemm`

Current status: draft G1. First SM120 NVFP4 W4A4 decode matvec source slice
compiles locally and passes deterministic correctness for `K=4096,12288`.
The public benchmark harness covers `K in {4096,12288}` and
`N in {1024,4096,12288}`.

Before v1 build window:

- Promote `build.toml.draft` only after source-extension tests and benchmark
  scripts are stable.
- Run `benchmarks/benchmark_nvfp4_w4a4_decode_matvec.py` against a built
  package artifact.
- Run `examples/nvfp4_w4a4_decode_matvec.py` against a built or Hub package.
- Add a fair baseline: PyTorch dequant+matmul for readability and
  CUTLASS/cuBLASLt or FlashRT internal low-bit baseline for serious claims.
- Decide whether v1 also includes warpsplit small-M or tiny FP8.

### `flashrt-fused-quant`

Current status: draft G1. The split and merged `SiLU(gate) * up` NVFP4
swizzled quantization source slice compiles locally and passes byte parity
against a fake-quant reference.
The public benchmark harness covers split and merged gate/up variants over the
v1 decode, small-batch, prefill, and VLA/video FFN shape grid.

Before v1 build window:

- Add memory-bandwidth benchmarks for decode, small batch, prefill, and
  VLA/video FFN hidden sizes.
- Run `benchmarks/benchmark_silu_mul_quant_nvfp4.py` against a built package
  artifact.
- Run `examples/swiglu_nvfp4_quant_block.py` against a built or Hub package.
- Decide whether v1 also includes residual/RMSNorm variants.

## Release Build Window

Run this only after every v1 package has stable source, tests, benchmarks, and
examples:

1. Clean stale build outputs and result symlinks.
2. Run `kernel-builder-docker check-config .` for every promoted package.
3. Run full `kernel-builder build` for all promoted packages.
4. Run `kernel-builder check-builds`.
5. Run package tests, benchmark CLIs, and examples against built artifacts.
6. Update every `VALIDATION.md` with exact variants, hardware, and failures.
7. Push one final v1-ready commit before upload.
