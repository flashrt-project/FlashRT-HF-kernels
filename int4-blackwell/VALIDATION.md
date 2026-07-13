# Validation

Release validation is intentionally exact because the undocumented operand
format is not visible in NVIDIA disassembly tools.

On RTX 5090 (SM120), driver 580.159.03 and CUDA 13.0.88:

- Rebuilding all four SM120 cubins is byte-for-byte deterministic.
- All 16 codebook values match their expected values with `rtol=0, atol=0`.
- All 128 accumulators in every codebook output tile are bitwise equal.
- E2M1 x E2M1, INT4 x E2M1, E2M1 x INT4, and INT4 x INT4 execute.
- The register-resident probe measures the same issue-rate class for NVFP4
  and uniform INT4.
- A runtime guard selects architecture-specific SM120/SM121 cubins and rejects
  other devices and drivers older than CUDA 13.0 support.
- The torch211/cu128 extension plus CUDA 13.0 cubin path is validated locally;
  the extension ABI does not require Torch itself to be a cu130 build.

This SM120 validation proves native decode semantics and issue rate. It does
not claim model-level accuracy.

SM121 cubins are generated independently with `sm_121a`. Their unpatched
`OMMA.SF` instruction encoding matches SM120, and all patch sites are found,
but SM121 remains a packaged release candidate until the exact codebook and
throughput tests run on a GB10/Spark device.

On NVIDIA Thor (SM110), CUDA 13.0.48 rejects the SM120 `OMMA.SF` instruction,
as expected. The separate CUTLASS tcgen05 backend was compiled for `sm_110a`
and validated with a full 16-code GEMM sweep. Dividing each 128 x 128 x 128
constant GEMM output by K gives exactly:

```text
0, 1, 2, 3, 4, 5, 6, 7, -0, -1, -2, -3, -4, -5, -6, -7
```

All elements in every output tile are uniform. The stock E2M1 descriptor
produces 64 for the `0.5 * 0.5 * 256` canary, while descriptor value zero
produces 256 for `1 * 1 * 256`, proving that the native decode changes rather
than a host-side reinterpretation. Hub cold-cache artifact validation is a
separate release gate and is recorded only after upload.

Random signed-INT4 GEMMs at `(M,N,K) = (128,128,128)`, `(128,256,256)`, and
`(256,128,128)` were compared with a FP32 PyTorch matmul rounded to BF16. All
three results were bit-exact (`max_error=0`, `mean_error=0`). These scale-one
tests cover packing, A/B layout, accumulation, and BF16 output independently
of the private physical scale-factor permutation.

SM100 and SM103 are compile targets, not runtime-validated claims.
