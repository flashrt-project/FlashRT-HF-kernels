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

This validation proves native decode semantics and issue rate. It does not
claim end-to-end GEMM or model accuracy because the package does not expose a
general GEMM operation.

SM121 cubins are generated independently with `sm_121a`. Their unpatched
`OMMA.SF` instruction encoding matches SM120, and all patch sites are found,
but SM121 remains a packaged release candidate until the exact codebook and
throughput tests run on a GB10/Spark device.

On NVIDIA Thor (SM110), CUDA 13.0 rejects the SM120 warp-level block-scale MMA
for `sm_110a`. NVIDIA CuTeDSL 4.6.0 NVFP4 GEMM passes on the same machine via
the separate `tcgen05`/`UTCOMMA.4X` path. E0M3 support on that path requires a
new codebook-validated backend; compiling the NVFP4 baseline alone is not
evidence of INT4 support.
