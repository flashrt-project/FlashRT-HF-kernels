# Validation

Release validation is intentionally exact because the undocumented operand
format is not visible in NVIDIA disassembly tools.

On RTX 5090 (SM120), driver 580.159.03 and CUDA 13.0.88:

- Rebuilding all four cubins is byte-for-byte deterministic.
- All 16 codebook values match their expected values with `rtol=0, atol=0`.
- All 128 accumulators in every codebook output tile are bitwise equal.
- E2M1 x E2M1, INT4 x E2M1, E2M1 x INT4, and INT4 x INT4 execute.
- The register-resident probe measures the same issue-rate class for NVFP4
  and uniform INT4.
- A runtime guard rejects devices other than SM120 and drivers older than
  CUDA 13.0 support.

This validation proves native decode semantics and issue rate. It does not
claim end-to-end GEMM or model accuracy because the package does not expose a
general GEMM operation.
