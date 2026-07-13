# Source provenance

- Source experiment: `../int4-dev-blackwell-test`.
- Validation date: 2026-07-12.
- Validated GPU: NVIDIA GeForce RTX 5090 (SM120).
- Validated driver/toolkit: 580.159.03 / CUDA 13.0.88.
- Copied source: `probe.cu` and `patch_cubin.py`.
- Generated data: baseline, A-only, B-only, and A+B patched cubins under
  `torch-ext/int4_blackwell/cubin/sm120/` and `sm121/`.
- tcgen05 GEMM base: FlashRT commit
  `df1d92936ee80b949ad91fdacaded6022b049549`, files
  `csrc/gemm/fp4/cutlass_fp4_gemm.{cu,cuh}`.
- Package-local CUTLASS 4.5.2 override:
  `csrc/cute/arch/mma_sm100_desc.hpp`, preserving the upstream BSD license and
  changing only the MXF4 descriptor value used for E2M1 from the public value
  one to the hardware-validated E0M3 value zero.

The patch locates `OMMA.SF` instructions in each ELF text section and sets
SASS bit 78 for operand A and/or bit 79 for operand B. CUDA disassemblers do
not expose the resulting E0M3 format, so release validation must execute all
16 code points and compare them exactly with the expected codebook.

The public binding adds Tensor outputs, SM120/SM121 architecture selection and
driver guards, current-device and current-stream handling, and explicit mode
selection. No FlashRT serving pointer or stream API is exposed.

The tcgen05 port renames the launcher into the package namespace, changes the
output to BF16, removes FlashRT runtime dependencies, and compiles only for
SM100a, SM103a, and SM110a with CUDA 13.0. SM103a is emitted with an explicit
`-gencode` because the current kernel-builder CUDA 13 default list does not
enumerate compute capability 10.3. CUTLASS 4.5.2 supplies the native CUDA 13
SM103/SM110 feature mapping; no global CUTLASS compatibility patch is used.
The SM110 path is validated by executing the complete codebook. SM100 and
SM103 require runtime validation before hardware claims are added.

The original SM120 operand-format discovery is credited to the Ling Team and
@im0qianqian in the package README and Kernel Card:
<https://zhuanlan.zhihu.com/p/2059376150565089368>

The release flake temporarily pins `LiangSu8899/kernels@0853469`, based on
`huggingface/kernels@570dcf4`. Its only source change updates the stale Nix
fixed-output hash for CUTLASS 4.5.2 from the specified value to the value
returned by the upstream archive. Switch the flake input back to upstream once
that one-line fix lands there.
