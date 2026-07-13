# Source provenance

- Source experiment: `../int4-dev-blackwell-test`.
- Validation date: 2026-07-12.
- Validated GPU: NVIDIA GeForce RTX 5090 (SM120).
- Validated driver/toolkit: 580.159.03 / CUDA 13.0.88.
- Copied source: `probe.cu` and `patch_cubin.py`.
- Generated data: baseline, A-only, B-only, and A+B patched cubins under
  `torch-ext/int4_blackwell/cubin/sm120/` and `sm121/`.

The patch locates `OMMA.SF` instructions in each ELF text section and sets
SASS bit 78 for operand A and/or bit 79 for operand B. CUDA disassemblers do
not expose the resulting E0M3 format, so release validation must execute all
16 code points and compare them exactly with the expected codebook.

The public binding adds Tensor outputs, SM120/SM121 architecture selection and
driver guards, current-device and current-stream handling, and explicit mode
selection. No FlashRT serving pointer or stream API is exposed.

The original SM120 operand-format discovery is credited to the Ling Team and
@im0qianqian in the package README and Kernel Card:
<https://zhuanlan.zhihu.com/p/2059376150565089368>
