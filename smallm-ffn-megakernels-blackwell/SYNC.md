# Source synchronization

The SM120 backend is derived from the production FlashRT megakernels introduced
by upstream commit `d8d5de79ba0cc9edb941a3b5244b158857678200`:

- `csrc/kernels/megakernel/action_ffn_megakernel_v6t_sm120.{cu,cuh}`
- `csrc/kernels/megakernel/und_ffn_megakernel_v5t_sm120.{cu,cuh}`
- `csrc/kernels/megakernel/und_ffn_megakernel_v5split_stage3_sm120.{cu,cuh}`

Package-local changes remove model naming from the Tensor API and add an SM110
backend under a separate `build.toml` target. SM120 retains the original
`kind::f8f6f4` MMA and fused software-grid-barrier path. SM110 uses the standard
E4M3 MMA encoding and an ordered quantize, up-projection, and down-projection
launch sequence because the original 288-CTA global barrier cannot make forward
progress on Thor's residency envelope. The GEMM epilogues remain fused.

The SM110 path is limited to the package's documented small-M contracts and is
selected only on compute capability 11.0. It is not an SM120 binary relabeled as
SM110. Both architecture targets share the same public Tensor API and are tested
as installed artifacts.
