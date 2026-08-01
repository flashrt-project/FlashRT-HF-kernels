{
  description = "Flake for FlashRT FP8 GQA cross-attention";
  # Temporary FlashRT fork of the same upstream builder revision with the
  # exact CUTLASS 4.4.2 dependency required by this FMHA source.
  inputs.kernel-builder.url = "github:flashrt-project/kernels/f33b6e0396618311e348cf38ddeaf556f7d2fde8";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs { inherit self; path = ./.; };
}
