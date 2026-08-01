{
  description = "Flake for FlashRT FP8 GQA cross-attention";
  # Temporary FlashRT fork of the same upstream builder revision with the
  # corrected CUTLASS 4.5.2 fixed-output hash.
  inputs.kernel-builder.url = "github:flashrt-project/kernels/fb59f9322bd36c4deaaeb5f33d7506b83a396cfa";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs { inherit self; path = ./.; };
}
