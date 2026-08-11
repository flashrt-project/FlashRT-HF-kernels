{
  description = "Flake for FlashRT FP8 GEMM kernels";

  inputs = {
    # Immutable compatibility builder for the maintained Torch 2.11 release
    # variants. It combines the pre-removal Torch 2.11 matrix with the
    # upstream CUTLASS 4.4/4.5 dependency fix.
    kernel-builder.url =
      "github:flashrt-project/kernels/68b9786fc2b27b8f246a7a4ea1ff9e2b864ebd6a";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
