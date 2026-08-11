{
  description = "Flake for FlashRT FP4 GEMM kernels";

  inputs = {
    # Immutable compatibility builder for the maintained Torch 2.11 release
    # variants, including the CUTLASS 4.4 dependency required by SM110.
    kernel-builder.url = "github:flashrt-project/kernels/68b9786fc2b27b8f246a7a4ea1ff9e2b864ebd6a";
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
