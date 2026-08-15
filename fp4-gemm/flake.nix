{
  description = "Flake for FlashRT FP4 GEMM kernels";

  inputs = {
    # Retains the maintained Torch 2.11 CUDA variants while providing both
    # CUTLASS 4.4 (SM110) and CUTLASS 4.5 (SM120) dependency mappings.
    kernel-builder.url = "github:flashrt-project/kernels/flashrt-torch211-cutlass44";
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
