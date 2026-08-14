{
  description = "Flake for FlashRT FP4 GEMM kernels";

  inputs = {
    # Includes the CUTLASS 4.4 dependency mapping required by the SM110 path.
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
