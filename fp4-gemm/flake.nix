{
  description = "Flake for FlashRT FP4 GEMM kernels";

  inputs = {
    # Includes the CUTLASS 4.4 dependency mapping required by the SM110 path.
    kernel-builder.url = "github:huggingface/kernels";
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
