{
  description = "Flake for FlashRT FP4 GEMM kernels";

  inputs = {
    # Includes the CUTLASS 4.4 dependency mapping required by the SM110 path.
    kernel-builder.url = "github:huggingface/kernels/81f55ea30fd8f819dcf93a3c934dd584c895bd2f";
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
