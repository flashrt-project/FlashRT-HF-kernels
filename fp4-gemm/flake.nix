{
  description = "Flake for FlashRT FP4 GEMM kernels";

  inputs = {
    # huggingface/kernels#741: CUTLASS 4.4.2 plus the corrected 4.5.2 hash.
    kernel-builder.url = "github:huggingface/kernels/870e825d881664e39f9287a27a74ef63ff3c545e";
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
