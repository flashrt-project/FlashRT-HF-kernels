{
  description = "Flake for FlashRT GEMM epilogue kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/final-triton-hashes";
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
