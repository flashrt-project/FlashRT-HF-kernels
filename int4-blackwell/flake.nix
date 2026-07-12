{
  description = "Flake for FlashRT SM120 INT4 primitives";

  inputs.kernel-builder.url = "github:huggingface/kernels";

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
