{
  description = "Flake for the FlashRT FA2 seqused runtime";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
