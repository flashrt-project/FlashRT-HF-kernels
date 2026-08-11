{
  description = "Flake for the FlashRT FA2 seqused runtime";

  inputs = {
    # This revision includes the Torch 2.13 x86 variants used by the release
    # matrix. Pin it so a builder matrix change cannot silently drop them.
    kernel-builder.url =
      "github:huggingface/kernels/81f55ea30fd8f819dcf93a3c934dd584c895bd2f";
  };

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
