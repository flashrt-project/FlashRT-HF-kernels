{
  description = "Flake for FlashRT causal Conv1D state kernels";

  inputs = {
    # This revision adds the Torch 2.13 CUDA 13.0 x86_64 release variant.
    # Keep it pinned so the published matrix remains reproducible.
    kernel-builder.url =
      "github:huggingface/kernels/81f55ea30fd8f819dcf93a3c934dd584c895bd2f";
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
