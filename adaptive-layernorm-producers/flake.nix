{
  description = "Flake for FlashRT adaptive LayerNorm producer kernels";

  inputs = {
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
