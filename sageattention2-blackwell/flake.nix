{
  description = "Flake for FlashRT SageAttention2 Blackwell kernels";

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
