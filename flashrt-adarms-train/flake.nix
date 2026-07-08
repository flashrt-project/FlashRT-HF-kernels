{
  description = "Flake for FlashRT AdaRMS training reference kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/432702bfbfbb17d3a1bd2c2743d004e21e769ab7";
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
