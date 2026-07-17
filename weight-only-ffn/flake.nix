{
  description = "Flake for FlashRT weight-only FFN kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/19aaa6421e674e9fecc352bbae6eab81d19a6bf4";
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
