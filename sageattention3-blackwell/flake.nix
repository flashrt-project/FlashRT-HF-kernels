{
  description = "Flake for FlashRT SageAttention3 Blackwell kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
