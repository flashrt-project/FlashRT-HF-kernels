{
  description = "Flake for FlashRT Blackwell fused MLP megakernels";
  inputs.kernel-builder.url = "github:huggingface/kernels";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs { inherit self; path = ./.; };
}
