{
  description = "Flake for FlashRT masked MHA runtime kernels";
  inputs.kernel-builder.url = "github:flashrt-project/kernels/68b9786fc2b27b8f246a7a4ea1ff9e2b864ebd6a";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs { inherit self; path = ./.; };
}
