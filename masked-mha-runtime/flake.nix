{
  description = "Flake for FlashRT masked MHA runtime kernels";
  inputs.kernel-builder.url = "github:huggingface/kernels/2c40e10d3da1392245ee81bd05b55fb3103b2b34";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs { inherit self; path = ./.; };
}
