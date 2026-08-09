{
  description = "Flake for FlashRT forward-only FA4 CuTe runtime";
  inputs.kernel-builder.url = "github:huggingface/kernels/81f55ea30fd8f819dcf93a3c934dd584c895bd2f";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs { inherit self; path = ./.; };
}
