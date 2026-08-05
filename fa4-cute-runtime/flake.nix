{
  description = "Flake for FlashRT forward-only FA4 CuTe runtime";
  inputs.kernel-builder.url = "github:huggingface/kernels/870e825d881664e39f9287a27a74ef63ff3c545e";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs { inherit self; path = ./.; };
}
