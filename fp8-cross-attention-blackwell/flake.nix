{
  description = "Flake for FlashRT FP8 GQA cross-attention";
  inputs.kernel-builder.url = "github:huggingface/kernels/633246310320d85def0c67d62c7912fd444a842f";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs { inherit self; path = ./.; };
}
