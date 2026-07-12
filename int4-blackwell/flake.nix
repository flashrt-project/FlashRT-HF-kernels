{
  inputs.kernel-builder.url = "github:huggingface/kernels/1";

  outputs = { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs ./.;
}
