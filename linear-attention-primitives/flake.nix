{
  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs = { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
    };
}
