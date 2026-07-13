{
  description = "Flake for FlashRT Blackwell INT4 primitives";

  # Temporary upstream hash fix: one-line CUTLASS 4.5.2 fixed-output update
  # based on huggingface/kernels main. Return to upstream after it lands.
  inputs.kernel-builder.url =
    "github:LiangSu8899/kernels/08534695226e512ad5f6abf537423df88531e661";

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
