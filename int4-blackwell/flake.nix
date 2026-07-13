{
  description = "Flake for FlashRT Blackwell INT4 primitives";

  # Temporary upstream hash fix: one-line CUTLASS 4.5.2 fixed-output update
  # based on huggingface/kernels main. Return to upstream after it lands.
  inputs.kernel-builder.url =
    "github:LiangSu8899/kernels/da73ab4c34bde4916c8efe88854722ed00c036bd";

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
