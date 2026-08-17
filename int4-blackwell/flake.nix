{
  description = "Flake for FlashRT Blackwell INT4 primitives";

  # Public immutable builder revision covering Torch 2.11-2.13 with the
  # merged CUTLASS 4.4/4.5 source hash fixes.
  inputs.kernel-builder.url =
    "github:flashrt-project/kernels/b39ca23f1b36383df00b27b3ffe1276cd5dbea85";

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
