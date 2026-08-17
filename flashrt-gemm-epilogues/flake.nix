{
  description = "Flake for FlashRT GEMM epilogue kernels";

  inputs = {
    kernel-builder.url =
      "github:flashrt-project/kernels/b39ca23f1b36383df00b27b3ffe1276cd5dbea85";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
