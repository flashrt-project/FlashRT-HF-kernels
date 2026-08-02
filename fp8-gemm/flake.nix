{
  description = "Flake for FlashRT FP8 GEMM kernels";

  inputs = {
    # Based on huggingface/kernels@e9152aa with only the stale CUTLASS 4.5.2
    # fixed-output hash updated. Return to upstream after the hash fix lands.
    kernel-builder.url =
      "github:LiangSu8899/kernels/d720fa90fb9cd92d1bc60a9dc5c55bef2aafabb8";
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
