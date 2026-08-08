# Source sync

- Upstream: `flashrt-project/FlashRT`
- Commit: `24df793f4fa2d50780aea03b644208c6e0cb4162`
- Source: `csrc/kernels/attention_mha_masked.cu`
- Local changes: Tensor validation and `torch.library` binding only.

The CUDA source was reverified byte-for-byte against FlashRT revision
`ff7795a88c257ccf7195c3c066ce69b7fe84be95`. SHA256 is
`751b1de5e227bfff00f57c30c0a5bfb8efefaeedd7230fb3b2d0c67a28773ff9`.
The explicit FP16/BF16 public entries are additive binding surfaces over this
same native implementation; the established `forward_static` entry remains
unchanged.
