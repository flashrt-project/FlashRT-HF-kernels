# Source Sync

Synced from `official/FlashRT`:

- `csrc/quantize/qkv_split_norm_rope_bf16.cu`: packed QKV split, Q/K RMSNorm,
  and RoPE math.
- `csrc/quantize/qkv_split_norm_rope_bf16.cuh`: public pointer-level contract.
- `csrc/kernels/qwen3_qkv_post_proc.cu`: decode Q staging and KV cache-write
  math, renamed to generic decode APIs for this package.
- `csrc/kernels/qwen3_qkv_post_proc.cuh`: decode pointer-level contract.

Local adaptation:

- Raw pointer APIs were converted to Tensor-based `torch.ops` bindings.
- The current package exposes no-bias Q/K, bias+Q/K/V, bias+cat-workspace,
  three-segment VLA joint workspace APIs, decode Q staging, direct KV-write,
  and device-position KV-write.
- The decode functions use generic public names and explicit shape validation.
  They keep the upstream `head_dim == 128` contract instead of pretending to be
  arbitrary-head-dimension kernels.
