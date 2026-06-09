# FlashRT QKV Cache RoPE

This package provides FlashRT QKV split, Q/K RMSNorm, and RoPE kernels for
Hugging Face Kernel Hub.

It is intended for VLA, VLM, and video-model attention staging where standard
attention kernels can handle the attention core, but the model still needs a
fast pre-attention QKV postprocess path.
It also includes single-token decode helpers for Q staging and direct KV cache
writes, plus a sequence GQA cache-write path for static decoder loops.

## Kernels

- `qkv_split_norm_rope_bf16`: split packed BF16 QKV, RMSNorm Q/K, apply RoPE to
  Q/K, and write BF16 Q/K tensors.
- `qkv_split_bias_norm_rope_v_bf16`: add packed QKV bias, RMSNorm Q/K, apply
  RoPE to Q/K, and materialize BF16 Q/K/V tensors.
- `qkv_split_bias_norm_rope_v_cat_bf16`: same video path, but writes directly
  into preallocated joint Q/K/V workspaces.
- `qkv_split_joint3_cat_bf16`: VLA-oriented path that fuses video/action/und
  QKV postprocess and writes one attention-ready joint Q/K/V workspace.
- `qkv_split_rope_kvcache_bf16`: split GQA packed QKV, apply interleaved RoPE
  to Q/K, and write K/V into preallocated sequence caches.
- `decode_q_norm_rope_stage_bf16`: RMSNorm Q, apply rotate-half RoPE, and
  write a decode Q staging buffer.
- `decode_k_norm_rope_kvwrite_bf16`: RMSNorm K, apply rotate-half RoPE, and
  write one K/V cache slot.
- `decode_k_norm_rope_kvwrite_devpos_bf16`: same KV write, but selects the
  cache slot from a CUDA int32 `cur_pos` tensor for graph-friendly decode.

The decode APIs are fixed to `head_dim == 128` and use BF16 `(64,)` cos/sin
vectors. Unsupported shapes are rejected at the Tensor binding layer.
The GQA sequence API uses BF16 interleaved `(seq_len, head_dim)` RoPE rows and
supports different Q and KV head counts.

## Hardware

- CUDA 12.8+
- BF16-capable NVIDIA GPUs

Current local source validation is on RTX 5090. Broader hardware rows should be
added after installed-artifact validation.

## Upstream

The serving source of truth remains FlashRT:

https://github.com/LiangSu8899/FlashRT
