# Examples

This directory contains Hub-facing usage examples that do not depend on the
FlashRT serving runtime.

- `qkv_postprocess_block.py`: minimal HF-style module for replacing packed-QKV
  split, Q/K RMSNorm, and interleaved RoPE with
  `qkv_split_norm_rope_bf16`.
- `model-block-note.md`: model-block explanation for the same replacement,
  including shape focus and claim boundaries.
