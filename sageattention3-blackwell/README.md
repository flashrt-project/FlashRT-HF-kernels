# SageAttention3 Blackwell

Speed-first FP4 attention for Blackwell, packaged from the Apache-2.0
[`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention) SageAttention3
implementation. This package is an optional accuracy tier; it does not replace
the higher-fidelity `flashrt/sageattention2-blackwell` default.

## Contract

- GPU: SM120a/SM121a Blackwell
- input layout: contiguous NHD `[batch, sequence, heads, head_dim]`
- head dimensions: 64 and 128
- Q/K/V sequence storage must be padded to a multiple of 128
- Q/K/V must follow SageAttention3 preprocessing: K sequence mean removal,
  Q block-mean removal, and caller-computed FP32 `delta_s`
- output dtype matches the input activation dtype (BF16 or FP16)
- all hot-path outputs are caller-owned through `Sage3Workspace`
- no MQA/GQA in this version

`per_block_mean=True` and `False` are both exposed. The package reports
`ACCURACY_PROFILE = "speed-first"`; callers should gate this implementation
against their own model-level quality contract.

## Usage

```python
from kernels import get_kernel

sage3 = get_kernel("flashrt/sageattention3-blackwell", version=1)
workspace = sage3.allocate_workspace(q_centered, k_centered, v_padded)

# Allocate before CUDA Graph capture.
sage3.prepare_qkv_fp4_nhd(q_centered, k_centered, v_padded, workspace)
out = sage3.blockscaled_fp4_attention_static(
    workspace,
    delta_s,
    unpadded_k=original_k_length,
    per_block_mean=True,
)
```

`out` is an NHD view over the caller-owned output buffer. The workspace object
must remain alive through execution and graph replay.

## Validation

The package test covers:

- each FP4 Q/K/V layout against the upstream quantization layout;
- output cosine against an FP32-softmax SDPA reference;
- both `per_block_mean` modes;
- caller-owned pointer stability and bitwise CUDA Graph replay;
- explicit rejection of unsupported layouts, dtypes, head dimensions and
  unpadded sequence storage.

The long video acceptance grid is `H=32`, `D=128`,
`S in {6144, 24576}`. Audio coverage uses `H=32`, `D=64` and padded sequence
lengths spanning 128 through 2688.
