# SageAttention3 Blackwell

Speed-first FP4 attention for Blackwell, packaged from the Apache-2.0
[`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention) SageAttention3
implementation. This package is an optional accuracy tier; it does not replace
the higher-fidelity `flashrt/sageattention2-blackwell` default.

## Contract

- GPU: SM120a/SM121a Blackwell
- input layout: contiguous NHD `[batch, sequence, heads, head_dim]`
- head dimension 64 with CUDA 12.8 or newer artifacts
- head dimensions 64 and 128 with CUDA 13.0 or newer artifacts
- preferred fused input is raw contiguous BF16 NHD; it accepts arbitrary
  positive sequence lengths and pads/crops internally
- the low-level API retains the pre-centered, 128-padded Q/K/V and FP32
  `delta_s` contract
- output dtype matches the input activation dtype (BF16 or FP16)
- all hot-path outputs are caller-owned through `Sage3Workspace`
- no MQA/GQA in this version

`per_block_mean=True` and `False` are both exposed. The package reports
`ACCURACY_PROFILE = "speed-first"`; callers should gate this implementation
against their own model-level quality contract.

## Quantization variants

The published v1 implementation is the Sage3 NVFP4/E2M1 path. It quantizes
Q/K/V into the layouts consumed by the block-scaled FP4 attention kernel and
is available through `sage3_prefill_fp4_bf16`.

An INT4RHT path is specified but is **not implemented or exported in v1**. The
candidate uses symmetric INT4 Q/K with per-16-element scales, applies a shared
Hadamard-128 rotation after RoPE and centering, and reuses SageAttention2's FP8
P/V path. The reserved API name is `sage3_prefill_int4rht_bf16`. It will only
become a public capability after passing reference cosine >= 0.9999,
model-capture cosine >= 0.997, and an all-in latency lower than SageAttention2
FP8V. Do not treat that reserved name as a callable v1 symbol.

## Usage

```python
from kernels import get_kernel

sage3 = get_kernel("flashrt/sageattention3-blackwell", version=1)
out_buffer = torch.empty_like(q)
workspace = sage3.allocate_fused_workspace(q, k, v, out=out_buffer)

# Allocate before CUDA Graph capture.
out = sage3.sage3_prefill_fp4_bf16(
    q, k, v, out=out_buffer, workspace=workspace, per_block_mean=True,
)
```

The fused entry manages K centering, Q block means/centering, FP4 Q/K/V
quantization and the BF16 correction matrix inside the package. Centering and
quantization use packaged CUDA operators; K reduction and correction GEMM use
allocation-free ATen/cuBLAS operations writing into the caller-owned workspace.
It avoids large caller-side center/pad staging tensors and lets the attention
core consume the BF16 correction directly. `out` is an NHD view over
caller-owned storage; the workspace must remain alive through execution and
graph replay.

The existing `allocate_workspace` / `prepare_qkv_fp4_nhd` /
`blockscaled_fp4_attention_static` APIs remain available for advanced callers
that already own preprocessed tensors.

Call `capabilities()` after loading the package and dispatch only head
dimensions listed in `head_dims`. CUDA 12.8 D128 builds are intentionally not
published as a supported performance tier: CUDA 12.8 spills the upstream D128
template to roughly 1 KiB of local stack per thread and is slower than SDPA.
Both the Python contract and the compiled operator reject that combination;
CUDA 13.0+ artifacts provide the validated D128 implementation.

Direct `out=` binding is zero-copy for 128-aligned lengths. For an unaligned
length, omit `out`; the returned tensor is a cropped view of the padded,
workspace-owned output.

## Validation

The package test covers:

- each FP4 Q/K/V layout against the upstream quantization layout;
- output cosine against an FP32-softmax SDPA reference;
- both `per_block_mean` modes;
- caller-owned pointer stability and bitwise CUDA Graph replay;
- explicit rejection of unsupported layouts, dtypes, head dimensions and
  unpadded sequence storage.
- fused-vs-low-level parity at aligned and non-aligned lengths;
- the raw-input fused path under bitwise CUDA Graph replay.

The long video acceptance grid is `H=32`, `D=128`,
`S in {6144, 24576}`. Audio coverage uses `H=32`, `D=64` and padded sequence
lengths spanning 128 through 2688.
