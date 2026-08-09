# adaptive-layernorm-producers Benchmark Results

Local source-build benchmark:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Runtime: local Torch 2.11 / CUDA 12.8 environment
- Command: `python adaptive-layernorm-producers/benchmarks/benchmark.py --backend source --iters 100`
- Baseline: PyTorch eager producer chain with equivalent operations.
- Status: source correctness passed before benchmark. Refresh this table after
  installed-artifact validation on each target hardware.

| Shape | Rows | Dim | AdaLN->FP8 us | Eager chain us | Speedup | LN->FP8 us | Eager LN chain us | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_action | 16 | 2048 | 4.117 | 65.010 | 15.79x | 3.962 | 48.006 | 12.12x |
| wan_video_short | 64 | 3072 | 4.133 | 63.425 | 15.35x | 4.117 | 47.826 | 11.62x |
| wan_video_ctx | 256 | 3072 | 4.140 | 69.821 | 16.86x | 4.108 | 55.611 | 13.54x |
| wan_video_2k | 2520 | 3072 | 12.330 | 263.276 | 21.35x | 10.267 | 218.926 | 21.32x |
| wan_video_4k | 4096 | 3072 | 18.465 | 463.733 | 25.11x | 16.412 | 394.908 | 24.06x |

## Six-way modulation producer

The new `adaln_modulation6_bf16` path was measured against its raw CUDA
launcher, the Tensor wrapper, eager, and `torch.compile`. Graph figures capture
32 identical launches per graph and divide replay time by 32, avoiding a graph
launch-floor artifact for the smallest row.

| Shape | Native us | Wrapper us | Graph native us | Graph wrapper us | Eager us | Compile us | Wrapper/native | Graph wrapper/native |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GROOT DiT B1 S51 D1536 | 2.221 | 3.052 | 1.536 | 1.663 | 48.804 | 43.536 | 1.374 | 1.083 |
| Motus B1 S2520 D3072 | 175.952 | 176.183 | 165.263 | 165.513 | 209.040 | 176.665 | 1.001 | 1.002 |
| video long B1 S5070 D3072 | 347.251 | 354.773 | 344.858 | 344.592 | 582.678 | 354.059 | 1.022 | 0.999 |

The direct S51 row exposes about `0.83 us` of dispatcher overhead. The
multi-launch CUDA Graph path reduces the absolute gap to `0.127 us`. Both
figures are retained; the direct row is not described as native-equivalent.

## NVIDIA Thor installed artifact

- Device: NVIDIA Thor, SM110, aarch64
- Runtime: PyTorch 2.11.0 + CUDA 13.0
- Variant: `torch211-cxx11-cu130-aarch64-linux`
- Source commit: `9b532b42ad71d0ef6bc79d1372e96910087faad4`
- Artifact code objects: SM87 and SM110a; measurements below are SM110a
- Kernel Hub `v1` artifact SHA256: `2b5542ae448332ebd4d1f2ca2f0030f88200061f8d2e64eb8581c314e7d7322c`

The full installed-artifact correctness matrix passed. The two per-token
producer entries were also measured against their raw registered ops. Outputs
were bit-exact in every row.

| Op | Shape | Raw us | Wrapper us | Wrapper/raw | Graph wrapper/raw us |
|---|---|---:|---:|---:|---:|
| per-token | GROOT DiT M51 D1536 | 6.154 | 6.148 | 0.999 | 6.211 / 6.224 |
| table per-token | GROOT DiT M51 D1536 | 6.166 | 6.150 | 0.997 | 7.127 / 6.607 |
| per-token | vision M105 D1152 | 8.204 | 8.207 | 1.000 | 8.203 / 8.377 |
| table per-token | vision M105 D1152 | 8.210 | 8.210 | 1.000 | 8.204 / 8.703 |
| per-token | video M2520 D3072 | 229.922 | 229.936 | 1.000 | 229.984 / 230.797 |
| table per-token | video M2520 D3072 | 230.174 | 229.929 | 0.999 | 229.719 / 230.431 |

The table reports the uploaded SM87+SM110a fat binary on Thor. SM87 is present
as native cubin code but is not included in these runtime measurements.

### GROOT N1.7 FP4 producer update

The additive FP4 producer from FlashRT
`24df793f4fa2d50780aea03b644208c6e0cb4162` was rebuilt as
`torch213-cxx11-cu130-aarch64-linux` on the same Thor host. Packed FP4 values
and SFA bytes were exact against the native staged contract. The Tensor
wrapper/raw registered-op measurement was `8.3094/8.1932 us` (`1.0142x`) in
direct mode and `5.3749/5.3797 us` (`0.9991x`) under CUDA Graph.
