# Results

## SM120 MMA WY KKT

RTX 5090, PyTorch `2.11.0+cu128`, 10 warmup and 100 measured launches:

| S | Scalar KKT | MMA KKT | Speedup |
| ---: | ---: | ---: | ---: |
| 2048 | 876.26 us | 26.65 us | 32.88x |

Both rows use the same tensors, output layout, stream, and package source
extension. See `VALIDATION.md` for the five-length numerical and graph gate.

These v5 rows are source-extension measurements. The v6 request-2 artifact
receipt is reported below.

## Speculative state stash

RTX 5090, PyTorch `2.11.0+cu128`, preallocated buffers:

| Shape | One stash scan | Four prefix re-advances | Speedup | Correctness |
| --- | ---: | ---: | ---: | --- |
| S=8, Hv/Hk/D=32/16/128 | 58.02 us | 119.89 us | 2.07x | bit-exact |

Every stash row used by the rollback gate equals the corresponding plain-core
prefix state bit for bit; CUDA Graph replay is also bit-identical.

## H32/H16 WY prefill

RTX 5090 source-extension triage, PyTorch `2.9.1+cu128`. Both rows use the
same seven native WY stages, static workspaces, fixed-order reductions, and
the same state-reset copy. H48 is the package's established native entry;
H32 uses the new parameterized entry.

| S | H32 WY us | H48 native WY us | H32 / H48 |
| ---: | ---: | ---: | ---: |
| 64 | 106.120 | 122.928 | 0.863 |
| 65 | 112.477 | 131.137 | 0.858 |
| 128 | 145.583 | 174.526 | 0.834 |
| 256 | 215.260 | 281.859 | 0.764 |

The prior serial 64x64 triangular solve took 2.18-2.43 ms by itself. Version
5 replaces it with a 64-thread shared-memory fixed-order solve; the complete
H32 pipeline is 20.6x faster at S=64 while preserving the reference metrics.

## Generic H32/H16 fused producer + chunk

Local source-extension benchmark on RTX 5090, PyTorch `2.9.1+cu128`. Both
paths reset the same BF16 recurrent state. `staged native` invokes the same
package's native split, gating, and shared-memory chunk kernels separately;
the fused path reads `conv_out/a/b` directly. Correctness is bit-identical.

| S | Fused native us | Staged native us | Speedup | Exact |
| ---: | ---: | ---: | ---: | --- |
| 1 | 12.293 | 17.288 | 1.41x | yes |
| 4 | 28.666 | 32.777 | 1.14x | yes |
| 64 | 362.683 | 366.435 | 1.01x | yes |

The fused chunk computes each per-token/head gate scalar once in thread 0 and
broadcasts it through shared memory; all 128 lanes no longer repeat the same
transcendentals. Regenerate these rows from the v5 built artifact before
making artifact-level performance claims.

## Full recurrent sequence entry

Source-extension benchmark on RTX 5090, PyTorch 2.9.1+cu128.

| Shape | Eager us | Compile us | Legacy per-token native us | Wrapper us | Raw sequence us |
|---|---:|---:|---:|---:|---:|
| `S65 H4 D128` | 7795.950 | 2092.266 | 798.974 | 306.614 | 305.986 |

The one-launch sequence entry is 6.82x faster than compile and 2.61x faster
than the prior per-token native launch loop. Wrapper overhead is 0.21%.

## WY request-2 primitives

RTX 5090 source artifact, PyTorch `2.11.0+cu128`, 20 warmups and 100 measured
launches with preallocated outputs:

| Entry | Shape | Kernel us | Reference us | Result |
| --- | --- | ---: | ---: | --- |
| `gdn_wy_norm_cumsum_pack_qk_v2_bf16` | `S=2044, Hk/Hv/D=16/48/128` | 33.02 | n/a | five logical outputs exact vs upgraded alias |
| `batched_unit_ltri_inv64_f32` | `B=512, 64x64` | 10.14 | 67.90 | max abs <= `1e-5` vs `solve_triangular` |

The full package gate passed `38/38` plus CUDA Graph, poisoned-tail,
unsupported-shape, and request-2 checks.

### v6 installed artifact

Cold Hub artifact on RTX 5090, PyTorch `2.13.0+cu130`, caller-owned outputs,
20 warmups and 100 measured launches:

| Entry | Shape | Artifact us | Source receipt us | Artifact/source |
| --- | --- | ---: | ---: | ---: |
| `gdn_wy_norm_cumsum_pack_qk_v2_bf16` | `S=2044, Hk/Hv/D=16/48/128` | 24.64 | 33.02 | 0.746x |

The installed full gate passed `38/38` plus CUDA Graph, poisoned-tail,
fail-fast, and `torch.compile(fullgraph=True)`. A separate cold-cache
`get_kernel(..., version=6)` run compiled and launched both request-2 entries.

## Streaming recurrent decode

RTX 5090 source-extension benchmark, PyTorch `2.11.0+cu128`, one token,
48 heads, head dimension 128. Both paths use caller-owned output and state
buffers and the same input tensors. Output and state are bit-exact.

| Entry | Pre-update Hub v6 us | Published v6 us | Speedup | Output | State |
| --- | ---: | ---: | ---: | --- | --- |
| `gdn_recurrent_inout_stream_bf16` | 12.30 | 6.153 | 2.00x | exact | exact |

The source receipt was `6.11 us`; the cold Hub artifact is within 1% of it.
The public `gated_delta_recurrent_inout_bf16` entry now dispatches to the same
stream implementation, so existing callers receive the register-pressure fix
without changing APIs.
