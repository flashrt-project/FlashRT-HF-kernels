# Results

Built-artifact benchmark results are pending for v4.

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
transcendentals. Regenerate these rows from the v4 built artifact before
making artifact-level performance claims.

The table below is a local source-extension triage benchmark on
`NVIDIA GeForce RTX 5090`, CUDA `12.8`, PyTorch `2.9.1+cu128`. It compares the
same package's baseline WY CUDA fallback path against the new v3 FLA-style
native CUDA MMA prefill path. It is not a HF Jobs artifact benchmark and should
be regenerated after upload.

Both paths include the same static state reset copy before each timed replay,
so the comparison is apples-to-apples within this package.

| S | Baseline WY fallback us | FLA MMA WY us | Speedup |
| ---: | ---: | ---: | ---: |
| 64 | 2495.942 | 1489.869 | 1.68x |
| 128 | 3559.574 | 1582.773 | 2.25x |
| 512 | 10511.024 | 2768.136 | 3.80x |
| 1024 | 19554.431 | 4174.474 | 4.68x |

## Full recurrent sequence entry

Source-extension benchmark on RTX 5090, PyTorch 2.9.1+cu128.

| Shape | Eager us | Compile us | Legacy per-token native us | Wrapper us | Raw sequence us |
|---|---:|---:|---:|---:|---:|
| `S65 H4 D128` | 7795.950 | 2092.266 | 798.974 | 306.614 | 305.986 |

The one-launch sequence entry is 6.82x faster than compile and 2.61x faster
than the prior per-token native launch loop. Wrapper overhead is 0.21%.
