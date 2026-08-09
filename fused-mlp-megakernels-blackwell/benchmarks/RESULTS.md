# Results

Source-extension triage on NVIDIA Thor SM110, CUDA 13, PyTorch 2.11.0+cu130.
The release table will be regenerated from the installed Hub artifact.

## SM100-family megakernel (production path)

| Shape | Eager us | Compile us | Wrapper us | Raw native us | vs compile |
|---|---:|---:|---:|---:|---:|
| `M768 N16384 K2048` | 1465.035 | 1296.318 | 1005.803 | 998.419 | 1.29x |

Wrapper overhead versus the raw native entry is 0.74%. The production row
passed with `max=0.00146484`, `p99=0.00012207`, `mean=0.00001121`, and
`cosine=0.99999988`.

## Portable SIMT fallback on NVIDIA Thor (SM110 installed artifact)

The SM100-family CUTLASS megakernel asserts at runtime on `sm_110a`, so the
installed artifact routes SM110 to the portable SIMT fallback
(`portable_geglu_simt.cu`). Measured against the installed
`<installed-dir>/fused-mlp-megakernels-blackwell` artifact on Thor:

| Shape | Eager us | Compile us | Wrapper us | Raw native us | vs eager |
|---|---:|---:|---:|---:|---:|
| `M768 N16384 K2048` | 1633.251 | 1358.933 | 85558.905 | 85538.818 | 0.019x |

Block-tiled SIMT (32x32 tile, shared-memory K chunks) runs ~22x faster than the
initial one-thread-per-output reference (1.89s -> 85.6ms). It remains a
correctness/compatibility path, not a production kernel: on `sm_110a` it is the
only available implementation (the CUTLASS megakernel cannot launch), and it is
~52x slower than eager. `sm_100`/`sm_103` continue to use the fused CUTLASS
megakernel; the gap is documented and the fallback is not used where a native
path exists.
