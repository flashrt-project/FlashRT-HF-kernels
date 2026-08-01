# Results

Source-extension triage on NVIDIA Thor SM110, CUDA 13, PyTorch 2.11.0+cu130.
The release table will be regenerated from the installed Hub artifact.

| Shape | Eager us | Compile us | Wrapper us | Raw native us | vs compile |
|---|---:|---:|---:|---:|---:|
| `M768 N16384 K2048` | 1465.035 | 1296.318 | 1005.803 | 998.419 | 1.29x |

Wrapper overhead versus the raw native entry is 0.74%. The production row
passed with `max=0.00146484`, `p99=0.00012207`, `mean=0.00001121`, and
`cosine=0.99999988`.
