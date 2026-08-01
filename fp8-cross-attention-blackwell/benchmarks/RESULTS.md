# Results

Source-extension triage on NVIDIA Thor SM110, CUDA 13, PyTorch 2.11.0+cu130.
The release table will be regenerated from the installed Hub artifact.

| Shape | Eager us | Compile us | Wrapper us | Raw native us | vs compile |
|---|---:|---:|---:|---:|---:|
| `B1 Sq786 Sk7984 Hq28 Hkv4 D128` | 17515.056 | 17102.983 | 1846.467 | 1846.438 | 9.26x |

The wrapper/native ratio is 1.000. This row passed the independent BF16 SDPA
reference with `max=0.00025749`, `p99=0.00012207`, `mean=0.00003710`, and
`cosine=0.99978602`.
