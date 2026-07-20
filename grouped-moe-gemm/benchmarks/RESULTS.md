# RTX 5090 source gate

CUDA 13.0 and a PyTorch 2.11 development environment; packed inputs and output
are preallocated. Values are kernel-call microseconds.

| tile rows | rows | N | K | latency |
|---:|---:|---:|---:|---:|
| 16 | 1024 | 1024 | 2048 | 75.80 |
| 64 block tile | 1024 | 1024 | 2048 | 18.57 |
| 64 block tile | 1024 | 4096 | 2048 | 58.82 |
| 64/N16 | 1024 | 80 | 256 | 5.58 |

Python packing/dequantization loops are correctness-only and are deliberately
not presented as a speedup baseline. The packaged CUDA files are byte-for-byte
copies of the accepted FlashRT implementations except local include paths;
built-artifact parity remains a post-Jobs gate.
