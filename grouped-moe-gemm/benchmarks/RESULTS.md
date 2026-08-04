# RTX 5090 source gate

CUDA 13.0 and a PyTorch 2.11 development environment; packed inputs and output
are preallocated. Values are kernel-call microseconds.

| tile rows | rows | N | K | latency |
|---:|---:|---:|---:|---:|
| 16 | 1024 | 1024 | 2048 | 75.80 |
| 64 block tile | 1024 | 1024 | 2048 | 18.57 |
| 64 block tile | 1024 | 4096 | 2048 | 58.82 |
| 64/N16 | 1024 | 80 | 256 | 5.58 |

Exact 35B-A3B projection rows, public Tensor wrapper versus the same native
FlashRT op through the raw dispatcher:

| Projection | rows | N | K | Native us | Public wrapper us | Ratio | Output |
|---|---:|---:|---:|---:|---:|---:|---|
| gate/up | 1024 | 1024 | 2048 | 18.454 | 18.480 | 1.0014 | bit exact |
| down | 1024 | 2048 | 512 | 16.411 | 16.423 | 1.0008 | bit exact |

The existing Hub v1 artifact was cold-loaded and repeated these rows at
18.522 us and 16.423 us respectively, both within 0.2% of its raw dispatcher.

Python packing/dequantization loops are correctness-only and are deliberately
not presented as a speedup baseline. The packaged CUDA files are byte-for-byte
copies of the accepted FlashRT implementations except local include paths;
built-artifact parity is a release gate.
