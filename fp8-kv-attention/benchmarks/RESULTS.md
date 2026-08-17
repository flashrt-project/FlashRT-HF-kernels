# FP8 KV Attention Results

## Hub v4 Built Artifact

RTX 5090, PyTorch `2.13.0+cu130`, artifact
`torch213-cxx11-cu130-x86_64-linux`, 20 warmups and 100 timed iterations:

| Profile | q_seq | context | Native us | Hub wrapper us | Cached-default us | Wrapper/native | Cosine | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24Q/4KV/D256 | 8 | 4096 | 16.507 | 16.552 | 16.511 | 1.003 | 0.99999529 | yes |
| 16Q/2KV/D256 | 8 | 4096 | 16.300 | 16.307 | 16.255 | 1.000 | 0.99999529 | yes |
| 32Q/8KV/D128 | 8 | 4096 | 11.134 | 11.106 | 11.101 | 0.997 | 0.99999154 | yes |
| 16Q/8KV/D128 | 8 | 4096 | 10.787 | 10.758 | 10.731 | 0.997 | 0.99999136 | yes |
| 32Q/16KV/D128 | 8 | 4096 | 14.110 | 14.111 | 14.090 | 1.000 | 0.99999160 | yes |

The native and Hub columns invoke the same page-32 CUDA implementation. The
receipt shows that packaging and the public Python wrapper add no measurable
hot-path regression.

## Source Acceptance

RTX 5090 source-extension acceptance row, PyTorch `2.11.0+cu128`:

| Profile | q_seq | context | Native us | Static wrapper us | Cached-default us | Equivalent compile us | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32Q/16KV/D128 | 8 | 4096 | 13.773 | 13.791 | 13.775 | 399.053 | 0.99999160 |
| 16Q/2KV/D256 | 8 | 4096 | 15.992 | 16.045 | 16.017 | 305.538 | 0.99999529 |

The native column invokes the same FlashRT XQA launch through a minimal C++
binding. Both wrapper rows exercise the public package API with preallocated
page table, mask, semaphore, scratch, and output tensors. The static row passes
SM count and strides explicitly; the cached-default row verifies that omitted
launch metadata does not reintroduce a per-call device-property query.

Do not publish headline speedup claims until source and installed-artifact
correctness both pass on the target hardware.
