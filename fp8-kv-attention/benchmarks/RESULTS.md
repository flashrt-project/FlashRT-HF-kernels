# FP8 KV Attention Results

Built-artifact benchmark rows are pending for the v4 page-32 release. The
table below is a source-extension acceptance receipt and is not a published
Hub-artifact claim.

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
