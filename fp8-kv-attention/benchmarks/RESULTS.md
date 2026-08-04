# FP8 KV Attention Results

Built-artifact benchmark rows are pending for v2.

RTX 5090 source-extension acceptance row, PyTorch `2.9.1+cu128`:

| Profile | q_seq | context | Native us | Wrapper us | Wrapper/native | Equivalent compile us | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32Q/16KV/D128 | 8 | 4096 | 13.761 | 13.756 | 0.9996 | 565.504 | 0.99999148 |

The native column invokes the same FlashRT XQA launch through a minimal C++
binding. The wrapper row exercises the public package API with preallocated
page table, mask, semaphore, scratch, and output tensors.

Do not publish headline speedup claims until source and installed-artifact
correctness both pass on the target hardware.
