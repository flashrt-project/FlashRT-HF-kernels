| Shape | B,L,H,D | FlashRT us | Eager us | vs eager | Q p99 | K p99 | Q cosine | K cosine | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| small | 1,64,8,128 | 5.149 | 163.642 | 31.78x | 0.007812 | 0.007812 | 0.99999630 | 0.99999619 | PASS |
| wan_1k | 1,1024,24,128 | 11.382 | 250.803 | 22.03x | 0.007812 | 0.007812 | 0.99999619 | 0.99999619 | PASS |
| wan_2520 | 1,2520,24,128 | 20.877 | 634.323 | 30.38x | 0.007812 | 0.007812 | 0.99999607 | 0.99999619 | PASS |
| wan_4096 | 1,4096,24,128 | 35.866 | 1476.803 | 41.18x | 0.007812 | 0.007812 | 0.99999619 | 0.99999613 | PASS |
| vl_512 | 1,512,16,128 | 6.621 | 156.986 | 23.71x | 0.007812 | 0.007812 | 0.99999619 | 0.99999630 | PASS |
| joint3_small | 1,76,8,128 | 5.459 | 332.618 | 60.93x | 0.007812 | 0.007812 | 0.99999666 | 0.99999678 | PASS |
| pi05_decoder_gqa_kvcache | 1,10,8,256 | 3.802 | 111.418 | 29.31x | 0.000000 | 0.000000 | 0.99999994 | 1.00000000 | PASS |
| joint3_vla | 1,2552,24,128 | 27.546 | 1017.469 | 36.94x | 0.007812 | 0.007812 | 0.99999624 | 0.99999624 | PASS |
| decode_q_stage_h24 | 1,1,24,128 | 3.478 | 82.906 | 23.83x | 0.007812 | 0.000000 | 0.99999648 | 1.00000000 | PASS |
| decode_kvwrite_h8 | 1,1,8,128 | 3.603 | 89.581 | 24.86x | 0.000000 | 0.007812 | 1.00000000 | 0.99999666 | PASS |
| decode_kvwrite_devpos_h8 | 1,1,8,128 | 3.594 | 92.224 | 25.66x | 0.000000 | 0.007812 | 1.00000000 | 0.99999642 | PASS |

## Per-head GQA compiled baseline

This row includes packed Q/K/V slicing and copies, per-head Q/K RMSNorm,
rotate-half RoPE, and Q/K/V workspace writes in both paths. The baseline uses
`torch.compile(fullgraph=True)`.

| Shape | B,S,QH,KVH,HD | FlashRT us | Compiled us | vs compiled | Q cosine | K cosine | V exact | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| per_head_gqa_n17 | 1,277,16,8,128 | 6.154 | 35.798 | 5.82x | 0.99999470 | 0.99999452 | yes | PASS |

## Fused packed QKV bias + split-half RoPE

RTX 5090, source wrapper, preallocated outputs. The eager and compiled
baselines include bias, Q/K/V split, rotate-half RoPE, BF16 conversion, and
all output materialization.

| Shape | B,S,QH,KVH,HD | Wrapper us | Eager us | Compile us | vs eager | vs compile | p99 | cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| groot_vit | 1,277,16,16,64 | 8.223 | 93.279 | 32.831 | 11.34x | 3.99x | 0.000000 | 0.99999988 |
| qwen3_vl_vision | 1,1024,16,16,72 | 24.599 | 91.964 | 43.277 | 3.74x | 1.76x | 0.000000 | 1.00000000 |
| lingbot_vision | 1,1024,16,16,80 | 24.588 | 96.880 | 42.366 | 3.94x | 1.72x | 0.000000 | 1.00000000 |
| lingbot_attention_fp16 | 1,51,16,8,80 | 3.139 | 91.745 | 49.093 | 29.23x | 15.64x | 0.000000 | 1.00000000 |
| qwen3_vl_text | 1,277,32,8,128 | 8.207 | 92.483 | 51.522 | 11.27x | 6.28x | 0.000000 | 1.00000000 |
| wan_video | 1,2520,24,24,128 | 84.029 | 453.525 | 119.426 | 5.40x | 1.42x | 0.000000 | 1.00000000 |

### Original FlashRT native parity

The internal native harness compiled
`official/FlashRT/csrc/kernels/lingbot_rope_qkv.cu` directly and compared the
pointer API with the Tensor wrapper using the same inputs and output buffers.

| Shape | Output | Wrapper us | Native us | Wrapper/native | Bitwise Q/K/V |
|---|---|---:|---:|---:|---|
| groot_vit | BF16 | 8.207 | 8.209 | 1.000 | yes |
| groot_vit | FP16 | 8.203 | 8.202 | 1.000 | yes |
| qwen3_vl_vision | BF16 | 24.578 | 24.582 | 1.000 | yes |
| qwen3_vl_vision | FP16 | 24.580 | 24.579 | 1.000 | yes |
| lingbot_vision | BF16 | 24.579 | 24.576 | 1.000 | yes |
| lingbot_vision | FP16 | 24.581 | 24.578 | 1.000 | yes |
| qwen3_vl_text | BF16 | 8.207 | 8.203 | 1.000 | yes |
| qwen3_vl_text | FP16 | 8.205 | 8.199 | 1.001 | yes |
| wan_video | BF16 | 83.974 | 84.021 | 0.999 | yes |
| wan_video | FP16 | 84.023 | 84.023 | 1.000 | yes |
