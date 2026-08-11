# Validation

Release requires FP16 and BF16 parity against PyTorch SDPA at key lengths
41, 277, 1024, 1025, and 2048; poisoned padded scratch; CUDA Graph replay;
source and installed-artifact runs on SM110 and SM120; and
native-vs-package timing on each published architecture.

The additive `attention_mha_fp16_masked` and
`attention_mha_bf16_masked` entries must remain bitwise-equal to
`forward_static`. The BF16 entry additionally checks `qkv_token_stride`
against the actual Q tensor stride so fused-QKV views fail early instead of
reading the wrong token row. Release gating includes poisoned padded columns,
bitwise graph replay, and an explicit/static latency ratio no greater than
1.05x.

On 2026-08-06, source full passed `10/10` on NVIDIA Thor (SM110), PyTorch
`2.13.0+cu130`, and CUDA 13.0. The additive `forward_seqused_static` rows cover
PI0.5 `(Sq,H,D)=(10,8,256)`, `Sk_max=456/968`, and device-resident
`valid_k=456/712/968`. Worst p99 absolute error was `0.000122`, worst cosine
was `0.99999982`, and graph replay was bitwise deterministic.

On 2026-08-08, the source and clean installed-artifact gates both passed
`10/10` on the same Thor class. The explicit FP16/BF16 entries measured
`1.005x/1.000x` of `forward_static`; BF16 fused-stride execution also passed
`torch.compile(fullgraph=True)` exactly. For the GROOT row
`Sq=Sk=41,H=32,D=48`, worst p99 absolute error was `0.003906` and cosine was
`0.99999213`; sequence lengths 1024, 1025, and 2048 also passed.

On 2026-08-11, the same full source matrix passed `10/10` on RTX 5090 with
PyTorch `2.11.0+cu128`. Worst GROOT-row p99 absolute error was `0.003906` and
cosine was `0.99999189`; graph replay remained bitwise deterministic. The
GROOT `(41,277,32,48)` profile was slower than SDPA and is therefore not a
default-takeover claim. The x86 artifact must repeat this gate after build.
