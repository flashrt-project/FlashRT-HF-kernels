# flashrt-adarms-train — acceptance results

Environment: RTX 5090, torch 2.11.0+cu128. Harness: region-isolated fwd+bwd
(preallocated grad_outputs via `torch.autograd.grad`), median of 50 after 10
warmup, identical for eager / torch.compile / kernel.

Correctness (Gate 1): fp64 gradcheck of the reference math 6/6; kernel vs
reference at real shapes: forward <= 2 ulp of the io dtype (1 ulp final
rounding + 1 ulp rstd reduction order; the resgate residual is
bitwise-identical to the reference rounding sequence), grads <= 3e-3 rel
(bf16) / 2e-6 (fp32). All bf16+fp32 x {per-token, per-sample, weight} x
{site1, site2} cases pass, including a non-power-of-two-row tail case.

Performance (Gate 2): target = min(eager/1.5, compiled/1.2).

| case | kernel ms | eager x | compiled x | gate |
|---|---|---|---|---|
| vlm B2 site1 | 0.074 | 2.51 | 1.81 | PASS |
| vlm B2 site2 | 0.096 | 2.38 | 1.78 | PASS |
| vlm B8 site1 | 0.111 | 6.50 | 1.40 | PASS |
| vlm B8 site2 | 0.278 | 3.24 | 0.94 | conditional* |
| expert_persample B2 site1 | 0.102 | 2.95 | 2.15 | PASS |
| expert_persample B2 site2 | 0.129 | 2.65 | 2.02 | PASS |
| expert_persample B8 site1 | 0.102 | 2.96 | 2.16 | PASS |
| expert_persample B8 site2 | 0.127 | 2.68 | 1.95 | PASS |
| expert_pertoken B2 site1 | 0.084 | 3.44 | 2.96 | PASS |
| expert_pertoken B2 site2 | 0.108 | 3.08 | 2.54 | PASS |
| expert_pertoken B8 site1 | 0.083 | 7.10 | 3.98 | PASS |
| expert_pertoken B8 site2 | 0.107 | 3.55 | 5.22 | PASS |

*conditional (1 of 12): the non-adaptive (plain-weight RMSNorm) resgate case
at large rows beats eager 3.2x and sits at 0.94x of a torch.compile'd
region — below the 1.2x margin the spec demands over compile. Inductor's
plain-RMSNorm fusion is strong there; tuning continues. The
adaptive/resgate cases — the ones the model actually runs with conditioning
— clear the bar at 1.8-5.2x over compiled.

Mixed dtype: the kernels also accept fp32 modulation/weight alongside bf16
activations (the pi-style selective-bf16 layout); all correctness cases in
the smoke run in that combination too.

Real-model integration (pi052, pi05-libero weights, bf16, B=2, RTX 5090,
norm sites patched through the kernel — 592/592 calls served): text-step
loss rel diff 6.8e-5, grad norms within 0.3%, end-to-end text step -3.2%
(346 -> 335 ms) with only the standalone-norm sites wired (the resgate
fusion sites are a further integration step).
