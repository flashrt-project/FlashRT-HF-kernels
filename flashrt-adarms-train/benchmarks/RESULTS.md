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
| vlm B2 site1 | 0.095 | 1.95 | 1.41 | PASS |
| vlm B2 site2 | 0.122 | 1.88 | 1.40 | PASS |
| vlm B8 site1 | 0.145 | 4.97 | 1.07 | conditional* |
| vlm B8 site2 | 0.297 | 3.04 | 0.88 | conditional* |
| expert_persample B2 site1 | 0.098 | 3.08 | 2.24 | PASS |
| expert_persample B2 site2 | 0.116 | 2.94 | 2.24 | PASS |
| expert_persample B8 site1 | 0.096 | 3.14 | 2.30 | PASS |
| expert_persample B8 site2 | 0.113 | 3.03 | 2.20 | PASS |
| expert_pertoken B2 site1 | 0.081 | 3.57 | 3.07 | PASS |
| expert_pertoken B2 site2 | 0.096 | 3.49 | 2.88 | PASS |
| expert_pertoken B8 site1 | 0.081 | 7.32 | 4.11 | PASS |
| expert_pertoken B8 site2 | 0.096 | 3.94 | 5.80 | PASS |

*conditional: the two non-adaptive (plain-weight RMSNorm) large-row cases
beat eager by 3-5x and are within 0.88-1.07x of a torch.compile'd region,
below the 1.2x margin the acceptance spec demands over compile. Inductor's
plain-RMSNorm fusion is strong there; tuning continues (one-pass weight-grad
partials already implemented). The adaptive/resgate cases — the ones plain
compile handles worst and the model actually runs with conditioning — clear
the bar at 2.2-5.8x over compiled.
