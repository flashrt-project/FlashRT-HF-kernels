import argparse

import torch

import int4_blackwell


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)

    props = torch.cuda.get_device_properties(0)

    if torch.cuda.get_device_capability(0) in {(10, 0), (10, 3), (11, 0)}:
        m = n = k = 128
        a_packed = torch.full((m, k // 2), 0x11, device="cuda", dtype=torch.uint8)
        b_packed = torch.full((n, k // 2), 0x11, device="cuda", dtype=torch.uint8)
        sfa = torch.full((m * k,), 0x38, device="cuda", dtype=torch.uint8)
        sfb = torch.full((n * k,), 0x38, device="cuda", dtype=torch.uint8)
        flops = 2 * m * n * k * args.iterations

        def run():
            return int4_blackwell.tcgen05_int4_gemm_bf16(a_packed, sfa, b_packed, sfb)

        for _ in range(args.repeats):
            run()
        torch.cuda.synchronize()
        for mode in ("e2m1", "a", "b", "ab"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            best_us = float("inf")
            for _ in range(args.repeats):
                start.record()
                run()
                end.record()
                end.synchronize()
                best_us = min(best_us, start.elapsed_time(end) * 1000.0)
            tflops = flops / (best_us * 1e-6) / 1e12
            print(f"{mode:5s} {best_us:9.3f} us {tflops:8.1f} TFLOPS")
        return

    blocks = props.multi_processor_count * 4
    warps = blocks * 8
    flops = warps * 4 * args.iterations * 2 * 16 * 8 * 64
    out = torch.empty((blocks, 256), device="cuda", dtype=torch.float32)
    for mode in ("e2m1", "a", "b", "ab"):
        int4_blackwell.mma_probe(
            mode, iterations=args.iterations, blocks=blocks, out=out
        )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best_ms = float("inf")
        for _ in range(args.repeats):
            start.record()
            int4_blackwell.mma_probe(
                mode,
                iterations=args.iterations,
                blocks=blocks,
                launches=args.launches,
                out=out,
            )
            end.record()
            end.synchronize()
            best_ms = min(best_ms, start.elapsed_time(end))
        per_launch_ms = best_ms / args.launches
        tflops = flops / (per_launch_ms * 1e-3) / 1e12
        print(f"{mode:5s} {per_launch_ms * 1e3:9.3f} us {tflops:8.1f} TFLOPS")


if __name__ == "__main__":
    main()
