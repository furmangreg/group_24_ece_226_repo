#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, ProfilerActivity

# ----------------------------
# Config
# ----------------------------
DTYPE = torch.float16
DEVICE = "cuda"

# GEMM: [B*T, H] x [H, 4H]
B = 8
T = 256
H = 4096
FFN = 4 * H

# Softmax: [B, heads, T, T]
HEADS = 32
SOFTMAX_T = 256

WARMUP_ITERS = 10
PROFILE_ITERS = 30

# Reduce allocator churn for GEMM/add by reusing outputs
REUSE_OUTPUTS = True


@dataclass
class SectionResult:
    name: str
    cuda_event_ms: float
    # Filled after profiler ends
    prof_cuda_ms: float = 0.0
    prof_cpu_ms: float = 0.0
    prof_cuda_alloc_bytes: int = 0  # cumulative churn attributed to op
    # Allocator peaks for the region
    peak_alloc_bytes: int = 0
    peak_reserved_bytes: int = 0
    # Model estimates
    est_bytes_moved: int = 0
    est_flops: int = 0


def sizeof_dtype(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    if dtype == torch.float64:
        return 8
    raise ValueError(f"Unsupported dtype: {dtype}")


def fmt_bytes(n: int) -> str:
    unit = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in unit:
        if x < 1024.0 or u == unit[-1]:
            return f"{x:,.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def fmt_ms(x: float) -> str:
    return f"{x:,.3f} ms"


def estimate_gemm(M: int, N: int, K: int, dtype: torch.dtype) -> Tuple[int, int]:
    el = sizeof_dtype(dtype)
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * el
    return flops, bytes_moved


def estimate_ewise_add(numel: int, dtype: torch.dtype) -> Tuple[int, int]:
    el = sizeof_dtype(dtype)
    flops = numel
    bytes_moved = (numel * 3) * el
    return flops, bytes_moved


def estimate_softmax(last_dim: int, outer: int, dtype: torch.dtype) -> Tuple[int, int]:
    el = sizeof_dtype(dtype)
    L = last_dim
    flops = (7 * L) * outer
    bytes_moved = (2 * L * outer) * el
    return flops, bytes_moved


def repeat(fn, iters: int):
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()


def run_section(name: str, fn, est_flops: int, est_bytes_moved: int) -> SectionResult:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    nvtx.range_push(name)
    start.record()
    fn()
    end.record()
    nvtx.range_pop()

    end.synchronize()
    cuda_event_ms = start.elapsed_time(end)

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    return SectionResult(
        name=name,
        cuda_event_ms=cuda_event_ms,
        peak_alloc_bytes=peak_alloc,
        peak_reserved_bytes=peak_reserved,
        est_bytes_moved=est_bytes_moved,
        est_flops=est_flops,
    )


def sum_prof_for_ops(op_table: List, op_exact_keys: List[str]) -> Tuple[float, float, int]:
    """
    Sum profiler stats ONLY for exact op keys (leaf ops) to avoid double-counting.
    Returns (cuda_ms, cpu_ms, self_cuda_memory_usage_bytes).
    """
    cuda_us = 0
    cpu_us = 0
    cuda_mem = 0

    keys = set(op_exact_keys)
    for evt in op_table:
        if evt.key in keys:
            cuda_us += getattr(evt, "cuda_time_total", 0) or 0
            cpu_us += getattr(evt, "cpu_time_total", 0) or 0
            cuda_mem += getattr(evt, "self_cuda_memory_usage", 0) or 0

    return cuda_us / 1000.0, cpu_us / 1000.0, int(cuda_mem)


def print_results(results: List[SectionResult]) -> None:
    print("\n=== Per-kernel results ===")
    for r in results:
        ai = (r.est_flops / r.est_bytes_moved) if r.est_bytes_moved else float("nan")
        bw_gbps_est = (r.est_bytes_moved / (r.cuda_event_ms / 1e3)) / 1e9 if r.cuda_event_ms > 0 else float("nan")
        tflops_est = (r.est_flops / (r.cuda_event_ms / 1e3)) / 1e12 if r.cuda_event_ms > 0 else float("nan")

        print(f"\n[{r.name}]")
        print(f"  CUDA-event time:           {fmt_ms(r.cuda_event_ms)}")
        print(f"  Profiler CUDA time total:  {fmt_ms(r.prof_cuda_ms)}  (leaf op only)")
        print(f"  Profiler CPU time total:   {fmt_ms(r.prof_cpu_ms)}   (leaf op only)")
        print(f"  Profiler CUDA alloc bytes: {fmt_bytes(r.prof_cuda_alloc_bytes)} (cumulative churn, self_cuda_memory_usage)")
        print(f"  Peak allocated (region):   {fmt_bytes(r.peak_alloc_bytes)} (peak live bytes)")
        print(f"  Peak reserved (region):    {fmt_bytes(r.peak_reserved_bytes)} (caching allocator)")
        print(f"  Est. bytes moved:          {fmt_bytes(r.est_bytes_moved)} (tensor-size model)")
        print(f"  Est. FLOPs:                {r.est_flops:,}")
        print(f"  Est. arithmetic intensity: {ai:,.3f} FLOPs/byte")
        print(f"  Est. effective bandwidth:  {bw_gbps_est:,.2f} GB/s  (est bytes / event time)")
        print(f"  Est. effective compute:    {tflops_est:,.2f} TFLOP/s (est flops / event time)")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Shapes
    M = B * T
    K = H
    N = FFN

    # Inputs
    X = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    W = torch.randn(K, N, device=DEVICE, dtype=DTYPE)

    A = torch.randn(M, N, device=DEVICE, dtype=DTYPE)
    B2 = torch.randn(M, N, device=DEVICE, dtype=DTYPE)

    logits = torch.randn(B, HEADS, SOFTMAX_T, SOFTMAX_T, device=DEVICE, dtype=DTYPE)

    # Optional reused outputs to reduce allocator churn
    gemm_out = torch.empty(M, N, device=DEVICE, dtype=DTYPE) if REUSE_OUTPUTS else None
    add_out = torch.empty(M, N, device=DEVICE, dtype=DTYPE) if REUSE_OUTPUTS else None

    # Estimates (per single call)
    gemm_flops, gemm_bytes = estimate_gemm(M, N, K, DTYPE)
    add_flops, add_bytes = estimate_ewise_add(A.numel(), DTYPE)
    soft_flops, soft_bytes = estimate_softmax(
        last_dim=SOFTMAX_T,
        outer=B * HEADS * SOFTMAX_T,
        dtype=DTYPE,
    )

    # Workloads
    def do_gemm():
        if gemm_out is None:
            _ = X @ W
        else:
            # out= avoids allocating a fresh output every iteration
            torch.matmul(X, W, out=gemm_out)

    def do_add():
        if add_out is None:
            _ = A + B2
        else:
            torch.add(A, B2, out=add_out)

    def do_softmax():
        _ = F.softmax(logits, dim=-1)

    # Warmup
    with nvtx.range("warmup"):
        for _ in range(WARMUP_ITERS):
            do_gemm()
            do_add()
            do_softmax()
    torch.cuda.synchronize()

    results: List[SectionResult] = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=False,
    ) as prof:
        results.append(
            run_section(
                name="GEMM",
                fn=lambda: repeat(do_gemm, PROFILE_ITERS),
                est_flops=gemm_flops * PROFILE_ITERS,
                est_bytes_moved=gemm_bytes * PROFILE_ITERS,
            )
        )
        results.append(
            run_section(
                name="ElementwiseAdd",
                fn=lambda: repeat(do_add, PROFILE_ITERS),
                est_flops=add_flops * PROFILE_ITERS,
                est_bytes_moved=add_bytes * PROFILE_ITERS,
            )
        )
        results.append(
            run_section(
                name="Softmax",
                fn=lambda: repeat(do_softmax, PROFILE_ITERS),
                est_flops=soft_flops * PROFILE_ITERS,
                est_bytes_moved=soft_bytes * PROFILE_ITERS,
            )
        )

    # Profiler finished: safe to query
    op_table = list(prof.key_averages())

    # Use leaf ops to avoid double-counting:
    # - GEMM leaf: aten::mm (or sometimes aten::addmm / aten::_scaled_mm depending on path)
    # - Softmax leaf: aten::_softmax
    # - Add leaf: aten::add
    leaf_map: Dict[str, List[str]] = {
        "GEMM": ["aten::mm", "aten::addmm", "aten::_scaled_mm"],  # include a few common GEMM leaves
        "ElementwiseAdd": ["aten::add"],
        "Softmax": ["aten::_softmax"],
    }

    for r in results:
        cuda_ms, cpu_ms, alloc_b = sum_prof_for_ops(op_table, leaf_map.get(r.name, []))
        r.prof_cuda_ms = cuda_ms
        r.prof_cpu_ms = cpu_ms
        r.prof_cuda_alloc_bytes = alloc_b

    print_results(results)

    print("\n=== Top ops by CUDA time (profiler) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    trace_path = "llm_kernels_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nWrote Chrome trace: {os.path.abspath(trace_path)}")
    print("Open with: chrome://tracing or Perfetto UI")


if __name__ == "__main__":
    main()