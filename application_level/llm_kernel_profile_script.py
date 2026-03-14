#!/usr/bin/env python3
"""
Profile 3 distinct "LLM-ish" GPU kernels in PyTorch:
  1) GEMM (matmul)
  2) Element-wise add
  3) Softmax

Uses:
  - NVTX ranges (labels in traces)
  - CUDA events (precise region timing)
  - torch.profiler (op-level CPU/CUDA time + allocator memory attribution)

Outputs:
  - Per-kernel region timing (CUDA events)
  - Per-kernel op summaries from torch.profiler (after profiler finishes)
  - Allocator peak memory stats per region
  - Simple model-based estimates of bytes moved + FLOPs

IMPORTANT LIMITATION:
  Without Nsight Compute / hardware counters, you cannot get true DRAM/L2 traffic,
  cache hit rates, achieved occupancy, etc. This script reports:
    - PyTorch allocator memory (real allocations/reserved)
    - Profiler self_cuda_memory_usage (allocator bytes attributed to ops)
    - Estimated bytes moved from tensor sizes (a model)
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, ProfilerActivity


# ----------------------------
# Config you might tweak
# ----------------------------
DTYPE = torch.float16
DEVICE = "cuda"

# GEMM: [B*T, H] x [H, 4H] ~ MLP projection-ish
B = 8
T = 256
H = 4096
FFN = 4 * H

# Softmax: attention logits [B, heads, T, T]
HEADS = 32
SOFTMAX_T = 256  # softmax along last dim

WARMUP_ITERS = 10
PROFILE_ITERS = 30


@dataclass
class SectionResult:
    name: str
    cuda_event_ms: float

    # Filled AFTER profiler finishes
    prof_cuda_ms: float = 0.0
    prof_cpu_ms: float = 0.0
    prof_cuda_alloc_bytes: int = 0

    # Region allocator stats
    peak_alloc_bytes: int = 0
    peak_reserved_bytes: int = 0

    # Simple model-based estimates
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
    """
    FLOPs ~ 2*M*N*K (FMA=2 flops).
    Bytes moved ~ read A + read B + write C (ignores reuse/caches/epilogues).
    """
    el = sizeof_dtype(dtype)
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * el
    return flops, bytes_moved


def estimate_ewise_add(numel: int, dtype: torch.dtype) -> Tuple[int, int]:
    """Add: ~1 flop/element. Bytes: read x + read y + write out."""
    el = sizeof_dtype(dtype)
    flops = numel
    bytes_moved = (numel * 3) * el
    return flops, bytes_moved


def estimate_softmax(last_dim: int, outer: int, dtype: torch.dtype) -> Tuple[int, int]:
    """
    Rough op-count model for softmax per row length L:
      max reduce ~ (L-1)
      subtract ~ L
      exp ~ L
      sum reduce ~ (L-1)
      divide ~ L
    ~ ~5L + 2(L-1) ~ about 7L (very rough).
    Bytes moved ~ read input + write output ~ 2*L elements per row (ignores temp).
    """
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
    # Reset per-region allocator peaks
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


def build_prof_op_table(prof) -> List:
    # Must only be called AFTER the profiler context exits.
    return list(prof.key_averages())


def sum_prof_for_ops(op_table: List, op_substrings: List[str]) -> Tuple[float, float, int]:
    """
    Sum profiler times/memory over events whose key contains any substring.
    Returns: (cuda_ms, cpu_ms, self_cuda_memory_usage_bytes)
    """
    cuda_us = 0
    cpu_us = 0
    cuda_mem = 0

    for evt in op_table:
        k = evt.key
        if any(s in k for s in op_substrings):
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
        print(f"  Profiler CUDA time total:  {fmt_ms(r.prof_cuda_ms)}  (matched ops only)")
        print(f"  Profiler CPU time total:   {fmt_ms(r.prof_cpu_ms)}   (matched ops only)")
        print(f"  Profiler CUDA alloc bytes: {fmt_bytes(r.prof_cuda_alloc_bytes)} (self_cuda_memory_usage)")
        print(f"  Peak allocated (region):   {fmt_bytes(r.peak_alloc_bytes)}")
        print(f"  Peak reserved (region):    {fmt_bytes(r.peak_reserved_bytes)}")
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

    # ----------------------------
    # Create inputs
    # ----------------------------
    # GEMM: X [M, K], W [K, N]
    M = B * T
    K = H
    N = FFN

    X = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    W = torch.randn(K, N, device=DEVICE, dtype=DTYPE)

    # Elementwise add: tensors same shape as GEMM output
    # Note: out-of-place add will allocate output; that's fine for profiling.
    Y = torch.randn(M, N, device=DEVICE, dtype=DTYPE)
    Z = torch.randn(M, N, device=DEVICE, dtype=DTYPE)

    # Softmax logits [B, heads, T, T]
    logits = torch.randn(B, HEADS, SOFTMAX_T, SOFTMAX_T, device=DEVICE, dtype=DTYPE)

    # ----------------------------
    # Estimate compute/memory (model-based)
    # ----------------------------
    gemm_flops, gemm_bytes = estimate_gemm(M, N, K, DTYPE)
    add_flops, add_bytes = estimate_ewise_add(Y.numel(), DTYPE)
    soft_flops, soft_bytes = estimate_softmax(
        last_dim=SOFTMAX_T,
        outer=B * HEADS * SOFTMAX_T,  # rows = B*heads*T
        dtype=DTYPE,
    )

    # ----------------------------
    # Define workloads
    # ----------------------------
    def do_gemm():
        _ = X @ W

    def do_add():
        _ = Y + Z

    def do_softmax():
        _ = F.softmax(logits, dim=-1)

    # ----------------------------
    # Warmup
    # ----------------------------
    with nvtx.range("warmup"):
        for _ in range(WARMUP_ITERS):
            do_gemm()
            do_add()
            do_softmax()
    torch.cuda.synchronize()

    # ----------------------------
    # Profile
    # ----------------------------
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

    # ----------------------------
    # Profiler has finished: safe to query key_averages()
    # ----------------------------
    op_table = build_prof_op_table(prof)

    # These substring matches are intentionally broad, but can overcount if your script
    # includes other ops with the same names. In this script it's pretty clean.
    op_map: Dict[str, List[str]] = {
        "GEMM": ["aten::matmul", "aten::mm", "aten::addmm", "aten::_scaled_mm"],
        # include add variants; exact keys vary by PyTorch version
        "ElementwiseAdd": ["aten::add"],
        "Softmax": ["aten::softmax", "aten::_softmax"],
    }

    for r in results:
        subs = op_map.get(r.name, [])
        cuda_ms, cpu_ms, alloc_b = sum_prof_for_ops(op_table, subs)
        r.prof_cuda_ms = cuda_ms
        r.prof_cpu_ms = cpu_ms
        r.prof_cuda_alloc_bytes = alloc_b

    # ----------------------------
    # Print results + a profiler table
    # ----------------------------
    print_results(results)

    print("\n=== Top ops by CUDA time (profiler) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    # Optional: export a trace for timeline inspection
    trace_path = "llm_kernels_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nWrote Chrome trace: {os.path.abspath(trace_path)}")
    print("Open with: chrome://tracing or Perfetto UI")


if __name__ == "__main__":
    main()