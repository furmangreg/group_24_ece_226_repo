#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity

# -----------------------
# Hardware: RTX 2070 roofline ceilings
# -----------------------
PEAK_BW_GBPS = 448.0          # RTX 2070 memory bandwidth (GB/s)
PEAK_FP32_TFLOPS = 7.9        # approx RTX 2070 FP32 peak (TFLOP/s)
PEAK_FP16_TC_TFLOPS = 59.7    # approx RTX 2070 Tensor (FP16) peak (TFLOP/s)

# -----------------------
# Workload config
# -----------------------
DTYPE = torch.float16
DEVICE = "cuda"

B, T, H = 8, 256, 4096
FFN = 4 * H
HEADS = 32
SOFTMAX_T = 256

WARMUP = 10
ITERS = 30

REUSE_OUTPUTS = True  # reduce allocator churn for GEMM/add


@dataclass
class KernelStats:
    name: str
    time_s: float
    flops: int
    bytes_moved: int
    oi: float
    tflops: float
    est_bw_gbps: float
    # profiler-derived (leaf-self) stats
    prof_self_cuda_ms: float = 0.0
    prof_self_cpu_ms: float = 0.0
    prof_self_cuda_alloc_bytes: int = 0
    peak_alloc_bytes: int = 0
    peak_reserved_bytes: int = 0


def dtype_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(dtype)


def est_gemm(M: int, N: int, K: int, dtype: torch.dtype) -> Tuple[int, int]:
    # FLOPs: 2MNK; bytes: A + B + C (lower bound model)
    el = dtype_bytes(dtype)
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * el
    return flops, bytes_moved


def est_add(numel: int, dtype: torch.dtype) -> Tuple[int, int]:
    el = dtype_bytes(dtype)
    flops = numel
    bytes_moved = 3 * numel * el
    return flops, bytes_moved


def est_softmax(L: int, outer: int, dtype: torch.dtype) -> Tuple[int, int]:
    # very rough: ~7L ops per row, bytes ~ read+write = 2L elements per row
    el = dtype_bytes(dtype)
    flops = (7 * L) * outer
    bytes_moved = (2 * L * outer) * el
    return flops, bytes_moved


def time_region_s(fn, nvtx_name: str) -> Tuple[float, int, int]:
    """Return (seconds, peak_alloc_bytes, peak_reserved_bytes) for fn()."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    with nvtx.range(nvtx_name):
        s.record()
        fn()
        e.record()

    e.synchronize()
    t_s = s.elapsed_time(e) / 1e3
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return t_s, peak_alloc, peak_reserved


def repeat(fn, iters: int):
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()


def sum_prof_self_for_ops(op_table, op_exact_keys: List[str]) -> Tuple[float, float, int]:
    """
    Sum SELF times for exact keys to avoid parent/child double-counting.
    Returns (self_cuda_ms, self_cpu_ms, self_cuda_memory_usage_bytes).
    """
    keys = set(op_exact_keys)
    self_cuda_us = 0
    self_cpu_us = 0
    cuda_mem = 0
    for evt in op_table:
        if evt.key in keys:
            self_cuda_us += getattr(evt, "self_cuda_time_total", 0) or 0
            self_cpu_us += getattr(evt, "self_cpu_time_total", 0) or 0
            cuda_mem += getattr(evt, "self_cuda_memory_usage", 0) or 0
    return self_cuda_us / 1000.0, self_cpu_us / 1000.0, int(cuda_mem)


def roofline_curve(peak_bw_gbps: float, peak_tflops: float, oi_vals: List[float]) -> List[float]:
    # P = min(peak_compute, peak_bw * OI)
    return [min(peak_tflops, peak_bw_gbps * oi) for oi in oi_vals]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

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

    # Reused outputs to reduce allocator churn
    gemm_out = torch.empty(M, N, device=DEVICE, dtype=DTYPE) if REUSE_OUTPUTS else None
    add_out = torch.empty(M, N, device=DEVICE, dtype=DTYPE) if REUSE_OUTPUTS else None

    # Workloads
    def do_gemm_once():
        if gemm_out is None:
            _ = X @ W
        else:
            torch.matmul(X, W, out=gemm_out)

    def do_add_once():
        if add_out is None:
            _ = A + B2
        else:
            torch.add(A, B2, out=add_out)

    def do_softmax_once():
        _ = F.softmax(logits, dim=-1)

    # Warmup
    for _ in range(WARMUP):
        do_gemm_once()
        do_add_once()
        do_softmax_once()
    torch.cuda.synchronize()

    # Model estimates per call
    fl_g, by_g = est_gemm(M, N, K, DTYPE)
    fl_a, by_a = est_add(A.numel(), DTYPE)
    fl_s, by_s = est_softmax(SOFTMAX_T, B * HEADS * SOFTMAX_T, DTYPE)

    # Multiply by iterations (we time ITERS repeats)
    fl_g *= ITERS; by_g *= ITERS
    fl_a *= ITERS; by_a *= ITERS
    fl_s *= ITERS; by_s *= ITERS

    # Profile (framework-level) + NVTX labels + CUDA-event timing
    results: List[KernelStats] = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        # GEMM region
        t_s, peak_alloc, peak_reserved = time_region_s(
            fn=lambda: repeat(do_gemm_once, ITERS),
            nvtx_name="GEMM",
        )
        oi = fl_g / by_g
        tflops = (fl_g / t_s) / 1e12
        bw = (by_g / t_s) / 1e9
        results.append(KernelStats("GEMM", t_s, fl_g, by_g, oi, tflops, bw,
                                   peak_alloc_bytes=peak_alloc, peak_reserved_bytes=peak_reserved))

        # Add region
        t_s, peak_alloc, peak_reserved = time_region_s(
            fn=lambda: repeat(do_add_once, ITERS),
            nvtx_name="ElementwiseAdd",
        )
        oi = fl_a / by_a
        tflops = (fl_a / t_s) / 1e12
        bw = (by_a / t_s) / 1e9
        results.append(KernelStats("ElementwiseAdd", t_s, fl_a, by_a, oi, tflops, bw,
                                   peak_alloc_bytes=peak_alloc, peak_reserved_bytes=peak_reserved))

        # Softmax region
        t_s, peak_alloc, peak_reserved = time_region_s(
            fn=lambda: repeat(do_softmax_once, ITERS),
            nvtx_name="Softmax",
        )
        oi = fl_s / by_s
        tflops = (fl_s / t_s) / 1e12
        bw = (by_s / t_s) / 1e9
        results.append(KernelStats("Softmax", t_s, fl_s, by_s, oi, tflops, bw,
                                   peak_alloc_bytes=peak_alloc, peak_reserved_bytes=peak_reserved))

    # Profiler finished: safe to query
    op_table = list(prof.key_averages())

    # Leaf ops to avoid parent/child nesting double counts
    leaf_map: Dict[str, List[str]] = {
        "GEMM": ["aten::mm", "aten::addmm", "aten::_scaled_mm"],
        "ElementwiseAdd": ["aten::add"],
        "Softmax": ["aten::_softmax"],
    }

    for ks in results:
        self_cuda_ms, self_cpu_ms, self_alloc_b = sum_prof_self_for_ops(op_table, leaf_map[ks.name])
        ks.prof_self_cuda_ms = self_cuda_ms
        ks.prof_self_cpu_ms = self_cpu_ms
        ks.prof_self_cuda_alloc_bytes = self_alloc_b

    # Print summary
    print("\n=== Kernel stats (CUDA events + model OI) ===")
    for ks in results:
        print(f"\n[{ks.name}]")
        print(f"  time (CUDA events):     {ks.time_s*1e3:,.3f} ms")
        print(f"  OI (FLOP/byte):         {ks.oi:,.6f}")
        print(f"  achieved perf:          {ks.tflops:,.3f} TFLOP/s (model flops / event time)")
        print(f"  est bandwidth:          {ks.est_bw_gbps:,.2f} GB/s (model bytes / event time)")
        print(f"  profiler self CUDA:     {ks.prof_self_cuda_ms:,.3f} ms (leaf op)")
        print(f"  profiler self CPU:      {ks.prof_self_cpu_ms:,.3f} ms (leaf op)")
        print(f"  profiler alloc churn:   {ks.prof_self_cuda_alloc_bytes / (1024**2):,.2f} MiB (self_cuda_memory_usage)")
        print(f"  peak allocated (region): {ks.peak_alloc_bytes / (1024**2):,.2f} MiB")
        print(f"  peak reserved (region):  {ks.peak_reserved_bytes / (1024**2):,.2f} MiB")

    print("\n=== Top ops by CUDA time (profiler) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    # Export trace
    trace_path = "llm_kernels_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nWrote Chrome trace: {os.path.abspath(trace_path)}")

    # -----------------------
    # Plot roofline(s)
    # -----------------------
    oi_vals = [10**x for x in [ -3, -2, -1, 0, 1, 2, 3, 4 ]]
    y_fp32 = roofline_curve(PEAK_BW_GBPS, PEAK_FP32_TFLOPS, oi_vals)
    y_fp16tc = roofline_curve(PEAK_BW_GBPS, PEAK_FP16_TC_TFLOPS, oi_vals)
    bw_slope = [(PEAK_BW_GBPS * oi) / 1000 for oi in oi_vals]

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")

    plt.plot(oi_vals, y_fp32, label=f"Roofline (FP32 peak={PEAK_FP32_TFLOPS:g} TF/s)")
    plt.plot(oi_vals, y_fp16tc, label=f"Roofline (FP16 TC peak={PEAK_FP16_TC_TFLOPS:g} TF/s)")
    plt.plot(oi_vals, bw_slope, linestyle="--", label=f"Bandwidth slope (BW={PEAK_BW_GBPS:g} GB/s)")

    for ks in results:
        plt.scatter([ks.oi], [ks.tflops])
        plt.text(ks.oi, ks.tflops, f"  {ks.name}", va="center")

    plt.xlabel("Operational Intensity (FLOPs / Byte)")
    plt.ylabel("Performance (TFLOP/s)")
    plt.title("RTX 2070 Roofline + LLM Kernels (model-based bytes)")
    plt.legend()
    plt.tight_layout()
    out = "roofline.png"
    plt.savefig(out, dpi=200)
    print(f"Saved plot: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()