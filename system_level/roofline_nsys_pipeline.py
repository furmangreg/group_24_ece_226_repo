#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -------- Tesla T4 theoretical ceilings --------
# Bandwidth 320 GB/s, FP32 8.1 TFLOP/s, FP16/BF16 TC 65 TFLOP/s
PEAK_BW_GBPS = 320.0
PEAK_FP32_TFLOPS = 8.1
PEAK_FP16_TC_TFLOPS = 65.0

# -------- A100-SXM4-40GB theoretical ceilings --------
# Bandwidth 1555 GB/s, FP32 19.5 TFLOP/s, FP16/BF16 TC 312 TFLOP/s
#PEAK_BW_GBPS = 1555.0
#PEAK_FP32_TFLOPS = 19.5
#PEAK_FP16_TC_TFLOPS = 312.0

@dataclass
class KernelSpec:
    name: str
    flops: float
    bytes_moved: float
    report_prefix: str

def sh(cmd: List[str], check=True) -> str:
    # Returns stdout text
    p = subprocess.run(cmd, check=check, text=True, capture_output=True)
    return p.stdout

def run_profile(prefix: str, kernel: str, iters: int, warmup: int, dtype: str,
                B: int, T: int, H: int, HEADS: int, SOFTMAX_T: int) -> str:
    """
    Creates prefix.qdrep (and prefix.sqlite).
    Use small traces: cuda + nvtx only, no sampling.
    """
    cmd = [
        "nsys", "profile",
        "-o", prefix,
        "-t", "cuda,nvtx",
        "--sample=none",
        "--trace-fork-before-exec=true",
        "python", "llm_kernels_workload.py",
        "--kernel", kernel,
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--dtype", dtype,
        "--B", str(B),
        "--T", str(T),
        "--H", str(H),
        "--HEADS", str(HEADS),
        "--SOFTMAX_T", str(SOFTMAX_T),
    ]
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return f"{prefix}.nsys-rep"
import csv
from io import StringIO

def nsys_kernel_sum_csv(nsys_rep: str) -> str:
    # CSV output is much easier to parse across versions
    cmd = ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", nsys_rep]
    return sh(cmd)

def parse_cuda_gpu_kern_sum_csv(csv_text: str) -> tuple[str, float, int]:
    """
    Parse cuda_gpu_kern_sum in CSV form.

    Returns (kernel_name, total_time_ms, instances)
    """
    # nsys often prints some non-CSV lines before/after; keep only CSV-ish lines
    lines = [ln for ln in csv_text.splitlines() if "," in ln and not ln.lower().startswith("report")]

    if not lines:
        raise RuntimeError("No CSV-like lines found in nsys stats output.")

    reader = csv.DictReader(StringIO("\n".join(lines)))
    rows = list(reader)
    if not rows:
        raise RuntimeError("CSV parsed but no rows found.")

    # Figure out which columns exist (varies by version)
    cols = {c.lower(): c for c in rows[0].keys()}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    name_col = pick("name", "kernel name", "kernel", "demangled name")
    inst_col = pick("instances", "calls", "count")
    total_col = pick("total time (ns)", "total time", "total time [ns]", "total time (us)", "total time (ms)")

    if name_col is None or inst_col is None or total_col is None:
        raise RuntimeError(f"Unexpected CSV columns: {list(rows[0].keys())}")

    # Convert total time to ns if needed
    total_header = total_col.lower()

    best = None  # (total_ns, instances, name)
    for r in rows:
        name = (r.get(name_col) or "").strip()
        if not name:
            continue

        # instances
        try:
            inst = int(str(r.get(inst_col)).replace(",", "").strip())
        except:
            continue

        # total time
        raw = str(r.get(total_col)).replace(",", "").strip()
        try:
            val = float(raw)
        except:
            continue

        # normalize to ns
        if "(ns)" in total_header or "[ns]" in total_header:
            total_ns = val
        elif "(us)" in total_header:
            total_ns = val * 1e3
        elif "(ms)" in total_header:
            total_ns = val * 1e6
        else:
            # if unknown, assume ns (most common)
            total_ns = val

        if best is None or total_ns > best[0]:
            best = (total_ns, inst, name)

    if best is None:
        raise RuntimeError("Could not select a dominant kernel row from CSV.")

    total_ms = best[0] / 1e6
    return best[2], total_ms, best[1]

# ---------- Analytical models (match workload shapes) ----------
def gemm_flops(M, N, K):
    return 2.0 * M * N * K

def gemm_bytes(M, N, K, dtype_bytes):
    return dtype_bytes * (M*K + K*N + M*N)

def add_flops(M, N):
    return float(M * N)

def add_bytes(M, N, dtype_bytes):
    return dtype_bytes * (3.0 * M * N)

def softmax_flops(B, HEADS, T):
    # Simple roofline-friendly model: ~5 FLOPs per element
    elems = B * HEADS * T * T
    return 5.0 * elems

def softmax_bytes(B, HEADS, T, dtype_bytes):
    # Read logits + write output (lower bound)
    elems = B * HEADS * T * T
    return dtype_bytes * (2.0 * elems)

def roofline_curve(peak_compute_tflops: float, oi_vals):
    return np.minimum(peak_compute_tflops, (PEAK_BW_GBPS * oi_vals) / 1000.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--HEADS", type=int, default=32)
    ap.add_argument("--SOFTMAX_T", type=int, default=256)
    args = ap.parse_args()

    dtype_bytes = 2 if args.dtype == "fp16" else 4

    B, T, H = args.B, args.T, args.H
    FFN = 4 * H
    HEADS = args.HEADS
    ST = args.SOFTMAX_T

    M = B * T
    K = H
    N = FFN

    specs = [
        KernelSpec("GEMM", gemm_flops(M, N, K), gemm_bytes(M, N, K, dtype_bytes), "nsys_gemm"),
        KernelSpec("ElementwiseAdd", add_flops(M, N), add_bytes(M, N, dtype_bytes), "nsys_add"),
        KernelSpec("Softmax", softmax_flops(B, HEADS, ST), softmax_bytes(B, HEADS, ST, dtype_bytes), "nsys_softmax"),
    ]

    results = {}

    # Profile each kernel separately so the stats are clean
    for spec in specs:
        kernel_flag = "gemm" if spec.name == "GEMM" else ("add" if spec.name == "ElementwiseAdd" else "softmax")
        qdrep = run_profile(
            prefix=spec.report_prefix,
            kernel=kernel_flag,
            iters=args.iters,
            warmup=args.warmup,
            dtype=args.dtype,
            B=B, T=T, H=H, HEADS=HEADS, SOFTMAX_T=ST
        )
        stats_csv = nsys_kernel_sum_csv(qdrep)
        dom_name, total_ms, instances = parse_cuda_gpu_kern_sum_csv(stats_csv)

        # Use total GPU kernel time of dominant kernel; divide by iters to get per-iter kernel time
        per_iter_ms = total_ms / max(1, instances)

        oi = spec.flops / spec.bytes_moved
        perf_tflops = (spec.flops / (per_iter_ms / 1000.0)) / 1e12

        results[spec.name] = {
            "dominant_kernel_name": dom_name,
            "instances": instances,
            "total_ms": total_ms,
            "per_iter_ms": per_iter_ms,
            "flops": spec.flops,
            "bytes": spec.bytes_moved,
            "oi": oi,
            "tflops": perf_tflops,
            "qdrep": f"{spec.report_prefix}.qdrep",
        }

    # Print summary
    print("\n=== Results (nsys time + analytic OI) ===\n")
    for k, r in results.items():
        print(f"[{k}]")
        print(f"  Dominant kernel:      {r['dominant_kernel_name']}")
        print(f"  Instances (counted):  {r['instances']}")
        print(f"  Total GPU time:       {r['total_ms']:.3f} ms")
        print(f"  Per-iter GPU time:    {r['per_iter_ms']:.6f} ms")
        print(f"  OI (FLOP/byte):       {r['oi']:.6g}")
        print(f"  Achieved perf:        {r['tflops']:.3f} TFLOP/s")
        print(f"  Bytes model:          {r['bytes']/1e9:.6f} GB")
        print()

    # Plot roofline
    oi_vals = np.logspace(-3, 4, 400)

    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.yscale("log")

    plt.plot(oi_vals, roofline_curve(PEAK_FP32_TFLOPS, oi_vals),
             label=f"FP32 roof ({PEAK_FP32_TFLOPS:g} TF/s)")
    plt.plot(oi_vals, roofline_curve(PEAK_FP16_TC_TFLOPS, oi_vals),
             label=f"FP16 Tensor roof ({PEAK_FP16_TC_TFLOPS:g} TF/s)")
    plt.plot(oi_vals, (PEAK_BW_GBPS * oi_vals) / 1000.0,
             linestyle="--", label=f"HBM BW slope ({PEAK_BW_GBPS:g} GB/s)")

    for name, r in results.items():
        plt.scatter([r["oi"]], [r["tflops"]])
        plt.text(r["oi"] * 1.05, r["tflops"] * 1.05, name, va="center")

    plt.xlabel("Operational Intensity (FLOPs / Byte)")
    plt.ylabel("Performance (TFLOP/s)")
    #plt.title("Roofline (A100) — time from Nsight Systems, OI from analytic model")
    plt.title("Roofline (T4) — time from Nsight Systems, OI from analytic model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roofline_nsys.png", dpi=200)
    print("Saved roofline_nsys.png")

    # Also save machine-readable JSON
    with open("roofline_nsys_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved roofline_nsys_results.json")

if __name__ == "__main__":
    main()