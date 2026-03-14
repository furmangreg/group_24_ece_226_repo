#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=["gemm", "add", "softmax"], required=True)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    # Shapes (keep defaults; driver assumes these unless you also update it)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--H", type=int, default=4096)
    p.add_argument("--HEADS", type=int, default=32)
    p.add_argument("--SOFTMAX_T", type=int, default=256)
    args = p.parse_args()

    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    torch.manual_seed(0)

    # Make matmul use tensor cores when applicable
    torch.backends.cuda.matmul.allow_tf32 = True

    B, T, H = args.B, args.T, args.H
    FFN = 4 * H
    HEADS = args.HEADS
    ST = args.SOFTMAX_T

    M = B * T
    K = H
    N = FFN

    # Allocate once
    X = torch.randn(M, K, device=device, dtype=dtype)
    W = torch.randn(K, N, device=device, dtype=dtype)

    A = torch.randn(M, N, device=device, dtype=dtype)
    B2 = torch.randn(M, N, device=device, dtype=dtype)

    logits = torch.randn(B, HEADS, ST, ST, device=device, dtype=dtype)

    # Reuse outputs for GEMM + ADD (avoid alloc noise)
    gemm_out = torch.empty(M, N, device=device, dtype=dtype)
    add_out = torch.empty(M, N, device=device, dtype=dtype)

    def run_gemm():
        torch.matmul(X, W, out=gemm_out)

    def run_add():
        torch.add(A, B2, out=add_out)

    def run_softmax():
        _ = F.softmax(logits, dim=-1)

    if args.kernel == "gemm":
        fn, name = run_gemm, "GEMM"
    elif args.kernel == "add":
        fn, name = run_add, "ElementwiseAdd"
    else:
        fn, name = run_softmax, "Softmax"

    # Warmup
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    # NVTX-labeled region
    nvtx.range_push(name)
    for _ in range(args.iters):
        fn()
    nvtx.range_pop()

    torch.cuda.synchronize()
    print("done")

if __name__ == "__main__":
    main()