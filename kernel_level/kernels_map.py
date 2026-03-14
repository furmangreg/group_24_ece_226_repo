import math
import torch

torch.set_grad_enabled(False)

def time_op(op_fn, iters=200, warmup=50):
    # Warmup
    for _ in range(warmup):
        op_fn()
    torch.cuda.synchronize()

    # CUDA events timing (accurate for async GPU work)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        op_fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    return ms / iters  # avg ms per iter

def gemm_case(M=4096, K=4096, N=4096, dtype=torch.float16):
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

    def op():
        return A @ B

    t_ms = time_op(op)

    # FLOPs for GEMM: 2*M*N*K (mul+add)
    flops = 2 * M * N * K

    # Lower-bound bytes moved from HBM:
    # read A + read B + write C
    bytes_moved = (A.numel() + B.numel() + (M*N)) * A.element_size()

    return ("GEMM", t_ms, flops, bytes_moved)

def add_case(B=32, S=2048, H=4096, dtype=torch.float16):
    X = torch.randn(B, S, H, device="cuda", dtype=dtype)
    Y = torch.randn(B, S, H, device="cuda", dtype=dtype)

    def op():
        return X + Y

    t_ms = time_op(op)

    # FLOPs: one add per element
    flops = X.numel()

    # Lower-bound bytes: read X + read Y + write out
    bytes_moved = (X.numel() + Y.numel() + X.numel()) * X.element_size()

    return ("ElementwiseAdd", t_ms, flops, bytes_moved)

def softmax_case(B=32, Hh=32, S=2048, dtype=torch.float16):
    # attention scores shape: (B, heads, S, S)
    # This is huge if S=2048; reduce to something realistic for profiling runtime.
    # We'll pick S=512 by default in main, and you can scale later.
    scores = torch.randn(B, Hh, S, S, device="cuda", dtype=dtype)

    def op():
        return torch.softmax(scores, dim=-1)

    t_ms = time_op(op, iters=100, warmup=20)

    # FLOPs for softmax per row (length S):
    # exp: S
    # sum: (S-1) adds
    # div: S
    # ~ (3S - 1) ops per row (rough)
    rows = B * Hh * S
    flops = rows * (3*S - 1)

    # Bytes lower-bound (very approximate): read scores + write output
    # (ignores extra reads/writes from multiple passes)
    bytes_moved = (scores.numel() + scores.numel()) * scores.element_size()

    return ("Softmax", t_ms, flops, bytes_moved)

def summarize(name, t_ms, flops, bytes_moved):
    sec = t_ms / 1e3
    tflops = flops / sec / 1e12
    gbps = bytes_moved / sec / 1e9
    oi = flops / bytes_moved  # FLOPs/byte
    print(f"{name:14s}  {t_ms:8.4f} ms   OI={oi:8.3f} FLOP/byte   {tflops:8.3f} TFLOP/s   {gbps:8.1f} GB/s")

if __name__ == "__main__":
    torch.manual_seed(0)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print()

    # GEMM: keep large to be compute-heavy
    summarize(*gemm_case(M=4096, K=4096, N=4096, dtype=torch.float16))

    # Add: use LLM-ish activation shape
    summarize(*add_case(B=32, S=2048, H=4096, dtype=torch.float16))

    # Softmax: S=512 to avoid absurd memory blowups; adjust later if needed
    summarize(*softmax_case(B=16, Hh=32, S=512, dtype=torch.float16))