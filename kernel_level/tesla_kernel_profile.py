"""
Roofline plot — uses analytic operational intensity and kernel runtime
to estimate achieved performance.

Run this in a Colab cell.

Before running, update the three duration values below using the
gpu__time_duration.sum column from your Nsight Compute CSV output
(values are in nanoseconds).
"""

import numpy as np
import matplotlib.pyplot as plt
# Peak performance numbers taken from NVIDIA specs

PEAK_FLOPS_FP16_TENSOR = 65e12   # 65 TFLOP/s using FP16 tensor cores
PEAK_FLOPS_FP32        = 8.1e12  # 8.1 TFLOP/s using FP32 CUDA cores
PEAK_BW                = 320e9   # 320 GB/s memory bandwidth

# Ridge point separates memory-bound vs compute-bound
RIDGE_POINT = PEAK_FLOPS_FP16_TENSOR / PEAK_BW

# Kernel dimensions (must match kernels.py)

B, T, H = 8, 256, 4096
HEADS   = 32
FFN     = 4 * H

# Derived matrix dimensions used in the kernels
M = B * T        # 2048
K = H            # 4096
N = FFN          # 16384

BYTES_PER_ELEM = 2   # FP16 uses 2 bytes per value

# Each output element does K multiply-adds
gemm_flops = 2 * M * N * K

# Memory read/write: X and W read, Y written
gemm_bytes = BYTES_PER_ELEM * (M * K + K * N + M * N)

# Operational intensity
gemm_oi = gemm_flops / gemm_bytes

# One addition per element
elem_flops = M * N

# Read A and B2, write Z
elem_bytes = BYTES_PER_ELEM * (M * N + M * N + M * N)

elem_oi = elem_flops / elem_bytes

num_elements  = B * HEADS * T * T

softmax_flops = 5 * num_elements

# Read logits and write output
softmax_bytes = BYTES_PER_ELEM * (num_elements + num_elements)

softmax_oi = softmax_flops / softmax_bytes

print("=" * 60)
print("Analytic Operational Intensity")
print("=" * 60)
print(f"GEMM:            {gemm_flops/1e9:.2f} GFLOP / {gemm_bytes/1e6:.1f} MB = {gemm_oi:.1f} FLOP/B")
print(f"Element-wise Add: {elem_flops/1e6:.2f} MFLOP / {elem_bytes/1e6:.1f} MB = {elem_oi:.3f} FLOP/B")
print(f"Softmax:          {softmax_flops/1e6:.2f} MFLOP / {softmax_bytes/1e6:.1f} MB = {softmax_oi:.2f} FLOP/B")
print(f"\nRidge point: {RIDGE_POINT:.1f} FLOP/B")
print("=" * 60)

gemm_duration_ns    = 13.08e6    # 13.08 ms
elem_duration_ns    = 0.79e6     # 0.79 ms
softmax_duration_ns = 0.35e6     # 0.35 ms

# Convert runtime into achieved FLOP/s
gemm_perf    = gemm_flops    / (gemm_duration_ns * 1e-9)
elem_perf    = elem_flops    / (elem_duration_ns * 1e-9)
softmax_perf = softmax_flops / (softmax_duration_ns * 1e-9)

print(f"\nAchieved Performance")
print(f"GEMM:             {gemm_perf/1e12:.2f} TFLOP/s")
print(f"Element-wise Add: {elem_perf/1e9:.2f} GFLOP/s")
print(f"Softmax:          {softmax_perf/1e9:.2f} GFLOP/s")

# Build roofline plot

kernels = [
    ("GEMM",             gemm_oi,    gemm_perf,    "red"),
    ("Element-wise Add", elem_oi,    elem_perf,    "green"),
    ("Softmax",          softmax_oi, softmax_perf, "orange"),
]

fig, ax = plt.subplots(figsize=(11, 7))

# Generate roofline curve
x = np.logspace(-2, 4, 1000)

# Memory roof
y_mem = PEAK_BW * x

# Compute roof
y_compute = np.full_like(x, PEAK_FLOPS_FP16_TENSOR)

# Final roofline = min(memory roof, compute roof)
y_roof = np.minimum(y_mem, y_compute)

ax.plot(x, y_roof / 1e12, 'b-', linewidth=2.5, label='Roofline (FP16 Tensor)')

# Reference lines
ax.axhline(y=PEAK_FLOPS_FP16_TENSOR / 1e12, color='blue', linestyle='--',
           alpha=0.3, label=f'Peak Compute: {PEAK_FLOPS_FP16_TENSOR/1e12:.0f} TFLOP/s')

ax.axvline(x=RIDGE_POINT, color='gray', linestyle=':', alpha=0.5,
           label=f'Ridge Point: {RIDGE_POINT:.0f} FLOP/B')

# Optional FP32 roofline
y_roof_fp32 = np.minimum(PEAK_BW * x, PEAK_FLOPS_FP32)

ax.plot(x, y_roof_fp32 / 1e12, 'c-', linewidth=2, alpha=0.4,
        label=f'Roofline (FP32 CUDA Core, {PEAK_FLOPS_FP32/1e12:.1f} TF/s)')

# Plot kernels
for name, oi, perf, color in kernels:
    ax.scatter(oi, perf / 1e12, s=180, c=color,
               edgecolors='black', linewidth=1.2)

    ax.annotate(name,
                (oi, perf / 1e12),
                textcoords="offset points",
                xytext=(12, 12),
                fontsize=11,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Plot formatting
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Operational Intensity [FLOP/Byte]', fontsize=13)
ax.set_ylabel('Performance [TFLOP/s]', fontsize=13)

ax.set_title('Roofline Model — Tesla T4 (FP16)\n'
             'Kernel Classification for LLM Operations',
             fontsize=13)

ax.legend(fontsize=9, loc='lower right')

ax.set_xlim(1e-2, 1e4)
ax.set_ylim(1e-3, 200)

ax.grid(True, which='both', alpha=0.2)

# Labels showing the two regions
ax.text(0.08, 0.55, 'MEMORY\nBOUND',
        fontsize=16, alpha=0.15,
        ha='center', transform=ax.transAxes)

ax.text(0.88, 0.55, 'COMPUTE\nBOUND',
        fontsize=16, alpha=0.15,
        ha='center', transform=ax.transAxes)

plt.tight_layout()

plt.savefig('/content/roofline_plot.png', dpi=200, bbox_inches='tight')

plt.show()

# ==============================================================
# 6. Classify kernels
# ==============================================================

print("\n" + "=" * 70)
print("KERNEL CLASSIFICATION")
print("=" * 70)

for name, oi, perf, _ in kernels:

    if oi > RIDGE_POINT:
        bound = "COMPUTE-BOUND"
        strategy = "Use tensor cores, mixed precision, or fuse operators"
    else:
        bound = "MEMORY-BOUND"
        strategy = "Reduce memory traffic (quantization or kernel fusion)"

    print(f"  {name:20s} | OI = {oi:>8.2f} FLOP/B | {bound}")
    print(f"  {'':20s} | {strategy}")

print("=" * 70)