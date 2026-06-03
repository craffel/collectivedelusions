import matplotlib.pyplot as plt
import numpy as np

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'text.usetex': False  # False for broader compatibility on headless CPUs
})

# ---------------- Figure 1: Calibration Budget N Sweep ----------------
N = [16, 64, 128]

# Weight Averaging Accuracies
wa_none = [38.46, 38.46, 38.46]
wa_sptaac = [45.21, 46.33, 45.89]
wa_taac = [24.27, 30.84, 31.85]
wa_fdsa = [42.08, 43.33, 43.14]
wa_sscc = [46.44, 46.54, 46.42]

# Task Arithmetic Accuracies
ta_none = [38.13, 38.13, 38.13]
ta_sptaac = [41.32, 42.11, 41.36]
ta_taac = [18.60, 20.01, 20.01]
ta_fdsa = [36.74, 37.95, 37.60]
ta_sscc = [42.66, 43.17, 43.03]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

# Plot Weight Averaging
ax1.plot(N, wa_sscc, marker='o', linewidth=2.5, color='#2ca02c', label='SSCC (Ours, Fused)')
ax1.plot(N, wa_sptaac, marker='s', linewidth=1.8, color='#1f77b4', label='SP-TAAC')
ax1.plot(N, wa_fdsa, marker='^', linewidth=1.8, color='#ff7f0e', label='FDSA (Hook-based)')
ax1.plot(N, wa_none, marker='x', linewidth=1.5, linestyle='--', color='#7f7f7f', label='None (Uncalibrated)')
ax1.plot(N, wa_taac, marker='d', linewidth=1.5, color='#d62728', label='TAAC (Sparsity Trap)')
ax1.set_title("Weight Averaging (WA)")
ax1.set_xlabel("Calibration Budget (N)")
ax1.set_ylabel("Average Multi-Task Accuracy (%)")
ax1.set_xticks(N)
ax1.set_xticklabels([str(n) for n in N])
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot Task Arithmetic
ax2.plot(N, ta_sscc, marker='o', linewidth=2.5, color='#2ca02c', label='SSCC (Ours, Fused)')
ax2.plot(N, ta_sptaac, marker='s', linewidth=1.8, color='#1f77b4', label='SP-TAAC')
ax2.plot(N, ta_fdsa, marker='^', linewidth=1.8, color='#ff7f0e', label='FDSA (Hook-based)')
ax2.plot(N, ta_none, marker='x', linewidth=1.5, linestyle='--', color='#7f7f7f', label='None (Uncalibrated)')
ax2.plot(N, ta_taac, marker='d', linewidth=1.5, color='#d62728', label='TAAC (Sparsity Trap)')
ax2.set_title(r"Task Arithmetic (TA, $\lambda = 0.3$)")
ax2.set_xlabel("Calibration Budget (N)")
ax2.set_xticks(N)
ax2.set_xticklabels([str(n) for n in N])
ax2.grid(True, linestyle='--', alpha=0.6)

# Put legend on the first plot or shared legend
ax2.legend(loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig("calibration_budget.pdf", bbox_inches='tight', dpi=300)
plt.close()
print("Saved calibration_budget.pdf")

# ---------------- Figure 2: Kernel Size Sweep (Capacity-Regularization) ----------------
k = [1, 3, 5, 7]
mnist = [53.08, 50.75, 40.47, 35.25]
fmnist = [58.82, 63.32, 63.34, 62.69]
cifar = [22.99, 25.56, 30.96, 32.59]
avg = [44.96, 46.54, 44.92, 43.51]

plt.figure(figsize=(6.5, 4.5))
plt.plot(k, avg, marker='o', linewidth=3.0, color='#2ca02c', label='Multi-Task Average')
plt.plot(k, mnist, marker='s', linewidth=1.8, color='#1f77b4', label='MNIST')
plt.plot(k, fmnist, marker='^', linewidth=1.8, color='#9467bd', label='Fashion-MNIST')
plt.plot(k, cifar, marker='d', linewidth=1.8, color='#d62728', label='CIFAR-10')

plt.title("The Capacity-Regularization Trade-Off (SSCC, N=64, WA)")
plt.xlabel("Convolutional Kernel Size ($k$)")
plt.ylabel("Classification Accuracy (%)")
plt.xticks(k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="best", frameon=True)

plt.tight_layout()
plt.savefig("kernel_sweep.pdf", bbox_inches='tight', dpi=300)
plt.close()
print("Saved kernel_sweep.pdf")
