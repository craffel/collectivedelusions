import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ----------------------------------
# Figure 1: Main Merging & R_l Ratio
# ----------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Data for Plot 1: Accuracy vs Lambda
lambdas = [0.1, 0.3, 0.5, 0.7, 1.0]
no_calib = [10.01, 11.53, 25.09, 39.72, 23.70]
repair = [10.21, 10.35, 28.29, 42.35, 29.22]
spttbc = [53.42, 73.27, 73.12, 70.36, 64.72]
shared_bc = [33.20, 58.92, 63.88, 63.40, 59.36]
ts_bc = [53.10, 73.64, 72.88, 70.27, 64.73]

ax1.plot(lambdas, no_calib, marker='o', linestyle='--', color='#d62728', label='No Calibration', linewidth=1.5)
ax1.plot(lambdas, repair, marker='s', linestyle='-.', color='#ff7f0e', label='REPAIR', linewidth=1.5)
ax1.plot(lambdas, shared_bc, marker='x', linestyle=':', color='#9467bd', label='Shared BN Calib (Offline)', linewidth=1.5)
ax1.plot(lambdas, spttbc, marker='^', linestyle=':', color='#2ca02c', label='SP-TTBC (Test-Time, batch=64)', linewidth=1.5)
ax1.plot(lambdas, ts_bc, marker='D', linestyle='-', color='#1f77b4', label='TS-BN Calib (Ours, Offline, batch=1)', linewidth=2.5)

ax1.set_title(r"Multi-Task Average Accuracy vs. Scale $\lambda$ (TA)", fontsize=12, fontweight='bold')
ax1.set_xlabel(r"Scaling Factor $\lambda$", fontsize=11)
ax1.set_ylabel("Average Accuracy (%)", fontsize=11)
ax1.set_ylim(0, 100)
ax1.set_xticks(lambdas)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

# Data for Plot 2: Frobenius Norm Ratios R_l (for lambda = 0.1, showing shrinkage)
layers = [
    "conv1", "l1.0.c1", "l1.0.c2", "l1.1.c1", "l1.1.c2",
    "l2.0.c1", "l2.0.c2", "l2.0.ds", "l2.1.c1", "l2.1.c2",
    "l3.0.c1", "l3.0.c2", "l3.0.ds", "l3.1.c1", "l3.1.c2",
    "l4.0.c1", "l4.0.c2", "l4.0.ds", "l4.1.c1", "l4.1.c2"
]
ratios_shrinkage = [
    0.9875, 0.9233, 0.8877, 0.9107, 0.8805,
    0.8695, 0.8158, 0.9467, 0.8309, 0.7978,
    0.7807, 0.7857, 0.8252, 0.8298, 0.7893,
    0.8483, 0.9361, 0.8498, 0.9419, 0.9241
]

ax2.bar(layers, ratios_shrinkage, color='#7f7f7f', edgecolor='black', alpha=0.8, width=0.6)
ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Original Expert Norm')
ax2.set_title(r"Frobenius Norm Ratio $R_l$ across Layers ($\lambda = 0.1$)", fontsize=12, fontweight='bold')
ax2.set_xlabel("ResNet-18 Layers", fontsize=11)
ax2.set_ylabel(r"Norm Ratio $R_l = \|W_{merged}\|_F^2 / \mathrm{mean}(\|W_{expert}\|_F^2)$", fontsize=11)
ax2.set_ylim(0.6, 1.1)
ax2.set_xticks(range(len(layers)))
ax2.set_xticklabels(layers, rotation=90, fontsize=9)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

plt.tight_layout()
plt.savefig("fnbc_results.png", dpi=300)
plt.close()
print("Main plots generated successfully as fnbc_results.png!")

# ------------------------------------------
# Figure 2: Ablation (Data Efficiency & Noise)
# ------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Data Efficiency
batches = [1, 2, 5, 10, 20, 50, 100]
accuracies_efficiency = [71.973, 73.117, 73.613, 73.620, 73.797, 73.823, 73.783]
# Convert batches to total samples (each batch is 64 samples)
samples = [b * 64 for b in batches]

ax1.plot(samples, accuracies_efficiency, marker='o', linestyle='-', color='#1f77b4', linewidth=2.0)
ax1.set_xscale('log')
ax1.set_title("TS-BC Accuracy vs. Number of Calibration Samples", fontsize=12, fontweight='bold')
ax1.set_xlabel("Calibration Sample Size (Log Scale)", fontsize=11)
ax1.set_ylabel("Multi-Task Average Accuracy (%)", fontsize=11)
ax1.set_ylim(65, 80)
ax1.set_xticks(samples)
ax1.set_xticklabels([str(s) for s in samples])
ax1.grid(True, which="both", ls="--", alpha=0.7)

# Add annotations for a few points
ax1.annotate("64 samples\n(71.97%)", xy=(64, 71.973), xytext=(80, 70.0),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.0, headwidth=6.0))
ax1.annotate("320 samples\n(73.61%)", xy=(320, 73.613), xytext=(400, 75.0),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.0, headwidth=6.0))

# Plot B: Noise Robustness under Covariate Shift
sigmas = [0.0, 0.1, 0.2, 0.3, 0.5]
no_calib_noise = [12.47, 12.43, 12.35, 12.27, 12.19]
spttbc_noise = [72.49, 71.91, 70.57, 69.29, 65.66]
tsbc_noise = [73.84, 71.65, 64.06, 47.34, 25.50]

ax2.plot(sigmas, no_calib_noise, marker='o', linestyle='--', color='#d62728', label='No Calibration', linewidth=1.5)
ax2.plot(sigmas, spttbc_noise, marker='^', linestyle=':', color='#2ca02c', label='SP-TTBC (Test-Time Adaptive, batch=64)', linewidth=1.5)
ax2.plot(sigmas, tsbc_noise, marker='D', linestyle='-', color='#1f77b4', label='TS-BC (Ours, Static Offline, batch=1)', linewidth=2.5)

ax2.set_title("Robustness to Test-Time Gaussian Noise", fontsize=12, fontweight='bold')
ax2.set_xlabel(r"Noise Standard Deviation $\sigma$", fontsize=11)
ax2.set_ylabel("Average Accuracy (%)", fontsize=11)
ax2.set_ylim(0, 100)
ax2.set_xticks(sigmas)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

plt.tight_layout()
plt.savefig("ablation_results.png", dpi=300)
plt.close()
print("Ablation plots generated successfully as ablation_results.png!")
