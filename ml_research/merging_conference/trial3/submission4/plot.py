import matplotlib.pyplot as plt
import numpy as np

# Set style settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Baselines
sgd_ta = 80.74
sam_ta = 79.17
sgd_proc = 15.776574
sam_proc = 15.779773

# Swept values
betas = [0.05, 0.10, 0.15, 0.20, 0.30]

spor_ta = [79.14, 79.77, 79.87, 79.46, 80.39]
spor_proc = [15.685993, 15.644077, 15.608322, 15.586096, 15.555408]

fg_inverse_ta = [79.97, 80.03, 79.77, 80.16, 79.45]
fg_inverse_proc = [15.668221, 15.636574, 15.604232, 15.592301, 15.542838]

# Plot 1: Merging Accuracy vs Beta
ax1.plot(betas, spor_ta, 'o-', linewidth=2.5, markersize=8, color='#4C72B0', label='Standard SPOR')
ax1.plot(betas, fg_inverse_ta, 's--', linewidth=2.5, markersize=8, color='#DD8452', label='FG-SPOR (Inverse)')

# Add horizontal baselines
ax1.axhline(y=sgd_ta, color='g', linestyle='-.', alpha=0.8, label='SGD Baseline (80.74%)')
ax1.axhline(y=sam_ta, color='r', linestyle=':', alpha=0.8, label='SAM Baseline (79.17%)')

ax1.set_xlabel(r'Regularization Coefficient $\beta$', fontsize=12, fontweight='bold')
ax1.set_ylabel('Full CIFAR-10 Merged Accuracy (%)', fontsize=12)
ax1.set_title('Merging Accuracy vs. Regularization Strength', fontsize=13, fontweight='bold')
ax1.set_xticks(betas)
ax1.set_xlim(0.03, 0.32)
ax1.set_ylim(78.0, 81.5)
ax1.legend(frameon=True, loc='lower left', fontsize=10)

# Plot 2: Procrustes Residual Norm vs Beta
ax2.plot(betas, spor_proc, 'o-', linewidth=2.5, markersize=8, color='#4C72B0', label='Standard SPOR')
ax2.plot(betas, fg_inverse_proc, 's--', linewidth=2.5, markersize=8, color='#DD8452', label='FG-SPOR (Inverse)')

# Add horizontal baselines
ax2.axhline(y=sgd_proc, color='g', linestyle='-.', alpha=0.8, label='SGD Baseline (15.7766)')
ax2.axhline(y=sam_proc, color='r', linestyle=':', alpha=0.8, label='SAM Baseline (15.7798)')

ax2.set_xlabel(r'Regularization Coefficient $\beta$', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Procrustes Residual Norm', fontsize=12)
ax2.set_title('Coordinate Drift vs. Regularization Strength', fontsize=13, fontweight='bold')
ax2.set_xticks(betas)
ax2.set_xlim(0.03, 0.32)
ax2.set_ylim(15.50, 15.82)
ax2.legend(frameon=True, loc='lower left', fontsize=10)

# Save plots
plt.tight_layout()
plt.savefig('experimental_results.png', dpi=300)
plt.close()
print("Successfully generated experimental_results.png line plots!")
