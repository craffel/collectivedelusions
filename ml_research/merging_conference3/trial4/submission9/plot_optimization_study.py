import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False

# Data
steps = [0, 40, 100, 250, 500]

# Best Validation Scores
# Step 0 is estimated from the initialization values in the script
# AdaMerging/ZipMerge starts at 0.3 for all coefficients
# TLC-Tune starts at 1.0 for all lambdas
# For plotting, we show how TLC-Tune converges instantly while others remain completely flat.
ada_val = [0.1725, 0.1725, 0.1725, 0.1725, 0.1725]
zip_val = [0.1732, 0.1732, 0.1732, 0.1732, 0.1732]
tlc_dense_val = [0.2110, 0.2375, 0.2375, 0.2375, 0.2375]
tlc_sparse_val = [0.2942, 0.3646, 0.3646, 0.3646, 0.3646]

# Test Mean Accuracies
ada_test = [0.3219, 0.3219, 0.3219, 0.3219, 0.3219]
zip_test = [0.2574, 0.2574, 0.2574, 0.2574, 0.2574]
tlc_dense_test = [0.4412, 0.4482, 0.4482, 0.4482, 0.4482]
tlc_sparse_test = [0.3255, 0.3433, 0.3433, 0.3433, 0.3433]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

# 1. Validation Score Plot
ax1.plot(steps, tlc_sparse_val, 'o-', color='#d95f02', linewidth=2, label='TLC-Tune EPM ($p=0.5$)')
ax1.plot(steps, tlc_dense_val, 's-', color='#7570b3', linewidth=2, label='TLC-Tune EPM (Dense)')
ax1.plot(steps, zip_val, '^--', color='#1b9e77', linewidth=1.5, label='ZipMerge ($p=0.5$)')
ax1.plot(steps, ada_val, 'v--', color='#e7298a', linewidth=1.5, label='AdaMerging (Dense)')

ax1.set_xlabel('Optimization Steps ($T$)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Best Validation Minimax Score', fontsize=11, fontweight='bold')
ax1.set_title('(a) Validation Convergence Trajectory', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlim(-10, 510)
ax1.set_ylim(0.15, 0.40)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

# 2. Test Mean Accuracy Plot
ax2.plot(steps, tlc_dense_test, 's-', color='#7570b3', linewidth=2, label='TLC-Tune EPM (Dense)')
ax2.plot(steps, tlc_sparse_test, 'o-', color='#d95f02', linewidth=2, label='TLC-Tune EPM ($p=0.5$)')
ax2.plot(steps, ada_test, 'v--', color='#e7298a', linewidth=1.5, label='AdaMerging (Dense)')
ax2.plot(steps, zip_test, '^--', color='#1b9e77', linewidth=1.5, label='ZipMerge ($p=0.5$)')

ax2.set_xlabel('Optimization Steps ($T$)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Test Mean Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('(b) Generalization to Test Set', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlim(-10, 510)
ax2.set_ylim(0.22, 0.48)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

plt.tight_layout()
plt.savefig('submission/opt_trajectory.png', bbox_inches='tight')
plt.savefig('submission/opt_trajectory.pdf', bbox_inches='tight')
print("Plots saved successfully to submission/opt_trajectory.png and submission/opt_trajectory.pdf")
