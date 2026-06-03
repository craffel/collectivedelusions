import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# Data
N_vals = [4, 16, 256]

# Full-Backbone Average Accuracies
full_taac = [10.22, 10.91, 10.09]
full_sptaac = [30.69, 22.28, 15.16]
full_qspa = [14.79, 12.09, 9.50]

# Local Layer 4 Average Accuracies
local_taac = [36.91, 40.55, 41.64]
local_sptaac = [42.94, 42.48, 41.06]
local_qspa = [43.41, 42.25, 41.12]

# Uncalibrated baseline
uncal = 40.17

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

# Plot 1: Full-Backbone Calibration
ax1.plot(N_vals, full_taac, marker='o', linestyle='-', color='#d62728', linewidth=1.8, label='TAAC')
ax1.plot(N_vals, full_sptaac, marker='s', linestyle='-', color='#bcbd22', linewidth=1.8, label='SP-TAAC')
ax1.plot(N_vals, full_qspa, marker='^', linestyle='-', color='#1f77b4', linewidth=1.8, label='QSPA (Ours)')
ax1.axhline(y=uncal, color='gray', linestyle='--', linewidth=1.2, label='Uncalibrated')
ax1.set_xscale('log')
ax1.set_xticks(N_vals)
ax1.set_xticklabels([str(n) for n in N_vals])
ax1.set_title('Full-Backbone Calibration', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Calibration Budget $N$ per task', fontsize=11, labelpad=5)
ax1.set_ylabel('Average Accuracy (%)', fontsize=11, labelpad=5)
ax1.grid(True, which="both", ls=":", alpha=0.5)
ax1.set_ylim(5, 50)

# Plot 2: Local Layer 4 Calibration
ax2.plot(N_vals, local_taac, marker='o', linestyle='-', color='#d62728', linewidth=1.8, label='TAAC')
ax2.plot(N_vals, local_sptaac, marker='s', linestyle='-', color='#bcbd22', linewidth=1.8, label='SP-TAAC')
ax2.plot(N_vals, local_qspa, marker='^', linestyle='-', color='#1f77b4', linewidth=1.8, label='QSPA (Ours)')
ax2.axhline(y=uncal, color='gray', linestyle='--', linewidth=1.2, label='Uncalibrated')
ax2.set_xscale('log')
ax2.set_xticks(N_vals)
ax2.set_xticklabels([str(n) for n in N_vals])
ax2.set_title('Local Layer 4 Calibration', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Calibration Budget $N$ per task', fontsize=11, labelpad=5)
ax2.grid(True, which="both", ls=":", alpha=0.5)
ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', shadow=False, framealpha=0.9)

plt.tight_layout()
plt.savefig('calibration_sweep.png', dpi=300, bbox_inches='tight')
plt.savefig('calibration_sweep.pdf', bbox_inches='tight')
print("Successfully generated calibration_sweep.png and calibration_sweep.pdf!")
