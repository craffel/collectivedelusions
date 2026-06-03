import matplotlib.pyplot as plt
import numpy as np

# Apply a clean style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.6,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
})

# ----------------------------------------------------
# Plot 1: CKA Representational Similarity Localization
# ----------------------------------------------------
blocks = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
wa_cka = [0.9999, 0.9484, 0.9156, 0.7091, 0.3805]
ta_cka = [0.9999, 0.9451, 0.9138, 0.7078, 0.3841]
inter_expert = [0.9998, 0.9043, 0.8262, 0.4665, 0.1457]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(blocks))

ax.plot(x, wa_cka, marker='o', color='#1f77b4', linestyle='-', label='WA vs Expert (Avg)')
ax.plot(x, ta_cka, marker='s', color='#ff7f0e', linestyle='--', label='TA vs Expert (Avg)')
ax.plot(x, inter_expert, marker='^', color='#2ca02c', linestyle=':', label='Inter-Expert (Avg)')

ax.set_xticks(x)
ax.set_xticklabels(blocks)
ax.set_ylabel('Linear CKA Similarity')
ax.set_xlabel('ResNet-18 Block / Depth')
ax.set_title('Representational CKA Similarity by Block')
ax.set_ylim(0.0, 1.05)
ax.legend(loc='lower left', frameon=True)
plt.tight_layout()
plt.savefig('cka_plot.png', dpi=300)
plt.savefig('cka_plot.pdf', bbox_inches='tight')
plt.close()

# ----------------------------------------------------
# Plot 2: Robustness to Calibration Data Scarcity
# ----------------------------------------------------
N_vals = [16, 32, 64, 128, 256, 512]

# WA Data
wa_ntaac = [77.15, 77.41, 78.05, 77.97, 78.04, 77.98]
wa_tnac_34 = [76.44, 76.63, 77.05, 77.15, 77.12, 77.08]
wa_tnac_4 = [75.56, 74.72, 74.81, 75.00, 74.88, 74.77]

# TA Data
ta_ntaac = [79.33, 79.44, 79.77, 79.80, 79.80, 79.90]
ta_tnac_34 = [78.51, 78.51, 78.78, 78.84, 78.71, 78.84]
ta_tnac_4 = [77.90, 77.12, 77.18, 77.30, 77.25, 77.20]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Weight Averaging subplot
ax1.plot(N_vals, wa_ntaac, marker='o', color='#d62728', linestyle='-', label='N-TAAC (Full)')
ax1.plot(N_vals, wa_tnac_34, marker='s', color='#1f77b4', linestyle='--', label='T-NAC (L3+L4)')
ax1.plot(N_vals, wa_tnac_4, marker='^', color='#ff7f0e', linestyle=':', label='T-NAC (L4 only)')
ax1.set_xscale('log')
ax1.set_xticks(N_vals)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel('Calibration Samples N per Task')
ax1.set_ylabel('Average Multi-task Accuracy (%)')
ax1.set_title('Weight Averaging (WA)')
ax1.set_ylim(73.5, 80.5)
ax1.legend(loc='lower right', frameon=True)

# Task Arithmetic subplot
ax2.plot(N_vals, ta_ntaac, marker='o', color='#d62728', linestyle='-', label='N-TAAC (Full)')
ax2.plot(N_vals, ta_tnac_34, marker='s', color='#1f77b4', linestyle='--', label='T-NAC (L3+L4)')
ax2.plot(N_vals, ta_tnac_4, marker='^', color='#ff7f0e', linestyle=':', label='T-NAC (L4 only)')
ax2.set_xscale('log')
ax2.set_xticks(N_vals)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.set_xlabel('Calibration Samples N per Task')
ax2.set_title('Task Arithmetic (TA, $\lambda=0.4$)')
ax2.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig('ablation_plot.png', dpi=300)
plt.savefig('ablation_plot.pdf', bbox_inches='tight')
plt.close()

print("Plots generated successfully!")
