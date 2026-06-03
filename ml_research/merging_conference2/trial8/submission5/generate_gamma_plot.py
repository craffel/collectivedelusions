import json
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.3,
})

# Gamma sweep data from our run
gammas = [1.0, 1.5, 2.0, 3.0]
avg_accs = [62.10, 62.07, 62.50, 62.20]

fig, ax = plt.subplots(figsize=(6, 3.0))
ax.plot(gammas, avg_accs, marker='o', linestyle='-', color='#d62728', linewidth=2, markersize=8, label='QR-IPR (INT8)')
ax.axhline(62.20, linestyle='--', color='#2ca02c', linewidth=1.5, label='FP32 Baseline (HNS/QR-IPR)')

ax.set_xlabel(r'Clamping Hyperparameter $\gamma$ (MAD Multiplier)')
ax.set_ylabel('Clean Average Accuracy (%)')
ax.set_title(r'Sensitivity Analysis of outlier clipping factor $\gamma$')
ax.set_xticks(gammas)
ax.set_ylim(61.5, 63.0)
ax.legend(frameon=True, facecolor='white', edgecolor='none')

for i, txt in enumerate(avg_accs):
    ax.annotate(f'{txt:.2f}%', (gammas[i], avg_accs[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='semibold')

plt.tight_layout()
plt.savefig("gamma_sensitivity.png", dpi=300)
plt.close()
print("Gamma sensitivity plot saved successfully as gamma_sensitivity.png!")
