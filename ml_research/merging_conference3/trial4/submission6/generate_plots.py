import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'font.family': 'sans-serif'
})

# Data
densities = [5, 10, 20, 50, 100]
sta_acc = [71.28, 77.53, 82.91, 86.91, 87.45]
r_sta_acc = [38.00, 62.60, 82.36, 88.81, 87.45]

ta_acc = 87.45
dare_acc = 87.48
ties_acc = 85.02

# Plot
fig, ax = plt.subplots(figsize=(6.2, 4.5))

# Plot Standard STA curve
ax.plot(densities, sta_acc, marker='o', linewidth=2, color='#1f77b4', label='Standard STA ($\lambda=0.3$)')

# Plot Rescaled STA curve
ax.plot(densities, r_sta_acc, marker='s', linestyle='-.', linewidth=2, color='#9467bd', label='Rescaled STA ($\lambda=0.3$)')

# Plot peak tuned STA point
ax.scatter(20, 90.53, marker='*', s=180, color='#e377c2', zorder=5, label='Tuned STA ($s=20\%, \lambda=0.8$) [90.53%]')

# Horizontal lines for baselines
ax.axhline(y=ta_acc, linestyle='--', color='#2ca02c', linewidth=1.5, label='Task Arithmetic (Full)')
ax.axhline(y=dare_acc, linestyle='-.', color='#ff7f0e', linewidth=1.5, label='DARE (p=0.8)')
ax.axhline(y=ties_acc, linestyle=':', color='#d62728', linewidth=1.5, label='TIES-Merging')

# Annotations
ax.annotate('90.53% (Peak)', xy=(20, 90.53), xytext=(24, 91.0),
            arrowprops=dict(arrowstyle="->", color='#e377c2', lw=1.5),
            color='#d147a3', fontweight='bold')

# Labels and limits
ax.set_xlabel('Survival Density $s$ (%)', fontweight='bold')
ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
ax.set_title('Average Performance vs. Survival Density', fontweight='bold', pad=12)
ax.set_xlim(0, 105)
ax.set_ylim(35, 95)
ax.set_xticks([5, 10, 20, 50, 100])

# Grid and legend
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', shadow=True)

plt.tight_layout()
plt.savefig('submission/sta_density_curve.png', dpi=300)
print("Plot successfully regenerated and saved to submission/sta_density_curve.png")
