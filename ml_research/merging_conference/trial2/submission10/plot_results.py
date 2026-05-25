import matplotlib.pyplot as plt
import numpy as np

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 16,
    'legend.fontsize': 11,
    'font.family': 'sans-serif',
    'text.usetex': False
})

# Color palette
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69']
colors_bar = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c']

# Figure 1: Gaussian Noise Accuracy Comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
methods = ['Static', 'SyMerge', 'SAM', 'ASAM', 'BF-ASAM\n(Ours)', 'R-BF-SAM\n(Ours)']
accuracies = [29.21, 35.35, 32.31, 30.41, 39.76, 40.21]

bars = ax.bar(methods, accuracies, color=colors_bar, edgecolor='black', width=0.6)

# Add value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Test-Time Adaptation Accuracy on Gaussian Noise (OOD)', pad=15, fontweight='bold')
ax.set_ylim(0, 50)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('gaussian_noise_comparison.pdf', bbox_inches='tight')
plt.close()

# Figure 2: Sensitivity to alpha_fisher
fig, ax1 = plt.subplots(figsize=(6, 4))
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
clean_accs = [64.15, 64.41, 64.45, 64.49, 64.48]
noise_accs = [38.14, 38.91, 39.59, 39.76, 39.78]

# Plot Gaussian Noise accuracy on left axis
color = '#e31a1c'
ax1.set_xlabel(r'Fisher Weighting Coefficient $\alpha_{\mathrm{fisher}}$ (Log Scale)', fontweight='bold')
ax1.set_ylabel('Gaussian Noise Accuracy (%)', color=color, fontweight='bold')
line1 = ax1.plot(alphas, noise_accs, marker='o', color=color, linewidth=2.5, markersize=8, label='Gaussian Noise (OOD)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')
ax1.grid(True, which="both", ls="--", alpha=0.5)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  
color = '#1f78b4'
ax2.set_ylabel('Clean Accuracy (%)', color=color, fontweight='bold')
line2 = ax2.plot(alphas, clean_accs, marker='s', color=color, linewidth=2.5, markersize=8, linestyle='--', label='Clean Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center')

plt.title(r'Sensitivity Analysis of $\alpha_{\mathrm{fisher}}$ under BF-ASAM', pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('sensitivity_alpha.pdf', bbox_inches='tight')
plt.close()

print("Plots successfully generated!")
