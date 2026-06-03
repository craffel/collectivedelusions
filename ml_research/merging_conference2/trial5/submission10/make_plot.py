import matplotlib.pyplot as plt
import numpy as np

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'text.usetex': False
})

# Budgets
N_vals = np.array([16, 64, 128])

# Data for WA
WA_uncal = 39.07
WA_methods = {
    'SP-TAAC': [38.51, 31.31, 38.42],
    'FDSA': [40.02, 39.64, 40.48],
    'C-FDSA': [33.93, 35.66, 36.34],
    'SMAC (Ours)': [65.91, 68.56, 69.18],
    'C-SMAC (Ours)': [63.33, 66.31, 67.23]
}

# Data for TA
TA_uncal = 40.63
TA_methods = {
    'SP-TAAC': [37.41, 31.34, 37.46],
    'FDSA': [33.74, 33.93, 34.17],
    'C-FDSA': [27.35, 27.48, 29.20],
    'SMAC (Ours)': [63.77, 66.65, 67.42],
    'C-SMAC (Ours)': [60.59, 63.98, 64.86]
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

# Colors and markers
colors = {
    'SP-TAAC': '#e74c3c',
    'FDSA': '#3498db',
    'C-FDSA': '#9b59b6',
    'SMAC (Ours)': '#2ecc71',
    'C-SMAC (Ours)': '#27ae60'
}
markers = {
    'SP-TAAC': 'o',
    'FDSA': 's',
    'C-FDSA': 'd',
    'SMAC (Ours)': '^',
    'C-SMAC (Ours)': 'v'
}

# Panel 1: WA
ax = axes[0]
ax.axhline(y=WA_uncal, color='grey', linestyle='--', linewidth=1.5, label='Uncalibrated Baseline')
for name, vals in WA_methods.items():
    ax.plot(N_vals, vals, marker=markers[name], color=colors[name], linewidth=2, markersize=7, label=name)
ax.set_title('Weight Averaging (WA)', fontweight='bold')
ax.set_xlabel('Calibration Budget (N)')
ax.set_ylabel('Average Accuracy (%)')
ax.set_xticks(N_vals)
ax.set_ylim(20, 75)
ax.grid(True, linestyle=':', alpha=0.6)

# Panel 2: TA
ax = axes[1]
ax.axhline(y=TA_uncal, color='grey', linestyle='--', linewidth=1.5, label='Uncalibrated Baseline')
for name, vals in TA_methods.items():
    ax.plot(N_vals, vals, marker=markers[name], color=colors[name], linewidth=2, markersize=7, label=name)
ax.set_title('Task Arithmetic (TA)', fontweight='bold')
ax.set_xlabel('Calibration Budget (N)')
ax.set_xticks(N_vals)
ax.set_ylim(20, 75)
ax.grid(True, linestyle=':', alpha=0.6)

# Gather handles and labels for a single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
# Adjust layout to fit legend
plt.subplots_adjust(bottom=0.18)

# Save as PDF for high-quality vector graphics
plt.savefig('results_plot.pdf', dpi=300, bbox_inches='tight')
print("Successfully saved results_plot.pdf")
