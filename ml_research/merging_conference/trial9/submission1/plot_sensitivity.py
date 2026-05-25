import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load sweep results
data = pd.read_csv("sweep_results.txt")

# Filter for the major coherence weights we want to plot
coherence_weights_to_plot = [0.005, 0.01, 0.02]

# Set up the matplotlib style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'text.usetex': False, # Avoid requiring LaTeX on host for plotting
    'font.family': 'sans-serif',
})

fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)

# Colors and markers for the plot
colors = {0.005: '#d62728', 0.01: '#1f77b4', 0.02: '#2ca02c'}
markers = {0.005: 'o', 0.01: 's', 0.02: '^'}

# Plot lines for each selected coherence weight
for cw in coherence_weights_to_plot:
    subset = data[data['Coherence_Weight'] == cw].sort_values('Variance_Weight')
    
    # We want to plot Overall Accuracy vs. Variance Weight
    x = subset['Variance_Weight'].values
    y = subset['Overall'].values
    
    label = f'$\\gamma_c = {cw}$'
    ax.plot(x, y, label=label, color=colors[cw], marker=markers[cw], linewidth=1.5, markersize=5)

# Add baseline indicators
# 1. Static Merging Baseline
static_acc = 57.62
ax.axhline(y=static_acc, color='grey', linestyle='--', linewidth=1.0, label='STATIC Baseline (57.62%)')

# 2. Uniform Merging Baseline
uniform_acc = 31.41
# We won't plot uniform on the main axis as it's too low and would squish the detail, but we can mention it in caption.

# Configure axes
ax.set_xlabel('Variance Weight ($\\gamma_v$)')
ax.set_ylabel('Overall Accuracy (%)')
ax.set_title('Hyperparameter Sensitivity of VAKP-BC (Ours)')
ax.set_xticks([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
ax.set_xticklabels(['0.0\n(BK-CoMerge)', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5'])
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_ylim(60.5, 66.0)

# Legend placement
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')

plt.tight_layout()

# Save the figure as both PDF (for vector quality in LaTeX) and PNG (for backup)
plt.savefig('hyperparameter_sensitivity.pdf', bbox_inches='tight')
plt.savefig('hyperparameter_sensitivity.png', bbox_inches='tight', dpi=300)
print("Plots generated successfully!")
