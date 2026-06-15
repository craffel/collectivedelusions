import json
import matplotlib.pyplot as plt
import os

# Set style for academic paper
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 14,
    'legend.fontsize': 11,
    'lines.linewidth': 2.0
})

# Load metrics
with open('results/real_physical_metrics.json', 'r') as f:
    results = json.load(f)

fig, ax = plt.subplots(figsize=(7, 4.8))

# Define colors and markers
methods_format = {
    'Task Arithmetic': {'color': '#7f7f7f', 'ls': ':', 'label': 'Task Arithmetic (Static, acc=81.5%)'},
    'AdaMerging': {'color': '#d62728', 'ls': '--', 'label': 'AdaMerging (Unconstrained, acc=78.0%)'},
    'PolyMerge': {'color': '#ff7f0e', 'ls': '-.', 'label': 'PolyMerge (Monomial, cond=389.3, acc=70.5%)'},
    'ChebyMerge': {'color': '#1f77b4', 'ls': '-', 'label': 'ChebyMerge (Ours, cond=2.7, acc=74.0%)'},
    'ChebyMerge + CSD': {'color': '#2ca02c', 'ls': '-', 'label': 'ChebyMerge + CSD (Ours, cond=2.7, acc=75.5%)'}
}

for method_name, data in results.items():
    if method_name in methods_format:
        fmt = methods_format[method_name]
        ax.plot(data['losses'], color=fmt['color'], linestyle=fmt['ls'], label=fmt['label'])

ax.set_xlabel('Optimization Step')
ax.set_ylabel('Unsupervised TTA Entropy Loss')
ax.set_title('Real Physical Model-Merging on CLIP ViT-B/32')
ax.legend(frameon=True, facecolor='white', edgecolor='lightgray')
ax.set_xlim(0, len(results['AdaMerging']['losses']) - 1)
plt.tight_layout()

# Save plot
os.makedirs('results', exist_ok=True)
os.makedirs('submission/results', exist_ok=True)
plt.savefig('results/fig3_physical_trajectory.png', dpi=300)
plt.savefig('submission/results/fig3_physical_trajectory.png', dpi=300)
print("Real physical trajectory plot generated and saved successfully!")
