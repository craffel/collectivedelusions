import json
import numpy as np
import matplotlib.pyplot as plt

# Load metrics
with open('experiment_metrics.json', 'r') as f:
    metrics = json.load(f)

datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
methods = {
    'task_arithmetic': 'Task Arithmetic',
    'ties_merging': 'TIES-Merging',
    'neta_alpha_1.0': 'NETA (Proposed)',
    'neta_alpha_0.5': 'NETA (\u03b1=0.5)',
    'layer_wise_adamerging': 'Layer-Wise AdaMerging'
}

# Color palette: elegant, clean minimalist colors
colors = ['#888888', '#E69F00', '#56B4E9', '#009E73', '#CC79A7']

# Extract data
means = {m: [] for m in methods}
stds = {m: [] for m in methods}

for method in methods:
    for ds in datasets:
        means[method].append(metrics[method][ds]['mean'] * 100)
        stds[method].append(metrics[method][ds]['std'] * 100)

x = np.arange(len(datasets))
width = 0.15

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

# Set grid
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

for i, (method_key, label) in enumerate(methods.items()):
    rects = ax.bar(
        x + (i - 2) * width,
        means[method_key],
        width,
        yerr=stds[method_key],
        label=label,
        color=colors[i],
        edgecolor='black',
        linewidth=0.8,
        capsize=3,
        zorder=3
    )

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Multi-Task Model Merging Performance (CLIP ViT-B/32)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
ax.set_ylim(60, 102)
ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=10)

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=300)
plt.savefig('submission/comparison_plot.png', dpi=300)
print("Successfully generated comparison_plot.png!")
