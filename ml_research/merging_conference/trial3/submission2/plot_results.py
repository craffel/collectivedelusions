import os
import matplotlib.pyplot as plt
import numpy as np

# Set style for academic publishing
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 9,
    'grid.alpha': 0.3,
})

# Parse results from results/summary.txt
# Header: Task,Corruption,Method,Acc_Before,Acc_After,Diff
# Method name in summary.txt: static, adamerging, symerge, ca-symerge
data = {}
with open("results/summary.txt", "r") as f:
    lines = f.readlines()
    for line in lines[1:]: # skip header
        parts = line.strip().split(",")
        if len(parts) == 6:
            task = parts[0].lower()
            corr = parts[1].lower()
            method = parts[2].lower()
            acc_after = float(parts[4])
            data[(task, corr, method)] = acc_after

tasks_display = ["MNIST", "FashionMNIST", "KMNIST"]
tasks_keys = ["mnist", "fashionmnist", "kmnist"]

corruptions_display = ["Clean", "Noise", "Rotation"]
corruptions_keys = ["none", "noise", "rotation"]

methods_display = ["Static", "AdaMerging", "Head-only TTA", "SyMerge (TTA)", "CA-SyMerge (Ours)"]
methods_keys = ["static", "adamerging", "head-tta", "symerge", "ca-symerge"]

# Create subplots (one for each task)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

# Colors: 5 distinct soft/academic colors
colors = ["#7293CB", "#D3A4F9", "#E17C05", "#AB63FA", "#84BA5B"]  
x = np.arange(len(corruptions_keys))
width = 0.15

for i, task_key in enumerate(tasks_keys):
    ax = axes[i]
    task_name = tasks_display[i]
    
    rects = []
    for m_idx, m_key in enumerate(methods_keys):
        vals = [data.get((task_key, corr_key, m_key), 0.0) for corr_key in corruptions_keys]
        # Shift the bar position based on its index
        offset = (m_idx - 2.0) * width
        r = ax.bar(x + offset, vals, width, label=methods_display[m_idx], color=colors[m_idx], edgecolor='black', linewidth=0.5)
        rects.append(r)
    
    ax.set_title(task_name, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions_display)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    if i == 0:
        ax.set_ylabel("Accuracy (%)", fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)

# Adjust layout and save
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/accuracy_comparison.pdf", bbox_inches='tight')
plt.savefig("results/accuracy_comparison.png", dpi=300, bbox_inches='tight')
print("Successfully generated and saved results/accuracy_comparison.pdf and results/accuracy_comparison.png")
