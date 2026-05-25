import matplotlib.pyplot as plt
import os
import numpy as np

os.makedirs("plots", exist_ok=True)

# Set styling for publication
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

# Data for core comparison
methods = ['Task Arithmetic (TA)', 'OrthoMerge (Global)', 'C-Ortho (OM-Conv-Only)']
std_accs = [84.30, 77.15, 85.58]
sam_accs = [81.91, 74.93, 83.36]

# Plot 1: Core Merging Accuracy Comparison
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(methods))
width = 0.35

rects1 = ax.bar(x - width/2, std_accs, width, label='Standard SGD Experts', color='#1f77b4', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, sam_accs, width, label='SAM Experts', color='#2ca02c', edgecolor='black', alpha=0.9)

ax.set_ylabel('Full CIFAR-10 Accuracy (%)', fontweight='bold')
ax.set_title('Resolving Representation Collapse with Convolutional-only OrthoMerge (C-Ortho)', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(60, 90)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('plots/merging_accuracy_comparison.png', dpi=300)
plt.close()

# Plot 2: Detailed Selective Merging Ablation
ablation_modes = [
    'Pure TA',
    'OM-FC-Only',
    'OM-All',
    'OM-Late-Layers',
    'OM-Early-Layers',
    'OM-Conv-No-Downsample',
    'OM-Conv-Only (C-Ortho)'
]
std_abl_accs = [84.30, 75.72, 77.15, 85.25, 84.66, 85.54, 85.58]
sam_abl_accs = [81.91, 72.85, 74.93, 83.16, 82.18, 83.32, 83.36]

fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(ablation_modes))
height = 0.35

rects_std = ax.barh(y - height/2, std_abl_accs, height, label='Standard SGD Experts', color='#3f72af', edgecolor='black', alpha=0.9)
rects_sam = ax.barh(y + height/2, sam_abl_accs, height, label='SAM Experts', color='#ff7e67', edgecolor='black', alpha=0.9)

ax.set_xlabel('Full CIFAR-10 Accuracy (%)', fontweight='bold')
ax.set_title('Ablation of Selective OrthoMerge Layer Configurations', pad=15)
ax.set_yticks(y)
ax.set_yticklabels(ablation_modes)
ax.set_xlim(65, 90)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')

# Add values to the bars
def autolabel_h(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.2f}%',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0),  # 5 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center', fontsize=8.5)

autolabel_h(rects_std)
autolabel_h(rects_sam)

plt.tight_layout()
plt.savefig('plots/selective_orthomerge_ablation.png', dpi=300)
plt.close()

print("Publication-quality plots generated and saved successfully!")
