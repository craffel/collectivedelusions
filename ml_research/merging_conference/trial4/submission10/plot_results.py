import matplotlib.pyplot as plt
import numpy as np
import torch

# Load results
results = torch.load("./results.pt")

corruptions = ["clean", "noise", "blur", "contrast"]
corr_labels = ["Clean", "Gaussian Noise", "Gaussian Blur", "Contrast"]
methods = ["static", "standard_tta", "ewc_tta", "s2c_merge", "fw_cms_tg_fwar", "fw_cms_tf_fwar"]
method_labels = ["Static Merged", "Standard TTA", "EWC-TTA", "S2C-Merge", "FW-CMS + FWAR (Ours, TG)", "FW-CMS + FWAR (Ours, TF)"]

x = np.arange(len(corruptions))
width = 0.13

# Setup plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Sequential Stream
for idx, (method, label) in enumerate(zip(methods, method_labels)):
    accs = [results["sequential"][c][method] for c in corruptions]
    ax1.bar(x + (idx - 2.5) * width, accs, width, label=label, alpha=0.9)

ax1.set_title("Sequential Stream Performance Comparison", fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(corr_labels, fontsize=10)
ax1.set_ylabel("Multi-Task Average Accuracy (%)", fontsize=11)
ax1.set_ylim(0, 100)

# Plot Alternating Stream
for idx, (method, label) in enumerate(zip(methods, method_labels)):
    accs = [results["alternating"][c][method] for c in corruptions]
    ax2.bar(x + (idx - 2.5) * width, accs, width, label=label, alpha=0.9)

ax2.set_title("Alternating Stream Performance Comparison", fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(corr_labels, fontsize=10)
ax2.set_ylabel("Multi-Task Average Accuracy (%)", fontsize=11)
ax2.set_ylim(0, 100)

# Add single legend at bottom
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, bbox_transform=fig.transFigure, bbox_to_anchor=(0.5, -0.05), fontsize=10)

plt.tight_layout()
plt.savefig("results_comparison.png", dpi=300, bbox_inches='tight')
print("Successfully plotted and saved results_comparison.png with FWAR results.")
