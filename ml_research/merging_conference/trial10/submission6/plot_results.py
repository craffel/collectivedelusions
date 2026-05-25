import matplotlib.pyplot as plt
import numpy as np
import os

# Load results
results = np.load("checkpoints/results.npy", allow_pickle=True).item()

methods = ["Method A", "Method B", "Method C", "Method D", "Method E", "Method F"]
domains = ["C-MN", "N-MN", "C-FN", "N-FN", "Nov-K", "Overall"]

# Setup colors matching professional papers (academic palette)
colors = ['#888888', '#D55E00', '#0072B2', '#F0E442', '#009E73', '#CC79A7']

x = np.arange(len(domains))
width = 0.12

fig, ax = plt.subplots(figsize=(10, 6))

for i, m in enumerate(methods):
    y_vals = [results[m][d] for d in domains]
    ax.bar(x + (i - 2.5) * width, y_vals, width, label=m, color=colors[i], edgecolor='black', linewidth=0.5)

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Test-Time Model Merging Performance Across Stream Segments', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(["Clean MNIST\n(C-MN)", "Noisy MNIST\n(N-MN)", "Clean Fashion\n(C-FN)", "Noisy Fashion\n(N-FN)", "OOD KMNIST\n(Nov-K)", "Overall\nStream"], fontsize=10)
ax.set_ylim(0, 115)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of the bars for Method E and Method F
for i, m in enumerate(["Method E", "Method F"]):
    y_vals = [results[m][d] for d in domains]
    idx = 4 + i # index in methods list (E is 4, F is 5)
    for j, val in enumerate(y_vals):
        ax.text(j + (idx - 2.5) * width, val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=8, rotation=90)

ax.legend(loc='upper right', frameon=True, shadow=False, edgecolor='black')
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/results.png", dpi=300)
print("Saved plots/results.png successfully!")
