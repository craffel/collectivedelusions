import numpy as np
import matplotlib.pyplot as plt

# Load results
data = np.load("experiment_results.npz", allow_pickle=True)
clean = data["clean"].item()
corrupted = data["corrupted"].item()

methods = ["Static", "TENT", "PC-Merge", "CPA-Merge", "PROTO-TTMM", "FP-OW (Ours)"]
tasks = ["CIFAR10", "SVHN", "FashionMNIST"]

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot Clean Results
ax = axes[0]
x = np.arange(len(tasks))
width = 0.12

for i, method in enumerate(methods):
    accs = [np.mean(clean[method]["task_accs"][t]) * 100. for t in tasks]
    ax.bar(x + i*width - len(methods)*width/2 + width/2, accs, width, label=method, alpha=0.9)

ax.set_title("Clean Test-Time Model Merging Performance", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["CIFAR-10 (Task A)", "SVHN (Task B)", "FashionMNIST (Novel - Task C)"], fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.tick_params(labelsize=11)

# Plot Corrupted Results
ax = axes[1]
for i, method in enumerate(methods):
    accs = [np.mean(corrupted[method]["task_accs"][t]) * 100. for t in tasks]
    ax.bar(x + i*width - len(methods)*width/2 + width/2, accs, width, label=method, alpha=0.9)

ax.set_title("Corrupted Test-Time Model Merging Performance", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["CIFAR-10 (Task A)", "SVHN (Task B)", "FashionMNIST (Novel - Task C)"], fontsize=11)
ax.set_ylim(0, 100)
ax.tick_params(labelsize=11)

# Add legend and layout
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=12, frameon=True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# Save figure
plt.savefig("results_plot.png", dpi=300)
print("Figure results_plot.png generated and saved successfully!")
