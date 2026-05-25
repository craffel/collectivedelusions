import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Set directories
SAVE_DIR = "/fsx/craffel/collectivedelusions/ml_research/merging_conference/trial4/submission3"
results_path = os.path.join(SAVE_DIR, "evaluation_results.pt")

if not os.path.exists(results_path):
    print("No evaluation results found yet. Please run evaluate_tta.py first.")
    exit(1)

# Load results
results = torch.load(results_path, map_location="cpu")

# Create figures folder
fig_dir = os.path.join(SAVE_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 1. Bar Plot: Accuracy Comparison
methods = ["static", "standard_tta", "s2c_merge", "ewc_tta", "mc_vti"]
method_labels = ["Static Merged", "Standard TTA", "S2C-Merge", "EWC-TTA", "MC-VTI (Ours)"]

seq_accs = [results["sequential"][m]["overall_accuracy"] for m in methods]
alt_accs = [results["alternating"][m]["overall_accuracy"] for m in methods]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, seq_accs, width, label='Sequential Stream (Severe Shift)', color='#1f77b4')
rects2 = ax.bar(x + width/2, alt_accs, width, label='Alternating Stream (High-Freq Shift)', color='#ff7f0e')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Overall Accuracy Comparison on MNIST-FashionMNIST-KMNIST Stream')
ax.set_xticks(x)
ax.set_xticklabels(method_labels)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Add value labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plot_path1 = os.path.join(fig_dir, "accuracy_comparison.png")
plt.savefig(plot_path1, dpi=300)
print(f"Saved accuracy comparison plot to {plot_path1}")
plt.close()

# 2. Line Plot: Merging Coefficients (Lambda) Trajectory for Ours (MC-VTI)
# We show the trajectory of lambda over the 150 sequential batches
lambdas = np.array(results["sequential"]["mc_vti"]["lambda_history"]) # Shape: (150, 3)

plt.figure(figsize=(12, 5))
plt.plot(lambdas[:, 0], label=r'$\lambda_1$ (MNIST Expert)', color='#2ca02c', linewidth=2)
plt.plot(lambdas[:, 1], label=r'$\lambda_2$ (FashionMNIST Expert)', color='#d62728', linewidth=2)
plt.plot(lambdas[:, 2], label=r'$\lambda_3$ (KMNIST Expert)', color='#9467bd', linewidth=2)

# Mark the task transition boundaries (at batch 50 and batch 100)
plt.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=100, color='gray', linestyle='--', alpha=0.7)

# Add text labels for tasks
plt.text(25, 1.05, 'MNIST Phase', ha='center', fontweight='bold', color='#2ca02c')
plt.text(75, 1.05, 'FashionMNIST Phase', ha='center', fontweight='bold', color='#d62728')
plt.text(125, 1.05, 'KMNIST Phase', ha='center', fontweight='bold', color='#9467bd')

plt.xlabel('Step (Batch Number)')
plt.ylabel(r'Merging Coefficients ($\lambda$)')
plt.title('Dynamic Merging Coefficient Trajectory (MC-VTI Ours) under Sequential Stream')
plt.xlim(0, 150)
plt.ylim(-0.05, 1.15)
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plot_path2 = os.path.join(fig_dir, "lambda_trajectory.png")
plt.savefig(plot_path2, dpi=300)
print(f"Saved lambda trajectory plot to {plot_path2}")
plt.close()
