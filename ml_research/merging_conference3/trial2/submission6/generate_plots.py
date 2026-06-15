import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
metrics_path = "results/metrics.json"
if not os.path.exists(metrics_path):
    print(f"Warning: {metrics_path} does not exist yet. Please run this script after the experiments complete.")
    exit(0)

with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Extract average accuracies and std devs for 8-bit and 4-bit
methods = [
    "fp16",
    "fp16_optimized_unquantized",
    "fp16_optimized_adam_unquantized",
    "q_then_m",
    "m_then_q",
    "fp16_optimized_quantized",
    "fp16_optimized_adam_quantized",
    "qmerge_es",
    "qmerge_adam"
]
method_labels = [
    "FP16 Merged",
    "AdaMerging (ES)",
    "AdaMerging (Adam)",
    "Q-then-M",
    "M-then-Q",
    "AdaMerging (Quant, ES)",
    "AdaMerging (Quant, Adam)",
    "Q-Merge (1+1 ES)",
    "Q-Merge (Adam GD)"
]

means_8 = []
stds_8 = []
means_4 = []
stds_4 = []

for m in methods:
    means_8.append(metrics["8"][m]["Average"]["mean"] * 100)
    stds_8.append(metrics["8"][m]["Average"]["std"] * 100)
    means_4.append(metrics["4"][m]["Average"]["mean"] * 100)
    stds_4.append(metrics["4"][m]["Average"]["std"] * 100)

# Set up matplotlib figure
plt.figure(figsize=(10, 6))
x = np.arange(len(methods))
width = 0.35

# Plot bars with error bars and distinct hatch patterns for grayscale/monochrome accessibility
plt.bar(x - width/2, means_8, width, yerr=stds_8, label="8-bit Quantization", color="#1f77b4", edgecolor='black', hatch='//', capsize=5)
plt.bar(x + width/2, means_4, width, yerr=stds_4, label="4-bit Quantization", color="#ff7f0e", edgecolor='black', hatch='\\\\', capsize=5)

# Style plot
plt.title("Q-Merge vs Baselines: Average Multi-Task Accuracy under Quantization", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Average Test Accuracy (%)", fontsize=12, fontweight='bold')
plt.xticks(x, method_labels, rotation=15, ha='right', fontsize=10, fontweight='bold')
plt.ylim(30, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc="lower right", fontsize=11, frameon=True, edgecolor='black')
plt.tight_layout()

# Save figure
plot_path = "results/qmerge_vs_baselines.png"
plt.savefig(plot_path, dpi=300)
print(f"Plot successfully generated and saved to {plot_path}!")
