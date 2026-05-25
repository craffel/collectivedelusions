import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from results.json
with open("results.json", "r") as f:
    results = json.load(f)

corruptions = ["none", "noise", "blur", "contrast", "rotation"]
plot_corruptions = ["Clean", "Noise", "Blur", "Contrast", "Rotation"]

std_experts = results["standard_experts"]
sam_experts = results["sam_experts"]

methods_map = {
    "Baseline (No TTA)": "baseline",
    "SyMerge (Std TTA)": "symerge",
    "SAT-SyMerge (SAM)": "sat-symerge",
    "O-LoRTA (Ours)": "o-lorta",
    "SD-O-LoRTA (Ours)": "sd-olorta"
}

std_data = {}
for display_name, json_key in methods_map.items():
    std_data[display_name] = [std_experts[json_key][corr]["avg"] for corr in corruptions]

sam_data = {}
for display_name, json_key in methods_map.items():
    sam_data[display_name] = [sam_experts[json_key][corr]["avg"] for corr in corruptions]

# Set standard plotting style for professional look
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

methods = ["Baseline (No TTA)", "SyMerge (Std TTA)", "SAT-SyMerge (SAM)", "O-LoRTA (Ours)", "SD-O-LoRTA (Ours)"]
colors = ["#4A5568", "#3182CE", "#E53E3E", "#319795", "#805AD5"] # Charcoal, Blue, Red, Teal, Purple

x = np.arange(len(plot_corruptions))
width = 0.15

# Plot 1: Standard Experts
for i, method in enumerate(methods):
    ax1.bar(x + (i - 2) * width, std_data[method], width, label=method, color=colors[i], edgecolor='black', linewidth=0.5)

ax1.set_title("Test-Time Adaptation under Standard Experts", fontsize=14, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(plot_corruptions, fontsize=12)
ax1.set_ylabel("Multi-Task Accuracy (%)", fontsize=12, fontweight='semibold')
ax1.set_ylim(20, 65)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: SAM-Trained Experts
for i, method in enumerate(methods):
    ax2.bar(x + (i - 2) * width, sam_data[method], width, label=method, color=colors[i], edgecolor='black', linewidth=0.5)

ax2.set_title("Test-Time Adaptation under SAM Experts", fontsize=14, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(plot_corruptions, fontsize=12)
ax2.set_ylabel("Multi-Task Accuracy (%)", fontsize=12, fontweight='semibold')
ax2.set_ylim(20, 65)
ax2.grid(True, linestyle='--', alpha=0.6)

# Add single legend for the entire figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=12, frameon=True)

plt.tight_layout()
fig.subplots_adjust(bottom=0.12)
plt.savefig("results_plot.png", dpi=300, bbox_inches='tight')
print("Plots generated successfully and saved to results_plot.png!")
