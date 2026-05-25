import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("src")
from merge_tta import run_tta

# Set seed for reproducibility
torch.manual_seed(42)

# Configurations
calib_sizes = [16, 32, 64, 128, 256, 512]
eval_configs = [
    {"task": "fashionmnist", "corruption": "noise"},
    {"task": "kmnist", "corruption": "rotation"},
    {"task": "mnist", "corruption": "rotation"}
]

# Run sweep
results = {f"{config['task']}_{config['corruption']}": [] for config in eval_configs}

print("Starting Calibration Samples Ablation Sweep...")
for size in calib_sizes:
    for config in eval_configs:
        task = config["task"]
        corr = config["corruption"]
        # Run TTA and get accuracy after
        _, acc_after, _ = run_tta(
            task_name=task,
            method="ca-symerge",
            corruption=corr,
            num_batches=15,
            calib_samples=size,
            gamma=15.0
        )
        key = f"{task}_{corr}"
        results[key].append(acc_after)

# Save results
os.makedirs("results", exist_ok=True)
with open("results/ablation_calib_samples.txt", "w") as f:
    f.write("calib_samples," + ",".join([f"{c['task']}_{c['corruption']}" for c in eval_configs]) + "\n")
    for idx, size in enumerate(calib_sizes):
        line_vals = [f"{results[f'{c['task']}_{c['corruption']}'][idx]:.2f}" for c in eval_configs]
        f.write(f"{size}," + ",".join(line_vals) + "\n")

print("Ablation results saved to results/ablation_calib_samples.txt")

# Set style for academic publishing
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.3,
})

plt.figure(figsize=(7, 4.5))

colors = ["#7293CB", "#E1974C", "#84BA5B"]
markers = ["o", "s", "^"]
labels = ["FashionMNIST (Noise)", "KMNIST (Rotation)", "MNIST (Rotation)"]

for idx, config in enumerate(eval_configs):
    key = f"{config['task']}_{config['corruption']}"
    vals = results[key]
    plt.plot(calib_sizes, vals, label=labels[idx], color=colors[idx], marker=markers[idx], linewidth=1.8, markersize=6)

plt.xlabel("Number of Calibration Samples ($N$)")
plt.ylabel("Accuracy After TTA (%)")
plt.title("Ablation of Calibration Sample Size ($N$)", fontweight='bold', pad=12)
plt.xscale('log')
plt.xticks(calib_sizes, labels=[str(s) for s in calib_sizes])
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc="best", framealpha=0.9)

plt.tight_layout()
plt.savefig("results/ablation_calib_samples.pdf", bbox_inches='tight')
plt.savefig("results/ablation_calib_samples.png", dpi=300, bbox_inches='tight')
print("Successfully generated and saved results/ablation_calib_samples.pdf and results/ablation_calib_samples.png")
