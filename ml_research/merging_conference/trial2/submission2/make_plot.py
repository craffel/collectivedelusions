import json
import os
import matplotlib.pyplot as plt

modes = ["standard", "sam", "so_lora", "so_lora_sam"]
labels = {
    "standard": "Standard LoRA",
    "sam": "SAM LoRA",
    "so_lora": "SO-LoRA (Ours)",
    "so_lora_sam": "SO-LoRA + SAM (Ours)"
}
colors = {
    "standard": "gray",
    "sam": "blue",
    "so_lora": "orange",
    "so_lora_sam": "red"
}
markers = {
    "standard": "o",
    "sam": "s",
    "so_lora": "^",
    "so_lora_sam": "D"
}

x_vals = [0.5, 0.7, 1.0]

plt.figure(figsize=(7, 5))

for mode in modes:
    filename = f"results_{mode}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        
        y_vals = []
        for key in ["coeffs_0.5_0.5", "coeffs_0.7_0.7", "coeffs_1.0_1.0"]:
            if key in data:
                y_vals.append(data[key]["avg_acc"])
        
        if len(y_vals) == 3:
            plt.plot(x_vals, y_vals, label=labels[mode], color=colors[mode], marker=markers[mode], linewidth=2, markersize=8)

plt.xlabel("Merging Coefficient ($\\lambda_1 = \\lambda_2$)", fontsize=12)
plt.ylabel("Multi-Task Test Accuracy (%)", fontsize=12)
plt.title("Model Merging Compatibility on Split CIFAR-10", fontsize=14, fontweight="bold")
plt.xticks(x_vals)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10, loc="lower right" if os.path.exists("results_so_lora_sam.json") else "best")
plt.tight_layout()

plt.savefig("results_plot.pdf", format="pdf")
plt.savefig("results_plot.png", format="png", dpi=300)
print("Saved plots to results_plot.pdf and results_plot.png!")
