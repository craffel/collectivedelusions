import matplotlib.pyplot as plt
import numpy as np

# Data
labels = [
    "Weight Averaging\n(WA Baseline)",
    "Single-Coeff\nAOS-CKA",
    "Single-Coeff\nOracle Peak",
    "M-AOS CKA\n(Ours)",
    "Multidimensional\nOracle Peak"
]

means = [34.58, 58.34, 58.43, 59.67, 60.08]
stds = [0.00, 0.19, 0.00, 0.23, 0.06]

# Colors
colors = ["#7f7f7f", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]

plt.figure(figsize=(9, 5.5))
bars = plt.bar(labels, means, yerr=stds, color=colors, capsize=8, edgecolor="black", alpha=0.85, width=0.6)

# Grid and Styling
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=12, fontweight="bold")
plt.title("AOS Merging Performance: Single-Coefficient vs. Multidimensional", fontsize=14, fontweight="bold", pad=15)
plt.ylim(30, 65)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.8, f"{height:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("maos_comparison.png", dpi=300)
plt.close()
print("Saved maos_comparison.png successfully.")
