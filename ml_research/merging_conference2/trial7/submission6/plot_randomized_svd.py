import json
import matplotlib.pyplot as plt
import numpy as np

# Load the benchmarking data
with open("results_randomized_svd.json", "r") as f:
    data = json.load(f)

full_time = data["full_svd"]["time"]
full_acc = data["full_svd"]["avg_acc"] * 100

r_svd_data = data["r_svd"]
fractions = [float(k) for k in r_svd_data.keys()]
times = [r_svd_data[k]["time"] for k in r_svd_data.keys()]
speedups = [r_svd_data[k]["speedup"] for k in r_svd_data.keys()]
errors = [r_svd_data[k]["frob_error"] * 100 for k in r_svd_data.keys()]
accuracies = [r_svd_data[k]["avg_acc"] * 100 for k in r_svd_data.keys()]

# Sort by fraction ascending
sorted_indices = np.argsort(fractions)
fractions = np.array(fractions)[sorted_indices]
times = np.array(times)[sorted_indices]
speedups = np.array(speedups)[sorted_indices]
errors = np.array(errors)[sorted_indices]
accuracies = np.array(accuracies)[sorted_indices]

# Create a professional figure with dual axes
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Rank Fraction ($r / \min(d_1, d_2)$)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Multi-Task Average Accuracy (%)', color=color, fontweight='bold', fontsize=12)
line1 = ax1.plot(fractions, accuracies, marker='o', linewidth=2.5, color=color, label='Average Accuracy (%)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('SVD Compute Speedup (x)', color=color, fontweight='bold', fontsize=12)
line2 = ax2.plot(fractions, speedups, marker='s', linewidth=2.5, color=color, linestyle='--', label='Speedup (x)')
ax2.tick_params(axis='y', labelcolor=color)

# Add error curve to ax1 (using right y-axis or a third y-axis, let's just plot on ax1 with a dotted line)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
color = 'tab:green'
ax3.set_ylabel('Frobenius Reconstruction Error (%)', color=color, fontweight='bold', fontsize=12)
line3 = ax3.plot(fractions, errors, marker='^', linewidth=2.5, color=color, linestyle='-.', label='Frobenius Error (%)')
ax3.tick_params(axis='y', labelcolor=color)

# Combine legends
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', fontsize=10)

plt.title('Randomized SPS-Merge (R-SPS-Merge) Computational & Performance Trade-offs', fontweight='bold', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('randomized_svd_tradeoff.png', dpi=300)
plt.close()
print("Trade-off plot generated and saved as randomized_svd_tradeoff.png.")
