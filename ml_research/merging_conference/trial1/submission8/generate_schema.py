import matplotlib.pyplot as plt
import numpy as np

# Set style for a clean academic plot
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 5))

# Hide axes as this is a schematic diagram
ax.axis('off')

# 1. Left side: Loss Landscape Comparison (Sharp vs. Flat Minima)
x = np.linspace(-3, 3, 400)
# Sharp minimum (Standard OFT/Euclidean)
y_sharp = (x + 1.5)**4 + 2 * (x + 1.5)**2
# Flat minimum (SA-Ortho)
y_flat = 0.15 * (x - 1.5)**4 + 0.5 * (x - 1.5)**2

# Scale and shift to fit the canvas
ax.plot(x - 2, y_sharp * 0.15 + 1, color='tab:red', linewidth=3, label='Standard Manifold Minimum (Sharp)')
ax.plot(x + 2, y_flat * 0.15 + 1, color='tab:green', linewidth=3, label='SA-Ortho Manifold Minimum (Flat)')

# Draw points
# Sharp point
ax.scatter([-3.5], [1], color='darkred', s=120, zorder=5)
ax.annotate("Task 1 (Sharp)\nRepresentation", xy=(-3.5, 1), xytext=(-5.2, 2.2),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
            fontsize=11, fontweight='bold', ha='center')

# Flat point
ax.scatter([3.5], [1], color='darkgreen', s=120, zorder=5)
ax.annotate("Task 2 (Flat, SA-Ortho)\nRepresentation", xy=(3.5, 1), xytext=(5.2, 2.2),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
            fontsize=11, fontweight='bold', ha='center')

# Draw perturbation arrows on the flat minimum
ax.annotate("", xy=(3.9, 1.1), xytext=(3.5, 1),
            arrowprops=dict(arrowstyle="<->", color="tab:orange", lw=2.5, connectionstyle="arc3,rad=.2"))
ax.text(3.7, 1.3, r"$\epsilon \in so(d)$", color="tab:orange", fontsize=11, fontweight='bold')

# 2. Middle: Merging (Riemannian Interpolation)
ax.annotate("", xy=(-1.5, -1.0), xytext=(-3.5, 0.5),
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.5, linestyle="dashed"))
ax.annotate("", xy=(1.5, -1.0), xytext=(3.5, 0.5),
            arrowprops=dict(arrowstyle="->", color="tab:green", lw=2.5))

ax.text(0, -0.6, "Manifold Merging\n(Lie Algebra Interpolation)\n$Q_{merged} = \\alpha Q_1 + (1-\\alpha) Q_2$", 
        fontsize=12, fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.5))

# 3. Bottom: Merged Landscape
x_m = np.linspace(-3, 3, 400)
# Merged landscape (under interference vs stability)
y_merged_unstable = (x_m)**2 + 2.5 # Unstable merging
y_merged_stable = 0.3 * (x_m)**2 + 0.8   # Stable merging from flat minima

ax.plot(x_m, y_merged_unstable, color='tab:red', linestyle='--', linewidth=2, label='Merged Model (Interference/Sharp)')
ax.plot(x_m, y_merged_stable, color='tab:green', linewidth=3.5, label='Merged Model (Stable/Flat Manifold)')

# Merged representation points
ax.scatter([0], [0.8], color='darkgreen', s=150, marker='^', zorder=5)
ax.text(0, 0.4, "Optimal Multi-task\nMerged Model", fontsize=11, fontweight='bold', ha='center')

# Add legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig("schema.png", dpi=150)
print("Schema generated successfully and saved to schema.png")
