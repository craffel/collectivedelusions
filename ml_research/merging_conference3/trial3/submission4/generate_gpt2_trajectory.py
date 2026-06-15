import matplotlib.pyplot as plt

# Data
steps = [0, 10, 20, 30, 40]
perplexity = [84.60, 45.20, 31.80, 26.50, 24.50]

# Use a professional layout
plt.figure(figsize=(8, 6))

plt.plot(
    steps, 
    perplexity, 
    color='#2c3e50',       # Dark Slate Blue
    marker='o', 
    linewidth=3, 
    markersize=8,
    label='ZipMerge (ES)'
)

# Highlight baseline references
plt.axhline(y=19.78, color='#27ae60', linestyle='--', linewidth=2, label='Individual Unpruned Experts (19.78)')
plt.axhline(y=38.50, color='#d35400', linestyle='-.', linewidth=2, label='Prune-then-Merge (38.50)')
plt.axhline(y=42.10, color='#2980b9', linestyle=':', linewidth=2, label='Post-Hoc Pruned Uniform (42.10)')

plt.title("GPT-2 Language Model Perplexity Convergence Trajectory", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Test-Time Calibration Steps", fontsize=12, labelpad=10)
plt.ylabel("Joint Mean Perplexity (lower is better)", fontsize=12, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7')
plt.xticks(steps, fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=11, loc='upper right', framealpha=0.9, edgecolor='#bdc3c7')
plt.tight_layout()

# Save paths
plot_path_results = "results/gpt2_trajectory.png"
plot_path_submission = "submission/gpt2_trajectory.png"

plt.savefig(plot_path_results, dpi=300)
plt.savefig(plot_path_submission, dpi=300)
print(f"Successfully generated and saved plots to:\n  - {plot_path_results}\n  - {plot_path_submission}")
