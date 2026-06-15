import matplotlib.pyplot as plt

# Data
sparsities_plot = [0.0, 0.5, 0.8]
methods_plot = ['Uniform', 'Ada-then-P', 'P-then-M', 'ZipMerge-STE', 'ZipMerge-ES']

# Hardcoded exact mean values from experiment_results.md
results_store = {
    0.0: {
        'Uniform': 13.17,
        'Ada-then-P': 13.30,
        'P-then-M': 13.17,
        'ZipMerge-STE': 13.30,
        'ZipMerge-ES': 13.30
    },
    0.5: {
        'Uniform': 11.89,  # M-then-P
        'Ada-then-P': 11.17,
        'P-then-M': 14.81,
        'ZipMerge-STE': 11.23,
        'ZipMerge-ES': 14.00
    },
    0.8: {
        'Uniform': 10.21,  # M-then-P
        'Ada-then-P': 10.73,
        'P-then-M': 16.97,
        'ZipMerge-STE': 11.32,
        'ZipMerge-ES': 10.47
    }
}

colors = {
    'Uniform': '#7f8c8d',       # Slate Gray
    'Ada-then-P': '#2980b9',    # Soft Blue
    'P-then-M': '#d35400',      # Dark Orange
    'ZipMerge-STE': '#c0392b',  # Crimson Red
    'ZipMerge-ES': '#27ae60'    # Emerald Green
}

markers = {
    'Uniform': 'o',
    'Ada-then-P': 's',
    'P-then-M': '^',
    'ZipMerge-STE': 'D',
    'ZipMerge-ES': 'v'
}

# Use a professional layout
plt.figure(figsize=(9, 7))

for method in methods_plot:
    means = [results_store[p][method] for p in sparsities_plot]
    plt.plot(
        sparsities_plot, 
        means, 
        label=method, 
        color=colors[method], 
        marker=markers[method], 
        linewidth=3, 
        markersize=10,
        markeredgewidth=1.5,
        markeredgecolor='black' if method in ['ZipMerge-STE', 'ZipMerge-ES', 'P-then-M'] else 'none'
    )
    
plt.title("ZipMerge Multi-Task Performance under Pruning Constraints", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Target Sparsity Ratio ($p$)", fontsize=14, labelpad=10)
plt.ylabel("Joint Mean Test Accuracy (%)", fontsize=14, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7')
plt.xticks(sparsities_plot, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(9.0, 18.0)
plt.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='#bdc3c7')
plt.tight_layout()

# Save paths
plot_path_results = "results/comparison_plot.png"
plot_path_submission = "submission/comparison_plot.png"

plt.savefig(plot_path_results, dpi=300)
plt.savefig(plot_path_submission, dpi=300)
print(f"Successfully generated and saved plots to:\n  - {plot_path_results}\n  - {plot_path_submission}")
