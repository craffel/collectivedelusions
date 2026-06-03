import matplotlib.pyplot as plt
import numpy as np

# Set style for professional ML papers
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# Define data for alpha sweep (Task-Specific Heads)
alphas = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
accuracies = np.array([43.82, 44.94, 42.42, 40.95, 38.33, 35.74, 32.27, 30.94])

# Plot 1: Hyperparameter Sweep of alpha
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(alphas, accuracies, marker='o', color='#1f77b4', linewidth=2, markersize=6, label='STDFS (Ours)')
ax.axhline(y=30.94, color='#d62728', linestyle='--', linewidth=1.5, label='Weight Averaging (WA)')
ax.axhline(y=30.91, color='#2ca02c', linestyle=':', linewidth=1.5, label='Task Arithmetic (TA, max)')

ax.set_xlabel(r'Low-Frequency Ratio ($\alpha$)')
ax.set_ylabel('Average Accuracy (%)')
ax.set_title('Impact of Low-Frequency Ratio on STDFS Merging')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')

# Adjust layout and save
plt.tight_layout()
plt.savefig('alpha_sweep.pdf', dpi=300)
plt.savefig('alpha_sweep.png', dpi=300)
plt.close()

# Plot 2: Method Comparison Bar Chart
datasets = ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'Average']
x = np.arange(len(datasets))
width = 0.15

# Accuracies of key methods under Task-Specific Heads
wa = [35.39, 29.04, 28.39, 30.94]
ta = [34.86, 28.36, 29.50, 30.91]
ties = [15.09, 12.51, 16.74, 14.78]
stdfs = [55.00, 43.99, 35.82, 44.94]
experts = [99.17, 91.65, 79.23, 90.02]

fig, ax = plt.subplots(figsize=(8, 4.5))

# Plot bars with professional colors
rects1 = ax.bar(x - 2*width, ties, width, label='TIES-Merging', color='#bcbd22', edgecolor='black', linewidth=0.5, alpha=0.85)
rects2 = ax.bar(x - width, wa, width, label='Weight Averaging', color='#d62728', edgecolor='black', linewidth=0.5, alpha=0.85)
rects3 = ax.bar(x, ta, width, label='Task Arithmetic', color='#2ca02c', edgecolor='black', linewidth=0.5, alpha=0.85)
rects4 = ax.bar(x + width, stdfs, width, label=r'STDFS (Ours, $\alpha=0.1$)', color='#1f77b4', edgecolor='black', linewidth=0.5, alpha=0.95)
rects5 = ax.bar(x + 2*width, experts, width, label='Individual Experts', color='#7f7f7f', edgecolor='black', linewidth=0.5, alpha=0.5)

# Add labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Performance Comparison Across Diverse Vision Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 110)
ax.grid(True, axis='y', linestyle=':', alpha=0.6)
ax.legend(loc='upper right', ncol=2)

# Helper to add value labels above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

# Only label our method and experts/WA to avoid cluttering
autolabel(rects2)
autolabel(rects4)

plt.tight_layout()
plt.savefig('method_comparison.pdf', dpi=300)
plt.savefig('method_comparison.png', dpi=300)
plt.close()

print("Plots generated successfully!")
