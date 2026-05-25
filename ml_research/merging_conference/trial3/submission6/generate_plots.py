import matplotlib.pyplot as plt
import numpy as np

# Data
modes = ['SGD', 'SAM', 'F-SAM (Inverse)', 'F-SAM (Direct)']
merged_acc = [90.81, 91.15, 84.47, 90.89]
procrustes_norm = [0.60824, 0.61124, 0.73379, 0.62910]
avg_expert_acc = [97.82, 98.17, 96.91, 98.21]

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Merged Accuracy vs Avg Expert Accuracy
x = np.arange(len(modes))
width = 0.35

rects1 = ax1.bar(x - width/2, avg_expert_acc, width, label='Avg Expert Acc', color='#4F81BD')
rects2 = ax1.bar(x + width/2, merged_acc, width, label='Merged Model Acc', color='#C0504D')

ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Expert vs. Merged Model Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(modes)
ax1.set_ylim(80, 100)
ax1.legend()

# Add value labels
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1, ax1)
autolabel(rects2, ax1)

# Plot 2: Procrustes Residual Norm
ax2.plot(modes, procrustes_norm, marker='o', linewidth=2.5, color='#8064A2', markersize=8)
ax2.set_ylabel('Average Procrustes Residual Norm')
ax2.set_title('Weight Space Geometric Distortion')
ax2.set_ylim(0.55, 0.80)

# Value labels on line
for i, val in enumerate(procrustes_norm):
    ax2.annotate(f'{val:.5f}', (modes[i], val), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results_plot.png', dpi=300)
plt.savefig('results_plot.pdf', dpi=300)
print("Plots generated successfully!")
