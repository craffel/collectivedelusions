import matplotlib.pyplot as plt
import numpy as np

# Set professional scientific plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0

# Data for Heterogeneous Stream
methods = ["Static", "EBER", "AdaMerging", "DR-Fisher", "IGGS-OW", "HAT-Merge (Ours)"]
clean_accs = [14.37, 41.50, 41.48, 41.44, 38.42, 67.25]
gaussian_accs = [12.81, 39.67, 39.69, 39.67, 32.94, 64.27]
contrast_accs = [9.65, 21.62, 21.21, 21.21, 21.46, 21.56]

x = np.arange(3)  # the label locations (Clean, Gaussian, Contrast)
width = 0.12  # the width of the bars

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

# Color palette matching top-tier publications
colors = ['#7f7f7f', '#aec7e8', '#ffbb78', '#2ca02c', '#d62728', '#1f77b4']

for idx, method in enumerate(methods):
    accs = [clean_accs[idx], gaussian_accs[idx], contrast_accs[idx]]
    offset = (idx - 2.5) * width
    rects = ax.bar(x + offset, accs, width, label=method, color=colors[idx], edgecolor='black', linewidth=0.5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Overall Accuracy (%)', fontweight='bold')
ax.set_title('Overall Performance on Heterogeneous (Mixed) Test Streams', fontweight='bold', fontsize=12, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(['Clean', 'Gaussian Noise', 'Contrast Shift'], fontweight='bold')
ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper right')

# Style grid and spines
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels on top of the bars for Ours and top baseline
for i in range(3):
    # HAT-Merge (Ours)
    ax.text(i + 2.5 * width, [clean_accs[5], gaussian_accs[5], contrast_accs[5]][i] + 1.0, 
            f"{[clean_accs[5], gaussian_accs[5], contrast_accs[5]][i]:.1f}%", 
            ha='center', va='bottom', fontsize=9, color='#1f77b4', fontweight='bold')
    # Best baseline (EBER/DR-Fisher)
    best_baseline = max(clean_accs[0:5][i], gaussian_accs[0:5][i], contrast_accs[0:5][i])
    ax.text(i - 1.5 * width, [clean_accs[1], gaussian_accs[1], contrast_accs[1]][i] + 1.0, 
            f"{[clean_accs[1], gaussian_accs[1], contrast_accs[1]][i]:.1f}%", 
            ha='center', va='bottom', fontsize=9, color='#7f7f7f')

plt.tight_layout()
plt.savefig('heterogeneous_stream_acc.png', bbox_inches='tight')
print("Saved plot as heterogeneous_stream_acc.png")
