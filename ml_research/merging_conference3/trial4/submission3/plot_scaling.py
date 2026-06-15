import matplotlib.pyplot as plt
import numpy as np

# Set style for professional academic publication (similar to ACM/IEEE/NeurIPS)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'text.usetex': False  # Avoid requiring full LaTeX for simple rendering
})

# Data from Appendix C Table 8 / 11
K = np.array([1, 5, 20, 100])
coexistence_mem = np.array([3.517, 3.584, 3.835, 5.176])
merging_mem = np.array([3.50, 3.50, 3.50, 3.50])

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# Plot Co-existence line
ax.plot(K, coexistence_mem, marker='o', markersize=6, color='#d62728', 
        linestyle='-', linewidth=2, label='Native-Format Co-existence (NF4 + FP16 Adapters)')

# Plot Merging line
ax.plot(K, merging_mem, marker='s', markersize=6, color='#1f77b4', 
        linestyle='--', linewidth=2, label='Weight-Space Merging (INT4, O(1) scaling)')

# Customization
ax.set_xlabel('Number of Active Tasks ($K$)', fontweight='bold')
ax.set_ylabel('Total Memory Footprint (GB)', fontweight='bold')
ax.set_title('Multi-Task Memory Scaling (7B parameter base model)', pad=12, fontweight='bold')
ax.set_xlim(-2, 105)
ax.set_ylim(3.0, 5.5)

# Adding annotations for key points
for k, val in zip([1, 100], [3.517, 5.176]):
    ax.annotate(f"{val:.3f} GB", (k, val), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='#d62728')

for k, val in zip([1, 100], [3.50, 3.50]):
    ax.annotate(f"{val:.2f} GB", (k, val), textcoords="offset points", xytext=(0,-15), ha='center', fontweight='bold', color='#1f77b4')

ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='lightgray')
plt.tight_layout()

# Save as PDF for vector graphics in LaTeX
plt.savefig('submission/memory_scaling.pdf', format='pdf', bbox_inches='tight')
plt.savefig('submission/memory_scaling.png', format='png', bbox_inches='tight')
print("Successfully generated memory_scaling.pdf and memory_scaling.png")
