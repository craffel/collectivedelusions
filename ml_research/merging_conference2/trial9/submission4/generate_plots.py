import matplotlib.pyplot as plt
import numpy as np

# Set style for academic paper
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'text.usetex': False  # Use default matplotlib math-text to avoid TeX requirement issues
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# ----------------------------------------------------------------------
# Panel (a): MLP Merging Accuracy (FP32 vs 8-bit Quantized)
# ----------------------------------------------------------------------
methods_mlp = ['WA', 'TA', 'S-IPR', 'WCPR', 'D-BWPA\n(Ours)', 'Full-BWPA\n(Ablation)']
fp32_mlp = [38.10, 40.10, 35.16, 41.96, 40.50, 16.82]
q8_mlp = [38.10, 40.12, 35.18, 41.98, 40.53, 16.74]

x = np.arange(len(methods_mlp))
width = 0.35

rects1 = ax1.bar(x - width/2, fp32_mlp, width, label='FP32', color='#3498db', edgecolor='black', alpha=0.9)
rects2 = ax1.bar(x + width/2, q8_mlp, width, label='8-bit PTQ', color='#e74c3c', edgecolor='black', alpha=0.9)

ax1.set_ylabel('Average Accuracy (%)')
ax1.set_title('(a) MLP Model Merging Accuracy (No BatchNorm)')
ax1.set_xticks(x)
ax1.set_xticklabels(methods_mlp)
ax1.legend(loc='upper right')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(0, 55)

# Add value labels on top of bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1, ax1)
autolabel(rects2, ax1)

# ----------------------------------------------------------------------
# Panel (b): ResNet-18 Merging Accuracy under Quantization
# ----------------------------------------------------------------------
# Show the difference with and without DE-BN under 8-bit PTQ
methods_res = ['WA', 'TA', 'Full-BWPA\n(Ablation)', 'D-BWPA\n(Ours)']
no_debn_res = [25.88, 29.29, 19.59, 32.06]
with_debn_res = [69.10, 69.82, 45.02, 67.81]

x_res = np.arange(len(methods_res))

rects3 = ax2.bar(x_res - width/2, no_debn_res, width, label='Without DE-BN', color='#95a5a6', edgecolor='black', alpha=0.9)
rects4 = ax2.bar(x_res + width/2, with_debn_res, width, label='With DE-BN (Corrected)', color='#2ecc71', edgecolor='black', alpha=0.9)

ax2.set_ylabel('Average Accuracy (%)')
ax2.set_title('(b) ResNet-18 8-bit Quantized Merging Accuracy')
ax2.set_xticks(x_res)
ax2.set_xticklabels(methods_res)
ax2.legend(loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(0, 85)

autolabel(rects3, ax2)
autolabel(rects4, ax2)

# Adjust layout
plt.tight_layout()

# Save figure as both PNG and PDF for LaTeX compatibility and high quality
plt.savefig('merging_comparison.pdf', dpi=300)
plt.savefig('merging_comparison.png', dpi=300)
print("Plots successfully generated and saved as merging_comparison.pdf and merging_comparison.png.")
