import numpy as np
import matplotlib.pyplot as plt

# Set style for academic paper
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'font.family': 'sans-serif',
    'text.usetex': False
})

# Create a figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# -------------------------------------------------------------
# SUBPLOT 1: Conceptual Loss Landscape (Sharp vs. Flat Minima)
# -------------------------------------------------------------
ax1 = axes[0]
w = np.linspace(-3, 3, 500)

# Training loss functions
L_flat_train = 0.08 * w**2 + 0.3
L_sharp_train = 0.8 * w**2 + 0.2

# Test loss functions under domain shift / corruption (shifted by delta_w = 1.0)
delta_w = 1.0
L_flat_test = 0.08 * (w - delta_w)**2 + 0.3
L_sharp_test = 0.8 * (w - delta_w)**2 + 0.2

# Plot training landscapes (solid lines)
ax1.plot(w, L_flat_train, color='#1f77b4', linestyle='-', linewidth=2.5, label='Flat Minimum (PW-SASLA - Train)')
ax1.plot(w, L_sharp_train, color='#d62728', linestyle='-', linewidth=2.5, label='Sharp Minimum (SyMerge - Train)')

# Plot test landscapes under shift (dashed lines)
ax1.plot(w, L_flat_test, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.plot(w, L_sharp_test, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7)

# Mark the optimized parameter values at w=0 (before shift)
ax1.scatter(0, L_flat_train[250], color='#1f77b4', s=60, zorder=5)
ax1.scatter(0, L_sharp_train[250], color='#d62728', s=60, zorder=5)

# Calculate index for shifted losses at w=0
idx_w0 = 250
loss_flat_shift = L_flat_test[idx_w0]
loss_sharp_shift = L_sharp_test[idx_w0]

# Mark the test losses at the optimized parameter values w=0
ax1.scatter(0, loss_flat_shift, color='#1f77b4', marker='X', s=80, zorder=5)
ax1.scatter(0, loss_sharp_shift, color='#d62728', marker='X', s=80, zorder=5)

# Add vertical arrows pointing from train loss to test loss at w=0
ax1.annotate('', xy=(0, loss_flat_shift), xytext=(0, L_flat_train[idx_w0]),
             arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5, ls=':'))
ax1.annotate('', xy=(0, loss_sharp_shift), xytext=(0, L_sharp_train[idx_w0]),
             arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.5, ls=':'))

# Labels and texts
ax1.text(-0.5, L_flat_train[idx_w0] - 0.05, 'Train Loss', color='#1f77b4', fontweight='bold', ha='right')
ax1.text(-0.5, loss_flat_shift + 0.05, 'OOD Test Loss', color='#1f77b4', fontweight='bold', ha='right')
ax1.text(0.5, L_sharp_train[idx_w0] - 0.05, 'Train Loss', color='#d62728', fontweight='bold', ha='left')
ax1.text(0.5, loss_sharp_shift + 0.05, 'OOD Test Loss', color='#d62728', fontweight='bold', ha='left')

ax1.set_title("A. Conceptual Landscape Under Domain Shift", fontsize=11, pad=10)
ax1.set_xlabel("Parameter Space (w)")
ax1.set_ylabel("Loss")
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(0.0, 2.5)
ax1.legend(loc='upper center', frameon=True, facecolor='white', framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# SUBPLOT 2: Empirical Results (Clean vs. Corrupted Average)
# -------------------------------------------------------------
ax2 = axes[1]

# Data from Table 1 and Table 2 in paper
methods = ['Task Arithmetic', 'AdaMerging', 'SyMerge', 'U-SASLA (Global)', 'PW-SASLA (Ours)']
clean_accs = [90.87, 91.40, 92.47, 92.50, 92.87]
corr_accs = [74.10, 74.67, 74.73, 73.37, 75.03]

x = np.arange(2)  # Two groups: Clean and Corrupted
width = 0.15      # Width of the bars

# Custom colors for the methods
colors = ['#7f7f7f', '#ff7f0e', '#d62728', '#2ca02c', '#1f77b4']

for i, method in enumerate(methods):
    y_vals = [clean_accs[i], corr_accs[i]]
    rects = ax2.bar(x + (i - 2) * width, y_vals, width, label=method, color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.9)
    # Add labels on top of the bars
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 2 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_title("B. Empirical Multi-Task Average Accuracy", fontsize=11, pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(['Clean Datasets (Average)', 'Corrupted Datasets (Average)'], fontweight='bold')
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(60, 100)  # Zoom in on the differences
ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('template/icml_numpapers.pdf', bbox_inches='tight', dpi=300)
print("Figure compiled and saved to template/icml_numpapers.pdf successfully!")
