import matplotlib.pyplot as plt
import numpy as np

# Set style for academic paper
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False  # avoid requiring latex installed for matplotlib
})

# Color palette
colors = {
    'joint': '#1f77b4',      # Muted Blue
    'seq': '#ff7f0e',        # Muted Orange
    'head': '#2ca02c',       # Muted Green
    'collapse': '#d62728'    # Red for collapse
}

# ----------------------------------------------------
# Plot 1: Temperature Sensitivity
# ----------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 2.8))

temps = [0.5, 1.0, 2.0, 5.0]
acc_joint = [43.00, 55.13, 54.20, 47.60]
acc_seq = [42.07, 56.87, 53.87, 48.47]
acc_head = [38.93, 51.67, 51.87, 43.67]

ax1.plot(temps, acc_joint, marker='o', linestyle='-', linewidth=2, color=colors['joint'], label='Joint TTA (SyMerge-style)')
ax1.plot(temps, acc_seq, marker='s', linestyle='--', linewidth=2, color=colors['seq'], label='Sequential TTA (Proposed)')
ax1.plot(temps, acc_head, marker='^', linestyle=':', linewidth=2, color=colors['head'], label='Head-only TTA')

ax1.set_xlabel('Distillation Temperature ($\\tau$)')
ax1.set_ylabel('Average Test Accuracy (%)')
ax1.set_title('Effect of Distillation Temperature on Adaptation Accuracy')
ax1.set_xticks(temps)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
ax1.set_ylim(35, 60)

plt.tight_layout()
fig1.savefig('fig_ablation_temp.pdf', dpi=300)
print("Saved fig_ablation_temp.pdf")

# ----------------------------------------------------
# Plot 2: Clamping Bound Sweep (Good Initialization)
# ----------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 2.8))

clamp_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', 'None']
x = np.arange(len(clamp_labels))

acc_joint_clamp = [55.40, 58.87, 59.47, 8.13, 8.13, 8.13]
acc_seq_clamp = [54.07, 58.07, 59.47, 8.13, 8.13, 8.13]

ax2.plot(x, acc_joint_clamp, marker='o', linestyle='-', linewidth=2, color=colors['joint'], label='Joint TTA (SyMerge-style)')
ax2.plot(x, acc_seq_clamp, marker='s', linestyle='--', linewidth=2, color=colors['seq'], label='Sequential TTA (Proposed)')

# Draw a red shaded background for the runaway collapse region (indices 3 to 5)
ax2.axvspan(2.5, 5.5, color='#ffcccc', alpha=0.5, label='Runaway Collapse (NaN)')

# Annotate the collapse region
ax2.text(4.0, 25, 'Numerical\nCollapse\n(NaN)', color=colors['collapse'], ha='center', va='center', fontweight='bold', fontsize=11)

ax2.set_xlabel('Clamping Upper Bound ($\\lambda_{\\max}$)')
ax2.set_ylabel('Average Test Accuracy (%)')
ax2.set_title('Clamping Bound Sweep under Good Initialization ($[0.3, 0.3, 0.3]$)')
ax2.set_xticks(x)
ax2.set_xticklabels(clamp_labels)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
ax2.set_ylim(0, 70)

plt.tight_layout()
fig2.savefig('fig_clamping_sweep.pdf', dpi=300)
print("Saved fig_clamping_sweep.pdf")
