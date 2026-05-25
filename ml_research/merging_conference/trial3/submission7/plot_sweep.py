import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Data
lrs = [0.05, 0.10, 0.20, 0.50]

# Sequential Stream Data
seq_static = [53.98, 53.98, 53.98, 53.98]
seq_std = [78.88, 82.40, 84.35, 85.42]
seq_l2_10 = [78.46, 82.15, 84.06, 85.15]
seq_l2_100 = [76.40, 79.85, 81.83, 82.33]
seq_ewc_10 = [78.88, 82.40, 84.35, 85.44]
seq_ewc_100 = [78.90, 82.38, 84.40, 85.40]

# Alternating Stream Data
alt_static = [53.98, 53.98, 53.98, 53.98]
alt_std = [65.17, 64.60, 64.62, 64.71]
alt_l2_10 = [64.19, 63.88, 63.81, 63.90]
alt_l2_100 = [60.06, 60.04, 60.04, 60.38]
alt_ewc_10 = [65.15, 64.60, 64.60, 64.69]
alt_ewc_100 = [65.12, 64.60, 64.60, 64.65]

# Plot Sequential Stream
ax1.plot(lrs, seq_static, 'k--', label='Static Merged', linewidth=1.5)
ax1.plot(lrs, seq_std, 'o-', color='#d95f02', label='Standard TTA ($\gamma=0$)', linewidth=2.0)
ax1.plot(lrs, seq_l2_100, 'v:', color='#e7298a', label='L2-TTA ($\gamma=100.0$)', linewidth=2.0)
ax1.plot(lrs, seq_ewc_100, 's-', color='#1b9e77', label='EWC-TTA ($\gamma=100.0$, Ours)', linewidth=2.5, markersize=7)

ax1.set_title('Sequential Stream (Severe Domain Shift)', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Coefficient Learning Rate ($\eta_\lambda$)', fontsize=11)
ax1.set_ylabel('Overall Accuracy (%)', fontsize=11)
ax1.set_xticks(lrs)
ax1.set_ylim(50, 90)
ax1.tick_params(axis='both', which='major', labelsize=10)

# Add data annotations for EWC vs L2 on Ax1
for x, y_ewc, y_l2 in zip(lrs, seq_ewc_100, seq_l2_100):
    ax1.annotate(f'{y_ewc:.1f}%', (x, y_ewc), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, fontweight='bold', color='#1b9e77')
    ax1.annotate(f'{y_l2:.1f}%', (x, y_l2), textcoords="offset points", xytext=(0,-13), ha='center', fontsize=9, color='#e7298a')

# Plot Alternating Stream
ax2.plot(lrs, alt_static, 'k--', label='Static Merged', linewidth=1.5)
ax2.plot(lrs, alt_std, 'o-', color='#d95f02', label='Standard TTA ($\gamma=0$)', linewidth=2.0)
ax2.plot(lrs, alt_l2_100, 'v:', color='#e7298a', label='L2-TTA ($\gamma=100.0$)', linewidth=2.0)
ax2.plot(lrs, alt_ewc_100, 's-', color='#1b9e77', label='EWC-TTA ($\gamma=100.0$, Ours)', linewidth=2.5, markersize=7)

ax2.set_title('Alternating Stream (High Frequency)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Coefficient Learning Rate ($\eta_\lambda$)', fontsize=11)
ax2.set_ylabel('Overall Accuracy (%)', fontsize=11)
ax2.set_xticks(lrs)
ax2.set_ylim(50, 70)
ax2.tick_params(axis='both', which='major', labelsize=10)

# Add data annotations for EWC vs L2 on Ax2
for x, y_ewc, y_l2 in zip(lrs, alt_ewc_100, alt_l2_100):
    ax2.annotate(f'{y_ewc:.1f}%', (x, y_ewc), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, fontweight='bold', color='#1b9e77')
    ax2.annotate(f'{y_l2:.1f}%', (x, y_l2), textcoords="offset points", xytext=(0,-13), ha='center', fontsize=9, color='#e7298a')

# Joint legend and title
ax1.legend(loc='lower right', frameon=True, fontsize=10)
plt.suptitle('TTA Performance Scaling with Coefficient Learning Rate ($\eta_\lambda$)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results_plot.png', dpi=300, bbox_inches='tight')
print("Successfully generated systematic results_plot.png!")
