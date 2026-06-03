import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# Set up matplotlib style for scientific papers
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False  # Keep False to avoid system latex dependencies
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# --- DATA: Main Calibration Sweep ---
# N values
N = [4, 16, 64, 128, 256]

# Accuracies without SFT (Mean Acc)
acc_no_sft = {
    'Uncalibrated': [34.17, 34.17, 34.17, 34.17, 34.17],
    'SP-TAAC (Global)': [27.49, 27.98, 27.41, 27.96, 28.14],
    'N-TAAC (Channel)': [10.12, 19.19, 25.45, 25.68, 25.67],
    'R-TAAC (Regularized)': [11.37, 12.45, 11.73, 11.74, 12.46],
    'HSC (Ours)': [28.68, 31.43, 33.51, 33.52, 34.35]
}

stds_no_sft = {
    'Uncalibrated': [0.00, 0.00, 0.00, 0.00, 0.00],
    'SP-TAAC (Global)': [1.70, 0.93, 0.33, 0.74, 0.19],
    'N-TAAC (Channel)': [0.96, 1.04, 0.86, 0.60, 0.81],
    'R-TAAC (Regularized)': [1.34, 0.78, 1.14, 0.38, 0.62],
    'HSC (Ours)': [2.24, 0.49, 0.55, 1.17, 1.13]
}

# Accuracies with SFT (Mean Acc + SFT)
acc_with_sft = {
    'Uncalibrated': [23.97, 43.64, 60.14, 66.15, 69.82],
    'SP-TAAC (Global)': [22.50, 32.76, 47.56, 53.63, 58.34],
    'N-TAAC (Channel)': [10.22, 16.88, 28.68, 32.28, 36.15],
    'R-TAAC (Regularized)': [14.37, 22.15, 35.63, 45.62, 47.81],
    'HSC (Ours)': [22.20, 33.87, 46.83, 52.70, 57.85]
}

stds_with_sft = {
    'Uncalibrated': [3.06, 3.39, 0.44, 2.61, 0.29],
    'SP-TAAC (Global)': [1.41, 1.66, 1.16, 1.29, 0.71],
    'N-TAAC (Channel)': [0.74, 2.68, 1.49, 0.29, 0.59],
    'R-TAAC (Regularized)': [3.25, 3.60, 0.80, 1.72, 2.44],
    'HSC (Ours)': [2.20, 2.42, 0.55, 0.99, 0.58]
}

colors = {
    'Uncalibrated': '#7f7f7f',       # Gray
    'SP-TAAC (Global)': '#2ca02c',   # Green
    'N-TAAC (Channel)': '#d62728',   # Red
    'R-TAAC (Regularized)': '#ff7f0e', # Orange
    'HSC (Ours)': '#1f77b4'          # Blue
}

markers = {
    'Uncalibrated': 'o',
    'SP-TAAC (Global)': 's',
    'N-TAAC (Channel)': '^',
    'R-TAAC (Regularized)': 'D',
    'HSC (Ours)': 'X'
}

# Figure 1: Mean Accuracy without SFT (Training-Free Robustness)
fig, ax = plt.subplots(figsize=(6, 3.2))
for method, accs in acc_no_sft.items():
    linewidth = 2.5 if 'Ours' in method else 1.5
    markersize = 8 if 'Ours' in method else 6
    alpha = 1.0 if 'Ours' in method else 0.85
    yerr = stds_no_sft[method]
    
    # Plot line with error bars
    ax.errorbar(N, accs, yerr=yerr, label=method, color=colors[method], marker=markers[method], 
                linewidth=linewidth, markersize=markersize, alpha=alpha, capsize=4, elinewidth=1.2)

ax.set_xscale('log')
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Calibration Set Size (N)')
ax.set_ylabel('Mean Test Accuracy (%)')
ax.set_title('Training-Free Activation Calibration')
ax.set_ylim(5, 42)
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', shadow=False)
plt.tight_layout()
plt.savefig('figures/calibration_sweep_no_sft.pdf', dpi=300)
plt.close()

# Figure 2: Mean Accuracy with SFT (Head-Adapted Performance)
fig, ax = plt.subplots(figsize=(6, 3.2))
for method, accs in acc_with_sft.items():
    linewidth = 2.5 if 'Ours' in method else 1.5
    markersize = 8 if 'Ours' in method else 6
    alpha = 1.0 if 'Ours' in method else 0.85
    yerr = stds_with_sft[method]
    
    # Plot line with error bars
    ax.errorbar(N, accs, yerr=yerr, label=method, color=colors[method], marker=markers[method], 
                linewidth=linewidth, markersize=markersize, alpha=alpha, capsize=4, elinewidth=1.2)

ax.set_xscale('log')
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Calibration Set Size (N)')
ax.set_ylabel('Mean Test Accuracy + SFT (%)')
ax.set_title('Supervised Fine-Tuning (SFT) Head Adaptation')
ax.set_ylim(5, 75)
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', shadow=False)
plt.tight_layout()
plt.savefig('figures/calibration_sweep_with_sft.pdf', dpi=300)
plt.close()


# --- PLOT 2: Split Index Sweep (at N=128) ---
split_indices = [0, 5, 10, 15, 19]
split_mean_acc = [11.74, 16.63, 27.72, 33.52, 29.29]
split_mean_std = [0.38, 1.00, 0.45, 1.17, 0.41]

split_sft_acc = [45.62, 48.29, 51.70, 52.70, 52.23]
split_sft_std = [1.72, 1.89, 1.33, 0.99, 1.05]

fig, ax = plt.subplots(figsize=(5.5, 3.0))
ax.errorbar(split_indices, split_mean_acc, yerr=split_mean_std, label='Without SFT (HSC)', 
            color='#1f77b4', marker='o', linewidth=2, markersize=7, capsize=4, elinewidth=1.2)
ax.errorbar(split_indices, split_sft_acc, yerr=split_sft_std, label='With SFT (HSC)', 
            color='#ff7f0e', marker='s', linewidth=2, markersize=7, capsize=4, elinewidth=1.2)
ax.axvline(x=15, linestyle=':', color='red', alpha=0.8, linewidth=1.5, label='Optimal Split (Block 4)')

ax.set_xlabel('Split Index (M)')
ax.set_ylabel('Mean Test Accuracy (%)')
ax.set_title('Split Index Sweep (Localization Illusion)')
ax.set_xticks(split_indices)
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig('figures/split_sweep.pdf', dpi=300)
plt.close()


# --- PLOT 3: Alpha Shrinkage Sweep (at N=128, Split=15) ---
alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
alpha_mean_acc = [32.46, 35.42, 33.52, 31.80, 31.46]
alpha_mean_std = [1.56, 1.30, 1.17, 0.60, 0.20]

alpha_sft_acc = [51.85, 54.95, 52.70, 49.27, 43.17]
alpha_sft_std = [1.99, 1.22, 0.99, 0.46, 1.59]

fig, ax = plt.subplots(figsize=(5.5, 3.0))
ax.errorbar(alphas, alpha_mean_acc, yerr=alpha_mean_std, label='Without SFT (HSC)', 
            color='#1f77b4', marker='o', linewidth=2, markersize=7, capsize=4, elinewidth=1.2)
ax.errorbar(alphas, alpha_sft_acc, yerr=alpha_sft_std, label='With SFT (HSC)', 
            color='#ff7f0e', marker='s', linewidth=2, markersize=7, capsize=4, elinewidth=1.2)
ax.axvline(x=0.25, linestyle=':', color='red', alpha=0.8, linewidth=1.5, label='Optimal Alpha (0.25)')

ax.set_xlabel('Shrinkage Parameter (Alpha)')
ax.set_ylabel('Mean Test Accuracy (%)')
ax.set_title('Shrinkage Interpolation (Regularization Dynamics)')
ax.set_xticks(alphas)
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig('figures/alpha_sweep.pdf', dpi=300)
plt.close()

print("All multi-seed figures successfully generated with error bars in figures/ folder!")
