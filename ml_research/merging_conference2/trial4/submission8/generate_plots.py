import matplotlib.pyplot as plt
import numpy as np
import json

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'text.usetex': False  # Avoid requiring full latex install for matplotlib
})

# ----------------- PLOT 1: Calibration Performance vs Budget -----------------
# Data
budgets = np.array([16, 64, 256])

# Accuracies (%)
wa_l2 = np.array([26.44, 44.00, 63.98])
wa_pr = np.array([40.71, 47.55, 65.02])
wfc_l2 = np.array([25.54, 41.00, 59.21])
wfc_pr = np.array([42.44, 50.21, 61.27])

# Baselines
wa_baseline = 37.51
experts_upper = 90.23

fig, ax = plt.subplots(figsize=(7, 4.5))

# Plot baselines as horizontal lines
ax.axhline(y=experts_upper, color='black', linestyle='--', linewidth=1.5, label='Individual Experts (Upper Bound)')
ax.axhline(y=wa_baseline, color='gray', linestyle=':', linewidth=1.5, label='Weight Averaging (WA) Baseline')

# Plot experimental lines
ax.plot(budgets, wa_l2, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='WA + L2-LSHA (Standard)')
ax.plot(budgets, wa_pr, marker='s', linestyle='-', color='#aec7e8', linewidth=2, label='WA + PR-LSHA (Prior-Reg)')
ax.plot(budgets, wfc_l2, marker='^', linestyle='-', color='#ff7f0e', linewidth=2, label='WFC + L2-LSHA (Standard)')
ax.plot(budgets, wfc_pr, marker='D', linestyle='-', color='#d62728', linewidth=2.5, label='WFC + PR-LSHA (Ours)')

# Labels and formatting
ax.set_xscale('log', base=2)
ax.set_xticks(budgets)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Calibration Budget $N$ (Log Scale)')
ax.set_ylabel('Multi-Task Average Accuracy (%)')
ax.set_title('Robustness under Calibration Data Scarcity')
ax.set_ylim(20, 95)
ax.set_xlim(12, 300)

# Grid styling
ax.grid(True, which="both", ls="--", color='lightgray')

# Legend and layout
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray')
plt.tight_layout()

# Save plot 1
plt.savefig('template/calibration_performance.pdf', dpi=300, bbox_inches='tight')
plt.savefig('template/calibration_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Successfully generated and saved calibration_performance.pdf!")


# ----------------- PLOT 2: Hyperparameter Sensitivity Sweep -----------------
# Load ablation sweep data
with open('ablation_results.json', 'r') as f:
    ablation_data = json.load(f)

# Sort out data points
regs = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
reg_labels = ['0', '0.1', '1', '10', '100', '1K', '10K']

configs = {
    'WA_L2-LSHA': {'backbone': 'WA', 'head': 'L2-LSHA', 'color': '#1f77b4', 'marker': 'o', 'label': 'WA + L2-LSHA'},
    'WA_PR-LSHA': {'backbone': 'WA', 'head': 'PR-LSHA', 'color': '#aec7e8', 'marker': 's', 'label': 'WA + PR-LSHA'},
    'WFC_L2-LSHA': {'backbone': 'WFC', 'head': 'L2-LSHA', 'color': '#ff7f0e', 'marker': '^', 'label': 'WFC + L2-LSHA'},
    'WFC_PR-LSHA': {'backbone': 'WFC', 'head': 'PR-LSHA', 'color': '#d62728', 'marker': 'D', 'label': 'WFC + PR-LSHA (Ours)'}
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
budgets_list = [16, 64, 256]

for idx, N in enumerate(budgets_list):
    ax = axes[idx]
    
    # Draw WA Baseline
    ax.axhline(y=wa_baseline, color='gray', linestyle=':', linewidth=1.5, label='WA Baseline' if idx == 0 else "")
    
    for conf_key, conf_info in configs.items():
        y_vals = []
        for r in regs:
            # Find the matching record
            match = None
            for item in ablation_data:
                if item['N'] == N and item['backbone'] == conf_info['backbone'] and item['head'] == conf_info['head'] and abs(item['reg'] - r) < 1e-5:
                    match = item
                    break
            if match is not None:
                y_vals.append(match['avg'] * 100.0)
            else:
                y_vals.append(np.nan)
        
        ax.plot(reg_labels, y_vals, marker=conf_info['marker'], linestyle='-', 
                color=conf_info['color'], linewidth=2, label=conf_info['label'] if idx == 0 else "")
    
    ax.set_title(f'Budget $N = {N}$')
    ax.set_xlabel('Regularization $\lambda$')
    ax.grid(True, which="both", ls="--", color='lightgray')
    if idx == 0:
        ax.set_ylabel('Multi-Task Average Accuracy (%)')
        ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='lightgray')

plt.tight_layout()
plt.savefig('template/sensitivity_sweep.pdf', dpi=300, bbox_inches='tight')
plt.savefig('template/sensitivity_sweep.png', dpi=300, bbox_inches='tight')
plt.close()
print("Successfully generated and saved sensitivity_sweep.pdf!")
