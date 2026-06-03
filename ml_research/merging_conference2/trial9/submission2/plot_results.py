import json
import matplotlib.pyplot as plt
import numpy as np

# Load experiment results
with open('experiment_results_fixed.json') as f:
    results = json.load(f)

# Unique methods, bits, environments, and bn_samples
methods = ['WA', 'Tuned TA', 'Tuned TA + DE-QC', 'S-IPR', 'WCPR', 'QR-IPR']
colors = {
    'WA': '#1f77b4',
    'Tuned TA': '#ff7f0e',
    'Tuned TA + DE-QC': '#2ca02c',
    'S-IPR': '#d62728',
    'WCPR': '#9467bd',
    'QR-IPR': '#8c564b'
}
markers = {
    'WA': 'o',
    'Tuned TA': 's',
    'Tuned TA + DE-QC': '^',
    'S-IPR': 'd',
    'WCPR': 'v',
    'QR-IPR': 'x'
}

# We want a multi-panel figure for Clean Environment: FP32, INT8, INT4
# X-axis: BN-Cal Samples (0, 16, 32)
# Y-axis: Multitask Average Accuracy (%)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
bits_list = [32, 8, 4]
titles = {32: 'FP32 (32-bit)', 8: 'INT8 (8-bit)', 4: 'INT4 (4-bit)'}

bn_samples_x = [0, 16, 32]

for i, bits in enumerate(bits_list):
    ax = axes[i]
    ax.set_title(titles[bits], fontsize=14, fontweight='bold')
    
    for method in methods:
        y_vals = []
        for bn in bn_samples_x:
            # Find the entry
            matched = [r for r in results if r['method'] == method and r['bits'] == bits and r['bn_samples'] == bn and r['env'] == 'Clean']
            if matched:
                y_vals.append(matched[0]['average'])
            else:
                y_vals.append(np.nan)
        
        ax.plot(bn_samples_x, y_vals, label=method, color=colors[method], marker=markers[method], linewidth=2, markersize=8)
    
    ax.set_xlabel('BN Calibration Samples', fontsize=12)
    if i == 0:
        ax.set_ylabel('Multitask Average Accuracy (%)', fontsize=12)
    
    ax.set_xticks(bn_samples_x)
    ax.set_xticklabels(['0 (None)', '16', '32'])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(20, 85)

# Add single legend for the entire figure at the top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig('results_plot.pdf', bbox_inches='tight')
print("Successfully generated results_plot.pdf!")
