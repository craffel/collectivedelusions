import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'text.usetex': False  # False to avoid needing system LaTeX during script execution
})

# Default fallback data (prior results)
regimes = ['FP32', 'INT8-Channel', 'INT4-Channel']
methods_display = ['DE-BN-16', 'DE-BN-32', 'DE-BN-64', 'RMS-BC-32x8 (Ours)', 'DEM-BC-32x8-b0.1 (Ours)']

means = {
    'DE-BN-16': [36.37, 36.36, 35.88],
    'DE-BN-32': [39.97, 40.05, 38.68],
    'DE-BN-64': [42.89, 42.89, 43.44],
    'RMS-BC-32x8 (Ours)': [43.66, 43.51, 44.20],
    'DEM-BC-32x8-b0.1 (Ours)': [49.40, 49.53, 49.96]
}

stds = {
    'DE-BN-16': [1.46, 1.44, 1.95],
    'DE-BN-32': [3.20, 3.17, 3.19],
    'DE-BN-64': [0.56, 0.48, 0.86],
    'RMS-BC-32x8 (Ours)': [0.38, 0.39, 0.07],
    'DEM-BC-32x8-b0.1 (Ours)': [0.21, 0.27, 0.03]
}

# Try loading dynamically from results.json
results_path = "results/results.json"
if os.path.exists(results_path):
    print(f"Loading dynamic results from {results_path}...")
    try:
        with open(results_path, "r") as f:
            res = json.load(f)
            
        # Extract sweeps for TA
        ta_data = {}
        for entry in res.get("sweeps", []):
            if entry.get("merging") == "TA":
                # Key: (precision, calibration) -> (mean, std)
                ta_data[(entry["precision"], entry["calibration"])] = (entry["mean_acc"] * 100, entry["std_acc"] * 100)
                
        # Fill means and stds dynamically if available
        new_means = {}
        new_stds = {}
        success = True
        
        # Mapping from display method to json calibration key
        method_map = {
            'DE-BN-16': 'DE-BN-16',
            'DE-BN-32': 'DE-BN-32',
            'DE-BN-64': 'DE-BN-64',
            'RMS-BC-32x8 (Ours)': 'RMS-BC-32x8',
            'DEM-BC-32x8-b0.1 (Ours)': 'DEM-BC-32x8-b0.1'
        }
        
        for disp_name, json_name in method_map.items():
            new_means[disp_name] = []
            new_stds[disp_name] = []
            for r in regimes:
                key = (r, json_name)
                if key in ta_data:
                    m_val, s_val = ta_data[key]
                    new_means[disp_name].append(m_val)
                    new_stds[disp_name].append(s_val)
                else:
                    success = False
                    print(f"Warning: Key {key} not found in results.json sweeps!")
                    break
                    
        if success:
            means = new_means
            stds = new_stds
            print("Successfully updated plot data from results.json!")
        else:
            print("Failed to completely update plot data from results.json. Using fallback.")
    except Exception as e:
        print(f"Error loading results.json: {e}. Using fallback.")

# Color palette (soft and professional)
colors = ['#aec7e8', '#ffbb78', '#98df8a', '#1f77b4', '#d62728']

# 1. Bar Chart: Accuracy and Variance Comparison
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(regimes))
width = 0.15

for i, method in enumerate(methods_display):
    offset = (i - len(methods_display)/2 + 0.5) * width
    ax.bar(x + offset, means[method], width, yerr=stds[method], 
           label=method, color=colors[i], capsize=4, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Multi-Task Average Accuracy (%)')
ax.set_title('Multi-Task Merging Performance & Stability under Quantization (Task Arithmetic)')
ax.set_xticks(x)
ax.set_xticklabels(regimes)
ax.set_ylim(25, 55)
ax.legend(loc='lower left', frameon=True)
plt.tight_layout()
plt.savefig('accuracy_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig('accuracy_comparison.png', format='png', bbox_inches='tight')
plt.close()

# 2. Line Chart: Standard Deviation / Uncertainty reduction
fig, ax = plt.subplots(figsize=(6, 4))
# Focus on INT4 precision where variance is most critical
int4_stds = [stds[m][-1] for m in methods_display]
x_labels = ['DE-BN-16', 'DE-BN-32', 'DE-BN-64', 'RMS-BC-32x8\n(Ours)', 'DEM-BC-32x8\n(Ours)']

ax.plot(x_labels, int4_stds, marker='o', color='#2ca02c', linewidth=2, markersize=8, label='Std. Dev. (%)')
ax.fill_between(x_labels, int4_stds, color='#2ca02c', alpha=0.1)
ax.set_ylabel('Standard Deviation across Seeds (%)')
ax.set_title('Variance Suppression under INT4-Channel Quantization')
ax.set_ylim(0, 4)
plt.tight_layout()
plt.savefig('variance_reduction.pdf', format='pdf', bbox_inches='tight')
plt.savefig('variance_reduction.png', format='png', bbox_inches='tight')
plt.close()

print("Plots successfully generated!")
