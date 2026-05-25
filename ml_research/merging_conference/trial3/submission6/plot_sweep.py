import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load original results
with open("results.json", "r") as f:
    results_orig = json.load(f)

# Initialize sweep data
# concept: SAM is gamma=0.0
sweep_gamma = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
sweep_merged_acc = []
sweep_procrustes = []
sweep_expert_acc = []

# Populate SAM as gamma=0.0
sweep_merged_acc.append(results_orig['sam']['merged_acc'])
sweep_procrustes.append(results_orig['sam']['avg_procrustes_norm'])
sweep_expert_acc.append(results_orig['sam']['avg_expert_acc'])

# Load gammas
gammas_to_load = [0.1, 0.25, 0.5, 1.5, 2.0]
loaded_results = {}

# We also have gamma=1.0 from original results
loaded_results[1.0] = results_orig['fsam_dir']

for g in gammas_to_load:
    filename = f"results_dir_gamma_{g}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            loaded_results[g] = data['fsam_dir']
    else:
        print(f"Warning: {filename} not found!")

# Sort and append to list
for g in sorted(sweep_gamma[1:]):
    if g in loaded_results:
        sweep_merged_acc.append(loaded_results[g]['merged_acc'])
        sweep_procrustes.append(loaded_results[g]['avg_procrustes_norm'])
        sweep_expert_acc.append(loaded_results[g]['avg_expert_acc'])
    else:
        # Placeholder in case some job isn't finished
        sweep_merged_acc.append(None)
        sweep_procrustes.append(None)
        sweep_expert_acc.append(None)

print("Sweep Gammas:", sweep_gamma)
print("Sweep Merged Accuracies:", sweep_merged_acc)
print("Sweep Procrustes Norms:", sweep_procrustes)

# Filter out None values just in case
valid_points = [(g, acc, proc, exp) for g, acc, proc, exp in zip(sweep_gamma, sweep_merged_acc, sweep_procrustes, sweep_expert_acc) if acc is not None]
if len(valid_points) > 0:
    g_plot, acc_plot, proc_plot, exp_plot = zip(*valid_points)
else:
    g_plot, acc_plot, proc_plot, exp_plot = [], [], [], []

# Generate publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Merged Accuracy vs Gamma
ax1.plot(g_plot, acc_plot, marker='s', linewidth=2.5, color='#C0504D', label='Merged Model Acc', markersize=8)
ax1.plot(g_plot, exp_plot, marker='^', linewidth=2.0, color='#4F81BD', linestyle='--', label='Avg Expert Acc', markersize=8)
ax1.axhline(y=results_orig['sgd']['merged_acc'], color='gray', linestyle=':', label='SGD Merged (90.81%)', linewidth=1.5)
ax1.set_xlabel('Fisher Weighting Strength (\\gamma)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Merged Model & Expert Performance vs. \\gamma')
ax1.set_xticks(sweep_gamma)
ax1.legend(frameon=True)
ax1.set_ylim(89.5, 99.0)

# Add value labels
for i, val in enumerate(acc_plot):
    ax1.annotate(f'{val:.2f}%', (g_plot[i], val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='#C0504D', fontweight='bold')

# Plot 2: Procrustes Residual Norm vs Gamma
ax2.plot(g_plot, proc_plot, marker='o', linewidth=2.5, color='#8064A2', markersize=8)
ax2.axhline(y=results_orig['sgd']['avg_procrustes_norm'], color='gray', linestyle=':', label='SGD Norm (0.60824)', linewidth=1.5)
ax2.set_xlabel('Fisher Weighting Strength (\\gamma)')
ax2.set_ylabel('Average Procrustes Residual Norm')
ax2.set_title('Weight Space Geometric Distortion vs. \\gamma')
ax2.set_xticks(sweep_gamma)
ax2.legend(frameon=True)

# Add value labels
for i, val in enumerate(proc_plot):
    ax2.annotate(f'{val:.5f}', (g_plot[i], val), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#8064A2', fontweight='bold')

plt.tight_layout()
plt.savefig('results_plot.png', dpi=300)
plt.savefig('results_plot.pdf', dpi=300)
print("Sweep plots generated and saved as results_plot.png/pdf!")

# Print markdown summary table
print("\n### Sweep Summary Table")
print("| Mode | Gamma (\\gamma) | Expert A Acc (%) | Expert B Acc (%) | Avg Expert Acc (%) | Avg Procrustes Norm | Merged Full Acc (%) |")
print("| --- | --- | --- | --- | --- | --- | --- |")
print(f"| SGD | - | {results_orig['sgd']['expert_A_acc']:.2f}% | {results_orig['sgd']['expert_B_acc']:.2f}% | {results_orig['sgd']['avg_expert_acc']:.2f}% | {results_orig['sgd']['avg_procrustes_norm']:.5f} | {results_orig['sgd']['merged_acc']:.2f}% |")
print(f"| SAM (concept. g=0) | 0.0 | {results_orig['sam']['expert_A_acc']:.2f}% | {results_orig['sam']['expert_B_acc']:.2f}% | {results_orig['sam']['avg_expert_acc']:.2f}% | {results_orig['sam']['avg_procrustes_norm']:.5f} | {results_orig['sam']['merged_acc']:.2f}% |")

for g in sorted(loaded_results.keys()):
    res = loaded_results[g]
    print(f"| Direct F-SAM | {g:.2f} | {res['expert_A_acc']:.2f}% | {res['expert_B_acc']:.2f}% | {res['avg_expert_acc']:.2f}% | {res['avg_procrustes_norm']:.5f} | {res['merged_acc']:.2f}% |")
