import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Load raw statistical results across runs
with open("statistical_results.json", "r") as f:
    results_data = json.load(f)

methods_keys = ['ta_default', 'ta', 'ties_default', 'ties', 'dare_default', 'dare', 'ada', 'svd', 'rms', 'pf_rms', 'pf_cw_rms']
methods_labels = [
    'Task Arithmetic (Default)',
    'Task Arithmetic (Tuned)',
    'Ties-Merging (Default)',
    'Ties-Merging (Tuned)',
    'DARE (Default)',
    'DARE (Tuned)',
    'AdaMerging',
    'SVD Isotropic',
    'RMS-Scale (Ours)',
    'PF-RMS (Ours)',
    'PF-CW-RMS (Ours)'
]

tasks = ['mnist', 'fashion', 'kmnist', 'avg']
task_labels = ['MNIST', 'FashionMNIST', 'KMNIST', 'Average Acc']
colors = ['#4F81BD', '#C0504D', '#9BBB59', '#8064A2']
hatches = ['', '', '', '//']

# Compute means and stds for each method and task
means = {t: [] for t in tasks}
stds = {t: [] for t in tasks}

for m in methods_keys:
    for t in tasks:
        values = results_data[m][t]
        means[t].append(np.mean(values))
        stds[t].append(np.std(values))

x = np.arange(len(methods_labels))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 6))

rects = []
for idx, (t, label, color, hatch) in enumerate(zip(tasks, task_labels, colors, hatches)):
    offset = (idx - 1.5) * width
    m_vals = means[t]
    s_vals = stds[t]
    
    # Plot bars with error bars representing standard deviation
    rect = ax.bar(x + offset, m_vals, width, yerr=s_vals, capsize=3, label=label, color=color, hatch=hatch, error_kw={'elinewidth':1.2, 'ecolor':'#333333'})
    rects.append(rect)

# Customize plot
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Comparison of Model Merging Methods on Multi-Task Classification (3-Seed Statistical Results with Std Dev)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods_labels, rotation=15, fontsize=10)
ax.set_ylim(0, 110)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(loc='lower left', fontsize=10)

# Add values above bars
def autolabel(rects_list):
    for rects in rects_list:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # vertical offset
                        textcoords="offset points",
                        va='bottom',
                        ha='center', fontsize=7, color='#222222')

autolabel(rects)

plt.tight_layout()
plt.savefig("results/fig1.png", dpi=300)
# Also copy to submission/results/fig1.png
os.makedirs("submission/results", exist_ok=True)
plt.savefig("submission/results/fig1.png", dpi=300)

print("Plot successfully regenerated and saved with statistical error bars to 'results/fig1.png' and 'submission/results/fig1.png'!")
