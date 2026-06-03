import matplotlib.pyplot as plt
import json
import numpy as np

# Load hyperparameter sweep data
with open("results_head_hyperparams.json", "r") as f:
    data = json.load(f)

seeds = ["42", "43", "44"]
epochs = [5, 10, 20]

# Extract average accuracies across seeds
def get_averages(lr, method):
    means = []
    stds = []
    for ep in epochs:
        config_key = f"lr={lr:.0e}_epochs={ep}"
        vals = [data[seed][config_key][method]["average"] for seed in seeds]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(means), np.array(stds)

# Setup figure
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

methods = {
    "Head-only Adaptation": {"color": "#1f77b4", "marker": "o", "label": "Head-only Adaptation"},
    "Post-BN LSC + Head Adapt": {"color": "#2ca02c", "marker": "s", "label": "Post-BN LSC + Head Adapt"},
    "Post-BN SMACS (tau=0.95) + Head Adapt": {"color": "#ff7f0e", "marker": "^", "label": "Post-BN SMACS (t=0.95) + Head Adapt"},
    "Pre-BN SMACS (tau=0.50) + Head Adapt": {"color": "#d62728", "marker": "d", "label": "Pre-BN SMACS (t=0.50) + Head Adapt"}
}

# Subplot 1: LR = 1e-3
lr1 = 1e-3
for method_name, style in methods.items():
    means, stds = get_averages(lr1, method_name)
    ax1.errorbar(epochs, means, yerr=stds, fmt=style["marker"]+"-", color=style["color"],
                 label=style["label"], capsize=4, elinewidth=1.5, markeredgewidth=1.5)
ax1.set_title("Optimization Scaling (Learning Rate = 1e-3)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Adaptation Epochs", fontsize=10)
ax1.set_ylabel("Average Multi-Task Accuracy (%)", fontsize=10)
ax1.set_xticks(epochs)
ax1.set_ylim(45, 76)
ax1.tick_params(labelsize=9)

# Subplot 2: LR = 1e-4
lr2 = 1e-4
for method_name, style in methods.items():
    means, stds = get_averages(lr2, method_name)
    ax2.errorbar(epochs, means, yerr=stds, fmt=style["marker"]+"-", color=style["color"],
                 label=style["label"], capsize=4, elinewidth=1.5, markeredgewidth=1.5)
ax2.set_title("Optimization Scaling (Learning Rate = 1e-4)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Adaptation Epochs", fontsize=10)
ax2.set_xticks(epochs)
ax2.set_ylim(45, 76)
ax2.tick_params(labelsize=9)

# Legend below
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=True)
plt.subplots_adjust(bottom=0.22, top=0.88, wspace=0.25)

plt.savefig("plot_head_hyperparams.pdf", bbox_inches='tight')
print("Successfully generated plot_head_hyperparams.pdf!")
