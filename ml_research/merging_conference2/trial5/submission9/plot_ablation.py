import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'text.usetex': False
})

layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
x = np.arange(len(layers))

# Data for WA
wa_mnist = [93.78, 91.88, 84.98, 75.57]
wa_fmnist = [86.68, 84.88, 87.44, 84.15]
wa_cifar = [96.82, 98.66, 98.51, 86.41]
wa_avg = [49.54, 48.82, 47.16, 42.85]

# Data for TA
ta_mnist = [93.61, 92.09, 84.21, 73.97]
ta_fmnist = [85.32, 85.80, 88.18, 84.46]
ta_cifar = [96.53, 98.55, 98.26, 88.38]
ta_avg = [46.51, 46.28, 44.26, 40.07]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

# Plot 1: Routing Accuracies (WA vs TA for each dataset)
ax1.plot(x, wa_mnist, marker='o', linestyle='-', color='#1f77b4', label='MNIST (WA)')
ax1.plot(x, ta_mnist, marker='o', linestyle='--', color='#1f77b4', alpha=0.7, label='MNIST (TA)')

ax1.plot(x, wa_fmnist, marker='s', linestyle='-', color='#ff7f0e', label='F-MNIST (WA)')
ax1.plot(x, ta_fmnist, marker='s', linestyle='--', color='#ff7f0e', alpha=0.7, label='F-MNIST (TA)')

ax1.plot(x, wa_cifar, marker='^', linestyle='-', color='#2ca02c', label='CIFAR-10 (WA)')
ax1.plot(x, ta_cifar, marker='^', linestyle='--', color='#2ca02c', alpha=0.7, label='CIFAR-10 (TA)')

ax1.set_title('Routing Accuracy vs. Routing Layer')
ax1.set_xlabel('Routing Layer')
ax1.set_ylabel('Routing Accuracy (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(layers)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower left', frameon=True)

# Plot 2: Downstream Multi-Task Average Accuracy
ax2.plot(x, wa_avg, marker='D', linestyle='-', color='#d62728', linewidth=2, label='Weight Averaging (WA)')
ax2.plot(x, ta_avg, marker='D', linestyle='-', color='#9467bd', linewidth=2, label='Task Arithmetic (TA)')

ax2.set_title('Downstream Multi-Task Avg Accuracy vs. Layer')
ax2.set_xlabel('Routing Layer')
ax2.set_ylabel('Average Accuracy (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(layers)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig('routing_ablation.pdf', bbox_inches='tight')
plt.savefig('routing_ablation.png', dpi=300, bbox_inches='tight')
print("Successfully saved routing_ablation.pdf and routing_ablation.png")
