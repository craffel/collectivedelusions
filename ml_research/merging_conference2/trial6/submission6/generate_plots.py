import matplotlib.pyplot as plt
import numpy as np
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'grid.alpha': 0.3
})

# Load the results
with open('experiment_results.json', 'r') as f:
    results = json.load(f)

scenarios = ['A_low_reg', 'B_std_reg', 'C_high_reg', 'D_l2sp_reg']
labels = ['Low Reg\n(WD=0)', 'Std Reg\n(WD=1e-4)', 'High Reg\n(WD=1e-2)', 'L2-SP Reg\n(lambda=1e-3)']

# Plot 1: Cosine Similarities of Weight Updates
plt.figure(figsize=(7, 4.5))
cos_sims_mnist_fmnist = [results[sc]['drift']['cos_sim_mnist_fmnist'] for sc in scenarios]
cos_sims_mnist_cifar = [results[sc]['drift']['cos_sim_mnist_cifar'] for sc in scenarios]
cos_sims_fmnist_cifar = [results[sc]['drift']['cos_sim_fmnist_cifar'] for sc in scenarios]

x = np.arange(len(scenarios))
width = 0.25

plt.bar(x - width, cos_sims_mnist_fmnist, width, label='MNIST vs F-MNIST', color='#4C72B0')
plt.bar(x, cos_sims_mnist_cifar, width, label='MNIST vs CIFAR-10', color='#55A868')
plt.bar(x + width, cos_sims_fmnist_cifar, width, label='F-MNIST vs CIFAR-10', color='#C44E52')

plt.ylabel('Cosine Similarity of Weight Updates')
plt.title('Expert Weight Update Alignment across Regularization Levels')
plt.xticks(x, labels)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig('weight_alignment.png', dpi=300)
plt.close()

# Plot 2: Merged Model Accuracies across Calibration Methods
plt.figure(figsize=(7.5, 5))
none_accs = [results[sc]['none']['average'] for sc in scenarios]
sptaac_accs = [results[sc]['sp-taac']['average'] for sc in scenarios]
hybrid_accs = [results[sc]['hybrid']['average'] for sc in scenarios]

plt.bar(x - width, none_accs, width, label='Uncalibrated (none)', color='#8C8C8C')
plt.bar(x, sptaac_accs, width, label='SP-TAAC Calibration', color='#DD8452')
plt.bar(x + width, hybrid_accs, width, label='Hybrid (SP-TAAC + SLR-WBC)', color='#4C72B0')

plt.ylabel('Average Merged Test Accuracy (%)')
plt.title('Multi-Task Merged Accuracy across Calibration Methods')
plt.xticks(x, labels)
plt.ylim(0, 55)
# Annotate individual bars
for i in range(len(scenarios)):
    plt.text(i - width, none_accs[i] + 1, f"{none_accs[i]:.1f}%", ha='center', fontsize=9)
    plt.text(i, sptaac_accs[i] + 1, f"{sptaac_accs[i]:.1f}%", ha='center', fontsize=9)
    plt.text(i + width, hybrid_accs[i] + 1, f"{hybrid_accs[i]:.1f}%", ha='center', fontsize=9)

plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left')
plt.tight_layout()
plt.savefig('merged_accuracies.png', dpi=300)
plt.close()

print("Plots successfully generated and saved as weight_alignment.png and merged_accuracies.png!")
