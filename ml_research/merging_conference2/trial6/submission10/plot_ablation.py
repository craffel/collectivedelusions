import matplotlib.pyplot as plt
import numpy as np

# Data
epochs = [10, 25, 50, 100, 150]
mnist_acc = [34.85, 71.12, 88.85, 95.56, 96.54]
fmnist_acc = [21.37, 59.08, 66.16, 72.35, 75.69]
cifar_acc = [41.78, 57.40, 63.45, 64.85, 65.14]
avg_acc = [32.67, 62.53, 72.82, 77.59, 79.12]

# Baselines (Horizontal Lines)
oracle = 83.15
real_joint = 70.15
ood_proxy = 58.25
pink_noise = 21.48
no_calib = 9.93

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.6
})

fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

# Plot curves for individual tasks and average
ax.plot(epochs, mnist_acc, marker='o', linestyle=':', color='#3498db', alpha=0.8, label='MNIST')
ax.plot(epochs, fmnist_acc, marker='s', linestyle=':', color='#9b59b6', alpha=0.8, label='Fashion-MNIST')
ax.plot(epochs, cifar_acc, marker='^', linestyle=':', color='#e67e22', alpha=0.8, label='CIFAR-10')
ax.plot(epochs, avg_acc, marker='D', linestyle='-', color='#2c3e50', linewidth=2.5, label='DF-Calib-Gen (Average)')

# Draw baselines
ax.axhline(y=oracle, color='#27ae60', linestyle='--', linewidth=1.5, label='Real Task-Specific Oracle (83.15%)')
ax.axhline(y=real_joint, color='#e74c3c', linestyle='-.', linewidth=1.5, label='Real Joint Multi-Task (70.15%)')
ax.axhline(y=ood_proxy, color='#f1c40f', linestyle='--', linewidth=1.0, label='OOD Proxy Calib (58.25%)')
ax.axhline(y=pink_noise, color='#7f8c8d', linestyle=':', linewidth=1.0, label='Pink-Noise Calib (21.48%)')

ax.set_xlabel('Generative Synthesis Epochs', fontweight='bold')
ax.set_ylabel('Multi-Task Accuracy (%)', fontweight='bold')
ax.set_title('Accuracy vs. Synthesis Epochs (TA $\lambda=0.5$)', fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(5, 95)
ax.set_xlim(5, 155)
ax.set_xticks(epochs)

# Legend positioning (two-column layout)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, shadow=False)

plt.tight_layout()
plt.savefig('epochs_ablation.pdf', bbox_inches='tight')
plt.savefig('epochs_ablation.png', bbox_inches='tight')
print("Plots saved successfully!")
