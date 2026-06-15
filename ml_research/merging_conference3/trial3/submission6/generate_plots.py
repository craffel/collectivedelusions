import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_width, fig_height = 6, 4

# 1. Plot ACM Lambdas (Layer Coefficients)
layers = np.arange(14)
mnist_lambdas = [0.0]*10 + [0.106, 0.196, 0.143, 0.170]
fmnist_lambdas = [0.0]*10 + [0.321, 0.151, 0.434, 2.167]
cifar_lambdas = [0.0]*10 + [0.595, 0.596, 0.797, 3.409]
svhn_lambdas = [0.0]*10 + [0.290, 0.352, 0.294, -1.494]

plt.figure(figsize=(7, 4.5))
plt.plot(layers, mnist_lambdas, marker='o', linewidth=2, color='#1f77b4', label='MNIST')
plt.plot(layers, fmnist_lambdas, marker='s', linewidth=2, color='#ff7f0e', label='FashionMNIST')
plt.plot(layers, cifar_lambdas, marker='^', linewidth=2, color='#2ca02c', label='CIFAR-10')
plt.plot(layers, svhn_lambdas, marker='d', linewidth=2, color='#d62728', label='SVHN')

plt.title('ACM Solved Layer-Wise Merging Coefficients ($\\Lambda^{l, *}$)', fontsize=12, fontweight='bold')
plt.xlabel('Layer Index $l$ (ViT-Tiny backbone)', fontsize=11)
plt.ylabel('Coefficient Value $\\Lambda_k^{l, *}$', fontsize=11)
plt.xticks(layers)
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig('submission/acm_lambdas.png', dpi=300)
plt.close()

# 2. Plot Simulation Comparison (Joint Average)
# Model I: Task Arithmetic=84.44, AdaMerging=83.45, RegCalMerge=86.02, PolyMerge=87.72, ACM=87.46
# Model II: Task Arithmetic=84.44, AdaMerging=79.07, RegCalMerge=82.15, PolyMerge=85.49, ACM=87.18
methods = ['Task Arith.', 'AdaMerging', 'RegCalMerge', 'PolyMerge', 'ACM (Ours)']
model1_accs = [84.44, 83.45, 86.02, 87.72, 87.46]
model2_accs = [84.44, 79.07, 82.15, 85.49, 87.18]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
rects1 = ax.bar(x - width/2, model1_accs, width, label='Model I (Convex Landscape)', color='#9ecae1', edgecolor='black', linewidth=0.7)
rects2 = ax.bar(x + width/2, model2_accs, width, label='Model II (Coupled Landscape)', color='#3182bd', edgecolor='black', linewidth=0.7)

ax.set_ylabel('Joint Average Accuracy (%)', fontsize=11)
ax.set_title('Simulation Results: Convex vs. Coupled Landscapes', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylim(70, 92)
ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

# Add value labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('submission/sim_comparison.png', dpi=300)
plt.close()

print("Plots generated successfully!")
