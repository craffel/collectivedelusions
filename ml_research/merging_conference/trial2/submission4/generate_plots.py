import json
import matplotlib.pyplot as plt
import os

# Set style for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# 1. Plot TTA Learning Rate Sweep
with open('sweep_results_lr.json', 'r') as f:
    lr_data = json.load(f)

lrs = [x['lr'] for x in lr_data]
cifar10_lr = [x['cifar10'] for x in lr_data]
svhn_lr = [x['svhn'] for x in lr_data]
avg_lr = [x['avg'] for x in lr_data]

plt.figure(figsize=(5, 3.8))
plt.plot(lrs, avg_lr, marker='o', linewidth=2.5, color='#1f77b4', label='Multi-Task Avg')
plt.plot(lrs, cifar10_lr, marker='s', linestyle='--', linewidth=1.5, color='#2ca02c', label='CIFAR-10')
plt.plot(lrs, svhn_lr, marker='^', linestyle='--', linewidth=1.5, color='#ff7f0e', label='SVHN')
plt.xscale('log')
plt.xlabel('TTA Learning Rate ($lr$)', fontsize=11)
plt.ylabel('Test Accuracy (%)', fontsize=11)
plt.title('Adaptation Learning Rate Sensitivity', fontsize=12, fontweight='bold', pad=10)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig('plot_lr.pdf', bbox_inches='tight', dpi=300)
plt.close()

# 2. Plot SAM Radius Sweep
with open('sweep_results_rho.json', 'r') as f:
    rho_data = json.load(f)

rhos = [x['rho'] for x in rho_data]
cifar10_rho = [x['cifar10'] for x in rho_data]
svhn_rho = [x['svhn'] for x in rho_data]
avg_rho = [x['avg'] for x in rho_data]

plt.figure(figsize=(5, 3.8))
plt.plot(rhos, avg_rho, marker='o', linewidth=2.5, color='#1f77b4', label='Multi-Task Avg')
plt.plot(rhos, cifar10_rho, marker='s', linestyle='--', linewidth=1.5, color='#2ca02c', label='CIFAR-10')
plt.plot(rhos, svhn_rho, marker='^', linestyle='--', linewidth=1.5, color='#ff7f0e', label='SVHN')
plt.xlabel('SAM Perturbation Radius ($\\rho$)', fontsize=11)
plt.ylabel('Test Accuracy (%)', fontsize=11)
plt.title('SAM Radius Sensitivity', fontsize=12, fontweight='bold', pad=10)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig('plot_rho.pdf', bbox_inches='tight', dpi=300)
plt.close()

# 3. Plot SOSR Weight Sweep
with open('sweep_results_beta.json', 'r') as f:
    beta_data = json.load(f)

betas = [x['beta'] for x in beta_data]
cifar10_beta = [x['cifar10'] for x in beta_data]
svhn_beta = [x['svhn'] for x in beta_data]
avg_beta = [x['avg'] for x in beta_data]

plt.figure(figsize=(5, 3.8))
plt.plot(betas, avg_beta, marker='o', linewidth=2.5, color='#1f77b4', label='Multi-Task Avg')
plt.plot(betas, cifar10_beta, marker='s', linestyle='--', linewidth=1.5, color='#2ca02c', label='CIFAR-10')
plt.plot(betas, svhn_beta, marker='^', linestyle='--', linewidth=1.5, color='#ff7f0e', label='SVHN')
plt.xlabel('SOSR Weight ($\\beta$)', fontsize=11)
plt.ylabel('Test Accuracy (%)', fontsize=11)
plt.title('SOSR Penalty Sensitivity', fontsize=12, fontweight='bold', pad=10)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig('plot_beta.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("Plots successfully generated and saved!")
