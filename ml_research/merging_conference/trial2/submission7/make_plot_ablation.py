import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("results.json", "r") as f:
    data = json.load(f)

ab_std_beta = data.get("ablation_standard", {})
ab_sam_beta = data.get("ablation_sam", {})
ab_std_rho = data.get("ablation_rho_standard", {})
ab_sam_rho = data.get("ablation_rho_sam", {})

betas = [0.001, 0.01, 0.1, 0.5, 1.0]
rhos = [0.01, 0.05, 0.1, 0.2]

# Extract beta accuracies
std_beta_clean = [ab_std_beta[str(b)]["none"]["avg"] for b in betas]
std_beta_blur = [ab_std_beta[str(b)]["blur"]["avg"] for b in betas]
sam_beta_clean = [ab_sam_beta[str(b)]["none"]["avg"] for b in betas]
sam_beta_blur = [ab_sam_beta[str(b)]["blur"]["avg"] for b in betas]

# Extract rho accuracies
std_rho_clean = [ab_std_rho[str(r)]["none"]["avg"] for r in rhos]
std_rho_blur = [ab_std_rho[str(r)]["blur"]["avg"] for r in rhos]
sam_rho_clean = [ab_sam_rho[str(r)]["none"]["avg"] for r in rhos]
sam_rho_blur = [ab_sam_rho[str(r)]["blur"]["avg"] for r in rhos]

# Set standard plotting style for professional look
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(14, 11))

# Subplot 1: Standard Experts - Beta Ablation
axs[0, 0].plot(betas, std_beta_clean, marker='o', linewidth=2.5, color='#3182CE', label='Clean (NONE)', markersize=8)
axs[0, 0].plot(betas, std_beta_blur, marker='s', linewidth=2.5, color='#319795', label='Gaussian Blur', markersize=8)
axs[0, 0].set_xscale('log')
axs[0, 0].set_xlabel(r'Regularization Strength $\beta$ (log scale)', fontsize=12, fontweight='semibold')
axs[0, 0].set_ylabel('Multi-Task Accuracy (%)', fontsize=12, fontweight='semibold')
axs[0, 0].set_title('Standard Experts - Beta Ablation', fontsize=13, fontweight='bold', pad=10)
axs[0, 0].set_xticks(betas)
axs[0, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axs[0, 0].grid(True, which="both", linestyle='--', alpha=0.6)
axs[0, 0].legend(fontsize=11)

# Subplot 2: SAM Experts - Beta Ablation
axs[0, 1].plot(betas, sam_beta_clean, marker='o', linewidth=2.5, color='#E53E3E', label='Clean (NONE)', markersize=8)
axs[0, 1].plot(betas, sam_beta_blur, marker='s', linewidth=2.5, color='#D69E2E', label='Gaussian Blur', markersize=8)
axs[0, 1].set_xscale('log')
axs[0, 1].set_xlabel(r'Regularization Strength $\beta$ (log scale)', fontsize=12, fontweight='semibold')
axs[0, 1].set_ylabel('Multi-Task Accuracy (%)', fontsize=12, fontweight='semibold')
axs[0, 1].set_title('SAM-Trained Experts - Beta Ablation', fontsize=13, fontweight='bold', pad=10)
axs[0, 1].set_xticks(betas)
axs[0, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axs[0, 1].grid(True, which="both", linestyle='--', alpha=0.6)
axs[0, 1].legend(fontsize=11)

# Subplot 3: Standard Experts - Rho Ablation
axs[1, 0].plot(rhos, std_rho_clean, marker='o', linewidth=2.5, color='#3182CE', label='Clean (NONE)', markersize=8)
axs[1, 0].plot(rhos, std_rho_blur, marker='s', linewidth=2.5, color='#319795', label='Gaussian Blur', markersize=8)
axs[1, 0].set_xscale('log')
axs[1, 0].set_xlabel(r'Perturbation Radius $\rho$ (log scale)', fontsize=12, fontweight='semibold')
axs[1, 0].set_ylabel('Multi-Task Accuracy (%)', fontsize=12, fontweight='semibold')
axs[1, 0].set_title('Standard Experts - Rho Ablation', fontsize=13, fontweight='bold', pad=10)
axs[1, 0].set_xticks(rhos)
axs[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axs[1, 0].grid(True, which="both", linestyle='--', alpha=0.6)
axs[1, 0].legend(fontsize=11)

# Subplot 4: SAM Experts - Rho Ablation
axs[1, 1].plot(rhos, sam_rho_clean, marker='o', linewidth=2.5, color='#E53E3E', label='Clean (NONE)', markersize=8)
axs[1, 1].plot(rhos, sam_rho_blur, marker='s', linewidth=2.5, color='#D69E2E', label='Gaussian Blur', markersize=8)
axs[1, 1].set_xscale('log')
axs[1, 1].set_xlabel(r'Perturbation Radius $\rho$ (log scale)', fontsize=12, fontweight='semibold')
axs[1, 1].set_ylabel('Multi-Task Accuracy (%)', fontsize=12, fontweight='semibold')
axs[1, 1].set_title('SAM-Trained Experts - Rho Ablation', fontsize=13, fontweight='bold', pad=10)
axs[1, 1].set_xticks(rhos)
axs[1, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axs[1, 1].grid(True, which="both", linestyle='--', alpha=0.6)
axs[1, 1].legend(fontsize=11)

plt.tight_layout()
plt.savefig("ablation_plot.png", dpi=300, bbox_inches='tight')
print("2x2 ablation plots generated successfully and saved to ablation_plot.png!")
