import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from run_experiments_v2 import (
    get_dct_matrix, get_covariance_matrix, get_optimal_profile,
    get_accuracy, generate_noise, idct_iii, optimize_tta_adam, L, K
)

device = torch.device("cpu")
Sigma = get_covariance_matrix(L, device)
Sigma_inv = torch.linalg.inv(Sigma)
lambda_stars = torch.stack([get_optimal_profile(k, L, device) for k in range(K)])
M_dct = get_dct_matrix(L, device)

seeds = list(range(42, 72))  # Exactly matching the 30 seeds protocol

# Sweeps
F_values = [1, 2, 3, 4, 5, 6, 8, 10, 12]
mu_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

lp_means = []
lp_stds = []

reg_means = []
reg_stds = []

print("Running hyperparameter sensitivity sweep for SpectralMerge-LP across 30 seeds...")
for F in F_values:
    run_accs = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        etas = torch.stack([generate_noise(L, device) for _ in range(K)])
        targets = lambda_stars + etas
        
        # SpectralMerge-LP F
        spec_lp_init = torch.zeros(K, F, device=device)
        spec_lp_init[:, 0] = 0.3 * (L ** 0.5)
        
        # Pad with zeros inside the lambda inside the forward loop
        f_spec_lp = lambda p: idct_iii(torch.cat([p, torch.zeros(K, L - F, device=device)], dim=1), M_dct)
        
        final_spec_lp = optimize_tta_adam(spec_lp_init, f_spec_lp, targets, Sigma_inv, steps=100, lr=0.01)
        accs = get_accuracy(final_spec_lp, lambda_stars, Sigma_inv, device)
        run_accs.append(sum(accs) / K)
        
    mean_val = np.mean(run_accs) * 100
    std_val = np.std(run_accs) * 100
    lp_means.append(mean_val)
    lp_stds.append(std_val)
    print(f"F = {F} | Accuracy: {mean_val:.2f}% +- {std_val:.2f}%")

print("\nRunning hyperparameter sensitivity sweep for SpectralMerge-Reg across 30 seeds...")
for mu in mu_values:
    run_accs = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        etas = torch.stack([generate_noise(L, device) for _ in range(K)])
        targets = lambda_stars + etas
        
        # SpectralMerge-Reg mu
        spec_reg_init = torch.zeros(K, L, device=device)
        spec_reg_init[:, 0] = 0.3 * (L ** 0.5)
        f_spec_reg = lambda p: idct_iii(p, M_dct)
        
        j_sq = torch.arange(L, dtype=torch.float32, device=device) ** 2
        reg_fn = lambda p: torch.sum(mu * j_sq * (p ** 2))
        
        final_spec_reg = optimize_tta_adam(spec_reg_init, f_spec_reg, targets, Sigma_inv, steps=100, lr=0.01, reg_fn=reg_fn)
        accs = get_accuracy(final_spec_reg, lambda_stars, Sigma_inv, device)
        run_accs.append(sum(accs) / K)
        
    mean_val = np.mean(run_accs) * 100
    std_val = np.std(run_accs) * 100
    reg_means.append(mean_val)
    reg_stds.append(std_val)
    print(f"mu = {mu} | Accuracy: {mean_val:.2f}% +- {std_val:.2f}%")

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Plot LP
lp_means = np.array(lp_means)
lp_stds = np.array(lp_stds)
ax1.plot(F_values, lp_means, marker='o', color='#1f77b4', linewidth=2, label='SpectralMerge-LP')
ax1.fill_between(F_values, lp_means - lp_stds, lp_means + lp_stds, color='#1f77b4', alpha=0.15)
ax1.set_xlabel("Low-Pass Cutoff Frequency ($F$)", fontsize=11)
ax1.set_ylabel("Multi-Task Average Accuracy (%)", fontsize=11)
ax1.set_title("SpectralMerge-LP Sensitivity", fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xticks(F_values)
ax1.set_ylim(84.0, 86.0)

# Highlight F=3 as optimal
ax1.axvline(x=3, color='crimson', linestyle=':', linewidth=1.5)
ax1.text(3.2, 84.2, 'Optimal F=3', color='crimson', fontsize=10, fontweight='bold')

# Plot Reg
reg_means = np.array(reg_means)
reg_stds = np.array(reg_stds)
ax2.plot(mu_values, reg_means, marker='s', color='#2ca02c', linewidth=2, label='SpectralMerge-Reg')
ax2.fill_between(mu_values, reg_means - reg_stds, reg_means + reg_stds, color='#2ca02c', alpha=0.15)
ax2.set_xscale('log')
ax2.set_xlabel("Regularization Strength ($\mu$)", fontsize=11)
ax2.set_ylabel("Multi-Task Average Accuracy (%)", fontsize=11)
ax2.set_title("SpectralMerge-Reg Sensitivity", fontsize=12, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_ylim(84.0, 86.0)

# Highlight mu=1.0 as optimal
ax2.axvline(x=1.0, color='crimson', linestyle=':', linewidth=1.5)
ax2.text(1.3, 84.2, 'Optimal $\mu$=1.0', color='crimson', fontsize=10, fontweight='bold')

plt.tight_layout()

# Save paths
os.makedirs("submission", exist_ok=True)
plot_path = "submission/hyperparameter_sensitivity.png"
plt.savefig(plot_path, dpi=300)
plt.savefig("submission/hyperparameter_sensitivity.pdf", dpi=300)
print(f"\nHyperparameter sensitivity plots successfully saved to {plot_path}")
