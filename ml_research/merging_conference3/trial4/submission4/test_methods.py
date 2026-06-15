import torch
import numpy as np

# Re-use our code
from run_experiments_v2 import (
    get_dct_matrix, get_covariance_matrix, get_optimal_profile,
    get_accuracy, generate_noise, idct_iii, optimize_tta_adam, L, K
)

device = torch.device("cpu")
Sigma = get_covariance_matrix(L, device)
Sigma_inv = torch.linalg.inv(Sigma)
lambda_stars = torch.stack([get_optimal_profile(k, L, device) for k in range(K)])
M_dct = get_dct_matrix(L, device)

# Vandermonde matrix for Poly-Val (d=2)
l_bar = torch.linspace(0.0, 1.0, L, device=device)
V_d2 = torch.stack([l_bar ** j for j in range(3)], dim=1) # (12, 3)

seeds = list(range(42, 47))

print("Evaluation on 5 seeds...")

acc_uniform = 0.0
acc_unconstrained = 0.0
acc_poly_d2 = 0.0
acc_spec_lp2 = 0.0
acc_spec_reg = 0.0

for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)
    
    etas = torch.stack([generate_noise(L, device) for _ in range(K)])
    targets = lambda_stars + etas
    
    # Uniform
    uniform_lambdas = torch.ones(K, L, device=device) * 0.3
    acc_uniform += sum(get_accuracy(uniform_lambdas, lambda_stars, Sigma_inv, device)) / K
    
    # Unconstrained
    unconstrained_init = torch.ones(K, L, device=device) * 0.3
    f_unconstrained = lambda p: p
    final_unconstrained = optimize_tta_adam(unconstrained_init, f_unconstrained, targets, Sigma_inv, steps=100, lr=0.01)
    acc_unconstrained += sum(get_accuracy(final_unconstrained, lambda_stars, Sigma_inv, device)) / K
    
    # Poly-Val d=2
    poly_init = torch.zeros(K, 3, device=device)
    poly_init[:, 0] = 0.3
    f_poly = lambda p: torch.matmul(p, V_d2.t())
    final_poly = optimize_tta_adam(poly_init, f_poly, targets, Sigma_inv, steps=100, lr=0.01)
    acc_poly_d2 += sum(get_accuracy(final_poly, lambda_stars, Sigma_inv, device)) / K
    
    # SpectralMerge-LP F=2
    spec_lp_init = torch.zeros(K, 2, device=device)
    spec_lp_init[:, 0] = 0.3 * (L ** 0.5)
    f_spec_lp = lambda p: idct_iii(torch.cat([p, torch.zeros(K, L - 2, device=device)], dim=1), M_dct)
    final_spec_lp = optimize_tta_adam(spec_lp_init, f_spec_lp, targets, Sigma_inv, steps=100, lr=0.01)
    acc_spec_lp2 += sum(get_accuracy(final_spec_lp, lambda_stars, Sigma_inv, device)) / K
    
    # SpectralMerge-Reg (mu=0.01)
    spec_reg_init = torch.zeros(K, L, device=device)
    spec_reg_init[:, 0] = 0.3 * (L ** 0.5)
    f_spec_reg = lambda p: idct_iii(p, M_dct)
    # penalty: sum(mu * j^2 * c_{k,j}^2)
    j_sq = torch.arange(L, dtype=torch.float32, device=device) ** 2
    reg_fn = lambda p: torch.sum(0.01 * j_sq * (p ** 2))
    final_spec_reg = optimize_tta_adam(spec_reg_init, f_spec_reg, targets, Sigma_inv, steps=100, lr=0.01, reg_fn=reg_fn)
    acc_spec_reg += sum(get_accuracy(final_spec_reg, lambda_stars, Sigma_inv, device)) / K

print(f"Uniform: {acc_uniform / len(seeds) * 100:.2f}%")
print(f"Unconstrained: {acc_unconstrained / len(seeds) * 100:.2f}%")
print(f"Poly-Val d=2: {acc_poly_d2 / len(seeds) * 100:.2f}%")
print(f"SpectralMerge-LP F=2: {acc_spec_lp2 / len(seeds) * 100:.2f}%")
print(f"SpectralMerge-Reg (mu=0.01): {acc_spec_reg / len(seeds) * 100:.2f}%")
