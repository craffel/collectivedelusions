import torch
import numpy as np

from run_experiments_v2 import (
    get_dct_matrix, get_covariance_matrix, get_optimal_profile,
    get_accuracy, generate_noise, idct_iii, L, K
)

device = torch.device("cpu")
Sigma = get_covariance_matrix(L, device)
Sigma_inv = torch.linalg.inv(Sigma)
lambda_stars = torch.stack([get_optimal_profile(k, L, device) for k in range(K)])
M_dct = get_dct_matrix(L, device)

# Vandermonde matrix for Poly-Val (d=2)
l_bar = torch.linspace(0.0, 1.0, L, device=device)
V_d2 = torch.stack([l_bar ** j for j in range(3)], dim=1) # (12, 3)

def optimize_val_adam(initial_param, forward_fn, targets, Sigma_inv, steps=150, lr=0.05, reg_fn=None):
    param = initial_param.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([param], lr=lr)
    for step in range(steps):
        optimizer.zero_grad()
        lambdas = forward_fn(param)
        
        # Validation loss
        loss = 0.0
        for k in range(K):
            e_k = lambdas[k] - targets[k]
            quad = torch.matmul(e_k, torch.matmul(Sigma_inv, e_k))
            cos_term = 0.03 * torch.sum(1.0 - torch.cos(10 * torch.pi * e_k))
            loss += 0.5 + 1.5 * quad + cos_term
            
        if reg_fn is not None:
            loss += reg_fn(param)
            
        loss.backward()
        optimizer.step()
    return forward_fn(param).detach()

seeds = list(range(42, 47))
M = 10

acc_layerwise = 0.0
acc_poly_d2 = 0.0
acc_spec_lp2 = 0.0
acc_spec_reg = 0.0

for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)
    
    # Generate validation noise scaled by 1/sqrt(M)
    etas_val = torch.stack([generate_noise(L, device) for _ in range(K)]) / (M ** 0.5)
    val_targets = lambda_stars + etas_val
    
    # 1. Layer-wise Search
    layer_init = torch.ones(K, L, device=device) * 0.3
    f_layer = lambda p: p
    final_layer = optimize_val_adam(layer_init, f_layer, val_targets, Sigma_inv, steps=150, lr=0.05)
    acc_layerwise += sum(get_accuracy(final_layer, lambda_stars, Sigma_inv, device)) / K
    
    # 2. Poly-Val d=2
    poly_init = torch.zeros(K, 3, device=device)
    poly_init[:, 0] = 0.3
    f_poly = lambda p: torch.matmul(p, V_d2.t())
    final_poly = optimize_val_adam(poly_init, f_poly, val_targets, Sigma_inv, steps=150, lr=0.05)
    acc_poly_d2 += sum(get_accuracy(final_poly, lambda_stars, Sigma_inv, device)) / K
    
    # 3. SpectralMerge-LP F=3
    spec_lp_init = torch.zeros(K, 3, device=device)
    spec_lp_init[:, 0] = 0.3 * (L ** 0.5)
    f_spec_lp = lambda p: idct_iii(torch.cat([p, torch.zeros(K, L - 3, device=device)], dim=1), M_dct)
    final_spec_lp = optimize_val_adam(spec_lp_init, f_spec_lp, val_targets, Sigma_inv, steps=150, lr=0.05)
    acc_spec_lp2 += sum(get_accuracy(final_spec_lp, lambda_stars, Sigma_inv, device)) / K
    
    # 4. SpectralMerge-Reg (mu=1.0)
    spec_reg_init = torch.zeros(K, L, device=device)
    spec_reg_init[:, 0] = 0.3 * (L ** 0.5)
    f_spec_reg = lambda p: idct_iii(p, M_dct)
    j_sq = torch.arange(L, dtype=torch.float32, device=device) ** 2
    reg_fn = lambda p: torch.sum(1.0 * j_sq * (p ** 2))
    final_spec_reg = optimize_val_adam(spec_reg_init, f_spec_reg, val_targets, Sigma_inv, steps=150, lr=0.05, reg_fn=reg_fn)
    acc_spec_reg += sum(get_accuracy(final_spec_reg, lambda_stars, Sigma_inv, device)) / K

print(f"OFS-Tune M={M} on 5 seeds:")
print(f"Layer-wise (unconstrained): {acc_layerwise / len(seeds) * 100:.2f}%")
print(f"Poly-Val d=2: {acc_poly_d2 / len(seeds) * 100:.2f}%")
print(f"SpectralMerge-LP F=3: {acc_spec_lp2 / len(seeds) * 100:.2f}%")
print(f"SpectralMerge-Reg (mu=1.0): {acc_spec_reg / len(seeds) * 100:.2f}%")
