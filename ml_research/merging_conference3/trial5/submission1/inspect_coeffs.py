import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

L = 12
K = 4
STEPS = 100
LAMBDA_RUG = 0.05
F_RUG = 15.0

def get_optimal_profile_stepwise(k, l):
    if k == 0:
        if l <= 3: return 0.2
        elif l <= 9: return 0.5
        else: return 0.8
    elif k == 1:
        if l <= 3: return 0.8
        elif l <= 9: return 0.5
        else: return 0.2
    elif k == 2:
        if l <= 3: return 0.3
        elif l <= 9: return 0.7
        else: return 0.4
    elif k == 3:
        if l <= 3: return 0.6
        elif l <= 9: return 0.4
        else: return 0.6
    else:
        return 0.5

def run_experiment(alpha_gnb):
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    A = torch.zeros((K, L), dtype=torch.float32)
    for k in range(K):
        for l in range(L):
            if l <= 3 or l >= 10:
                A[k, l] = torch.distributions.Uniform(0.8, 1.2).sample()
            else:
                A[k, l] = torch.distributions.Uniform(0.2, 0.4).sample()
    B = 0.5 * A
    c = torch.mean(A, dim=0)
    
    alpha_opt = torch.zeros((K, L), dtype=torch.float32)
    for k in range(K):
        for l in range(L):
            opt_val = get_optimal_profile_stepwise(k, l)
            eps = torch.distributions.Normal(0, 0.02).sample()
            alpha_opt[k, l] = torch.clamp(torch.tensor(opt_val) + eps, 0.0, 1.0)
            
    D = torch.zeros((K, K, L), dtype=torch.float32)
    for l in range(L):
        D[0, 1, l] = D[1, 0, l] = torch.distributions.Uniform(0.01, 0.03).sample()
        D[2, 3, l] = D[3, 2, l] = torch.distributions.Uniform(0.20, 0.30).sample()
        D[0, 3, l] = D[3, 0, l] = torch.distributions.Uniform(0.12, 0.18).sample()
        D[1, 2, l] = D[2, 1, l] = torch.distributions.Uniform(0.15, 0.21).sample()
        D[0, 2, l] = D[2, 0, l] = torch.distributions.Uniform(0.10, 0.15).sample()
        D[1, 3, l] = D[3, 1, l] = torch.distributions.Uniform(0.10, 0.15).sample()

    eta = torch.distributions.Normal(0, 0.10).sample((K, L))
    
    def get_tta_objective_vectorized(coeffs):
        lambda_shifted = coeffs - eta
        term_s = A * ((lambda_shifted - alpha_opt) ** 2) + B * ((lambda_shifted - alpha_opt) ** 4)
        diff = lambda_shifted.unsqueeze(1) - lambda_shifted.unsqueeze(0)
        diff_sq = diff ** 2
        interference_sum = torch.sum(D * diff_sq, dim=1)
        sens_loss = torch.sum(term_s + interference_sum)
        cosine_penalty = LAMBDA_RUG * torch.sum(torch.cos(F_RUG * lambda_shifted))
        return sens_loss + cosine_penalty

    # RCR-Merge
    coeffs_rcr = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_rcr = torch.optim.Adam([coeffs_rcr], lr=0.05)
    weight_l = torch.sqrt(c[1:] * c[:-1])
    with torch.enable_grad():
        coeffs_init = torch.ones((K, L), dtype=torch.float32) * 0.3
        coeffs_init.requires_grad_(True)
        tta_loss_init = get_tta_objective_vectorized(coeffs_init)
        grad_tta = torch.autograd.grad(tta_loss_init, coeffs_init)[0]
        norm_grad_tta = torch.norm(grad_tta)

        delta_gnb = 0.05
        pert_pattern = torch.tensor([(-1.0)**li for li in range(L)], dtype=torch.float32)
        coeffs_pert = 0.3 + delta_gnb * pert_pattern.unsqueeze(0).repeat(K, 1)
        coeffs_pert.requires_grad_(True)
        diff_sq_pert = (coeffs_pert[:, 1:] - coeffs_pert[:, :-1]) ** 2
        rcr_penalty_pert = torch.sum(weight_l * diff_sq_pert)
        grad_rcr = torch.autograd.grad(rcr_penalty_pert, coeffs_pert)[0]
        norm_grad_rcr = torch.norm(grad_rcr)
        beta_rcr = (alpha_gnb * norm_grad_tta / norm_grad_rcr).item()

    for step in range(STEPS):
        optimizer_rcr.zero_grad()
        tta_loss = get_tta_objective_vectorized(coeffs_rcr)
        diff_sq = (coeffs_rcr[:, 1:] - coeffs_rcr[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_l * diff_sq)
        anchor_penalty = torch.sum((coeffs_rcr - 0.3) ** 2)
        total_loss = tta_loss + beta_rcr * rcr_penalty + 0.01 * anchor_penalty
        total_loss.backward()
        optimizer_rcr.step()
        with torch.no_grad():
            coeffs_rcr.clamp_(0.0, 1.0)

    mse = torch.mean((coeffs_rcr - alpha_opt)**2).item()
    print(f"Alpha={alpha_gnb}, Beta={beta_rcr:.4f} => MSE={mse:.5f}")

for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    run_experiment(alpha)
