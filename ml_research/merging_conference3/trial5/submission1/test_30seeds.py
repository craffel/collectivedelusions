import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

L = 12
K = 4
NUM_SEEDS = 30
LR = 0.01
STEPS = 100
LAMBDA_RUG = 0.05
F_RUG = 15.0

BASE_ACC = torch.tensor([94.68, 82.71, 94.04, 78.37], dtype=torch.float32)
DELTA_ACC = torch.tensor([5.32, 17.29, 5.96, 21.63], dtype=torch.float32)

s_l = torch.tensor([0.6]*4 + [1.0]*6 + [1.6]*2, dtype=torch.float32)
Sigma = torch.zeros((L, L), dtype=torch.float32)
for i in range(L):
    for j in range(L):
        Sigma[i, j] = torch.sqrt(s_l[i] * s_l[j]) * (0.5 ** abs(i - j))
Sigma_inv = torch.inverse(Sigma)

def get_optimal_profile(k, l_norm):
    if k == 0:
        return 0.3 + 0.4 * l_norm
    elif k == 1:
        return 0.7 - 0.4 * l_norm
    elif k == 2:
        return 0.2 + 1.2 * l_norm - 1.2 * (l_norm ** 2)
    elif k == 3:
        return 0.55
    else:
        return 0.5

def run_experiment_for_seed(seed, alpha_gnb, gamma_anchor, lr_rcr):
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
            l_norm = l / (L - 1)
            opt_val = get_optimal_profile(k, l_norm)
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

    def evaluate_accuracy_vectorized(coeffs):
        d = coeffs - alpha_opt
        d_init = torch.ones_like(coeffs) * 0.3 - alpha_opt
        dist_opt = torch.sum(d * torch.matmul(d, Sigma_inv), dim=1)
        dist_init = torch.sum(d_init * torch.matmul(d_init, Sigma_inv), dim=1)
        accs = BASE_ACC + DELTA_ACC * (1.0 - dist_opt / dist_init)
        accs = torch.clamp(accs, 0.0, 100.0)
        return accs.tolist()

    coeffs_rcr = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_rcr = torch.optim.Adam([coeffs_rcr], lr=lr_rcr)
    
    weight_l = torch.sqrt(c[1:] * c[:-1])
    
    # GNB
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
        total_loss = tta_loss + beta_rcr * rcr_penalty + gamma_anchor * anchor_penalty
        total_loss.backward()
        optimizer_rcr.step()
        with torch.no_grad():
            coeffs_rcr.clamp_(0.0, 1.0)
            
    return evaluate_accuracy_vectorized(coeffs_rcr)

if __name__ == "__main__":
    for alpha, gamma, lr in [(5.0, 0.01, 0.05), (5.0, 0.01, 0.02), (5.0, 0.02, 0.05)]:
        all_accs = []
        for seed in range(1, 31):
            accs = run_experiment_for_seed(seed, alpha, gamma, lr)
            all_accs.append(np.mean(accs))
        print(f"Alpha={alpha}, Gamma={gamma}, LR={lr} => 30-seed average = {np.mean(all_accs):.4f}% ± {np.std(all_accs):.4f}%")
