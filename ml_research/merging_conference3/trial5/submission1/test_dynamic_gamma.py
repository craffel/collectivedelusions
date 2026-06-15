import torch
import torch.nn as nn
import numpy as np

# Set random seeds for reproducibility
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
Sigma_id_inv = torch.eye(L)

def get_optimal_profile_blockwise(k, l):
    if k == 0:
        if l <= 3: return 0.1
        elif l <= 7: return 0.95
        else: return 0.1
    elif k == 1:
        if l <= 3: return 0.9
        elif l <= 7: return 0.1
        else: return 0.9
    elif k == 2:
        if l <= 3: return 0.2
        elif l <= 7: return 0.85
        else: return 0.5
    elif k == 3:
        if l <= 3: return 0.8
        elif l <= 7: return 0.2
        else: return 0.6
    else:
        return 0.5

def run_experiment_for_seed(seed, alpha_drift=0.00005):
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
            opt_val = get_optimal_profile_blockwise(k, l)
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

    eta = torch.distributions.Normal(0, 0.05).sample((K, L))
    
    def get_tta_objective_vectorized(coeffs):
        lambda_shifted = coeffs - eta
        term_s = A * ((lambda_shifted - alpha_opt) ** 2) + B * ((lambda_shifted - alpha_opt) ** 4)
        diff = lambda_shifted.unsqueeze(1) - lambda_shifted.unsqueeze(0)
        diff_sq = diff ** 2
        interference_sum = torch.sum(D * diff_sq, dim=1)
        sens_loss = torch.sum(term_s + interference_sum)
        cosine_penalty = LAMBDA_RUG * torch.sum(torch.cos(F_RUG * lambda_shifted))
        return sens_loss + cosine_penalty

    def evaluate_accuracy(coeffs, use_euclidean=False):
        d = coeffs - alpha_opt
        d_init = torch.ones_like(coeffs) * 0.3 - alpha_opt
        inv_m = Sigma_id_inv if use_euclidean else Sigma_inv
        dist_opt = torch.sum(d * torch.matmul(d, inv_m), dim=1)
        dist_init = torch.sum(d_init * torch.matmul(d_init, inv_m), dim=1)
        accs = BASE_ACC + DELTA_ACC * (1.0 - dist_opt / dist_init)
        accs = torch.clamp(accs, 0.0, 100.0)
        return accs.tolist()

    weight_l = torch.sqrt(c[1:] * c[:-1])
    
    # Gradient Norm Balancing for beta
    delta_gnb = 0.05
    pert_pattern = torch.tensor([(-1.0)**li for li in range(L)], dtype=torch.float32)
    coeffs_pert = 0.3 + delta_gnb * pert_pattern.unsqueeze(0).repeat(K, 1)
    coeffs_pert.requires_grad_(True)
    diff_sq_pert = (coeffs_pert[:, 1:] - coeffs_pert[:, :-1]) ** 2
    rcr_penalty_pert = torch.sum(weight_l * diff_sq_pert)
    grad_rcr = torch.autograd.grad(rcr_penalty_pert, coeffs_pert)[0]
    norm_grad_rcr = torch.norm(grad_rcr).item()

    # Base TTA Gradient
    with torch.enable_grad():
        coeffs_init = torch.ones((K, L), dtype=torch.float32) * 0.3
        coeffs_init.requires_grad_(True)
        tta_loss_init = get_tta_objective_vectorized(coeffs_init)
        grad_tta = torch.autograd.grad(tta_loss_init, coeffs_init)[0]
        norm_grad_tta = torch.norm(grad_tta).item()

    beta_rcr = 0.5 * norm_grad_tta / norm_grad_rcr

    # Standard GNB with hardcoded gamma = 0.001
    coeffs_fixed = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_fixed = torch.optim.Adam([coeffs_fixed], lr=0.08)
    for step in range(STEPS):
        optimizer_fixed.zero_grad()
        tta_loss = get_tta_objective_vectorized(coeffs_fixed)
        diff_sq = (coeffs_fixed[:, 1:] - coeffs_fixed[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_l * diff_sq)
        anchor_penalty = torch.sum((coeffs_fixed - 0.3) ** 2)
        total_loss = tta_loss + beta_rcr * rcr_penalty + 0.001 * anchor_penalty
        total_loss.backward()
        optimizer_fixed.step()
        with torch.no_grad():
            coeffs_fixed.clamp_(0.0, 1.0)

    # GNB with self-scaling gamma_anchor
    epsilon_gnb = 0.1
    coeffs_drift = 0.3 + epsilon_gnb * torch.ones((K, L), dtype=torch.float32)
    coeffs_drift.requires_grad_(True)
    anchor_penalty_drift = torch.sum((coeffs_drift - 0.3) ** 2)
    grad_anchor = torch.autograd.grad(anchor_penalty_drift, coeffs_drift)[0]
    norm_grad_anchor = torch.norm(grad_anchor).item()

    gamma_anchor = (alpha_drift * norm_grad_tta / norm_grad_anchor)

    coeffs_self = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_self = torch.optim.Adam([coeffs_self], lr=0.08)
    for step in range(STEPS):
        optimizer_self.zero_grad()
        tta_loss = get_tta_objective_vectorized(coeffs_self)
        diff_sq = (coeffs_self[:, 1:] - coeffs_self[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_l * diff_sq)
        anchor_penalty = torch.sum((coeffs_self - 0.3) ** 2)
        total_loss = tta_loss + beta_rcr * rcr_penalty + gamma_anchor * anchor_penalty
        total_loss.backward()
        optimizer_self.step()
        with torch.no_grad():
            coeffs_self.clamp_(0.0, 1.0)

    return {
        'fixed_coupled': evaluate_accuracy(coeffs_fixed, use_euclidean=False),
        'self_coupled': evaluate_accuracy(coeffs_self, use_euclidean=False),
        'fixed_euclidean': evaluate_accuracy(coeffs_fixed, use_euclidean=True),
        'self_euclidean': evaluate_accuracy(coeffs_self, use_euclidean=True),
        'gamma': gamma_anchor
    }

for alpha_drift in [0.00001, 0.00003, 0.00005, 0.0001, 0.0003]:
    fixed_coupled_res = []
    self_coupled_res = []
    fixed_eucl_res = []
    self_eucl_res = []
    gammas = []
    
    for s in range(1, 31):
        r = run_experiment_for_seed(s, alpha_drift=alpha_drift)
        fixed_coupled_res.append(np.mean(r['fixed_coupled']))
        self_coupled_res.append(np.mean(r['self_coupled']))
        fixed_eucl_res.append(np.mean(r['fixed_euclidean']))
        self_eucl_res.append(np.mean(r['self_euclidean']))
        gammas.append(r['gamma'])
        
    print(f"\n--- GNB SELF-SCALING ANCHOR EVALUATION (alpha_drift={alpha_drift}) ---")
    print(f"Average Computed Gamma: {np.mean(gammas):.6f}")
    print(f"Fixed Gamma=0.001 Coupled: {np.mean(fixed_coupled_res):.4f}% +/- {np.std(fixed_coupled_res):.4f}%")
    print(f"Self-Scaled Gamma Coupled: {np.mean(self_coupled_res):.4f}% +/- {np.std(self_coupled_res):.4f}%")
    print(f"Fixed Gamma=0.001 Euclidean: {np.mean(fixed_eucl_res):.4f}% +/- {np.std(fixed_eucl_res):.4f}%")
    print(f"Self-Scaled Gamma Euclidean: {np.mean(self_eucl_res):.4f}% +/- {np.std(self_eucl_res):.4f}%")
