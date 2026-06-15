import torch
import torch.nn as nn
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

L = 12
K = 4
NUM_SEEDS = 30
LR = 0.08
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

def run_experiment_for_seed(seed, drift_scale=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initial sensitivities at theta_0
    A_init = torch.zeros((K, L), dtype=torch.float32)
    for k in range(K):
        for l in range(L):
            if l <= 3 or l >= 10:
                A_init[k, l] = torch.distributions.Uniform(0.8, 1.2).sample()
            else:
                A_init[k, l] = torch.distributions.Uniform(0.2, 0.4).sample()
    
    # Drift direction representing coordinate updates during adaptation
    A_drift_dir = torch.distributions.Normal(0, A_init).sample()
    
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
    
    # Get sensitivities as a function of time step t to simulate parameter-drift
    def get_A_at_step(t):
        progress = float(t) / STEPS
        # Curvature drifts by drift_scale
        return torch.clamp(A_init + drift_scale * progress * A_drift_dir, 0.05, 5.0)

    def get_tta_objective_step(coeffs, A_t):
        B_t = 0.5 * A_t
        lambda_shifted = coeffs - eta
        term_s = A_t * ((lambda_shifted - alpha_opt) ** 2) + B_t * ((lambda_shifted - alpha_opt) ** 4)
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

    # Pre-trained curvatures at theta_0
    c_init = torch.mean(A_init, dim=0)
    weight_init = torch.sqrt(c_init[1:] * c_init[:-1])

    # --- 1. Static RCR-Merge (Uses pre-trained curvature throughout) ---
    # Gradient Norm Balancing at step 0
    delta_gnb = 0.05
    pert_pattern = torch.tensor([(-1.0)**li for li in range(L)], dtype=torch.float32)
    coeffs_pert = 0.3 + delta_gnb * pert_pattern.unsqueeze(0).repeat(K, 1)
    coeffs_pert.requires_grad_(True)
    diff_sq_pert = (coeffs_pert[:, 1:] - coeffs_pert[:, :-1]) ** 2
    rcr_penalty_pert = torch.sum(weight_init * diff_sq_pert)
    grad_rcr = torch.autograd.grad(rcr_penalty_pert, coeffs_pert)[0]
    norm_grad_rcr = torch.norm(grad_rcr).item()

    with torch.enable_grad():
        coeffs_init = torch.ones((K, L), dtype=torch.float32) * 0.3
        coeffs_init.requires_grad_(True)
        tta_loss_init = get_tta_objective_step(coeffs_init, A_init)
        grad_tta = torch.autograd.grad(tta_loss_init, coeffs_init)[0]
        norm_grad_tta = torch.norm(grad_tta).item()

    beta_rcr = 0.5 * norm_grad_tta / norm_grad_rcr
    
    epsilon_gnb = 0.1
    coeffs_drift = 0.3 + epsilon_gnb * torch.ones((K, L), dtype=torch.float32)
    coeffs_drift.requires_grad_(True)
    anchor_penalty_drift = torch.sum((coeffs_drift - 0.3) ** 2)
    grad_anchor = torch.autograd.grad(anchor_penalty_drift, coeffs_drift)[0]
    norm_grad_anchor = torch.norm(grad_anchor).item()
    gamma_anchor = (0.0003 * norm_grad_tta / norm_grad_anchor)

    coeffs_static = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_static = torch.optim.Adam([coeffs_static], lr=LR)
    for step in range(STEPS):
        optimizer_static.zero_grad()
        A_step = get_A_at_step(step)
        tta_loss = get_tta_objective_step(coeffs_static, A_step)
        
        diff_sq = (coeffs_static[:, 1:] - coeffs_static[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_init * diff_sq)
        anchor_penalty = torch.sum((coeffs_static - 0.3) ** 2)
        
        total_loss = tta_loss + beta_rcr * rcr_penalty + gamma_anchor * anchor_penalty
        total_loss.backward()
        optimizer_static.step()
        with torch.no_grad():
            coeffs_static.clamp_(0.0, 1.0)

    # --- 2. Dynamic RCR-Merge (Oracle updates FIM curvature at each step) ---
    coeffs_dynamic = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_dynamic = torch.optim.Adam([coeffs_dynamic], lr=LR)
    for step in range(STEPS):
        optimizer_dynamic.zero_grad()
        A_step = get_A_at_step(step)
        tta_loss = get_tta_objective_step(coeffs_dynamic, A_step)
        
        # True drifted curvatures
        c_step = torch.mean(A_step, dim=0)
        weight_step = torch.sqrt(c_step[1:] * c_step[:-1])
        
        # We re-evaluate GNB based on step curvature
        diff_sq_pert_t = (coeffs_pert[:, 1:] - coeffs_pert[:, :-1]) ** 2
        rcr_penalty_pert_t = torch.sum(weight_step * diff_sq_pert_t)
        grad_rcr_t = torch.autograd.grad(rcr_penalty_pert_t, coeffs_pert)[0]
        norm_grad_rcr_t = torch.norm(grad_rcr_t).item()
        
        beta_rcr_t = 0.5 * norm_grad_tta / norm_grad_rcr_t
        
        diff_sq = (coeffs_dynamic[:, 1:] - coeffs_dynamic[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_step * diff_sq)
        anchor_penalty = torch.sum((coeffs_dynamic - 0.3) ** 2)
        
        total_loss = tta_loss + beta_rcr_t * rcr_penalty + gamma_anchor * anchor_penalty
        total_loss.backward()
        optimizer_dynamic.step()
        with torch.no_grad():
            coeffs_dynamic.clamp_(0.0, 1.0)

    acc_static_coupled = evaluate_accuracy(coeffs_static, use_euclidean=False)
    acc_dynamic_coupled = evaluate_accuracy(coeffs_dynamic, use_euclidean=False)
    acc_static_eucl = evaluate_accuracy(coeffs_static, use_euclidean=True)
    acc_dynamic_eucl = evaluate_accuracy(coeffs_dynamic, use_euclidean=True)

    return {
        'static_coupled': np.mean(acc_static_coupled),
        'dynamic_coupled': np.mean(acc_dynamic_coupled),
        'static_eucl': np.mean(acc_static_eucl),
        'dynamic_eucl': np.mean(acc_dynamic_eucl)
    }

for drift in [0.1, 0.3, 0.5]:
    static_coupled_res = []
    dynamic_coupled_res = []
    static_eucl_res = []
    dynamic_eucl_res = []
    
    for s in range(1, NUM_SEEDS + 1):
        r = run_experiment_for_seed(s, drift_scale=drift)
        static_coupled_res.append(r['static_coupled'])
        dynamic_coupled_res.append(r['dynamic_coupled'])
        static_eucl_res.append(r['static_eucl'])
        dynamic_eucl_res.append(r['dynamic_eucl'])
        
    print(f"\n--- PARAMETER DRIFT ROBUSTNESS EVALUATION (Drift Scale={drift*100:.0f}%) ---")
    print(f"Static RCR-Merge Coupled:  {np.mean(static_coupled_res):.4f}% +/- {np.std(static_coupled_res):.4f}%")
    print(f"Dynamic RCR-Merge Coupled: {np.mean(dynamic_coupled_res):.4f}% +/- {np.std(dynamic_coupled_res):.4f}%")
    print(f"Static RCR-Merge Euclidean:  {np.mean(static_eucl_res):.4f}% +/- {np.std(static_eucl_res):.4f}%")
    print(f"Dynamic RCR-Merge Euclidean: {np.mean(dynamic_eucl_res):.4f}% +/- {np.std(dynamic_eucl_res):.4f}%")
