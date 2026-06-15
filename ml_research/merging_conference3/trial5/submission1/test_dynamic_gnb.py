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

def run_experiment_for_seed(seed):
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

    # Pre-compute constant regularizer gradient norm at pert state
    weight_l = torch.sqrt(c[1:] * c[:-1])
    delta_gnb = 0.05
    pert_pattern = torch.tensor([(-1.0)**li for li in range(L)], dtype=torch.float32)
    coeffs_pert = 0.3 + delta_gnb * pert_pattern.unsqueeze(0).repeat(K, 1)
    coeffs_pert.requires_grad_(True)
    diff_sq_pert = (coeffs_pert[:, 1:] - coeffs_pert[:, :-1]) ** 2
    rcr_penalty_pert = torch.sum(weight_l * diff_sq_pert)
    grad_rcr = torch.autograd.grad(rcr_penalty_pert, coeffs_pert)[0]
    norm_grad_rcr = torch.norm(grad_rcr).item()

    # Static RCR-Merge (beta is evaluated once at step 0)
    coeffs_static = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_static = torch.optim.Adam([coeffs_static], lr=0.08)
    
    with torch.enable_grad():
        coeffs_init = torch.ones((K, L), dtype=torch.float32) * 0.3
        coeffs_init.requires_grad_(True)
        tta_loss_init = get_tta_objective_vectorized(coeffs_init)
        grad_tta = torch.autograd.grad(tta_loss_init, coeffs_init)[0]
        norm_grad_tta = torch.norm(grad_tta).item()
        
    beta_static = 0.5 * norm_grad_tta / norm_grad_rcr
    
    for step in range(STEPS):
        optimizer_static.zero_grad()
        tta_loss = get_tta_objective_vectorized(coeffs_static)
        diff_sq = (coeffs_static[:, 1:] - coeffs_static[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_l * diff_sq)
        anchor_penalty = torch.sum((coeffs_static - 0.3) ** 2)
        total_loss = tta_loss + beta_static * rcr_penalty + 0.001 * anchor_penalty
        total_loss.backward()
        optimizer_static.step()
        with torch.no_grad():
            coeffs_static.clamp_(0.0, 1.0)

    # Dynamic RCR-Merge (beta is dynamically re-scaled at each step t based on gradient norms)
    coeffs_dynamic = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_dynamic = torch.optim.Adam([coeffs_dynamic], lr=0.08)
    
    for step in range(STEPS):
        optimizer_dynamic.zero_grad()
        # To compute current gradient norm without interfering with optimizer step,
        # we compute current TTA loss on a detached version or we can just compute it directly.
        # Computing it on detached version allows us to get the gradient norm safely.
        coeffs_temp = coeffs_dynamic.clone().detach()
        coeffs_temp.requires_grad_(True)
        tta_loss_temp = get_tta_objective_vectorized(coeffs_temp)
        grad_temp = torch.autograd.grad(tta_loss_temp, coeffs_temp)[0]
        norm_grad_tta_t = torch.norm(grad_temp).item()
        
        # Dynamic beta scaling with decay limit to prevent dividing by zero or going too small
        beta_t = 0.5 * norm_grad_tta_t / norm_grad_rcr
        # Enforce a minimal baseline regularization to prevent collapsing to fully unconstrained near convergence
        beta_t = max(beta_t, 0.05 * beta_static)
        
        tta_loss = get_tta_objective_vectorized(coeffs_dynamic)
        diff_sq = (coeffs_dynamic[:, 1:] - coeffs_dynamic[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_l * diff_sq)
        anchor_penalty = torch.sum((coeffs_dynamic - 0.3) ** 2)
        total_loss = tta_loss + beta_t * rcr_penalty + 0.001 * anchor_penalty
        total_loss.backward()
        optimizer_dynamic.step()
        with torch.no_grad():
            coeffs_dynamic.clamp_(0.0, 1.0)

    return {
        'Static_coupled': evaluate_accuracy(coeffs_static, use_euclidean=False),
        'Dynamic_coupled': evaluate_accuracy(coeffs_dynamic, use_euclidean=False),
        'Static_euclidean': evaluate_accuracy(coeffs_static, use_euclidean=True),
        'Dynamic_euclidean': evaluate_accuracy(coeffs_dynamic, use_euclidean=True),
    }

static_coupled_res = []
dynamic_coupled_res = []
static_eucl_res = []
dynamic_eucl_res = []

for s in range(1, 31):
    r = run_experiment_for_seed(s)
    static_coupled_res.append(np.mean(r['Static_coupled']))
    dynamic_coupled_res.append(np.mean(r['Dynamic_coupled']))
    static_eucl_res.append(np.mean(r['Static_euclidean']))
    dynamic_eucl_res.append(np.mean(r['Dynamic_euclidean']))

print("\n--- DYNAMIC GNB EVALUATION (30 Seeds Avg Acc) ---")
print(f"Static RCR-Merge Coupled: {np.mean(static_coupled_res):.4f}% +/- {np.std(static_coupled_res):.4f}%")
print(f"Dynamic RCR-Merge Coupled: {np.mean(dynamic_coupled_res):.4f}% +/- {np.std(dynamic_coupled_res):.4f}%")
print(f"Static RCR-Merge Euclidean: {np.mean(static_eucl_res):.4f}% +/- {np.std(static_eucl_res):.4f}%")
print(f"Dynamic RCR-Merge Euclidean: {np.mean(dynamic_eucl_res):.4f}% +/- {np.std(dynamic_eucl_res):.4f}%")
