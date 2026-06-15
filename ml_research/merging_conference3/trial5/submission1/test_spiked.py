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
Sigma_id_inv = torch.eye(L)

def get_optimal_profile_spiked(k, l):
    # Modular specialist architecture: all layers baseline 0.3, except one middle layer with sharp specialization
    baseline = 0.3
    if k == 0:
        return 0.85 if l == 5 else baseline
    elif k == 1:
        return 0.10 if l == 6 else baseline
    elif k == 2:
        return 0.90 if l == 7 else baseline
    elif k == 3:
        return 0.05 if l == 4 else baseline
    else:
        return baseline

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
            opt_val = get_optimal_profile_spiked(k, l)
            eps = torch.distributions.Normal(0, 0.01).sample() # tight around spike
            alpha_opt[k, l] = torch.clamp(torch.tensor(opt_val) + eps, 0.0, 1.0)
            
    D = torch.zeros((K, K, L), dtype=torch.float32)
    for l in range(L):
        D[0, 1, l] = D[1, 0, l] = torch.distributions.Uniform(0.01, 0.03).sample()
        D[2, 3, l] = D[3, 2, l] = torch.distributions.Uniform(0.20, 0.30).sample()
        D[0, 3, l] = D[3, 0, l] = torch.distributions.Uniform(0.12, 0.18).sample()
        D[1, 2, l] = D[2, 1, l] = torch.distributions.Uniform(0.15, 0.21).sample()
        D[0, 2, l] = D[2, 0, l] = torch.distributions.Uniform(0.10, 0.15).sample()
        D[1, 3, l] = D[3, 1, l] = torch.distributions.Uniform(0.10, 0.15).sample()

    eta = torch.distributions.Normal(0, 0.05).sample((K, L)) # moderate transductive noise
    
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

    # Uniform baseline
    coeffs_uniform = torch.ones((K, L), dtype=torch.float32) * 0.3
    
    # Unconstrained AdaMerging (Adam, lr=0.01)
    coeffs_ada = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_ada = torch.optim.Adam([coeffs_ada], lr=0.01)
    for step in range(STEPS):
        optimizer_ada.zero_grad()
        loss = get_tta_objective_vectorized(coeffs_ada)
        loss.backward()
        optimizer_ada.step()
        with torch.no_grad():
            coeffs_ada.clamp_(0.0, 1.0)
            
    # PolyMerge (d=2)
    poly_params = nn.Parameter(torch.zeros((K, 3), dtype=torch.float32))
    with torch.no_grad():
        poly_params[:, 0] = 0.3
    optimizer_poly = torch.optim.Adam([poly_params], lr=LR)
    l_norm_vec = torch.tensor([l / (L - 1) for l in range(L)], dtype=torch.float32)
    for step in range(STEPS):
        optimizer_poly.zero_grad()
        coeffs_poly_step = poly_params[:, 0:1] + poly_params[:, 1:2] * l_norm_vec.unsqueeze(0) + poly_params[:, 2:3] * (l_norm_vec.unsqueeze(0) ** 2)
        loss = get_tta_objective_vectorized(coeffs_poly_step)
        loss.backward()
        optimizer_poly.step()
    coeffs_poly_final = poly_params[:, 0:1] + poly_params[:, 1:2] * l_norm_vec.unsqueeze(0) + poly_params[:, 2:3] * (l_norm_vec.unsqueeze(0) ** 2)
    coeffs_poly_final = coeffs_poly_final.clone().detach().clamp(0.0, 1.0)

    # TV-Regularized (beta=0.5, lr=0.02)
    coeffs_tv = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_tv = torch.optim.Adam([coeffs_tv], lr=0.02)
    for step in range(STEPS):
        optimizer_tv.zero_grad()
        tta_loss = get_tta_objective_vectorized(coeffs_tv)
        tv_penalty = torch.sum((coeffs_tv[:, 1:] - coeffs_tv[:, :-1]) ** 2)
        total_loss = tta_loss + 0.5 * tv_penalty
        total_loss.backward()
        optimizer_tv.step()
        with torch.no_grad():
            coeffs_tv.clamp_(0.0, 1.0)

    # RCR-Merge (Ours) (alpha_gnb=1.0, lr=0.05, anchor=0.005)
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

        beta_rcr = (1.0 * norm_grad_tta / norm_grad_rcr).item()
        
    for step in range(STEPS):
        optimizer_rcr.zero_grad()
        tta_loss = get_tta_objective_vectorized(coeffs_rcr)
        diff_sq = (coeffs_rcr[:, 1:] - coeffs_rcr[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_l * diff_sq)
        anchor_penalty = torch.sum((coeffs_rcr - 0.3) ** 2)
        total_loss = tta_loss + beta_rcr * rcr_penalty + 0.005 * anchor_penalty
        total_loss.backward()
        optimizer_rcr.step()
        with torch.no_grad():
            coeffs_rcr.clamp_(0.0, 1.0)

    return {
        'Uniform_coupled': evaluate_accuracy(coeffs_uniform, use_euclidean=False),
        'Ada_coupled': evaluate_accuracy(coeffs_ada, use_euclidean=False),
        'Poly_coupled': evaluate_accuracy(coeffs_poly_final, use_euclidean=False),
        'TV_coupled': evaluate_accuracy(coeffs_tv, use_euclidean=False),
        'RCR_coupled': evaluate_accuracy(coeffs_rcr, use_euclidean=False),
        
        'Uniform_euclidean': evaluate_accuracy(coeffs_uniform, use_euclidean=True),
        'Ada_euclidean': evaluate_accuracy(coeffs_ada, use_euclidean=True),
        'Poly_euclidean': evaluate_accuracy(coeffs_poly_final, use_euclidean=True),
        'TV_euclidean': evaluate_accuracy(coeffs_tv, use_euclidean=True),
        'RCR_euclidean': evaluate_accuracy(coeffs_rcr, use_euclidean=True),
    }

# Run 30 seeds
coupled_res = {k: [] for k in ['Uniform', 'Ada', 'Poly', 'TV', 'RCR']}
euclidean_res = {k: [] for k in ['Uniform', 'Ada', 'Poly', 'TV', 'RCR']}

for s in range(1, 31):
    r = run_experiment_for_seed(s)
    coupled_res['Uniform'].append(np.mean(r['Uniform_coupled']))
    coupled_res['Ada'].append(np.mean(r['Ada_coupled']))
    coupled_res['Poly'].append(np.mean(r['Poly_coupled']))
    coupled_res['TV'].append(np.mean(r['TV_coupled']))
    coupled_res['RCR'].append(np.mean(r['RCR_coupled']))

    euclidean_res['Uniform'].append(np.mean(r['Uniform_euclidean']))
    euclidean_res['Ada'].append(np.mean(r['Ada_euclidean']))
    euclidean_res['Poly'].append(np.mean(r['Poly_euclidean']))
    euclidean_res['TV'].append(np.mean(r['TV_euclidean']))
    euclidean_res['RCR'].append(np.mean(r['RCR_euclidean']))

print("\n--- MODULAR SPECIALIST SPIKED LANDSCAPE: COUPLED COVARIANCE (30 Seeds Avg Acc) ---")
for k, v in coupled_res.items():
    print(f"{k:<10}: {np.mean(v):.4f}% +/- {np.std(v):.4f}%")

print("\n--- MODULAR SPECIALIST SPIKED LANDSCAPE: DECOUPLED EUCLIDEAN (30 Seeds Avg Acc) ---")
for k, v in euclidean_res.items():
    print(f"{k:<10}: {np.mean(v):.4f}% +/- {np.std(v):.4f}%")
