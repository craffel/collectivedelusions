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
STEPS = 2000
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

def run_experiment_for_seed(seed, drift_scale=0.4, threshold=0.03):
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
        # Introduce some non-stationary transductive scale changes to simulate streaming shifts
        # E.g., Task 1 and 2 sensitivities switch over time
        phase_shift = np.sin(2.0 * np.pi * progress)
        multiplier = 1.0 + 0.3 * phase_shift
        return torch.clamp(multiplier * (A_init + drift_scale * progress * A_drift_dir), 0.05, 5.0)

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

    # GNB at step 0 for Static
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

    beta_rcr_static = 0.5 * norm_grad_tta / norm_grad_rcr
    
    epsilon_gnb = 0.1
    coeffs_drift = 0.3 + epsilon_gnb * torch.ones((K, L), dtype=torch.float32)
    coeffs_drift.requires_grad_(True)
    anchor_penalty_drift = torch.sum((coeffs_drift - 0.3) ** 2)
    grad_anchor = torch.autograd.grad(anchor_penalty_drift, coeffs_drift)[0]
    norm_grad_anchor = torch.norm(grad_anchor).item()
    gamma_anchor = (0.0003 * norm_grad_tta / norm_grad_anchor)

    # --- 1. Unconstrained AdaMerging (Catastrophic Collapse) ---
    coeffs_unconstrained = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_unconstrained = torch.optim.Adam([coeffs_unconstrained], lr=LR)
    for step in range(STEPS):
        optimizer_unconstrained.zero_grad()
        A_step = get_A_at_step(step)
        tta_loss = get_tta_objective_step(coeffs_unconstrained, A_step)
        tta_loss.backward()
        optimizer_unconstrained.step()
        with torch.no_grad():
            coeffs_unconstrained.clamp_(0.0, 1.0)

    # --- 2. Static RCR-Merge (Maintains fixed coordinate system) ---
    coeffs_static = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_static = torch.optim.Adam([coeffs_static], lr=LR)
    for step in range(STEPS):
        optimizer_static.zero_grad()
        A_step = get_A_at_step(step)
        tta_loss = get_tta_objective_step(coeffs_static, A_step)
        
        diff_sq = (coeffs_static[:, 1:] - coeffs_static[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_init * diff_sq)
        anchor_penalty = torch.sum((coeffs_static - 0.3) ** 2)
        
        total_loss = tta_loss + beta_rcr_static * rcr_penalty + gamma_anchor * anchor_penalty
        total_loss.backward()
        optimizer_static.step()
        with torch.no_grad():
            coeffs_static.clamp_(0.0, 1.0)

    # --- 3. Threshold-Triggered Local Charting RCR-Merge (Ours) ---
    coeffs_triggered = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_triggered = torch.optim.Adam([coeffs_triggered], lr=LR)
    
    # Active state for the local chart
    anchor_center = torch.ones((K, L), dtype=torch.float32) * 0.3
    weight_active = weight_init.clone()
    beta_active = beta_rcr_static
    trigger_count = 0
    
    for step in range(STEPS):
        optimizer_triggered.zero_grad()
        A_step = get_A_at_step(step)
        tta_loss = get_tta_objective_step(coeffs_triggered, A_step)
        
        # Check if coordinate drift exceeds threshold: mean squared drift of coefficients
        with torch.no_grad():
            mean_sq_drift = torch.mean((coeffs_triggered - anchor_center) ** 2).item()
            if mean_sq_drift >= threshold:
                # Trip trigger! Establish a new local chart on the manifold
                trigger_count += 1
                anchor_center = coeffs_triggered.clone().detach()
                
                # Re-estimate the local FIM trace (with some small simulation estimation noise)
                c_step = torch.mean(A_step, dim=0)
                estimation_noise = torch.distributions.Normal(0, 0.05).sample((L,))
                c_estimated = torch.clamp(c_step + estimation_noise, 0.05, 5.0)
                weight_active = torch.sqrt(c_estimated[1:] * c_estimated[:-1])
                
                # Re-run GNB to scale beta for the new local chart
                with torch.enable_grad():
                    diff_sq_pert_t = (coeffs_pert[:, 1:] - coeffs_pert[:, :-1]) ** 2
                    rcr_penalty_pert_t = torch.sum(weight_active * diff_sq_pert_t)
                    grad_rcr_t = torch.autograd.grad(rcr_penalty_pert_t, coeffs_pert)[0]
                    norm_grad_rcr_t = torch.norm(grad_rcr_t).item()
                
                beta_active = 0.5 * norm_grad_tta / norm_grad_rcr_t
        
        diff_sq = (coeffs_triggered[:, 1:] - coeffs_triggered[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_active * diff_sq)
        anchor_penalty = torch.sum((coeffs_triggered - anchor_center) ** 2)
        
        total_loss = tta_loss + beta_active * rcr_penalty + gamma_anchor * anchor_penalty
        total_loss.backward()
        optimizer_triggered.step()
        with torch.no_grad():
            coeffs_triggered.clamp_(0.0, 1.0)

    acc_unconstrained = evaluate_accuracy(coeffs_unconstrained, use_euclidean=False)
    acc_unconstrained_eucl = evaluate_accuracy(coeffs_unconstrained, use_euclidean=True)
    acc_static_coupled = evaluate_accuracy(coeffs_static, use_euclidean=False)
    acc_triggered_coupled = evaluate_accuracy(coeffs_triggered, use_euclidean=False)
    
    acc_static_eucl = evaluate_accuracy(coeffs_static, use_euclidean=True)
    acc_triggered_eucl = evaluate_accuracy(coeffs_triggered, use_euclidean=True)

    return {
        'unconstrained': np.mean(acc_unconstrained),
        'unconstrained_eucl': np.mean(acc_unconstrained_eucl),
        'static_coupled': np.mean(acc_static_coupled),
        'triggered_coupled': np.mean(acc_triggered_coupled),
        'static_eucl': np.mean(acc_static_eucl),
        'triggered_eucl': np.mean(acc_triggered_eucl),
        'triggers': trigger_count
    }

print("Running 30-seed simulation of long-term adaptational stream with non-stationarity...")
unconstrained_res = []
unconstrained_eucl_res = []
static_coupled_res = []
triggered_coupled_res = []
static_eucl_res = []
triggered_eucl_res = []
trigger_counts = []

for s in range(1, NUM_SEEDS + 1):
    r = run_experiment_for_seed(s, drift_scale=0.4, threshold=0.03)
    unconstrained_res.append(r['unconstrained'])
    unconstrained_eucl_res.append(r['unconstrained_eucl'])
    static_coupled_res.append(r['static_coupled'])
    triggered_coupled_res.append(r['triggered_coupled'])
    static_eucl_res.append(r['static_eucl'])
    triggered_eucl_res.append(r['triggered_eucl'])
    trigger_counts.append(r['triggers'])

print(f"\n--- LONG-TERM adaptational STREAM EVALUATION (2000 steps, Drift=40%, Thresh=0.03) ---")
print(f"Unconstrained AdaMerging Coupled:   {np.mean(unconstrained_res):.4f}% +/- {np.std(unconstrained_res):.4f}%")
print(f"Static RCR-Merge Coupled:           {np.mean(static_coupled_res):.4f}% +/- {np.std(static_coupled_res):.4f}%")
print(f"Triggered RCR-Merge Coupled:        {np.mean(triggered_coupled_res):.4f}% +/- {np.std(triggered_coupled_res):.4f}%")
print(f"Unconstrained AdaMerging Euclidean: {np.mean(unconstrained_eucl_res):.4f}% +/- {np.std(unconstrained_eucl_res):.4f}%")
print(f"Static RCR-Merge Euclidean:         {np.mean(static_eucl_res):.4f}% +/- {np.std(static_eucl_res):.4f}%")
print(f"Triggered RCR-Merge Euclidean:      {np.mean(triggered_eucl_res):.4f}% +/- {np.std(triggered_eucl_res):.4f}%")
print(f"Average Trigger Events Tripped:     {np.mean(trigger_counts):.2f}")
