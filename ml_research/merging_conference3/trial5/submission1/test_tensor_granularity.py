import torch
import torch.nn as nn
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

L = 12       # Number of blocks
D_dim = 24   # 24 total weight groups (12 attention, 12 MLP)
K = 4        # Number of tasks
NUM_SEEDS = 30
LR = 0.08    # Tuned learning rate
STEPS = 100
LAMBDA_RUG = 0.05
F_RUG = 15.0

BASE_ACC = torch.tensor([94.68, 82.71, 94.04, 78.37], dtype=torch.float32)
DELTA_ACC = torch.tensor([5.32, 17.29, 5.96, 21.63], dtype=torch.float32)

# Base block-wise sensitivity scales
s_l = torch.tensor([0.6]*4 + [1.0]*6 + [1.6]*2, dtype=torch.float32)

# Tensor-wise sensitivity scales (Attention is 1.5x more sensitive, MLP is 0.5x)
s_prime = torch.zeros(D_dim, dtype=torch.float32)
for l in range(L):
    s_prime[2*l] = s_l[l] * 1.5       # Attention
    s_prime[2*l+1] = s_l[l] * 0.5     # MLP

# Build 24x24 Covariance matrices
Sigma_T = torch.zeros((D_dim, D_dim), dtype=torch.float32)
for i in range(D_dim):
    for j in range(D_dim):
        # High-frequency spatial coupling across weight groups
        Sigma_T[i, j] = torch.sqrt(s_prime[i] * s_prime[j]) * (0.5 ** (abs(i - j) / 2.0))
Sigma_T_inv = torch.inverse(Sigma_T)
Sigma_T_id_inv = torch.eye(D_dim)

def get_optimal_profile_blockwise(k, l):
    # Base block-wise optimal profile (same as stepwise/modular)
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
    
    # 24-dimensional sensitivity matrices
    A = torch.zeros((K, D_dim), dtype=torch.float32)
    for k in range(K):
        for l in range(L):
            # Attention (2*l) is peaky, MLP (2*l+1) is flatter
            if l <= 3 or l >= 10:
                A[k, 2*l] = torch.distributions.Uniform(1.2, 1.8).sample()     # Peaky Attention
                A[k, 2*l+1] = torch.distributions.Uniform(0.4, 0.6).sample()   # Robust MLP
            else:
                A[k, 2*l] = torch.distributions.Uniform(0.3, 0.5).sample()     # Inner Attention
                A[k, 2*l+1] = torch.distributions.Uniform(0.1, 0.2).sample()   # Inner MLP
    
    B = 0.5 * A
    c = torch.mean(A, dim=0) # Curvatures for all 24 weight groups
    
    # Target profile (Attention and MLP share optimal targets within each block)
    alpha_opt = torch.zeros((K, D_dim), dtype=torch.float32)
    for k in range(K):
        for l in range(L):
            opt_val = get_optimal_profile_blockwise(k, l)
            # Both get slightly independent perturbation noise
            eps_att = torch.distributions.Normal(0, 0.02).sample()
            eps_mlp = torch.distributions.Normal(0, 0.02).sample()
            alpha_opt[k, 2*l] = torch.clamp(torch.tensor(opt_val) + eps_att, 0.0, 1.0)
            alpha_opt[k, 2*l+1] = torch.clamp(torch.tensor(opt_val) + eps_mlp, 0.0, 1.0)
            
    D = torch.zeros((K, K, D_dim), dtype=torch.float32)
    for i in range(D_dim):
        D[0, 1, i] = D[1, 0, i] = torch.distributions.Uniform(0.01, 0.03).sample()
        D[2, 3, i] = D[3, 2, i] = torch.distributions.Uniform(0.20, 0.30).sample()
        D[0, 3, i] = D[3, 0, i] = torch.distributions.Uniform(0.12, 0.18).sample()
        D[1, 2, i] = D[2, 1, i] = torch.distributions.Uniform(0.15, 0.21).sample()
        D[0, 2, i] = D[2, 0, i] = torch.distributions.Uniform(0.10, 0.15).sample()
        D[1, 3, i] = D[3, 1, i] = torch.distributions.Uniform(0.10, 0.15).sample()

    # Transductive adaptation noise
    eta = torch.distributions.Normal(0, 0.05).sample((K, D_dim))
    
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
        inv_m = Sigma_T_id_inv if use_euclidean else Sigma_T_inv
        dist_opt = torch.sum(d * torch.matmul(d, inv_m), dim=1)
        dist_init = torch.sum(d_init * torch.matmul(d_init, inv_m), dim=1)
        accs = BASE_ACC + DELTA_ACC * (1.0 - dist_opt / dist_init)
        accs = torch.clamp(accs, 0.0, 100.0)
        return accs.tolist()

    # Weight matrices for spatial TV penalties
    weight_prime = torch.sqrt(c[1:] * c[:-1]) # Adjacent weights on all 24 components

    # --- 1. Uniform Baseline ---
    coeffs_uni = torch.ones((K, D_dim), dtype=torch.float32) * 0.3
    acc_uni_coupled = evaluate_accuracy(coeffs_uni, use_euclidean=False)
    acc_uni_eucl = evaluate_accuracy(coeffs_uni, use_euclidean=True)

    # --- 2. Unconstrained AdaMerging ---
    coeffs_ada = nn.Parameter(torch.ones((K, D_dim), dtype=torch.float32) * 0.3)
    optimizer_ada = torch.optim.Adam([coeffs_ada], lr=LR)
    for _ in range(STEPS):
        optimizer_ada.zero_grad()
        loss = get_tta_objective_vectorized(coeffs_ada)
        loss.backward()
        optimizer_ada.step()
        with torch.no_grad():
            coeffs_ada.clamp_(0.0, 1.0)
    acc_ada_coupled = evaluate_accuracy(coeffs_ada, use_euclidean=False)
    acc_ada_eucl = evaluate_accuracy(coeffs_ada, use_euclidean=True)

    # --- 3. Layer-wise Scalar RCR-Merge ---
    # Coefficients are forced to be identical between Attention and MLP within each block
    coeffs_layer_scalar = nn.Parameter(torch.ones((K, L), dtype=torch.float32) * 0.3)
    optimizer_layer = torch.optim.Adam([coeffs_layer_scalar], lr=LR)
    
    # Curvatures for layer-wise scalar setup
    c_layer = torch.zeros(L, dtype=torch.float32)
    for l in range(L):
        # Average sensitivity of the block
        c_layer[l] = 0.5 * (c[2*l] + c[2*l+1])
    weight_layer = torch.sqrt(c_layer[1:] * c_layer[:-1])

    # Gradient Norm Balancing for Layer-wise Scalar
    delta_gnb = 0.05
    pert_pattern_L = torch.tensor([(-1.0)**li for li in range(L)], dtype=torch.float32)
    coeffs_pert_L = 0.3 + delta_gnb * pert_pattern_L.unsqueeze(0).repeat(K, 1)
    coeffs_pert_L.requires_grad_(True)
    diff_sq_pert_L = (coeffs_pert_L[:, 1:] - coeffs_pert_L[:, :-1]) ** 2
    rcr_penalty_pert_L = torch.sum(weight_layer * diff_sq_pert_L)
    grad_rcr_L = torch.autograd.grad(rcr_penalty_pert_L, coeffs_pert_L)[0]
    norm_grad_rcr_L = torch.norm(grad_rcr_L).item()

    with torch.enable_grad():
        coeffs_init_L = torch.ones((K, L), dtype=torch.float32) * 0.3
        coeffs_init_L.requires_grad_(True)
        # Re-project to 24-dim to evaluate TTA objective
        coeffs_init_L_24 = coeffs_init_L.repeat_interleave(2, dim=1)
        tta_loss_init_L = get_tta_objective_vectorized(coeffs_init_L_24)
        grad_tta_L = torch.autograd.grad(tta_loss_init_L, coeffs_init_L)[0]
        norm_grad_tta_L = torch.norm(grad_tta_L).item()

    beta_rcr_L = 0.5 * norm_grad_tta_L / norm_grad_rcr_L
    
    # Auto Anchor weight scaling for layer-wise scalar
    epsilon_gnb = 0.1
    coeffs_drift_L = 0.3 + epsilon_gnb * torch.ones((K, L), dtype=torch.float32)
    coeffs_drift_L.requires_grad_(True)
    anchor_penalty_drift_L = torch.sum((coeffs_drift_L - 0.3) ** 2)
    grad_anchor_L = torch.autograd.grad(anchor_penalty_drift_L, coeffs_drift_L)[0]
    norm_grad_anchor_L = torch.norm(grad_anchor_L).item()
    gamma_anchor_L = (0.0003 * norm_grad_tta_L / norm_grad_anchor_L)

    for _ in range(STEPS):
        optimizer_layer.zero_grad()
        # Expand 12 coefficients to 24-dim
        coeffs_expanded = coeffs_layer_scalar.repeat_interleave(2, dim=1)
        tta_loss = get_tta_objective_vectorized(coeffs_expanded)
        
        diff_sq = (coeffs_layer_scalar[:, 1:] - coeffs_layer_scalar[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_layer * diff_sq)
        anchor_penalty = torch.sum((coeffs_layer_scalar - 0.3) ** 2)
        
        total_loss = tta_loss + beta_rcr_L * rcr_penalty + gamma_anchor_L * anchor_penalty
        total_loss.backward()
        optimizer_layer.step()
        with torch.no_grad():
            coeffs_layer_scalar.clamp_(0.0, 1.0)
            
    coeffs_layer_final = coeffs_layer_scalar.repeat_interleave(2, dim=1)
    acc_layer_coupled = evaluate_accuracy(coeffs_layer_final, use_euclidean=False)
    acc_layer_eucl = evaluate_accuracy(coeffs_layer_final, use_euclidean=True)

    # --- 4. Tensor-wise Granularity RCR-Merge ---
    # Coefficients are separate for Attention and MLP, and optimized independently (24-dim)
    coeffs_tensor = nn.Parameter(torch.ones((K, D_dim), dtype=torch.float32) * 0.3)
    optimizer_tensor = torch.optim.Adam([coeffs_tensor], lr=LR)

    # Gradient Norm Balancing for Tensor-wise
    pert_pattern_T = torch.tensor([(-1.0)**ti for ti in range(D_dim)], dtype=torch.float32)
    coeffs_pert_T = 0.3 + delta_gnb * pert_pattern_T.unsqueeze(0).repeat(K, 1)
    coeffs_pert_T.requires_grad_(True)
    diff_sq_pert_T = (coeffs_pert_T[:, 1:] - coeffs_pert_T[:, :-1]) ** 2
    rcr_penalty_pert_T = torch.sum(weight_prime * diff_sq_pert_T)
    grad_rcr_T = torch.autograd.grad(rcr_penalty_pert_T, coeffs_pert_T)[0]
    norm_grad_rcr_T = torch.norm(grad_rcr_T).item()

    with torch.enable_grad():
        coeffs_init_T = torch.ones((K, D_dim), dtype=torch.float32) * 0.3
        coeffs_init_T.requires_grad_(True)
        tta_loss_init_T = get_tta_objective_vectorized(coeffs_init_T)
        grad_tta_T = torch.autograd.grad(tta_loss_init_T, coeffs_init_T)[0]
        norm_grad_tta_T = torch.norm(grad_tta_T).item()

    beta_rcr_T = 0.5 * norm_grad_tta_T / norm_grad_rcr_T

    # Auto Anchor weight scaling for Tensor-wise
    coeffs_drift_T = 0.3 + epsilon_gnb * torch.ones((K, D_dim), dtype=torch.float32)
    coeffs_drift_T.requires_grad_(True)
    anchor_penalty_drift_T = torch.sum((coeffs_drift_T - 0.3) ** 2)
    grad_anchor_T = torch.autograd.grad(anchor_penalty_drift_T, coeffs_drift_T)[0]
    norm_grad_anchor_T = torch.norm(grad_anchor_T).item()
    gamma_anchor_T = (0.0003 * norm_grad_tta_T / norm_grad_anchor_T)

    for _ in range(STEPS):
        optimizer_tensor.zero_grad()
        tta_loss = get_tta_objective_vectorized(coeffs_tensor)
        
        diff_sq = (coeffs_tensor[:, 1:] - coeffs_tensor[:, :-1]) ** 2
        rcr_penalty = torch.sum(weight_prime * diff_sq)
        anchor_penalty = torch.sum((coeffs_tensor - 0.3) ** 2)
        
        total_loss = tta_loss + beta_rcr_T * rcr_penalty + gamma_anchor_T * anchor_penalty
        total_loss.backward()
        optimizer_tensor.step()
        with torch.no_grad():
            coeffs_tensor.clamp_(0.0, 1.0)
            
    acc_tensor_coupled = evaluate_accuracy(coeffs_tensor, use_euclidean=False)
    acc_tensor_eucl = evaluate_accuracy(coeffs_tensor, use_euclidean=True)

    return {
        'uni_coupled': np.mean(acc_uni_coupled),
        'uni_eucl': np.mean(acc_uni_eucl),
        'ada_coupled': np.mean(acc_ada_coupled),
        'ada_eucl': np.mean(acc_ada_eucl),
        'layer_coupled': np.mean(acc_layer_coupled),
        'layer_eucl': np.mean(acc_layer_eucl),
        'tensor_coupled': np.mean(acc_tensor_coupled),
        'tensor_eucl': np.mean(acc_tensor_eucl)
    }

results = {
    'uni_coupled': [], 'uni_eucl': [],
    'ada_coupled': [], 'ada_eucl': [],
    'layer_coupled': [], 'layer_eucl': [],
    'tensor_coupled': [], 'tensor_eucl': []
}

for seed in range(1, NUM_SEEDS + 1):
    r = run_experiment_for_seed(seed)
    for k in results:
        results[k].append(r[k])

print("\n=== TENSOR-WISE VS LAYER-WISE GRANULARITY SIMULATION (30 SEEDS) ===")
print(f"Uniform Coupled:               {np.mean(results['uni_coupled']):.4f}% +/- {np.std(results['uni_coupled']):.4f}%")
print(f"Unconstrained Ada Coupled:    {np.mean(results['ada_coupled']):.4f}% +/- {np.std(results['ada_coupled']):.4f}%")
print(f"L-RCR-Merge Coupled (Layer):   {np.mean(results['layer_coupled']):.4f}% +/- {np.std(results['layer_coupled']):.4f}%")
print(f"T-RCR-Merge Coupled (Tensor):  {np.mean(results['tensor_coupled']):.4f}% +/- {np.std(results['tensor_coupled']):.4f}%")
print("-" * 60)
print(f"Uniform Euclidean:             {np.mean(results['uni_eucl']):.4f}% +/- {np.std(results['uni_eucl']):.4f}%")
print(f"Unconstrained Ada Euclidean:  {np.mean(results['ada_eucl']):.4f}% +/- {np.std(results['ada_eucl']):.4f}%")
print(f"L-RCR-Merge Euclidean (Layer): {np.mean(results['layer_eucl']):.4f}% +/- {np.std(results['layer_eucl']):.4f}%")
print(f"T-RCR-Merge Euclidean (Tensor):{np.mean(results['tensor_eucl']):.4f}% +/- {np.std(results['tensor_eucl']):.4f}%")
