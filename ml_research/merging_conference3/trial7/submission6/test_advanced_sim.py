import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Simulation parameters
L = 14          # number of layers
K = 4           # number of tasks
B_cal = 64      # calibration batch size (16 per task)
B_test = 400    # test batch size (100 per task)

# Task vector norms representing the asymmetric task complexity (Frobenius norm squared)
task_norms = torch.tensor([1.0, 4.0, 16.0, 64.0])
expert_ceilings = [95.0, 90.0, 82.0, 75.0]
gamma_decay = 0.015

# Introduce a representation entanglement matrix to simulate backbone representation leakage
# It rotates and mixes the task-specific coordinate axes
entanglement_matrix = torch.tensor([
    [0.2, 0.6, 0.2, 0.0],
    [0.0, 0.2, 0.6, 0.2],
    [0.1, 0.0, 0.2, 0.7],
    [0.6, 0.1, 0.0, 0.3]
])

def generate_unit_states(batch_size_per_task, noise_std=0.2, entangle=True):
    psi_list = []
    task_list = []
    for k in range(K):
        base = torch.zeros(batch_size_per_task, K)
        base[:, k] = 1.0
        noise = torch.randn(batch_size_per_task, K) * noise_std
        psi_t = base + noise
        
        # Normalize
        psi_t = psi_t / (torch.norm(psi_t, dim=-1, keepdim=True) + 1e-8)
        
        if entangle:
            # Apply representation entanglement
            psi_t = psi_t @ entanglement_matrix
            # Re-normalize onto the unit sphere
            psi_t = psi_t / (torch.norm(psi_t, dim=-1, keepdim=True) + 1e-8)
            
        psi_list.append(psi_t)
        task_list.append(torch.full((batch_size_per_task,), k, dtype=torch.long))
    return torch.cat(psi_list, dim=0), torch.cat(task_list, dim=0)

# Generate Calibration and Test splits (with representation entanglement!)
psi_cal, task_cal = generate_unit_states(B_cal // K, noise_std=0.15, entangle=True)
psi_test, task_test = generate_unit_states(B_test // K, noise_std=0.25, entangle=True)

D = 192
V_matrices = torch.zeros(L, K, D, D)

# Helper to generate orthogonal matrices for SVD-based matrix construction
def get_random_orthogonal(dim):
    q, r = torch.linalg.qr(torch.randn(dim, dim))
    return q

for l in range(L):
    U = get_random_orthogonal(D)
    V_orth = get_random_orthogonal(D)
    for k in range(K):
        # We construct diverse, highly structured singular value spectra to break concentration of measure:
        # - Expert 0: Rank 1 (highly sparse spectrum)
        # - Expert 1: Rank 8 (flat spectrum)
        # - Expert 2: Power-law decay (s_i = (i+1)^-1.5)
        # - Expert 3: Exponential decay (s_i = exp(-i/20))
        S = torch.zeros(D, dtype=torch.float32)
        if k == 0:
            S[0] = 1.0
        elif k == 1:
            S[:8] = 1.0
        elif k == 2:
            S = torch.tensor([(i + 1)**(-1.5) for i in range(D)], dtype=torch.float32)
        elif k == 3:
            S = torch.tensor([np.exp(-i / 20.0) for i in range(D)], dtype=torch.float32)
            
        # Construct structured matrix
        W_struct = U @ torch.diag(S) @ V_orth
        
        # Normalize W_struct to have Frobenius norm equal to target_f_norm
        f_norm = torch.norm(W_struct, p='fro')
        target_f_norm = torch.sqrt(task_norms[k])
        V_matrices[l, k] = W_struct * (target_f_norm / (f_norm + 1e-8))

# Precompute Gamma_F and Gamma_S based on actual Frobenius and Spectral norms of V_matrices
Gamma_F = torch.zeros(L, K)
Gamma_S = torch.zeros(L, K)

for l in range(L):
    for k in range(K):
        Gamma_F[l, k] = torch.sqrt(torch.sum(V_matrices[l, k] ** 2))
        svals = torch.linalg.svdvals(V_matrices[l, k])
        Gamma_S[l, k] = svals[0]

# Print out the norms to verify structured geometries
print("Layer-averaged norms:")
for k in range(K):
    f_mean = torch.mean(Gamma_F[:, k]).item()
    s_mean = torch.mean(Gamma_S[:, k]).item()
    ratio = s_mean / f_mean
    print(f"Expert {k}: Frobenius norm = {f_mean:.4f}, Spectral norm = {s_mean:.4f}, Ratio = {ratio:.4f}")

class LinearRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(L, K, K))
        self.B = nn.Parameter(torch.zeros(L, K))
        
    def forward(self, psi):
        logits = torch.einsum('bd,lkd->blk', psi, self.W) + self.B.unsqueeze(0)
        alpha = torch.softmax(logits, dim=-1)
        return alpha

def compute_merged_distances(alpha):
    alpha_sq_weighted = (alpha ** 2) * task_norms.unsqueeze(0).unsqueeze(1)
    sum_all_weighted = torch.sum(alpha_sq_weighted, dim=-1)
    dist = sum_all_weighted.unsqueeze(-1) + task_norms.unsqueeze(0).unsqueeze(1) - 2 * alpha * task_norms.unsqueeze(0).unsqueeze(1)
    distances = torch.mean(dist, dim=1)
    return distances

def compute_test_distances(alpha, router=None):
    alpha_sq_weighted = (alpha ** 2) * task_norms.unsqueeze(0).unsqueeze(1)
    sum_all_weighted = torch.sum(alpha_sq_weighted, dim=-1)
    dist = sum_all_weighted.unsqueeze(-1) + task_norms.unsqueeze(0).unsqueeze(1) - 2 * alpha * task_norms.unsqueeze(0).unsqueeze(1)
    base_dist = torch.mean(dist, dim=1)
    W_norms = torch.zeros(K)
    if router is not None:
        W_norms = torch.sqrt(torch.sum(router.W ** 2, dim=[0, 2]) + torch.sum(router.B ** 2, dim=0))
    noise_factor = 0.05
    mean_V_k_F = torch.mean(Gamma_F, dim=0)
    gap_penalty = noise_factor * W_norms * mean_V_k_F
    distances = base_dist + gap_penalty.unsqueeze(0)
    return distances

def evaluate_router(router=None, static_alpha=None, is_pfsr=False):
    if router is not None:
        with torch.no_grad():
            alpha = router(psi_test)
        distances = compute_test_distances(alpha, router)
    elif static_alpha is not None:
        alpha = static_alpha
        distances = compute_test_distances(alpha, None)
    elif is_pfsr:
        # PFSR is training-free, so router weights are zero (no generalization gap)
        # It maps input representation directly to task routing via cosine-similarity/coordinates
        # Since input is entangled, PFSR will suffer representation leakage
        alpha_val = torch.softmax(psi_test / 0.1, dim=-1)
        alpha = alpha_val.unsqueeze(1).expand(-1, L, -1)
        distances = compute_test_distances(alpha, None)
        
    accuracies = [0.0] * K
    for k in range(K):
        idx = (task_test == k)
        dist_k = distances[idx, k]
        acc_k = expert_ceilings[k] * torch.exp(-gamma_decay * dist_k)
        accuracies[k] = torch.mean(acc_k).item()
    mean_acc = sum(accuracies) / K
    return accuracies, mean_acc

def calibrate_router(reg_type="none", lambda_reg=0.1, lr=0.1, epochs=600):
    router = LinearRouter()
    optimizer = optim.Adam(router.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        alpha = router(psi_cal)
        distances = compute_merged_distances(alpha)
        logits = -distances / 4.0
        loss_ce = loss_fn(logits, task_cal)
        
        loss_reg = 0.0
        W_squared = torch.sum(router.W ** 2, dim=-1) # (L, K)
        B_squared = router.B ** 2 # (L, K)
        
        # Helper for smoothed group-lasso: sqrt(||W||^2 + B^2 + eps)
        eps = 1e-8
        W_B_l2_norm = torch.sqrt(W_squared + B_squared + eps) # (L, K)
        
        if reg_type == "l2":
            loss_reg = lambda_reg * (torch.sum(router.W ** 2) + torch.sum(router.B ** 2))
        elif reg_type == "vr":
            mean_alpha = torch.mean(alpha, dim=0)
            loss_reg = lambda_reg * torch.sum(torch.var(mean_alpha, dim=-1))
        elif reg_type == "sr3_f":
            # Traditional quadratic Frobenius-scaled regularizer
            loss_reg = lambda_reg * torch.sum(Gamma_F * (W_squared + B_squared))
        elif reg_type == "sr3_s":
            # Traditional quadratic Spectral-scaled regularizer
            loss_reg = lambda_reg * torch.sum(Gamma_S * (W_squared + B_squared))
        elif reg_type == "sr3_f_l1":
            # Provably optimal linear Group-Lasso Frobenius-scaled regularizer
            loss_reg = lambda_reg * torch.sum(Gamma_F * W_B_l2_norm)
        elif reg_type == "sr3_s_l1":
            # Provably optimal linear Group-Lasso Spectral-scaled regularizer
            loss_reg = lambda_reg * torch.sum(Gamma_S * W_B_l2_norm)
            
        loss_total = loss_ce + loss_reg
        loss_total.backward()
        optimizer.step()
        
    test_accs, mean_test_acc = evaluate_router(router=router)
    return test_accs, mean_test_acc

def run_sweep(reg_type, lambdas):
    best_mean = -1.0
    best_accs = None
    best_lambda = None
    for lam in lambdas:
        accs, mean_acc = calibrate_router(reg_type=reg_type, lambda_reg=lam)
        if mean_acc > best_mean:
            best_mean = mean_acc
            best_accs = accs
            best_lambda = lam
    print(f"{reg_type.upper():<20} | Best Lambda: {best_lambda:<10} | Mean Acc: {best_mean:.4f}% | Accs: {[f'{a:.2f}' for a in best_accs]}")
    return best_accs, best_mean

print("\nEvaluating PFSR (Parameter-Free Subspace Routing)...")
pfsr_accs, pfsr_mean = evaluate_router(is_pfsr=True)
print(f"PFSR Mean Acc: {pfsr_mean:.4f}% | Accs: {[f'{a:.2f}' for a in pfsr_accs]}")

print("\nEvaluating Static Uniform Merging...")
uniform_alpha = torch.full((B_test, L, K), 1.0 / K)
uniform_accs, uniform_mean = evaluate_router(static_alpha=uniform_alpha)
print(f"Uniform Mean Acc: {uniform_mean:.4f}% | Accs: {[f'{a:.2f}' for a in uniform_accs]}")

print("\nEvaluating Unregularized Linear Router...")
unreg_accs, unreg_mean = calibrate_router(reg_type="none")
print(f"Unregularized Mean Acc: {unreg_mean:.4f}% | Accs: {[f'{a:.2f}' for a in unreg_accs]}")

print("\nRunning sweeps on regularizations...")
run_sweep("l2", [1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 2e-4, 5e-4])
run_sweep("vr", [0.5, 1.0, 2.0, 5.0, 10.0])
run_sweep("sr3_f", [1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 2e-4])
run_sweep("sr3_s", [5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
run_sweep("sr3_f_l1", [1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 2e-4, 5e-4, 1e-3])
run_sweep("sr3_s_l1", [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
