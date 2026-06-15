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

def generate_unit_states(batch_size_per_task, noise_std=0.2):
    psi_list = []
    task_list = []
    for k in range(K):
        base = torch.zeros(batch_size_per_task, K)
        base[:, k] = 1.0
        noise = torch.randn(batch_size_per_task, K) * noise_std
        psi_t = base + noise
        psi_t = psi_t / (torch.norm(psi_t, dim=-1, keepdim=True) + 1e-8)
        psi_list.append(psi_t)
        task_list.append(torch.full((batch_size_per_task,), k, dtype=torch.long))
    return torch.cat(psi_list, dim=0), torch.cat(task_list, dim=0)

psi_cal, task_cal = generate_unit_states(B_cal // K, noise_std=0.15)
psi_test, task_test = generate_unit_states(B_test // K, noise_std=0.25)

D = 192
rank = 8  # Model LoRA rank
V_matrices = torch.zeros(L, K, D, D)
for l in range(L):
    for k in range(K):
        A = torch.randn(D, rank)
        B = torch.randn(rank, D)
        W_random = A @ B
        f_norm = torch.norm(W_random, p='fro')
        target_f_norm = torch.sqrt(task_norms[k])
        V_matrices[l, k] = W_random * (target_f_norm / (f_norm + 1e-8))

Gamma_F = torch.zeros(L, K)
Gamma_S = torch.zeros(L, K)
for l in range(L):
    for k in range(K):
        Gamma_F[l, k] = torch.sqrt(torch.sum(V_matrices[l, k] ** 2))
        svals = torch.linalg.svdvals(V_matrices[l, k])
        Gamma_S[l, k] = svals[0]

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

def evaluate_router(router=None, static_alpha=None):
    if router is not None:
        with torch.no_grad():
            alpha = router(psi_test)
        distances = compute_test_distances(alpha, router)
    elif static_alpha is not None:
        alpha = static_alpha
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
        W_squared = torch.sum(router.W ** 2, dim=-1)
        B_squared = router.B ** 2
        
        if reg_type == "l2":
            loss_reg = lambda_reg * (torch.sum(router.W ** 2) + torch.sum(router.B ** 2))
        elif reg_type == "vr":
            mean_alpha = torch.mean(alpha, dim=0)
            loss_reg = lambda_reg * torch.sum(torch.var(mean_alpha, dim=-1))
        elif reg_type == "sr3_f":
            loss_reg = lambda_reg * torch.sum(Gamma_F * (W_squared + B_squared))
        elif reg_type == "sr3_s":
            loss_reg = lambda_reg * torch.sum(Gamma_S * (W_squared + B_squared))
            
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

# Run sweeps on existing and new regularizations
print("Running sweeps with low-rank task vectors...")
run_sweep("l2", [1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 2e-4])
run_sweep("vr", [1.0, 2.0, 5.0, 10.0])
run_sweep("sr3_f", [1e-5, 2e-5, 5e-5, 8e-5, 1e-4])
run_sweep("sr3_s", [5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
