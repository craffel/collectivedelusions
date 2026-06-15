import numpy as np
import torch
import math

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

D_in = 192
D_out = 192
P = D_in * D_out
p_sparsity = 0.05 # 5% critical coordinates per task

def generate_uncorrelated_task_vectors(K):
    # Independent Gaussian task vectors
    task_vectors = []
    for _ in range(K):
        V_k = torch.randn(D_out, D_in) / math.sqrt(D_in)
        task_vectors.append(V_k)
    return task_vectors

def generate_correlated_task_vectors(K, rank=10):
    # Correlated task vectors sharing a low-rank/low-dimensional subspace
    # Base shared directions
    U_shared = torch.randn(D_out, rank)
    V_shared = torch.randn(D_in, rank)
    
    task_vectors = []
    for _ in range(K):
        # Each task is a combination of shared subspace + a small independent perturbation
        task_coeffs = torch.randn(rank, 1)
        V_k_shared = torch.matmul(U_shared, torch.matmul(torch.diag(task_coeffs.squeeze()), V_shared.t()))
        V_k_pert = torch.randn(D_out, D_in) / math.sqrt(D_in) * 0.1
        task_vectors.append(V_k_shared + V_k_pert)
    return task_vectors

def compute_union_sparsity(task_vectors, p):
    # Compute individual masks
    masks = []
    for V_k in task_vectors:
        abs_V_k = torch.abs(V_k)
        threshold = torch.quantile(abs_V_k, 1.0 - p)
        mask_k = (abs_V_k >= threshold).float()
        masks.append(mask_k)
    
    # Compute union mask
    union_mask = torch.clamp(torch.sum(torch.stack(masks), dim=0), 0.0, 1.0)
    union_size = torch.sum(union_mask).item()
    total_elements = union_mask.numel()
    return union_size / total_elements

K_list = [1, 2, 4, 8, 12, 16]

print("Running Shared Union Gating Scalability Simulation (Sparsity Target = 5.0%):")
print(f"{'K':<5} | {'Uncorrelated Union (%)':<24} | {'Correlated Union (%)':<22} | {'Linear Scaling Bound (K*p) (%)':<30}")
print("-" * 90)

for K in K_list:
    uncorrelated_vectors = generate_uncorrelated_task_vectors(K)
    correlated_vectors = generate_correlated_task_vectors(K)
    
    uncor_ratio = compute_union_sparsity(uncorrelated_vectors, p_sparsity) * 100.0
    cor_ratio = compute_union_sparsity(correlated_vectors, p_sparsity) * 100.0
    linear_bound = min(100.0, K * p_sparsity * 100.0)
    
    print(f"{K:<5} | {uncor_ratio:<24.2f}% | {cor_ratio:<22.2f}% | {linear_bound:<30.2f}%")
