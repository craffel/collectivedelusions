import numpy as np
import torch
import torch.nn as nn
import json

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_random_orthogonal_matrix(d):
    # Generate a random orthogonal matrix using QR decomposition
    A = torch.randn(d, d)
    Q, R = torch.linalg.qr(A)
    # Ensure determinant is 1 (proper rotation)
    dists = torch.diag(R)
    ph = dists / torch.abs(dists)
    Q = Q * ph
    return Q

def generate_task_data_misaligned(d=128, num_tasks=5, rotation_scale=0.1):
    # Base pretrained model weights
    W0 = torch.randn(d, d) / np.sqrt(d)
    
    tasks = []
    for i in range(num_tasks):
        # Unaligned expert weights
        U = torch.randn(d, 10) / np.sqrt(d)
        V = torch.randn(d, 10) / np.sqrt(d)
        task_vector = torch.matmul(U, V.T) + 0.05 * torch.randn(d, d)
        
        # Aligned target
        W_target_aligned = W0 + task_vector
        
        # Introduce a task-specific rotation matrix P_i
        # Since we want to control the amount of rotation, we can interpolate P_i with Identity
        P = generate_random_orthogonal_matrix(d)
        # Interpolate in Lie algebra or just take a small rotation
        # Let's make it a small rotation: P = exp(rotation_scale * skew)
        skew = torch.randn(d, d)
        skew = 0.5 * (skew - skew.T)
        P = torch.linalg.matrix_exp(rotation_scale * skew)
        
        # The actual expert weight that we observe (which is rotated/misaligned)
        W_target = torch.matmul(P, W_target_aligned)
        
        # Diagonal Fisher Information (power-law decay)
        F_log = torch.randn(d, d) * 1.5 + 2.0
        F = torch.exp(F_log)
        F = F / F.mean()
        
        tasks.append({
            "W_target": W_target, # Observational expert weights
            "F": F,
            "P": P, # Ground truth rotation
            "W_target_aligned": W_target_aligned
        })
        
    return W0, tasks

from run_experiments import compute_loss, solve_standard_procrustes, solve_weighted_procrustes, cayley_transform, inv_cayley_transform

def run_experiment_seed(seed, d=128, num_tasks=5, rotation_scale=0.1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    W0, tasks = generate_task_data_misaligned(d, num_tasks, rotation_scale)
    
    # --- 1. Pretrained Baseline ---
    losses_pretrained = [compute_loss(W0, t["W_target"], t["F"]) for t in tasks]
    
    # --- 2. Task Arithmetic (Euclidean averaging) ---
    task_vectors = [t["W_target"] - W0 for t in tasks]
    W_ta = W0 + sum(task_vectors) / num_tasks
    losses_ta = [compute_loss(W_ta, t["W_target"], t["F"]) for t in tasks]
    
    # --- 3. Ties Merging Baseline ---
    pruned_vectors = []
    for tv in task_vectors:
        flat_tv = tv.flatten()
        threshold = torch.quantile(torch.abs(flat_tv), 0.8)
        mask = torch.abs(tv) >= threshold
        pruned_vectors.append(tv * mask)
    W_ties = W0 + sum(pruned_vectors) / num_tasks
    losses_ties = [compute_loss(W_ties, t["W_target"], t["F"]) for t in tasks]
    
    # --- 4. OrthoMerge Baseline ---
    R_list_om = []
    rho_list_om = []
    for t in tasks:
        R = solve_standard_procrustes(t["W_target"], W0)
        R_list_om.append(R)
        rho_list_om.append(t["W_target"] - torch.matmul(R, W0))
        
    # Merge rotations in Lie algebra (no-scaling vs scaling)
    # We will try both and see which works better
    for use_scaling in [True, False]:
        Q_list_om = [cayley_transform(R) for R in R_list_om]
        Q_sum_om = sum(Q_list_om)
        if use_scaling:
            norms_om = sum([torch.linalg.matrix_norm(Q, 'fro') for Q in Q_list_om])
            sum_norm_om = torch.linalg.matrix_norm(Q_sum_om, 'fro')
            scale_om = norms_om / (sum_norm_om + 1e-6)
        else:
            scale_om = 1.0
        Q_merged_om = scale_om * (Q_sum_om / num_tasks)
        R_merged_om = inv_cayley_transform(Q_merged_om)
        
        rho_merged_om = sum(rho_list_om) / num_tasks
        W_om = torch.matmul(R_merged_om, W0) + rho_merged_om
        losses_om = [compute_loss(W_om, t["W_target"], t["F"]) for t in tasks]
        if use_scaling:
            loss_om_scaled = np.mean(losses_om)
        else:
            loss_om_unscaled = np.mean(losses_om)
    
    # --- 5. FOMM (Ours) ---
    R_list_fomm = []
    rho_list_fomm = []
    for t in tasks:
        R = solve_weighted_procrustes(t["W_target"], W0, t["F"], lr=0.02, num_iters=200)
        R_list_fomm.append(R)
        rho_list_fomm.append(t["W_target"] - torch.matmul(R, W0))
        
    for use_scaling in [True, False]:
        Q_list_fomm = [cayley_transform(R) for R in R_list_fomm]
        Q_sum_fomm = sum(Q_list_fomm)
        if use_scaling:
            norms_fomm = sum([torch.linalg.matrix_norm(Q, 'fro') for Q in Q_list_fomm])
            sum_norm_fomm = torch.linalg.matrix_norm(Q_sum_fomm, 'fro')
            scale_fomm = norms_fomm / (sum_norm_fomm + 1e-6)
        else:
            scale_fomm = 1.0
        Q_merged_fomm = scale_fomm * (Q_sum_fomm / num_tasks)
        R_merged_fomm = inv_cayley_transform(Q_merged_fomm)
        
        rho_merged_fomm = sum(rho_list_fomm) / num_tasks
        W_fomm = torch.matmul(R_merged_fomm, W0) + rho_merged_fomm
        losses_fomm = [compute_loss(W_fomm, t["W_target"], t["F"]) for t in tasks]
        if use_scaling:
            loss_fomm_scaled = np.mean(losses_fomm)
        else:
            loss_fomm_unscaled = np.mean(losses_fomm)
            
    return {
        "pretrained": np.mean(losses_pretrained),
        "ta": np.mean(losses_ta),
        "ties": np.mean(losses_ties),
        "orthomerge_scaled": loss_om_scaled,
        "orthomerge_unscaled": loss_om_unscaled,
        "fomm_scaled": loss_fomm_scaled,
        "fomm_unscaled": loss_fomm_unscaled
    }

print("Evaluating with rotation_scale = 0.05")
print(run_experiment_seed(42, rotation_scale=0.05))

print("\nEvaluating with rotation_scale = 0.1")
print(run_experiment_seed(42, rotation_scale=0.1))

print("\nEvaluating with rotation_scale = 0.2")
print(run_experiment_seed(42, rotation_scale=0.2))
