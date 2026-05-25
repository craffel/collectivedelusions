import numpy as np
import torch
import torch.nn as nn
import json

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_task_data(d=128, num_tasks=5):
    # Base pretrained model weights
    W0 = torch.randn(d, d) / np.sqrt(d)
    
    tasks = []
    for i in range(num_tasks):
        # Expert weights are base weights + a task-specific update
        # To make it realistic, the task vector is low-rank + some sparse noise
        U = torch.randn(d, 10) / np.sqrt(d)
        V = torch.randn(d, 10) / np.sqrt(d)
        task_vector = torch.matmul(U, V.T) + 0.05 * torch.randn(d, d)
        W_target = W0 + task_vector
        
        # Diagonal Fisher Information represents parameter sensitivity
        # Realistic Fisher values follow a log-normal (power-law decay) distribution
        F_log = torch.randn(d, d) * 1.5 + 2.0  # Mean=2.0, SD=1.5
        F = torch.exp(F_log)
        F = F / F.mean()  # Normalize
        
        tasks.append({
            "W_target": W_target,
            "F": F
        })
        
    return W0, tasks

def compute_loss(W, W_target, F):
    # Quadratic approximation of the task loss based on Fisher Information
    loss = 0.5 * torch.sum(F * (W - W_target)**2)
    return loss.item()

def solve_standard_procrustes(W_target, W0):
    # Standard Orthogonal Procrustes: min_R ||W_target - R W0||_F^2 s.t. R^T R = I
    # Analytical solution via SVD: R = U V^T where U S V^T = SVD(W_target W0^T)
    M = torch.matmul(W_target, W0.T)
    U, S, Vh = torch.linalg.svd(M)
    R = torch.matmul(U, Vh)
    return R

def solve_weighted_procrustes(W_target, W0, F, lr=0.01, num_iters=100):
    # Weighted Orthogonal Procrustes: min_R ||(W_target - R W0) * F^0.5||_F^2 s.t. R^T R = I
    # We solve this using Riemannian Gradient Descent on the Orthogonal Group O(d)
    d = W0.shape[0]
    
    # Initialize with standard Procrustes to start close to a good solution
    R = solve_standard_procrustes(W_target, W0).clone()
    
    for i in range(num_iters):
        # 1. Compute Euclidean Gradient
        # Loss f(R) = 0.5 * ||(W_target - R W0) * F^0.5||_F^2
        # d f / d R = - ((W_target - R W0) * F) W0^T
        diff = W_target - torch.matmul(R, W0)
        grad_E = - torch.matmul(diff * F, W0.T)
        
        # 2. Project Euclidean gradient onto the Tangent Space of O(d) at R
        # grad_R = R * skew(R^T * grad_E) = 0.5 * (grad_E - R * grad_E^T * R)
        grad_R = 0.5 * (grad_E - torch.matmul(torch.matmul(R, grad_E.T), R))
        
        # Riemannian Gradient Clipping
        grad_norm = torch.linalg.matrix_norm(grad_R, 'fro')
        if grad_norm > 1.0 or torch.isnan(grad_norm):
            grad_R = grad_R / (grad_norm + 1e-8)
            
        # 3. Take Riemannian step
        R_next = R - lr * grad_R
        
        # 4. Retract back onto O(d) using SVD
        U, S, Vh = torch.linalg.svd(R_next)
        R = torch.matmul(U, Vh)
        
    return R

def cayley_transform(R):
    # Cayley transform: Q = (R - I)(R + I)^-1
    d = R.shape[0]
    I = torch.eye(d)
    inv_part = torch.linalg.inv(R + I + 1e-6 * I)
    Q = torch.matmul(R - I, inv_part)
    # Force skew-symmetry to handle numerical errors
    Q = 0.5 * (Q - Q.T)
    return Q

def inv_cayley_transform(Q):
    # Inverse Cayley transform: R = (I + Q)(I - Q)^-1
    d = Q.shape[0]
    I = torch.eye(d)
    inv_part = torch.linalg.inv(I - Q)
    R = torch.matmul(I + Q, inv_part)
    return R

def run_experiment_seed(seed, d=128, num_tasks=5):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    W0, tasks = generate_task_data(d, num_tasks)
    
    # --- 1. Pretrained Baseline ---
    losses_pretrained = [compute_loss(W0, t["W_target"], t["F"]) for t in tasks]
    
    # --- 2. Task Arithmetic (Euclidean averaging) ---
    task_vectors = [t["W_target"] - W0 for t in tasks]
    W_ta = W0 + sum(task_vectors) / num_tasks
    losses_ta = [compute_loss(W_ta, t["W_target"], t["F"]) for t in tasks]
    
    # --- 3. Ties Merging Baseline (simplified) ---
    # Keep top 20% of weights by magnitude for each task vector, average them
    pruned_vectors = []
    for tv in task_vectors:
        flat_tv = tv.flatten()
        threshold = torch.quantile(torch.abs(flat_tv), 0.8)
        mask = torch.abs(tv) >= threshold
        pruned_vectors.append(tv * mask)
    W_ties = W0 + sum(pruned_vectors) / num_tasks
    losses_ties = [compute_loss(W_ties, t["W_target"], t["F"]) for t in tasks]
    
    # --- 4. OrthoMerge Baseline ---
    # Decouple using standard Procrustes
    R_list_om = []
    rho_list_om = []
    for t in tasks:
        R = solve_standard_procrustes(t["W_target"], W0)
        R_list_om.append(R)
        rho_list_om.append(t["W_target"] - torch.matmul(R, W0))
        
    # Merge rotations in Lie algebra
    Q_list_om = [cayley_transform(R) for R in R_list_om]
    Q_sum_om = sum(Q_list_om)
    # Magnitude correction scaling
    norms_om = sum([torch.linalg.matrix_norm(Q, 'fro') for Q in Q_list_om])
    sum_norm_om = torch.linalg.matrix_norm(Q_sum_om, 'fro')
    scale_om = norms_om / (sum_norm_om + 1e-6)
    Q_merged_om = scale_om * (Q_sum_om / num_tasks)
    R_merged_om = inv_cayley_transform(Q_merged_om)
    
    # Merge residuals
    rho_merged_om = sum(rho_list_om) / num_tasks
    W_om = torch.matmul(R_merged_om, W0) + rho_merged_om
    losses_om = [compute_loss(W_om, t["W_target"], t["F"]) for t in tasks]
    
    # --- 5. FOMM (Our Fisher-Weighted Orthogonal Merging) ---
    # Decouple using Fisher-weighted Procrustes
    R_list_fomm = []
    rho_list_fomm = []
    for t in tasks:
        R = solve_weighted_procrustes(t["W_target"], W0, t["F"])
        R_list_fomm.append(R)
        rho_list_fomm.append(t["W_target"] - torch.matmul(R, W0))
        
    # Merge rotations in Lie algebra
    Q_list_fomm = [cayley_transform(R) for R in R_list_fomm]
    Q_sum_fomm = sum(Q_list_fomm)
    norms_fomm = sum([torch.linalg.matrix_norm(Q, 'fro') for Q in Q_list_fomm])
    sum_norm_fomm = torch.linalg.matrix_norm(Q_sum_fomm, 'fro')
    scale_fomm = norms_fomm / (sum_norm_fomm + 1e-6)
    Q_merged_fomm = scale_fomm * (Q_sum_fomm / num_tasks)
    R_merged_fomm = inv_cayley_transform(Q_merged_fomm)
    
    # Merge residuals
    rho_merged_fomm = sum(rho_list_fomm) / num_tasks
    W_fomm = torch.matmul(R_merged_fomm, W0) + rho_merged_fomm
    losses_fomm = [compute_loss(W_fomm, t["W_target"], t["F"]) for t in tasks]
    
    # --- 6. FW-TA (Fisher-Weighted Task Arithmetic) ---
    F_sum = sum([t["F"] for t in tasks]) + 1e-6
    W_fw = torch.zeros_like(W0)
    for t in tasks:
        W_fw += t["W_target"] * t["F"]
    W_fw = W_fw / F_sum
    losses_fw = [compute_loss(W_fw, t["W_target"], t["F"]) for t in tasks]
    
    return {
        "pretrained": np.mean(losses_pretrained),
        "ta": np.mean(losses_ta),
        "ties": np.mean(losses_ties),
        "orthomerge": np.mean(losses_om),
        "fomm": np.mean(losses_fomm),
        "fw_ta": np.mean(losses_fw)
    }

# Run over 5 seeds
seeds = [42, 100, 2026, 7, 999]
results = []
for seed in seeds:
    res = run_experiment_seed(seed)
    results.append(res)
    print(f"Seed {seed}: {res}")

# Calculate average and std dev
summary = {}
for key in ["pretrained", "ta", "ties", "orthomerge", "fomm", "fw_ta"]:
    vals = [res[key] for res in results]
    summary[key] = {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals))
    }

print("\n=== FINAL RESULTS SUMMARY ===")
print(json.dumps(summary, indent=2))

with open("results.json", "w") as f:
    json.dump({"runs": results, "summary": summary}, f, indent=2)
