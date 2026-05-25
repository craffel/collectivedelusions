import numpy as np
import torch
import torch.nn as nn

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

from run_experiments import compute_loss, solve_standard_procrustes, solve_weighted_procrustes, cayley_transform, inv_cayley_transform
from test_misalignment import generate_task_data_misaligned

def test_unrotated_merging(rotation_scale=0.1):
    W0, tasks = generate_task_data_misaligned(d=128, num_tasks=5, rotation_scale=rotation_scale)
    num_tasks = len(tasks)
    
    # 1. TA
    task_vectors = [t["W_target"] - W0 for t in tasks]
    W_ta = W0 + sum(task_vectors) / num_tasks
    loss_ta = np.mean([compute_loss(W_ta, t["W_target"], t["F"]) for t in tasks])
    
    # 2. OrthoMerge (standard)
    R_list_om = []
    rho_list_om = []
    for t in tasks:
        R = solve_standard_procrustes(t["W_target"], W0)
        R_list_om.append(R)
        rho_list_om.append(t["W_target"] - torch.matmul(R, W0))
    Q_list_om = [cayley_transform(R) for R in R_list_om]
    Q_sum_om = sum(Q_list_om)
    Q_merged_om = Q_sum_om / num_tasks
    R_merged_om = inv_cayley_transform(Q_merged_om)
    rho_merged_om = sum(rho_list_om) / num_tasks
    W_om = torch.matmul(R_merged_om, W0) + rho_merged_om
    loss_om = np.mean([compute_loss(W_om, t["W_target"], t["F"]) for t in tasks])
    
    # 3. OrthoMerge with Unrotated Residuals
    # \tilde{\rho}_i = R_i^T W_i - W_0
    rho_tilde_list = []
    for R, t in zip(R_list_om, tasks):
        rho_tilde = torch.matmul(R.T, t["W_target"]) - W0
        rho_tilde_list.append(rho_tilde)
    rho_tilde_merged = sum(rho_tilde_list) / num_tasks
    W_om_unrotated = torch.matmul(R_merged_om, W0 + rho_tilde_merged)
    loss_om_unrotated = np.mean([compute_loss(W_om_unrotated, t["W_target"], t["F"]) for t in tasks])
    
    # 4. FOMM with Unrotated Residuals (Fisher-weighted)
    R_list_fomm = []
    for t in tasks:
        R = solve_weighted_procrustes(t["W_target"], W0, t["F"])
        R_list_fomm.append(R)
    Q_list_fomm = [cayley_transform(R) for R in R_list_fomm]
    Q_sum_fomm = sum(Q_list_fomm)
    Q_merged_fomm = Q_sum_fomm / num_tasks
    R_merged_fomm = inv_cayley_transform(Q_merged_fomm)
    
    rho_tilde_fomm_list = []
    for R, t in zip(R_list_fomm, tasks):
        rho_tilde = torch.matmul(R.T, t["W_target"]) - W0
        rho_tilde_fomm_list.append(rho_tilde)
        
    # Fisher-weighted averaging of unrotated residuals
    # F in base space is rotated from F in target space: F_base = R.T @ F @ R (approx, or we can just use F_i directly or average them)
    # Let's try simple average first
    rho_tilde_fomm_merged = sum(rho_tilde_fomm_list) / num_tasks
    W_fomm_unrotated = torch.matmul(R_merged_fomm, W0 + rho_tilde_fomm_merged)
    loss_fomm_unrotated = np.mean([compute_loss(W_fomm_unrotated, t["W_target"], t["F"]) for t in tasks])
    
    print(f"Rotation Scale {rotation_scale}:")
    print(f"  TA: {loss_ta:.4f}")
    print(f"  OrthoMerge (standard): {loss_om:.4f}")
    print(f"  OrthoMerge (unrotated): {loss_om_unrotated:.4f}")
    print(f"  FOMM (unrotated): {loss_fomm_unrotated:.4f}")

test_unrotated_merging(0.05)
test_unrotated_merging(0.1)
test_unrotated_merging(0.2)
