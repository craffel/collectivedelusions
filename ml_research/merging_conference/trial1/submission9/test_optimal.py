import numpy as np
import torch
import json

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

from run_experiments import compute_loss, solve_standard_procrustes, solve_weighted_procrustes, cayley_transform, inv_cayley_transform
from test_misalignment import generate_task_data_misaligned

def test_optimal_fwm_with_rotation(rotation_scale=0.1):
    W0, tasks = generate_task_data_misaligned(d=128, num_tasks=5, rotation_scale=rotation_scale)
    num_tasks = len(tasks)
    
    # 1. Task Arithmetic (standard)
    task_vectors = [t["W_target"] - W0 for t in tasks]
    W_ta = W0 + sum(task_vectors) / num_tasks
    loss_ta = np.mean([compute_loss(W_ta, t["W_target"], t["F"]) for t in tasks])
    
    # 2. Fisher-Weighted Task Arithmetic (FW-TA)
    W_obs = [t["W_target"] for t in tasks]
    F_sum = sum([t["F"] for t in tasks]) + 1e-6
    W_fw = torch.zeros_like(W0)
    for t in tasks:
        W_fw += t["W_target"] * t["F"]
    W_fw = W_fw / F_sum
    loss_fw = np.mean([compute_loss(W_fw, t["W_target"], t["F"]) for t in tasks])
    
    # 3. FOMM with Fisher-Weighted Residuals and standard rotation merging
    R_list_fomm = []
    rho_list_fomm = []
    for t in tasks:
        R = solve_weighted_procrustes(t["W_target"], W0, t["F"])
        R_list_fomm.append(R)
        rho_list_fomm.append(t["W_target"] - torch.matmul(R, W0))
        
    Q_list_fomm = [cayley_transform(R) for R in R_list_fomm]
    Q_sum_fomm = sum(Q_list_fomm)
    Q_merged_fomm = Q_sum_fomm / num_tasks
    R_merged_fomm = inv_cayley_transform(Q_merged_fomm)
    
    # Fisher-Weighted Residual Merging
    F_sum_fomm = sum([t["F"] for t in tasks]) + 1e-6
    rho_weighted = torch.zeros_like(rho_list_fomm[0])
    for t_idx, rho in enumerate(rho_list_fomm):
        F = tasks[t_idx]["F"]
        rho_weighted += rho * F
    rho_merged_fomm = rho_weighted / F_sum_fomm
    W_fomm = torch.matmul(R_merged_fomm, W0) + rho_merged_fomm
    loss_fomm = np.mean([compute_loss(W_fomm, t["W_target"], t["F"]) for t in tasks])
    
    # 4. Fisher-Weighted Orthogonal Model Merging (FW-FOMM - Our New Formulation!)
    # We can do Fisher-weighted merging of Q_i as well!
    # Let's weight the Lie algebra elements Q_i using the sum of Fisher information for each task
    task_weights = [t["F"].sum().item() for t in tasks]
    total_task_weight = sum(task_weights) + 1e-6
    task_weights = [w / total_task_weight for w in task_weights]
    
    Q_weighted_sum = torch.zeros_like(Q_list_fomm[0])
    for t_idx, Q in enumerate(Q_list_fomm):
        Q_weighted_sum += task_weights[t_idx] * Q
    
    # Scale correction
    norms_fomm = sum([task_weights[t_idx] * torch.linalg.matrix_norm(Q, 'fro') for t_idx, Q in enumerate(Q_list_fomm)])
    sum_norm_fomm = torch.linalg.matrix_norm(Q_weighted_sum, 'fro')
    scale_fomm = norms_fomm / (sum_norm_fomm + 1e-6)
    Q_merged_fw = scale_fomm * Q_weighted_sum
    R_merged_fw = inv_cayley_transform(Q_merged_fw)
    
    W_fomm_fw = torch.matmul(R_merged_fw, W0) + rho_merged_fomm
    loss_fomm_fw = np.mean([compute_loss(W_fomm_fw, t["W_target"], t["F"]) for t in tasks])
    
    print(f"Rotation Scale {rotation_scale}:")
    print(f"  Standard TA: {loss_ta:.4f}")
    print(f"  Fisher-Weighted TA (FW-TA): {loss_fw:.4f}")
    print(f"  FOMM (standard average): {loss_fomm:.4f}")
    print(f"  FOMM (FW-FOMM): {loss_fomm_fw:.4f}")

test_optimal_fwm_with_rotation(0.05)
test_optimal_fwm_with_rotation(0.1)
test_optimal_fwm_with_rotation(0.2)
