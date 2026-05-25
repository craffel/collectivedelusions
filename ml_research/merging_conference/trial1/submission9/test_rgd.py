import numpy as np
import torch
import torch.nn as nn
import json

np.random.seed(42)
torch.manual_seed(42)

from run_experiments import generate_task_data, compute_loss, solve_standard_procrustes, solve_weighted_procrustes, cayley_transform, inv_cayley_transform

W0, tasks = generate_task_data(d=128, num_tasks=5)

# Test RGD convergence for the first task
W_target = tasks[0]["W_target"]
F = tasks[0]["F"]

print("--- Testing RGD Convergence on Task 0 ---")
# Let's run solve_weighted_procrustes but print the loss at intervals
R = solve_standard_procrustes(W_target, W0)
initial_loss = 0.5 * torch.sum(F * (W_target - torch.matmul(R, W0))**2).item()
print(f"Initial (Standard Procrustes) Weighted Loss: {initial_loss:.4f}")

# Custom tracking solver
R_opt = R.clone()
lr = 0.01
for i in range(100):
    diff = W_target - torch.matmul(R_opt, W0)
    grad_E = - torch.matmul(diff * F, W0.T)
    grad_R = 0.5 * (grad_E - torch.matmul(torch.matmul(R_opt, grad_E.T), R_opt))
    
    # Gradient clipping
    grad_norm = torch.linalg.matrix_norm(grad_R, 'fro')
    if grad_norm > 1.0 or torch.isnan(grad_norm):
        grad_R = grad_R / (grad_norm + 1e-8)
        
    R_temp = R_opt - lr * grad_R
    U, S, Vh = torch.linalg.svd(R_temp)
    R_opt = torch.matmul(U, Vh)
    if (i + 1) % 10 == 0 or i < 5:
        curr_loss = 0.5 * torch.sum(F * (W_target - torch.matmul(R_opt, W0))**2).item()
        print(f"Iteration {i+1}: Loss = {curr_loss:.4f}")

# Let's check with learning rate 0.05 or different iterations
print("\n--- Testing RGD with different learning rates and iterations ---")
for test_lr in [0.005, 0.01, 0.02, 0.05, 0.1]:
    R_opt = R.clone()
    for i in range(200):
        diff = W_target - torch.matmul(R_opt, W0)
        grad_E = - torch.matmul(diff * F, W0.T)
        grad_R = 0.5 * (grad_E - torch.matmul(torch.matmul(R_opt, grad_E.T), R_opt))
        
        # Gradient clipping
        grad_norm = torch.linalg.matrix_norm(grad_R, 'fro')
        if grad_norm > 1.0 or torch.isnan(grad_norm):
            grad_R = grad_R / (grad_norm + 1e-8)
            
        R_temp = R_opt - test_lr * grad_R
        U, S, Vh = torch.linalg.svd(R_temp)
        R_opt = torch.matmul(U, Vh)
    final_loss = 0.5 * torch.sum(F * (W_target - torch.matmul(R_opt, W0))**2).item()
    print(f"LR {test_lr}, 200 iters: Final Loss = {final_loss:.4f}")

# Let's check the effect of Lie Algebra scale
print("\n--- Testing Lie Algebra scale impact on original experiments ---")
def run_experiment_with_params(scale_type="orthomerge"):
    # Run over seed 42
    W0, tasks = generate_task_data(d=128, num_tasks=5)
    num_tasks = len(tasks)
    
    # 1. TA
    task_vectors = [t["W_target"] - W0 for t in tasks]
    W_ta = W0 + sum(task_vectors) / num_tasks
    loss_ta = np.mean([compute_loss(W_ta, t["W_target"], t["F"]) for t in tasks])
    
    # 2. OrthoMerge
    R_list_om = []
    rho_list_om = []
    for t in tasks:
        R = solve_standard_procrustes(t["W_target"], W0)
        R_list_om.append(R)
        rho_list_om.append(t["W_target"] - torch.matmul(R, W0))
    Q_list_om = [cayley_transform(R) for R in R_list_om]
    Q_sum_om = sum(Q_list_om)
    
    if scale_type == "orthomerge":
        norms_om = sum([torch.linalg.matrix_norm(Q, 'fro') for Q in Q_list_om])
        sum_norm_om = torch.linalg.matrix_norm(Q_sum_om, 'fro')
        scale_om = norms_om / (sum_norm_om + 1e-6)
    else:
        scale_om = 1.0 # No scaling
        
    Q_merged_om = scale_om * (Q_sum_om / num_tasks)
    R_merged_om = inv_cayley_transform(Q_merged_om)
    rho_merged_om = sum(rho_list_om) / num_tasks
    W_om = torch.matmul(R_merged_om, W0) + rho_merged_om
    loss_om = np.mean([compute_loss(W_om, t["W_target"], t["F"]) for t in tasks])
    
    # 3. FOMM
    R_list_fomm = []
    rho_list_fomm = []
    for t in tasks:
        R = solve_weighted_procrustes(t["W_target"], W0, t["F"])
        R_list_fomm.append(R)
        rho_list_fomm.append(t["W_target"] - torch.matmul(R, W0))
    Q_list_fomm = [cayley_transform(R) for R in R_list_fomm]
    Q_sum_fomm = sum(Q_list_fomm)
    
    if scale_type == "orthomerge":
        norms_fomm = sum([torch.linalg.matrix_norm(Q, 'fro') for Q in Q_list_fomm])
        sum_norm_fomm = torch.linalg.matrix_norm(Q_sum_fomm, 'fro')
        scale_fomm = norms_fomm / (sum_norm_fomm + 1e-6)
    else:
        scale_fomm = 1.0 # No scaling
        
    Q_merged_fomm = scale_fomm * (Q_sum_fomm / num_tasks)
    R_merged_fomm = inv_cayley_transform(Q_merged_fomm)
    rho_merged_fomm = sum(rho_list_fomm) / num_tasks
    W_fomm = torch.matmul(R_merged_fomm, W0) + rho_merged_fomm
    loss_fomm = np.mean([compute_loss(W_fomm, t["W_target"], t["F"]) for t in tasks])
    
    print(f"Scale type: {scale_type} -> TA loss: {loss_ta:.4f}, OrthoMerge loss: {loss_om:.4f}, FOMM loss: {loss_fomm:.4f}")

run_experiment_with_params("orthomerge")
run_experiment_with_params("none")
