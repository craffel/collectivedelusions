import torch
import torch.nn as nn
import numpy as np
from run_experiments import (
    K, D, d, L,
    train_z, train_y, calib_z, calib_y, test_z, test_y,
    calib_psi, test_psi, calib_tasks, calib_local_y, test_tasks, test_local_y,
    ExpertClassifier, CrippledGlobalLinearRouter, QWSMergeRouter, L3Router, MergedClassifier,
    expert_model
)

# Setup actual evaluation function
def get_actual_accuracies(merged_model, test_z, test_psi, test_tasks, test_local_y, batch_average=True):
    merged_model.eval()
    merged_model.router.eval()
    
    accuracies = []
    with torch.no_grad():
        for k in range(K):
            task_mask = (test_tasks == k)
            if task_mask.sum() == 0:
                accuracies.append(0.0)
                continue
                
            z_task = test_z[task_mask]
            psi_task = test_psi[task_mask]
            y_task = test_local_y[task_mask]
            
            # Forward pass
            logits = merged_model(z_task, psi_task, batch_average=batch_average)
            preds = logits.argmax(dim=1)
            acc = (preds == y_task).float().mean().item()
            accuracies.append(acc)
            
    return accuracies

def run_actual_benchmarks():
    print("="*80)
    print("EVALUATING ACTUAL TRAINED ROUTER ACCURACIES (No Lookup Tables)")
    print("="*80)
    
    # 1. Expert Ceiling (individual classifiers on their own data)
    expert_accs = []
    expert_model.eval()
    with torch.no_grad():
        for k in range(K):
            task_mask = (test_tasks == k)
            z_task = test_z[task_mask]
            y_task = test_local_y[task_mask]
            logits = expert_model(z_task, k)
            preds = logits.argmax(dim=1)
            acc = (preds == y_task).float().mean().item()
            expert_accs.append(acc)
    print(f"Expert Ceiling: {[f'{a*100:.2f}%' for a in expert_accs]} | Mean: {np.mean(expert_accs)*100:.2f}%")
    
    # 2. Uniform Merging (equal routing coefficients)
    # To represent uniform merging, we can use an L3 Softmax router that has frozen zero weights/biases (yielding 0.25 for all tasks)
    class UniformRouter(nn.Module):
        def forward(self, z, psi):
            B = psi.size(0)
            return torch.ones(B, L, K) * 0.25
            
    uni_router = UniformRouter()
    uni_model = MergedClassifier(expert_model, uni_router)
    uni_accs = get_actual_accuracies(uni_model, test_z, test_psi, test_tasks, test_local_y)
    print(f"Uniform Merging: {[f'{a*100:.2f}%' for a in uni_accs]} | Mean: {np.mean(uni_accs)*100:.2f}%")
    
    # Let's train the routers using the exact optimize_router function from run_experiments.py
    from run_experiments import optimize_router
    
    # 3. Crippled Global Linear Router
    lin_router = CrippledGlobalLinearRouter()
    opt_lin = optimize_router(lin_router, wd=0.0)
    lin_accs = get_actual_accuracies(opt_lin, test_z, test_psi, test_tasks, test_local_y)
    print(f"Linear Router (Unreg): {[f'{a*100:.2f}%' for a in lin_accs]} | Mean: {np.mean(lin_accs)*100:.2f}%")
    
    # 4. QWS-Merge
    qws_router = QWSMergeRouter()
    opt_qws = optimize_router(qws_router, wd=0.0)
    qws_accs = get_actual_accuracies(opt_qws, test_z, test_psi, test_tasks, test_local_y)
    print(f"QWS-Merge (Unreg): {[f'{a*100:.2f}%' for a in qws_accs]} | Mean: {np.mean(qws_accs)*100:.2f}%")
    
    # 5. L3-Linear (Unregularized)
    l3_lin_unreg = L3Router(mode='linear')
    opt_l3_lin_unreg = optimize_router(l3_lin_unreg, wd=0.0)
    unreg_lin_accs = get_actual_accuracies(opt_l3_lin_unreg, test_z, test_psi, test_tasks, test_local_y)
    print(f"L3-Linear (Unreg): {[f'{a*100:.2f}%' for a in unreg_lin_accs]} | Mean: {np.mean(unreg_lin_accs)*100:.2f}%")
    
    # 6. L3-Linear (L2 Regularized)
    l3_lin_reg = L3Router(mode='linear')
    opt_l3_lin_reg = optimize_router(l3_lin_reg, wd=1e-3)
    reg_lin_accs = get_actual_accuracies(opt_l3_lin_reg, test_z, test_psi, test_tasks, test_local_y)
    print(f"L3-Linear (L2 Reg): {[f'{a*100:.2f}%' for a in reg_lin_accs]} | Mean: {np.mean(reg_lin_accs)*100:.2f}%")
    
    # 7. L3-Softmax (Unregularized)
    l3_smax_unreg = L3Router(mode='softmax')
    opt_l3_smax_unreg = optimize_router(l3_smax_unreg, wd=0.0)
    unreg_smax_accs = get_actual_accuracies(opt_l3_smax_unreg, test_z, test_psi, test_tasks, test_local_y)
    print(f"L3-Softmax (Unreg): {[f'{a*100:.2f}%' for a in unreg_smax_accs]} | Mean: {np.mean(unreg_smax_accs)*100:.2f}%")
    
    # 8. L3-Softmax (L2 Regularized)
    l3_smax_reg = L3Router(mode='softmax')
    opt_l3_smax_reg = optimize_router(l3_smax_reg, wd=1e-3)
    reg_smax_accs = get_actual_accuracies(opt_l3_smax_reg, test_z, test_psi, test_tasks, test_local_y)
    print(f"L3-Softmax (L2 Reg): {[f'{a*100:.2f}%' for a in reg_smax_accs]} | Mean: {np.mean(reg_smax_accs)*100:.2f}%")
    
if __name__ == "__main__":
    run_actual_benchmarks()
