import os
import torch
import torch.nn as nn
import numpy as np
from train import (
    SEEDS, get_datasets, pretrain_experts, L3SoftmaxRouterZero,
    train_router, evaluate_router, evaluate_uniform
)

def main():
    print("Starting Subspace Overlap Sensitivity Sweep across 10 independent seeds...")
    rhos = [0.0, 0.1, 0.25, 0.33, 0.5, 0.75, 0.9]
    models = ['Uniform', 'L3_Softmax_WellReg', 'VR_Router']
    
    # Structure to hold all sweep results: {model_name: {rho: [acc1, acc2, ...]}}
    sweep_results = {m: {r: [] for r in rhos} for m in models}
    
    # To save time and be highly efficient, let's execute seed by seed and rho by rho
    for rho in rhos:
        print(f"\nEvaluating Subspace Overlap rho = {rho:.2f}...")
        for seed in SEEDS:
            # Generate dataset for this specific rho and seed
            train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, prototypes = get_datasets(seed, rho=rho)
            
            # Pre-train actual experts under this specific rho and seed
            W_base, b_base, expert_weights, expert_biases, expert_accuracies = pretrain_experts(seed, train_x, train_y, test_x, test_y)
            
            # Calculate dynamic router weight/bias offsets (task vectors)
            V_weights = [expert_weights[k] - W_base for k in range(4)]
            V_biases = [expert_biases[k] - b_base for k in range(4)]
            
            # Use random projection
            p_proj = torch.randn(192, 4)
            p_proj = p_proj / torch.norm(p_proj, dim=0, keepdim=True)
            
            # Shuffle test set for heterogeneous evaluation
            perm = torch.randperm(test_x.shape[0])
            test_x_shuf = test_x[perm]
            test_y_shuf = test_y[perm]
            test_tasks_shuf = test_tasks[perm]
            
            # 1. Uniform Merging
            uniform_acc = evaluate_uniform(test_x_shuf, test_y_shuf, test_tasks_shuf, W_base, b_base, V_weights, V_biases)
            sweep_results['Uniform'][rho].append(uniform_acc)
            
            # 2. Well-Regularized Softmax Router (lambda_var = 0.0)
            wellreg_router = L3SoftmaxRouterZero(14, 4, 4)
            train_router(wellreg_router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, epochs=100, lr=1e-3, lambda_wd=1e-2, lambda_var=0.0)
            wellreg_acc = evaluate_router(wellreg_router, test_x_shuf, test_y_shuf, test_tasks_shuf, W_base, b_base, V_weights, V_biases, p_proj, 'hetero', 256, is_vr_router=True)
            sweep_results['L3_Softmax_WellReg'][rho].append(wellreg_acc)
            
            # 3. VR_Router (lambda_var = 1.0)
            vr_router = L3SoftmaxRouterZero(14, 4, 4)
            train_router(vr_router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, epochs=100, lr=1e-3, lambda_wd=1e-2, lambda_var=1.0)
            vr_acc = evaluate_router(vr_router, test_x_shuf, test_y_shuf, test_tasks_shuf, W_base, b_base, V_weights, V_biases, p_proj, 'hetero', 256, is_vr_router=True)
            sweep_results['VR_Router'][rho].append(vr_acc)
            
        print(f"Results for rho = {rho:.2f}:")
        for m in models:
            m_accs = np.array(sweep_results[m][rho]) * 100
            print(f"  {m:<20} | Joint Mean: {np.mean(m_accs):.2f}% ± {np.std(m_accs):.2f}%")

    print("\n\n--- FINAL SUBSPACE OVERLAP SENSITIVITY TABLE ---")
    print("| Subspace Overlap (rho) | Uniform Merging | L3_Softmax_WellReg | VR_Router |")
    print("| :---: | :---: | :---: | :---: |")
    for r in rhos:
        line = f"| {r:.2f} "
        for m in models:
            m_accs = np.array(sweep_results[m][r]) * 100
            line += f"| {np.mean(m_accs):.2f}% ± {np.std(m_accs):.2f}% "
        line += "|"
        print(line)

    # Save output to a text file for easy extraction and inclusion
    with open("results/subspace_overlap_sweep_results.txt", "w") as f:
        f.write("# Subspace Overlap (rho) Sensitivity Sweep Results across 10 Seeds\n\n")
        f.write("| Subspace Overlap (rho) | Uniform Merging | L3_Softmax_WellReg | VR_Router |\n")
        f.write("| :---: | :---: | :---: | :---: |\n")
        for r in rhos:
            line = f"| {r:.2f} "
            for m in models:
                m_accs = np.array(sweep_results[m][r]) * 100
                line += f"| {np.mean(m_accs):.2f}% ± {np.std(m_accs):.2f}% "
            line += "|\n"
            f.write(line)

if __name__ == "__main__":
    main()
