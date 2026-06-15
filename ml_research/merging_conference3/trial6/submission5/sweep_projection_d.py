import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from train import get_datasets, pretrain_experts, L3SoftmaxRouterZero, train_router, evaluate_router, SEEDS

# Use the same cache as train.py to keep it fast
expert_cache = {}

def get_cached_experts_and_datasets(seed):
    if seed in expert_cache:
        return expert_cache[seed]
    train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, prototypes = get_datasets(seed, rho=0.33)
    W_base, b_base, expert_weights, expert_biases, expert_accuracies = pretrain_experts(seed, train_x, train_y, test_x, test_y)
    expert_cache[seed] = (train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, prototypes,
                          W_base, b_base, expert_weights, expert_biases, expert_accuracies)
    return expert_cache[seed]

def main():
    print("Starting Sweep over Projection Subspace Dimension d...")
    dims = [2, 4, 8, 16]
    results = {d: [] for d in dims}

    for d in dims:
        print(f"Sweeping d = {d} across 10 seeds...")
        for seed in SEEDS:
            (train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, prototypes,
             W_base, b_base, expert_weights, expert_biases, expert_accuracies) = get_cached_experts_and_datasets(seed)
            
            V_weights = [expert_weights[k] - W_base for k in range(4)]
            V_biases = [expert_biases[k] - b_base for k in range(4)]
            
            # Generate random projection of shape [192, d]
            torch.manual_seed(seed)
            p_proj = torch.randn(192, d)
            p_proj = p_proj / torch.norm(p_proj, dim=0, keepdim=True)
            
            # Shuffle test set for Hetero evaluation
            perm = torch.randperm(test_x.shape[0])
            test_x_shuf = test_x[perm]
            test_y_shuf = test_y[perm]
            test_tasks_shuf = test_tasks[perm]
            
            # Train Well-Regularized Router on d-dimensional projection
            router = L3SoftmaxRouterZero(14, 4, d)
            train_router(router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, 
                         epochs=100, lr=1e-3, lambda_wd=1e-2, lambda_var=0.0)
            
            # Evaluate under Hetero B=1 (where vectorization collapse would occur)
            acc = evaluate_router(router, test_x_shuf, test_y_shuf, test_tasks_shuf, W_base, b_base, V_weights, V_biases, p_proj, 
                                  'hetero', 1, is_vr_router=True)
            results[d].append(acc)
            
    print("\n--- PROJECTION DIMENSION SWEEP RESULTS (B=1) ---")
    for d in dims:
        mean_acc = np.mean(results[d]) * 100
        std_acc = np.std(results[d]) * 100
        print(f"d = {d:<2} | Hetero (B=1) Joint Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

if __name__ == "__main__":
    main()
