import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from train import (
    SEEDS, get_cached_experts_and_datasets, train_router, evaluate_router,
    L3SoftmaxRouterZero, evaluate_uniform
)

class L3MLPSoftmaxRouterZero(nn.Module):
    def __init__(self, L, K, d, hidden_dim=8):
        super().__init__()
        # Tiny initialization near zero to preserve maximum-entropy uniform prior
        # while allowing symmetry breaking during backprop.
        self.W1 = nn.Parameter(torch.randn(L, hidden_dim, d) * 1e-4)
        self.b1 = nn.Parameter(torch.zeros(L, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(L, K, hidden_dim) * 1e-4)
        self.b2 = nn.Parameter(torch.zeros(L, K))
        
    def forward(self, psi):
        # psi is (B, d)
        h = torch.einsum('bd,lhd->blh', psi, self.W1) + self.b1 # (B, L, hidden_dim)
        h = torch.tanh(h)
        out = torch.einsum('blh,lkh->blk', h, self.W2) + self.b2 # (B, L, K)
        return torch.softmax(out, dim=2)

def run_mlp_experiment():
    print("Running 2-layer MLP Router experiment across all 10 seeds...")
    
    results = {
        'homo': [],
        'hetero_256': [],
        'hetero_1': []
    }
    
    batch_sizes = [1, 8, 32, 128, 512]
    stress_results = {b: [] for b in batch_sizes}
    
    for seed in SEEDS:
        (train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, prototypes,
         W_base, b_base, expert_weights, expert_biases, expert_accuracies) = get_cached_experts_and_datasets(seed)
         
        V_weights = [expert_weights[k] - W_base for k in range(4)]
        V_biases = [expert_biases[k] - b_base for k in range(4)]
        
        p_proj = torch.randn(192, 4)
        p_proj = p_proj / torch.norm(p_proj, dim=0, keepdim=True)
        
        perm = torch.randperm(test_x.shape[0])
        test_x_shuf = test_x[perm]
        test_y_shuf = test_y[perm]
        test_tasks_shuf = test_tasks[perm]
        
        # Instantiate 2-layer MLP Router
        mlp_router = L3MLPSoftmaxRouterZero(14, 4, 4, hidden_dim=8)
        # Train with standard weight decay and zero-initializer philosophy
        train_router(mlp_router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, 
                     epochs=100, lr=1e-3, lambda_wd=1e-2, lambda_var=0.0)
        
        results['homo'].append(evaluate_router(mlp_router, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj, 'homo', 256, is_vr_router=True))
        results['hetero_256'].append(evaluate_router(mlp_router, test_x_shuf, test_y_shuf, test_tasks_shuf, W_base, b_base, V_weights, V_biases, p_proj, 'hetero', 256, is_vr_router=True))
        results['hetero_1'].append(evaluate_router(mlp_router, test_x_shuf, test_y_shuf, test_tasks_shuf, W_base, b_base, V_weights, V_biases, p_proj, 'hetero', 1, is_vr_router=True))
        
        # Heterogeneity stress test sizes
        for b in batch_sizes:
            stress_results[b].append(evaluate_router(mlp_router, test_x_shuf, test_y_shuf, test_tasks_shuf, W_base, b_base, V_weights, V_biases, p_proj, 'hetero', b, is_vr_router=True))

    print("\n--- 2-LAYER MLP ROUTER STATISTICAL SIGNIFICANCE RESULTS (Mean ± Std %) ---")
    homo_mean, homo_std = np.mean(results['homo']) * 100, np.std(results['homo']) * 100
    het256_mean, het256_std = np.mean(results['hetero_256']) * 100, np.std(results['hetero_256']) * 100
    het1_mean, het1_std = np.mean(results['hetero_1']) * 100, np.std(results['hetero_1']) * 100
    print(f"MLP_Router | Homo (B=256): {homo_mean:.2f} ± {homo_std:.2f}% | Hetero (B=256): {het256_mean:.2f} ± {het256_std:.2f}% | Hetero (B=1): {het1_mean:.2f} ± {het1_std:.2f}%")
    
    print("\n--- 2-LAYER MLP HETEROGENEITY STRESS TEST ---")
    for b in batch_sizes:
        b_mean, b_std = np.mean(stress_results[b]) * 100, np.std(stress_results[b]) * 100
        print(f"Batch size B={b:<3} | Accuracy: {b_mean:.2f} ± {b_std:.2f}%")

if __name__ == '__main__':
    run_mlp_experiment()
