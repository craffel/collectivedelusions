import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from run_experiments import (
    D, K, C, L, D_PROJ, set_seed, generate_sandbox_data, train_experts, 
    evaluate_merged_model, evaluate_uniform_merging
)

# Define standard MoE routing models operating directly on raw features z in R^192
class RawSoftmaxMoERouter(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.linear = nn.Linear(D, K)
        
    def forward(self, z):
        # z: Shape (B, D)
        scores = self.linear(z) # (B, K)
        alpha = torch.softmax(scores, dim=-1) # (B, K)
        # Repeat across L layers to match evaluation setup
        return alpha.unsqueeze(1).repeat(1, L, 1) # (B, L, K)

class RawTop1MoERouter(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.linear = nn.Linear(D, K)
        
    def forward(self, z):
        # z: Shape (B, D)
        scores = self.linear(z) # (B, K)
        # We will apply a continuous softmax during training, but hard Top-1 during evaluation.
        # However, to be highly realistic to MoE, we can just return soft scores, and the evaluation
        # function handles top1_gating=True. So the router itself returns softmax scores.
        alpha = torch.softmax(scores, dim=-1)
        return alpha.unsqueeze(1).repeat(1, L, 1)

def train_moe_router(cal_splits, experts, router, lambda_wd=1e-3, epochs=100, lr=1e-2):
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=lambda_wd)
    criterion = nn.CrossEntropyLoss()
    
    # Collect all calibration samples
    X_cal_all = torch.cat([cal_splits[k][0] for k in range(K)], dim=0)
    y_cal_all = torch.cat([cal_splits[k][1] for k in range(K)], dim=0)
    
    router.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Router outputs dynamic coefficients for each sample
        # Since Raw routers operate directly on raw features z, we pass X_cal_all
        alpha = router(X_cal_all)  # (B_cal, L, K)
        bar_alpha_all = alpha.mean(dim=1)  # (B_cal, K)
        
        # Dynamic parameter merging and forward pass
        loss = 0.0
        for b in range(X_cal_all.shape[0]):
            bar_alpha_b = bar_alpha_all[b]
            W_merged = torch.zeros(C, D)
            b_merged = torch.zeros(C)
            for t in range(K):
                W_merged += bar_alpha_b[t] * experts[t].weight
                b_merged += bar_alpha_b[t] * experts[t].bias
                
            logits = torch.matmul(W_merged, X_cal_all[b]) + b_merged
            loss += criterion(logits.unsqueeze(0), y_cal_all[b].unsqueeze(0))
            
        loss = loss / X_cal_all.shape[0]
        loss.backward()
        optimizer.step()

def run_moe_ablation():
    seeds = [1, 2, 3, 4, 5]
    results = {
        "raw_softmax_moe": [],
        "raw_top1_moe": []
    }
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        splits = generate_sandbox_data(seed, num_train=1000, num_test=250, num_cal=16, leakage=0.0)
        
        # Train experts (they have standard ceilings)
        experts, ceilings = train_experts(splits["train"], splits["test"])
        
        # 1. Train Raw Softmax MoE Gating Router on raw features
        raw_softmax_router = RawSoftmaxMoERouter(D, K)
        train_moe_router(splits["cal"], experts, raw_softmax_router, lambda_wd=1e-3, epochs=100, lr=1e-2)
        
        # Evaluate standard homogeneous
        # Note: evaluate_merged_model takes the router. To make evaluate_merged_model bypass
        # state projection for Raw routers, we can temporarily wrap the evaluate_merged_model call
        # or we can pass a dummy projection matrix P, but wait, evaluate_merged_model does:
        # if isinstance(router, GlobalLinearRouter):
        #     alpha = router(all_z)
        # So we can define our routers as subclasses of GlobalLinearRouter or register them similarly.
        # Let's inspect run_experiments.py: GlobalLinearRouter is checked using isinstance(router, GlobalLinearRouter).
        # We can subclass GlobalLinearRouter! That is extremely elegant.
        
    print("Done checking setup.")

if __name__ == "__main__":
    # Let's write the fully subclassed versions to run seamlessly with evaluate_merged_model:
    from run_experiments import GlobalLinearRouter
    
    class RawSoftmaxMoERouterSub(GlobalLinearRouter):
        def __init__(self, D, K):
            super().__init__(D, K)
            self.linear = nn.Linear(D, K)
        def forward(self, z):
            scores = self.linear(z)
            alpha = torch.softmax(scores, dim=-1)
            return alpha.unsqueeze(1).repeat(1, L, 1)
            
    class RawTop1MoERouterSub(GlobalLinearRouter):
        def __init__(self, D, K):
            super().__init__(D, K)
            self.linear = nn.Linear(D, K)
        def forward(self, z):
            scores = self.linear(z)
            alpha = torch.softmax(scores, dim=-1)
            return alpha.unsqueeze(1).repeat(1, L, 1)

    # Let's run the actual training and evaluation across 5 seeds:
    seeds = [1, 2, 3, 4, 5]
    results = {
        "raw_softmax_moe": [],
        "raw_top1_moe": []
    }
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        splits = generate_sandbox_data(seed, num_train=1000, num_test=250, num_cal=16, leakage=0.0)
        experts, ceilings = train_experts(splits["train"], splits["test"])
        
        # Dummy P for the signature, though unused for GlobalLinearRouter subclasses
        P = torch.zeros(D, D_PROJ)
        
        # Train Raw-Softmax
        raw_soft = RawSoftmaxMoERouterSub(D, K)
        train_moe_router(splits["cal"], experts, raw_soft, lambda_wd=1e-3, epochs=100, lr=1e-2)
        accs_soft, mean_soft = evaluate_merged_model(splits["test"], experts, raw_soft, P, "homogeneous")
        results["raw_softmax_moe"].append([accs_soft[i] for i in range(K)] + [mean_soft])
        print(f"Raw Softmax MoE Gating Joint Mean: {mean_soft*100:.2f}%")
        
        # Train Raw-Top1 (same training, but evaluate with top1_gating=True)
        raw_top1 = RawTop1MoERouterSub(D, K)
        train_moe_router(splits["cal"], experts, raw_top1, lambda_wd=1e-3, epochs=100, lr=1e-2)
        accs_top1, mean_top1 = evaluate_merged_model(splits["test"], experts, raw_top1, P, "homogeneous", top1_gating=True)
        results["raw_top1_moe"].append([accs_top1[i] for i in range(K)] + [mean_top1])
        print(f"Raw Top-1 MoE Gating Joint Mean: {mean_top1*100:.2f}%")
        
    print("\n================ FINAL RESULTS (Standard MoE Gating Ablation) ================")
    for key in results:
        data = np.array(results[key])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        print(f"{key}: MNIST={mean[0]*100:.2f}±{std[0]*100:.2f}%, F-MNIST={mean[1]*100:.2f}±{std[1]*100:.2f}%, CIFAR={mean[2]*100:.2f}±{std[2]*100:.2f}%, SVHN={mean[3]*100:.2f}±{std[3]*100:.2f} | Joint Mean={mean[4]*100:.2f}±{std[4]*100:.2f}%")
        
    with open("results/moe_gating_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
