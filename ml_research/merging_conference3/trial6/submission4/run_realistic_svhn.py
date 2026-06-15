import torch
import torch.nn as nn
import numpy as np
import os
import json
from run_experiments import (
    D, K, C, L, D_PROJ, set_seed, train_experts, compute_pca_matrix, 
    project_states, compute_task_anchors, L3LinearRouter, train_router, 
    evaluate_merged_model, evaluate_uniform_merging
)

def generate_sandbox_data_realistic_svhn(seed, num_train=1000, num_test=250, num_cal=16, leakage=0.0):
    set_seed(seed)
    
    # Noise levels to achieve realistic expert ceilings:
    # SVHN sigma is changed from 0.95 to 0.15 to achieve an ~85-90% ceiling.
    sigmas = [0.01, 0.12, 0.18, 0.15]
    
    # Partition D=192 dimensions into K=4 disjoint subspaces of size 48 each
    subspace_dim = D // K  # 48
    
    # Generate task/class prototypes
    prototypes = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            base_coords = torch.randn(subspace_dim)
            base_coords = base_coords / (torch.norm(base_coords) + 1e-8)
            prototypes[k, c, k * subspace_dim : (k + 1) * subspace_dim] = (1.0 - leakage) * base_coords
            prototypes[k, c] = prototypes[k, c] / (torch.norm(prototypes[k, c]) + 1e-8)
            
    # Generate splits
    splits = {"train": [], "test": [], "cal": []}
    
    for k in range(K):
        sigma = sigmas[k]
        
        # Train split
        train_feats = []
        train_labels = []
        for _ in range(num_train):
            c = torch.randint(0, C, (1,)).item()
            noise = torch.randn(D) * sigma
            x = prototypes[k, c] + noise
            train_feats.append(x)
            train_labels.append(c)
        splits["train"].append((torch.stack(train_feats), torch.tensor(train_labels)))
        
        # Test split
        test_feats = []
        test_labels = []
        for _ in range(num_test):
            c = torch.randint(0, C, (1,)).item()
            noise = torch.randn(D) * sigma
            x = prototypes[k, c] + noise
            test_feats.append(x)
            test_labels.append(c)
        splits["test"].append((torch.stack(test_feats), torch.tensor(test_labels)))
        
        # Calibration split
        cal_feats = []
        cal_labels = []
        for _ in range(num_cal):
            c = torch.randint(0, C, (1,)).item()
            noise = torch.randn(D) * sigma
            x = prototypes[k, c] + noise
            cal_feats.append(x)
            cal_labels.append(c)
        splits["cal"].append((torch.stack(cal_feats), torch.tensor(cal_labels)))
        
    return splits

def run_realistic_svhn_evaluation():
    seeds = [1, 2, 3, 4, 5]
    results = {
        "expert_ceiling": [],
        "uniform_merging": [],
        "l3_linear_l2": [],
        "l3_linear_tsar": [],
        "l3_linear_tsar_pcgrad": []
    }
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        splits = generate_sandbox_data_realistic_svhn(seed, num_train=1000, num_test=250, num_cal=16, leakage=0.0)
        
        # Train experts
        experts, ceilings = train_experts(splits["train"], splits["test"])
        results["expert_ceiling"].append(ceilings)
        
        # Compute projection matrix P
        X_cal_all = torch.cat([splits["cal"][k][0] for k in range(K)], dim=0)
        P = compute_pca_matrix(X_cal_all, d=D_PROJ)
        
        # Anchors
        anchors = compute_task_anchors(splits["cal"], P)
        
        # 1. Static Uniform
        uni_accs, uni_mean = evaluate_uniform_merging(splits["test"], experts)
        results["uniform_merging"].append([uni_accs[i] for i in range(K)] + [uni_mean])
        print(f"Uniform Merging Mean: {uni_mean*100:.2f}%")
        
        # 2. L3-Linear + L2
        l3_l2 = L3LinearRouter(L, K, d=D_PROJ)
        train_router(splits["cal"], experts, l3_l2, P, lambda_wd=1e-3, epochs=100, lr=1e-2)
        l2_accs, l2_mean = evaluate_merged_model(splits["test"], experts, l3_l2, P, "homogeneous")
        results["l3_linear_l2"].append([l2_accs[i] for i in range(K)] + [l2_mean])
        print(f"L2 Router Mean: {l2_mean*100:.2f}%")
        
        # 3. L3-Linear + TSAR
        l3_tsar = L3LinearRouter(L, K, d=D_PROJ)
        train_router(splits["cal"], experts, l3_tsar, P, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors, epochs=100, lr=1e-2)
        tsar_accs, tsar_mean = evaluate_merged_model(splits["test"], experts, l3_tsar, P, "homogeneous")
        results["l3_linear_tsar"].append([tsar_accs[i] for i in range(K)] + [tsar_mean])
        print(f"TSAR Router Mean: {tsar_mean*100:.2f}%")
        
        # 4. L3-Linear + TSAR + PCGrad
        l3_tsar_pcgrad = L3LinearRouter(L, K, d=D_PROJ)
        train_router(splits["cal"], experts, l3_tsar_pcgrad, P, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors, epochs=100, lr=1e-2, pcgrad=True)
        pc_accs, pc_mean = evaluate_merged_model(splits["test"], experts, l3_tsar_pcgrad, P, "homogeneous")
        results["l3_linear_tsar_pcgrad"].append([pc_accs[i] for i in range(K)] + [pc_mean])
        print(f"TSAR + PCGrad Router Mean: {pc_mean*100:.2f}%")
        
    print("\n================ FINAL RESULTS (Realistic SVHN Expert) ================")
    for key in results:
        data = np.array(results[key])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        if key == "expert_ceiling":
            print(f"Expert Ceiling: MNIST={mean[0]*100:.2f}%, F-MNIST={mean[1]*100:.2f}%, CIFAR={mean[2]*100:.2f}%, SVHN={mean[3]*100:.2f}% | Mean={np.mean(mean)*100:.2f}%")
        else:
            print(f"{key}: MNIST={mean[0]*100:.2f}±{std[0]*100:.2f}%, F-MNIST={mean[1]*100:.2f}±{std[1]*100:.2f}%, CIFAR={mean[2]*100:.2f}±{std[2]*100:.2f}%, SVHN={mean[3]*100:.2f}±{std[3]*100:.2f}% | Joint Mean={mean[4]*100:.2f}±{std[4]*100:.2f}%")
            
    # Save results
    with open("results/realistic_svhn_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_realistic_svhn_evaluation()
