import torch
import numpy as np
import math
import random
from run_experiments import (
    generate_sandbox_data, train_experts, compute_pca_matrix,
    compute_task_anchors, L3LinearRouter, train_router,
    evaluate_merged_model, set_seed, D, K, L, D_PROJ
)

def compute_random_gaussian_matrix(D, d):
    # Generate random Gaussian matrix
    P = torch.randn(D, d)
    # Perform QR decomposition to get tight orthonormal projection
    Q, R = torch.linalg.qr(P)
    return Q[:, :d]

def run_ablation():
    seeds = [10, 11, 12, 13, 14]
    sizes = [16, 32, 64, 128]
    
    # Structure to hold raw results: size -> method -> list of joint means
    raw_results = {
        sz: {"PCA": [], "Gaussian": []} for sz in sizes
    }
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        # Generates basic splits and trains experts
        splits_128 = generate_sandbox_data(seed, num_cal=128)
        experts, _ = train_experts(splits_128["train"], splits_128["test"])
        
        for sz in sizes:
            # Sub-sample the calibration split for size sz
            set_seed(seed)
            cal_splits_sz = []
            for k in range(K):
                X_cal, y_cal = splits_128["cal"][k]
                indices = torch.randperm(X_cal.shape[0])[:sz]
                cal_splits_sz.append((X_cal[indices], y_cal[indices]))
                
            all_cal_z_sz = torch.cat([cal_splits_sz[k][0] for k in range(K)], dim=0)
            
            # --- PCA PROJECTION ---
            P_pca = compute_pca_matrix(all_cal_z_sz, d=D_PROJ)
            anchors_pca = compute_task_anchors(cal_splits_sz, P_pca)
            
            tsar_pca = L3LinearRouter(L, K, D_PROJ, activation="identity")
            train_router(cal_splits_sz, experts, tsar_pca, P_pca, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors_pca, epochs=100, lr=1e-2)
            _, acc_pca = evaluate_merged_model(splits_128["test"], experts, tsar_pca, P_pca, "homogeneous")
            raw_results[sz]["PCA"].append(acc_pca)
            
            # --- RANDOM GAUSSIAN PROJECTION ---
            set_seed(seed) # set seed to ensure reproducibility
            P_rand = compute_random_gaussian_matrix(D, D_PROJ)
            anchors_rand = compute_task_anchors(cal_splits_sz, P_rand)
            
            tsar_rand = L3LinearRouter(L, K, D_PROJ, activation="identity")
            train_router(cal_splits_sz, experts, tsar_rand, P_rand, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors_rand, epochs=100, lr=1e-2)
            _, acc_rand = evaluate_merged_model(splits_128["test"], experts, tsar_rand, P_rand, "homogeneous")
            raw_results[sz]["Gaussian"].append(acc_rand)
            
            print(f"Size {sz} - PCA: {acc_pca*100:.2f}%, Gaussian: {acc_rand*100:.2f}%")
            
    print("\n================ EXTENDED ABLATION RESULTS ================")
    for sz in sizes:
        pca_arr = np.array(raw_results[sz]["PCA"]) * 100
        gauss_arr = np.array(raw_results[sz]["Gaussian"]) * 100
        
        print(f"B_cal = {sz}:")
        print(f"  PCA Projection:             {np.mean(pca_arr):.2f} ± {np.std(pca_arr):.2f}%")
        print(f"  Random Gaussian Projection: {np.mean(gauss_arr):.2f} ± {np.std(gauss_arr):.2f}%")

if __name__ == "__main__":
    run_ablation()
