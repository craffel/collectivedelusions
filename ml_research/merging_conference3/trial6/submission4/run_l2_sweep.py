import torch
import numpy as np
import random
from run_experiments import (
    generate_sandbox_data, train_experts, compute_pca_matrix,
    L3LinearRouter, train_router, evaluate_merged_model, set_seed, D_PROJ
)

def run_l2_sweep():
    seeds = [10, 11, 12, 13, 14]
    wd_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    sz = 64  # Calibration size B_cal = 64
    
    # Structure to hold results: lambda_wd -> list of joint means
    results = {wd: [] for wd in wd_values}
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        splits = generate_sandbox_data(seed, num_cal=64)
        experts, _ = train_experts(splits["train"], splits["test"])
        
        all_cal_z = torch.cat([splits["cal"][k][0] for k in range(4)], dim=0)
        P_pca = compute_pca_matrix(all_cal_z, d=D_PROJ)
        
        for wd in wd_values:
            set_seed(seed)
            router = L3LinearRouter(14, 4, D_PROJ, activation="identity")
            
            # Train router with the current weight decay and zero anchor regularization
            train_router(
                splits["cal"], experts, router, P_pca,
                lambda_wd=wd, lambda_anchor=0.0, anchors=None,
                epochs=100, lr=1e-2
            )
            
            _, joint_acc = evaluate_merged_model(splits["test"], experts, router, P_pca, "homogeneous")
            results[wd].append(joint_acc)
            print(f"  WD = {wd}: Joint Mean Accuracy = {joint_acc*100:.2f}%")
            
    print("\n================ L2 WEIGHT DECAY SWEEP RESULTS ================")
    for wd in wd_values:
        acc_arr = np.array(results[wd]) * 100
        print(f"lambda_wd = {wd}: {np.mean(acc_arr):.2f} ± {np.std(acc_arr):.2f}%")

if __name__ == "__main__":
    run_l2_sweep()
