import torch
import numpy as np
import math
import sys
from test_scale_K import train_router_scaled, evaluate_model_scaled

def main():
    print("=== Running Empirical Analysis: Impact of T_cal on K=16 Expert Pool ===", flush=True)
    K_val = 16
    T_vals = [32, 128, 256]
    seeds = [101, 102, 103] # Reduced to 3 seeds for speed, but still statistically robust
    
    m1_mapping = [0] * 11
    m11_mapping = list(range(11))
    
    for T_cal in T_vals:
        print(f"\nEvaluating T_cal = {T_cal} under K = {K_val} task experts:", flush=True)
        
        m1_hom_accs, m1_het_accs = [], []
        m11_hom_accs, m11_het_accs = [], []
        
        for idx, seed in enumerate(seeds):
            print(f"  Running Seed {seed} ({idx+1}/{len(seeds)})...", end="", flush=True)
            
            # Global M=1
            r_m1, sigs_m1, ind_m1, sigmas, biases = train_router_scaled(
                K_val, 1, m1_mapping, seed, T_cal=T_cal
            )
            acc_hom, _ = evaluate_model_scaled(
                K_val, 1, r_m1, sigs_m1, ind_m1, sigmas, biases, seed, 'homogeneous'
            )
            acc_het, _ = evaluate_model_scaled(
                K_val, 1, r_m1, sigs_m1, ind_m1, sigmas, biases, seed, 'heterogeneous'
            )
            m1_hom_accs.append(acc_hom)
            m1_het_accs.append(acc_het)
            
            # LDS-Kinetics M=11
            r_m11, sigs_m11, ind_m11, _, _ = train_router_scaled(
                K_val, 11, m11_mapping, seed, T_cal=T_cal
            )
            acc_hom11, _ = evaluate_model_scaled(
                K_val, 11, r_m11, sigs_m11, ind_m11, sigmas, biases, seed, 'homogeneous'
            )
            acc_het11, _ = evaluate_model_scaled(
                K_val, 11, r_m11, sigs_m11, ind_m11, sigmas, biases, seed, 'heterogeneous'
            )
            m11_hom_accs.append(acc_hom11)
            m11_het_accs.append(acc_het11)
            
            print(" Done.", flush=True)
            
        print(f"  [T_cal={T_cal}] Global M=1:", flush=True)
        print(f"    Homo Acc:  {np.mean(m1_hom_accs):.4f}% ± {np.std(m1_hom_accs):.4f}%", flush=True)
        print(f"    Hetero Acc: {np.mean(m1_het_accs):.4f}% ± {np.std(m1_het_accs):.4f}%", flush=True)
        
        print(f"  [T_cal={T_cal}] LDS-Kinetics M=11:", flush=True)
        print(f"    Homo Acc:  {np.mean(m11_hom_accs):.4f}% ± {np.std(m11_hom_accs):.4f}%", flush=True)
        print(f"    Hetero Acc: {np.mean(m11_het_accs):.4f}% ± {np.std(m11_het_accs):.4f}%", flush=True)
        
        diff_hom = np.mean(m11_hom_accs) - np.mean(m1_hom_accs)
        diff_het = np.mean(m11_het_accs) - np.mean(m1_het_accs)
        print(f"  [DIVERGENCE GAP (M11 - M1)] Homo Gap: {diff_hom:.4f}%, Hetero Gap: {diff_het:.4f}%", flush=True)

if __name__ == "__main__":
    main()
