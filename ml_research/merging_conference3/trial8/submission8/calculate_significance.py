import numpy as np
import torch
from scipy import stats
from run_experiments import get_extracted_features, ExperimentRunner

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running statistical significance tests on device: {device}...")
    
    features = get_extracted_features(device)
    runner = ExperimentRunner(features)
    
    seeds = list(range(42, 62))
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
    sample_sizes = [8, 16, 32, 64, 128, 256]
    
    # ----------------------------------------------------
    # Experiment 1 Significance ($M=2$, $N=64$)
    # ----------------------------------------------------
    print("\n=== Experiment 1: Robustness to Covariate Shift (M=2, N=64) ===")
    print(f"{'Noise':6s} | {'SRC-DE Mean':11s} | {'Unreg Mean':11s} | {'p (vs Unreg)':12s} | {'Ridge Mean':11s} | {'p (vs Ridge)':12s} | {'Tuned Mean':11s} | {'p (vs Tuned)':12s}")
    print("-" * 115)
    
    for noise in noise_levels:
        aucs_src = []
        aucs_unreg = []
        aucs_ridge = []
        aucs_tuned = []
        for s in seeds:
            avg_aucs, _, _ = runner.run_evaluation(N_calib=64, noise_var=noise, n_components=2, seed=s)
            aucs_src.append(avg_aucs["SRC-DE"])
            aucs_unreg.append(avg_aucs["Unreg GMM"])
            aucs_ridge.append(avg_aucs["Ridge GMM"])
            aucs_tuned.append(avg_aucs["Tuned Ridge GMM"])
            
        _, p_unreg = stats.ttest_rel(aucs_src, aucs_unreg)
        _, p_ridge = stats.ttest_rel(aucs_src, aucs_ridge)
        _, p_tuned = stats.ttest_rel(aucs_src, aucs_tuned)
        
        print(f"{noise:<6.2f} | {np.mean(aucs_src):<11.4f} | {np.mean(aucs_unreg):<11.4f} | {p_unreg:<12.4e} | {np.mean(aucs_ridge):<11.4f} | {p_ridge:<12.4e} | {np.mean(aucs_tuned):<11.4f} | {p_tuned:<12.4e}")

    # ----------------------------------------------------
    # Experiment 2 Significance ($M=2$, $\sigma^2=0.05$)
    # ----------------------------------------------------
    print("\n=== Experiment 2: Sample Complexity Map (M=2, noise=0.05) ===")
    print(f"{'Size N':6s} | {'SRC-DE Mean':11s} | {'Unreg Mean':11s} | {'p (vs Unreg)':12s} | {'Ridge Mean':11s} | {'p (vs Ridge)':12s} | {'Tuned Mean':11s} | {'p (vs Tuned)':12s}")
    print("-" * 115)
    
    for N in sample_sizes:
        aucs_src = []
        aucs_unreg = []
        aucs_ridge = []
        aucs_tuned = []
        for s in seeds:
            avg_aucs, _, _ = runner.run_evaluation(N_calib=N, noise_var=0.05, n_components=2, seed=s)
            aucs_src.append(avg_aucs["SRC-DE"])
            aucs_unreg.append(avg_aucs["Unreg GMM"])
            aucs_ridge.append(avg_aucs["Ridge GMM"])
            aucs_tuned.append(avg_aucs["Tuned Ridge GMM"])
            
        _, p_unreg = stats.ttest_rel(aucs_src, aucs_unreg)
        _, p_ridge = stats.ttest_rel(aucs_src, aucs_ridge)
        _, p_tuned = stats.ttest_rel(aucs_src, aucs_tuned)
        
        print(f"{N:<6d} | {np.mean(aucs_src):<11.4f} | {np.mean(aucs_unreg):<11.4f} | {p_unreg:<12.4e} | {np.mean(aucs_ridge):<11.4f} | {p_ridge:<12.4e} | {np.mean(aucs_tuned):<11.4f} | {p_tuned:<12.4e}")

if __name__ == "__main__":
    main()
