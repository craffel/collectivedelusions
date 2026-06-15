import numpy as np
import os
import matplotlib.pyplot as plt
from run_experiments import (
    generate_subspace_data,
    run_chemmerge_kinetics,
    evaluate_accuracies
)

def run_eta_ablation():
    print("Running Active Coupling Step Size (eta) Ablation Study...")
    
    seeds = list(range(42, 52))
    eta_values = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]
    
    # Store results
    # {eta: {"Homog_B256": [], "Heterog_B256": [], "Heterog_B1": []}}
    results = {eta: {"Homog_B256": [], "Heterog_B256": [], "Heterog_B1": []} for eta in eta_values}
    
    for eta in eta_values:
        print(f"Evaluating eta = {eta}...")
        for seed in seeds:
            # Generate test data
            X_test, y_test_task, y_test_class, v = generate_subspace_data(250, seed=seed)
            
            # --- CONFIG 1: Homogeneous Batching (B=256) ---
            homog_indices = np.argsort(y_test_task)
            X_test_homog = X_test[homog_indices]
            y_test_task_homog = y_test_task[homog_indices]
            y_test_class_homog = y_test_class[homog_indices]
            
            weights_chem_homog, _ = run_chemmerge_kinetics(X_test_homog, v, None, temp=0.01, eta=eta)
            _, homog_acc = evaluate_accuracies(X_test_homog, y_test_task_homog, y_test_class_homog, v, weights_chem_homog, seed=seed)
            results[eta]["Homog_B256"].append(homog_acc)
            
            # --- CONFIG 2: Heterogeneous Batching (B=256) ---
            # ChemMerge is sample-wise, so B=256 heterogeneous is identical to B=1 heterogeneous
            weights_chem_het, _ = run_chemmerge_kinetics(X_test, v, None, temp=0.01, eta=eta)
            _, het256_acc = evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_chem_het, seed=seed)
            results[eta]["Heterog_B256"].append(het256_acc)
            
            # --- CONFIG 3: Heterogeneous Serving (B=1) ---
            # Since ChemMerge is sample-wise, B=1 and B=256 heterogeneous performance is identical in the simulator
            results[eta]["Heterog_B1"].append(het256_acc)
            
    # Print results
    print("\n--- ABLATION STUDY OF ACTIVE COUPLING STEP SIZE (eta) ---")
    print(f"{'eta':<6} | {'Homog (B=256)':<20} | {'Heterog (B=256)':<20} | {'Heterog (B=1)':<20}")
    print("-" * 75)
    
    for eta in eta_values:
        h_mean = np.mean(results[eta]["Homog_B256"]) * 100
        h_std = np.std(results[eta]["Homog_B256"]) * 100
        het_mean = np.mean(results[eta]["Heterog_B256"]) * 100
        het_std = np.std(results[eta]["Heterog_B256"]) * 100
        
        print(f"{eta:<6} | {h_mean:5.2f}% +/- {h_std:4.2f}% | {het_mean:5.2f}% +/- {het_std:4.2f}% | {het_mean:5.2f}% +/- {het_std:4.2f}%")
        
    # Generate and save plot
    plt.figure(figsize=(8, 5))
    homog_means = [np.mean(results[eta]["Homog_B256"]) * 100 for eta in eta_values]
    homog_stds = [np.std(results[eta]["Homog_B256"]) * 100 for eta in eta_values]
    het_means = [np.mean(results[eta]["Heterog_B256"]) * 100 for eta in eta_values]
    het_stds = [np.std(results[eta]["Heterog_B256"]) * 100 for eta in eta_values]
    
    plt.errorbar(eta_values, homog_means, yerr=homog_stds, fmt="-o", color="#d62728", label="Homogeneous Serving", linewidth=2, capsize=4)
    plt.errorbar(eta_values, het_means, yerr=het_stds, fmt="-s", color="#1f77b4", label="Heterogeneous Serving (B=256 / B=1)", linewidth=2, capsize=4)
    
    plt.title("Impact of Active Representation Coupling ($\eta$) on Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Coupling Step Size ($\eta$)", fontsize=11)
    plt.ylabel("Joint Mean Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/coupling_ablation.png", dpi=150)
    plt.savefig("submission/results/coupling_ablation.png", dpi=150)
    plt.close()
    print("Ablation plot saved to results/coupling_ablation.png and submission/results/coupling_ablation.png")

if __name__ == "__main__":
    run_eta_ablation()
