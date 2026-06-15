import os
import numpy as np
import matplotlib.pyplot as plt
from run_experiments import (
    generate_subspace_data,
    run_chemmerge_kinetics,
    evaluate_accuracies
)

def run_sensitivity_analysis():
    print("Starting Hyperparameter Sensitivity Analysis for ChemMerge...")
    
    seeds = list(range(42, 47)) # Use 5 seeds for efficient but statistically sound sweep
    os.makedirs("results", exist_ok=True)
    os.makedirs("submission/results", exist_ok=True)
    
    # 1. Sweep delta_t (Step Size)
    # Default k_decay = 0.3, so stability bound is delta_t < 2 / 1.3 = 1.538
    delta_t_vals = [0.1, 0.4, 0.8, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5]
    delta_t_results = {dt: [] for dt in delta_t_vals}
    
    for dt in delta_t_vals:
        for seed in seeds:
            X_test, y_test_task, y_test_class, v = generate_subspace_data(250, seed=seed)
            # Run ChemMerge kinetics with specific delta_t
            weights, _ = run_chemmerge_kinetics(X_test, v, None, temp=0.01, delta_t=dt, k_decay=0.3, eta=0.0)
            _, acc = evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights, seed=seed)
            delta_t_results[dt].append(acc * 100)
            
    # 2. Sweep k_decay (Decay Rate)
    # Default delta_t = 1.5. Bound is delta_t < 2 / (1 + k_decay) => 1 + k_decay < 2/1.5 = 1.33 => k_decay < 0.33
    # If delta_t = 1.5, then k_decay > 0.33 might cause instability! Let's check!
    k_decay_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    k_decay_results = {kd: [] for kd in k_decay_vals}
    
    for kd in k_decay_vals:
        for seed in seeds:
            X_test, y_test_task, y_test_class, v = generate_subspace_data(250, seed=seed)
            weights, _ = run_chemmerge_kinetics(X_test, v, None, temp=0.01, delta_t=1.5, k_decay=kd, eta=0.0)
            _, acc = evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights, seed=seed)
            k_decay_results[kd].append(acc * 100)
            
    # 3. Sweep temp (Reaction Temperature tau)
    # Default delta_t = 1.5, k_decay = 0.3
    temp_vals = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3]
    temp_results = {t: [] for t in temp_vals}
    
    for t in temp_vals:
        for seed in seeds:
            X_test, y_test_task, y_test_class, v = generate_subspace_data(250, seed=seed)
            weights, _ = run_chemmerge_kinetics(X_test, v, None, temp=t, delta_t=1.5, k_decay=0.3, eta=0.0)
            _, acc = evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights, seed=seed)
            temp_results[t].append(acc * 100)
            
    # --- Print Summary of Findings ---
    print("\n--- SENSITIVITY SWEEP SUMMARIES ---")
    print("\n1. Step Size delta_t (Bound limit: ~1.538):")
    for dt in delta_t_vals:
        means = np.mean(delta_t_results[dt])
        stds = np.std(delta_t_results[dt])
        print(f"  dt = {dt:3.1f} : {means:5.2f}% +/- {stds:4.2f}%")
        
    print("\n2. Decay Rate k_decay:")
    for kd in k_decay_vals:
        means = np.mean(k_decay_results[kd])
        stds = np.std(k_decay_results[kd])
        print(f"  kd = {kd:3.1f} : {means:5.2f}% +/- {stds:4.2f}%")
        
    print("\n3. Temperature tau:")
    for t in temp_vals:
        means = np.mean(temp_results[t])
        stds = np.std(temp_results[t])
        print(f"  tau = {t:5.3f} : {means:5.2f}% +/- {stds:4.2f}%")
        
    # --- Generate Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Subplot A: delta_t sensitivity
    dt_means = [np.mean(delta_t_results[dt]) for dt in delta_t_vals]
    dt_stds = [np.std(delta_t_results[dt]) for dt in delta_t_vals]
    axes[0].errorbar(delta_t_vals, dt_means, yerr=dt_stds, fmt="o-", color="#d62728", linewidth=2, capsize=4)
    axes[0].axvline(1.538, linestyle="--", color="blue", alpha=0.7, label=r"Stability Limit ($\approx 1.538$)")
    axes[0].set_title(r"Sensitivity to Step Size $\Delta t$", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(r"Virtual Step Size ($\Delta t$)", fontsize=10)
    axes[0].set_ylabel("Joint Mean Accuracy (%)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9, loc="lower left")
    
    # Subplot B: k_decay sensitivity
    kd_means = [np.mean(k_decay_results[kd]) for kd in k_decay_vals]
    kd_stds = [np.std(k_decay_results[kd]) for kd in k_decay_vals]
    axes[1].errorbar(k_decay_vals, kd_means, yerr=kd_stds, fmt="s-", color="#1f77b4", linewidth=2, capsize=4)
    # Under delta_t=1.5, stability limit for kd is kd < 2/1.5 - 1 = 0.33
    axes[1].axvline(0.333, linestyle="--", color="blue", alpha=0.7, label="Stability Limit (kd < 0.33)")
    axes[1].set_title(r"Sensitivity to Decay Rate $k_{\text{decay}}$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel(r"Decay Rate ($k_{\text{decay}}$)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9, loc="lower left")
    
    # Subplot C: temp sensitivity
    t_means = [np.mean(temp_results[t]) for t in temp_vals]
    t_stds = [np.std(temp_results[t]) for t in temp_vals]
    axes[2].errorbar(temp_vals, t_means, yerr=t_stds, fmt="^-", color="#2ca02c", linewidth=2, capsize=4)
    axes[2].set_xscale("log")
    axes[2].set_title(r"Sensitivity to Temperature $\tau$", fontsize=11, fontweight="bold")
    axes[2].set_xlabel(r"Reaction Temperature ($\tau$)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Hyperparameter Sensitivity & Discretization Stability Analysis of ChemMerge", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    plt.savefig("results/parameter_sensitivity.png", dpi=150)
    plt.savefig("submission/results/parameter_sensitivity.png", dpi=150)
    plt.close()
    print("Parameter sensitivity plots saved to results/parameter_sensitivity.png and submission/results/parameter_sensitivity.png")

if __name__ == "__main__":
    run_sensitivity_analysis()
