import os
import numpy as np
import matplotlib.pyplot as plt
from run_experiments import (
    generate_subspace_data,
    run_chemmerge_kinetics,
    run_chemmerge_kinetics_exponential,
    evaluate_accuracies
)

def run_exponential_ablation():
    print("Starting Discretization Scheme Ablation: Explicit Euler vs. Exponential Integrator...")
    
    seeds = list(range(42, 47)) # Use 5 seeds for statistical robustness
    os.makedirs("results", exist_ok=True)
    os.makedirs("submission/results", exist_ok=True)
    
    delta_t_vals = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    
    euler_results = {dt: [] for dt in delta_t_vals}
    exp_results = {dt: [] for dt in delta_t_vals}
    
    for dt in delta_t_vals:
        for seed in seeds:
            # Generate heterogeneous test data
            X_test, y_test_task, y_test_class, v = generate_subspace_data(250, seed=seed)
            
            # 1. Explicit Euler
            weights_euler, _ = run_chemmerge_kinetics(
                X_test, v, expected_sims=None, temp=0.01, delta_t=dt, k_decay=0.3, eta=0.0
            )
            _, acc_euler = evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_euler, seed=seed)
            euler_results[dt].append(acc_euler * 100)
            
            # 2. Exponential Integrator
            weights_exp, _ = run_chemmerge_kinetics_exponential(
                X_test, v, expected_sims=None, temp=0.01, delta_t=dt, k_decay=0.3, eta=0.0
            )
            _, acc_exp = evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_exp, seed=seed)
            exp_results[dt].append(acc_exp * 100)
            
    # --- Print Summary of Findings ---
    print("\n" + "="*70)
    print(f"{'Step Size (dt)':<15} | {'Explicit Euler (Clip)':<25} | {'Exponential (Exact)':<25}")
    print("="*70)
    for dt in delta_t_vals:
        euler_mean = np.mean(euler_results[dt])
        euler_std = np.std(euler_results[dt])
        exp_mean = np.mean(exp_results[dt])
        exp_std = np.std(exp_results[dt])
        print(f"{dt:<15.1f} | {euler_mean:6.2f}% +/- {euler_std:4.2f}% | {exp_mean:6.2f}% +/- {exp_std:4.2f}%")
    print("="*70)
    
    # --- Generate Plot ---
    plt.figure(figsize=(8, 5))
    
    euler_means = [np.mean(euler_results[dt]) for dt in delta_t_vals]
    euler_stds = [np.std(euler_results[dt]) for dt in delta_t_vals]
    
    exp_means = [np.mean(exp_results[dt]) for dt in delta_t_vals]
    exp_stds = [np.std(exp_results[dt]) for dt in delta_t_vals]
    
    plt.errorbar(delta_t_vals, euler_means, yerr=euler_stds, fmt="o-", color="#d62728", label="Explicit Euler (with projection)", linewidth=2, capsize=4)
    plt.errorbar(delta_t_vals, exp_means, yerr=exp_stds, fmt="s-", color="#1f77b4", label="Exponential Integrator (exact)", linewidth=2, capsize=4)
    
    # Stability boundary of unclipped Euler: dt < 1.538
    plt.axvline(1.538, linestyle="--", color="grey", alpha=0.8, label=r"Euler Stability Limit ($\approx 1.538$)")
    
    plt.title("Discretization Comparison: Explicit Euler vs. Exponential Integrator", fontsize=12, fontweight="bold")
    plt.xlabel(r"Virtual Step Size ($\Delta t$)", fontsize=11)
    plt.ylabel("Joint Mean Accuracy (%)", fontsize=11)
    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("results/exponential_vs_euler.png", dpi=150)
    plt.savefig("submission/results/exponential_vs_euler.png", dpi=150)
    plt.close()
    
    print("\nComparison plot successfully saved to results/exponential_vs_euler.png and submission/results/exponential_vs_euler.png")

if __name__ == "__main__":
    run_exponential_ablation()
