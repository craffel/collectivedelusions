import numpy as np
import os
import matplotlib.pyplot as plt
from run_experiments import (
    train_linear_router,
    get_linear_router_weights,
    get_pfsr_weights,
    get_sps_zca_weights,
    run_chemmerge_kinetics,
    evaluate_accuracies
)

def generate_non_orthogonal_data(num_samples, rho=0.0, seed=42):
    """
    Generates 192-dimensional representation vectors for K=4 tasks,
    with a controlled non-orthogonality/overlap parameter rho in [0, 1].
    rho = 0 corresponds to fully orthogonal coordinate blocks.
    """
    np.random.seed(seed)
    
    # Base orthogonal vectors
    v_orth = []
    for k in range(4):
        vk = np.zeros(192)
        vk[k*48 : (k+1)*48] = 1.0 / np.sqrt(48)
        v_orth.append(vk)
    v_orth = np.array(v_orth)
    
    # Common shared vector
    v_shared = np.zeros(192)
    v_shared[:192] = 1.0 / np.sqrt(192)
    
    # Task centroids with overlap rho
    v = []
    for k in range(4):
        # Blend orthogonal task signature with the shared vector
        vk = np.sqrt(1 - rho) * v_orth[k] + np.sqrt(rho) * v_shared
        vk = vk / np.linalg.norm(vk) # Re-normalize
        v.append(vk)
    v = np.array(v)
    
    sigmas = [0.05, 0.15, 0.40, 1.20]
    
    X = []
    y_true_task = []
    y_true_class = []
    
    for k in range(4):
        for _ in range(num_samples):
            noise = np.random.normal(0, sigmas[k], 192)
            xk = v[k] + noise
            xk = xk / np.linalg.norm(xk)
            
            X.append(xk)
            y_true_task.append(k)
            y_true_class.append(np.random.randint(0, 10))
            
    return np.array(X), np.array(y_true_task), np.array(y_true_class), v

def run_overlap_evaluation():
    print("Running Non-Orthogonal Manifold / Overlap Evaluation...")
    
    seeds = list(range(42, 52))
    rho_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    methods = ["Uniform Merging", "SABLE", "SPS-ZCA", "ChemMerge (Ours)"]
    results = {m: {rho: [] for rho in rho_values} for m in methods}
    
    # Expected similarity scaling for SPS-ZCA (re-calibrated for non-orthogonality)
    expected_sims = np.array([0.82, 0.43, 0.177, 0.06])
    
    for rho in rho_values:
        print(f"Evaluating task overlap rho = {rho}...")
        for seed in seeds:
            # Generate non-orthogonal split
            X_test, y_test_task, y_test_class, v = generate_non_orthogonal_data(250, rho=rho, seed=seed)
            
            # 1. Uniform Merging
            weights_uniform = np.ones((1000, 4)) * 0.25
            
            # 2. SABLE
            weights_sable = get_pfsr_weights(X_test, v, temp=0.05)
            
            # 3. SPS-ZCA
            weights_zca = get_sps_zca_weights(X_test, v, expected_sims + rho * (1 - expected_sims), temp=0.001)
            
            # 4. ChemMerge (Ours)
            weights_chem, _ = run_chemmerge_kinetics(X_test, v, None, temp=0.01)
            
            # Evaluate Heterogeneous B=1 serving
            _, results["Uniform Merging"][rho].append(evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_uniform, seed=seed)[1])
            _, results["SABLE"][rho].append(evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_sable, seed=seed)[1])
            _, results["SPS-ZCA"][rho].append(evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_zca, seed=seed)[1])
            _, results["ChemMerge (Ours)"][rho].append(evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_chem, seed=seed)[1])
            
    print("\n--- PERFORMANCE UNDER CONTROLLED TASK OVERLAP (RHO) ---")
    print(f"{'rho':<5} | {'Uniform':<12} | {'SABLE':<12} | {'SPS-ZCA':<12} | {'ChemMerge (Ours)':<15}")
    print("-" * 70)
    for rho in rho_values:
        u_mean = np.mean(results["Uniform Merging"][rho]) * 100
        s_mean = np.mean(results["SABLE"][rho]) * 100
        z_mean = np.mean(results["SPS-ZCA"][rho]) * 100
        c_mean = np.mean(results["ChemMerge (Ours)"][rho]) * 100
        print(f"{rho:<5} | {u_mean:5.2f}%      | {s_mean:5.2f}%      | {z_mean:5.2f}%      | {c_mean:5.2f}%")
        
    # Plot results
    plt.figure(figsize=(8, 5))
    colors = ["#7f7f7f", "#17becf", "#bcbd22", "#d62728"]
    markers = ["o", "^", "s", "D"]
    
    for m, col, mark in zip(methods, colors, markers):
        means = [np.mean(results[m][rho]) * 100 for rho in rho_values]
        stds = [np.std(results[m][rho]) * 100 for rho in rho_values]
        plt.errorbar(rho_values, means, yerr=stds, fmt=f"-{mark}", color=col, label=m, linewidth=2, capsize=4)
        
    plt.title("Robustness of Model Merging under Entangled Task Manifolds", fontsize=12, fontweight="bold")
    plt.xlabel(r"Task Overlap / Entanglement ($\rho$)", fontsize=11)
    plt.ylabel("Joint Mean Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/entangled_robustness.png", dpi=150)
    plt.savefig("submission/results/entangled_robustness.png", dpi=150)
    plt.close()
    print("Overlap robustness plot saved to results/entangled_robustness.png and submission/results/entangled_robustness.png")

if __name__ == "__main__":
    run_overlap_evaluation()
