import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_num_threads(1)
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json
import os

# Constants
L = 12  # Number of layers
K = 4   # Number of tasks
SEEDS = list(range(42, 52))  # 10 seeds: 42 to 51 inclusive
M = 10  # Validation sample size

DATASETS = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
BASELINES = np.array([0.9271, 0.8164, 0.9017, 0.7324])
DELTAS = np.array([0.015, 0.040, 0.025, 0.055])

# Layer sensitivity matrix configuration
def get_sensitivity_matrix():
    s = np.zeros(L)
    s[0:4] = 0.6   # Early layers
    s[4:8] = 1.0   # Middle layers
    s[8:12] = 1.6  # Late layers
    
    Sigma = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            Sigma[i, j] = np.sqrt(s[i] * s[j]) * (0.5 ** abs(i - j))
    
    Sigma_inv = np.linalg.inv(Sigma)
    return Sigma, Sigma_inv

SIGMA, SIGMA_INV = get_sensitivity_matrix()

# True optimal target profiles
def get_true_target(k):
    l_bar = np.linspace(0, 1, L)
    if k == 0:    # MNIST
        return 0.5 - 0.25 * l_bar
    elif k == 1:  # FashionMNIST
        return 0.2 + 0.35 * np.sin(np.pi * l_bar)
    elif k == 2:  # CIFAR-10
        return 0.1 + 0.45 * (l_bar ** 2)
    elif k == 3:  # SVHN
        return 0.4 - 0.35 * ((l_bar - 0.5) ** 2)
    else:
        raise ValueError(f"Unknown task {k}")

TRUE_TARGETS = np.stack([get_true_target(k) for k in range(K)])

# Simulated Accuracy (Generalization on True target)
def get_accuracy(lambda_val, k):
    if lambda_val.ndim == 2:
        l_k = lambda_val[k]
    else:
        l_k = lambda_val
    l_k = np.clip(l_k, 0.0, 1.0)
    
    d_k = l_k - TRUE_TARGETS[k]
    d_0 = 0.3 - TRUE_TARGETS[k]
    
    dist_k = d_k.T @ SIGMA_INV @ d_k
    dist_0 = d_0.T @ SIGMA_INV @ d_0
    
    acc = BASELINES[k] + DELTAS[k] * (1.0 - dist_k / dist_0)
    return max(0.0, acc)

# Simulated Noisy Labeled Validation Set Loss with Validation Bias (Selection Bias / Domain Shift)
def get_validation_loss(lambda_val, k, seed, M, bias_scale=0.0, bias_type="isotropic", sigma_val=0.15, sigma_cov=0.1):
    rng = np.random.default_rng(seed + k * 500 + M)
    nu = rng.normal(0, sigma_val / np.sqrt(M), size=L)
    
    rng_bias = np.random.default_rng(k * 999)
    if bias_type == "isotropic":
        v_bias = rng_bias.normal(0, bias_scale, size=L)
    elif bias_type == "structured_late":
        v_bias = np.zeros(L)
        # Apply bias only to layers 8-12
        v_bias[8:12] = rng_bias.normal(0, bias_scale * np.sqrt(L / 4.0), size=4)
    else:
        v_bias = np.zeros(L)
        
    t_val = TRUE_TARGETS[k] + nu + v_bias
    
    B = rng.normal(0, sigma_cov / np.sqrt(M), size=(L, L))
    E = B.T @ B
    Sigma_val = SIGMA + E
    Sigma_val_inv = np.linalg.inv(Sigma_val)
    
    l_k = np.clip(lambda_val[k], 0.0, 1.0) if lambda_val.ndim == 2 else np.clip(lambda_val, 0.0, 1.0)
    diff = l_k - t_val
    loss = diff.T @ Sigma_val_inv @ diff
    return loss

def synthesize_lambdas(params, search_space):
    lambdas = np.zeros((K, L))
    if search_space == "gt_merge":
        for k in range(K):
            lambdas[k, :] = params[k]
    elif search_space == "poly_1":
        l_bar = np.linspace(0, 1, L)
        for k in range(K):
            lambdas[k, :] = params[k*2] + params[k*2+1] * l_bar
    elif search_space == "poly_2":
        l_bar = np.linspace(0, 1, L)
        for k in range(K):
            lambdas[k, :] = params[k*3] + params[k*3+1] * l_bar + params[k*3+2] * (l_bar ** 2)
    elif search_space == "layer_wise":
        lambdas = params.reshape(K, L)
    else:
        raise ValueError(f"Unknown search space {search_space}")
    return np.clip(lambdas, 0.0, 1.0)

def run_ofs_tune_fast(search_space, M, seed, bias_scale=0.0, bias_type="isotropic"):
    if search_space == "gt_merge":
        dim = K
        bounds = [(0.0, 1.0)] * dim
        x0 = [0.3] * dim
    elif search_space == "poly_1":
        dim = K * 2
        bounds = []
        x0 = []
        for _ in range(K):
            bounds.extend([(0.0, 1.0), (-1.0, 1.0)])
            x0.extend([0.3, 0.0])
    elif search_space == "poly_2":
        dim = K * 3
        bounds = []
        x0 = []
        for _ in range(K):
            bounds.extend([(0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])
            x0.extend([0.3, 0.0, 0.0])
    elif search_space == "layer_wise":
        dim = K * L
        bounds = [(0.0, 1.0)] * dim
        x0 = [0.3] * dim
    else:
        raise ValueError(f"Unknown search space {search_space}")
    
    def val_objective(params):
        lambdas = synthesize_lambdas(params, search_space)
        total_loss = 0.0
        for k in range(K):
            total_loss += get_validation_loss(lambdas, k, seed, M, bias_scale=bias_scale, bias_type=bias_type)
        return total_loss / K

    # Tight maxiter limit for fast convergence
    max_iters = 60 if search_space != "layer_wise" else 30
    res = minimize(
        val_objective,
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': max_iters}
    )
    return synthesize_lambdas(res.x, search_space)

def main():
    print("Initializing Sequential Multi-Panel Validation Bias / Domain Shift Sweep...")
    
    bias_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    uniform_lambdas = np.full((K, L), 0.3)
    
    # Initialize aggregated structures
    agg_iso = {k: {b: [] for b in bias_values} for k in ["uniform", "gt", "poly1", "poly2", "layer"]}
    agg_struct = {k: {b: [] for b in bias_values} for k in ["uniform", "gt", "poly1", "poly2", "layer"]}
    
    print("\nRunning Sweeps sequentially over 10 seeds with real-time updates...")
    for seed in SEEDS:
        print(f"--- Processing Seed {seed} ---", flush=True)
        uniform_acc = np.mean([get_accuracy(uniform_lambdas, k) for k in range(K)])
        for b in bias_values:
            # Uniform baseline (unaffected by validation bias)
            agg_iso["uniform"][b].append(uniform_acc)
            agg_struct["uniform"][b].append(uniform_acc)
            
            # --- ISOTROPIC BIAS ---
            gt_l = run_ofs_tune_fast("gt_merge", M, seed, bias_scale=b, bias_type="isotropic")
            agg_iso["gt"][b].append(np.mean([get_accuracy(gt_l, k) for k in range(K)]))
            
            p1_l = run_ofs_tune_fast("poly_1", M, seed, bias_scale=b, bias_type="isotropic")
            agg_iso["poly1"][b].append(np.mean([get_accuracy(p1_l, k) for k in range(K)]))
            
            p2_l = run_ofs_tune_fast("poly_2", M, seed, bias_scale=b, bias_type="isotropic")
            agg_iso["poly2"][b].append(np.mean([get_accuracy(p2_l, k) for k in range(K)]))
            
            layer_l = run_ofs_tune_fast("layer_wise", M, seed, bias_scale=b, bias_type="isotropic")
            agg_iso["layer"][b].append(np.mean([get_accuracy(layer_l, k) for k in range(K)]))
            
            # --- STRUCTURED LATE-LAYER BIAS ---
            gt_l_s = run_ofs_tune_fast("gt_merge", M, seed, bias_scale=b, bias_type="structured_late")
            agg_struct["gt"][b].append(np.mean([get_accuracy(gt_l_s, k) for k in range(K)]))
            
            p1_l_s = run_ofs_tune_fast("poly_1", M, seed, bias_scale=b, bias_type="structured_late")
            agg_struct["poly1"][b].append(np.mean([get_accuracy(p1_l_s, k) for k in range(K)]))
            
            p2_l_s = run_ofs_tune_fast("poly_2", M, seed, bias_scale=b, bias_type="structured_late")
            agg_struct["poly2"][b].append(np.mean([get_accuracy(p2_l_s, k) for k in range(K)]))
            
            layer_l_s = run_ofs_tune_fast("layer_wise", M, seed, bias_scale=b, bias_type="structured_late")
            agg_struct["layer"][b].append(np.mean([get_accuracy(layer_l_s, k) for k in range(K)]))
            
        print(f"Seed {seed} finished successfully.", flush=True)
            
    print("All validation bias sweeps complete!")
    
    # Process Statistics
    processed_iso = {
        b: {
            "uniform_mean": np.mean(agg_iso["uniform"][b]), "uniform_std": np.std(agg_iso["uniform"][b]),
            "gt_mean": np.mean(agg_iso["gt"][b]), "gt_std": np.std(agg_iso["gt"][b]),
            "poly1_mean": np.mean(agg_iso["poly1"][b]), "poly1_std": np.std(agg_iso["poly1"][b]),
            "poly2_mean": np.mean(agg_iso["poly2"][b]), "poly2_std": np.std(agg_iso["poly2"][b]),
            "layer_mean": np.mean(agg_iso["layer"][b]), "layer_std": np.std(agg_iso["layer"][b])
        } for b in bias_values
    }
    
    processed_struct = {
        b: {
            "uniform_mean": np.mean(agg_struct["uniform"][b]), "uniform_std": np.std(agg_struct["uniform"][b]),
            "gt_mean": np.mean(agg_struct["gt"][b]), "gt_std": np.std(agg_struct["gt"][b]),
            "poly1_mean": np.mean(agg_struct["poly1"][b]), "poly1_std": np.std(agg_struct["poly1"][b]),
            "poly2_mean": np.mean(agg_struct["poly2"][b]), "poly2_std": np.std(agg_struct["poly2"][b]),
            "layer_mean": np.mean(agg_struct["layer"][b]), "layer_std": np.std(agg_struct["layer"][b])
        } for b in bias_values
    }
    
    # Save results to file
    with open("validation_bias_results.json", "w") as f:
        json.dump({"isotropic_bias_sweep": processed_iso, "structured_bias_sweep": processed_struct}, f, indent=2)
    print("Saved aggregated results to validation_bias_results.json")
    
    # Plotting: Two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bias_array = np.array(bias_values) * 100
    
    # Panel 1: Isotropic Bias Sweep
    ax1.axhline(y=np.mean(agg_iso["uniform"][0.0])*100, color="black", linestyle="--", label="Task Arithmetic Baseline", lw=1.5)
    ax1.errorbar(bias_array, [processed_iso[b]["layer_mean"]*100 for b in bias_values], 
                 yerr=[processed_iso[b]["layer_std"]*100 for b in bias_values], 
                 label="OFS-Tune (Layer-wise, 48-D)", marker="s", color="#d62728", capsize=4, lw=1.5)
    ax1.errorbar(bias_array, [processed_iso[b]["poly2_mean"]*100 for b in bias_values], 
                 yerr=[processed_iso[b]["poly2_std"]*100 for b in bias_values], 
                 label="OFS-Tune (Poly-Val $d=2$, 12-D)", marker="^", color="#2ca02c", capsize=4, lw=1.5)
    ax1.errorbar(bias_array, [processed_iso[b]["poly1_mean"]*100 for b in bias_values], 
                 yerr=[processed_iso[b]["poly1_std"]*100 for b in bias_values], 
                 label="OFS-Tune (Poly-Val $d=1$, 8-D)", marker="o", color="#1f77b4", capsize=4, lw=2)
    ax1.errorbar(bias_array, [processed_iso[b]["gt_mean"]*100 for b in bias_values], 
                 yerr=[processed_iso[b]["gt_std"]*100 for b in bias_values], 
                 label="OFS-Tune (GT-Merge, 4-D)", marker="v", color="#ff7f0e", capsize=4, lw=1.5)
                 
    ax1.set_title("A: Isotropic Gaussian Validation Target Shift", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Validation Shift / Target Bias Scale (%)", fontsize=10)
    ax1.set_ylabel("Simulated Multi-Task Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9, loc="lower left")
    
    # Panel 2: Structured Late-Layer Bias Sweep
    ax2.axhline(y=np.mean(agg_struct["uniform"][0.0])*100, color="black", linestyle="--", label="Task Arithmetic Baseline", lw=1.5)
    ax2.errorbar(bias_array, [processed_struct[b]["layer_mean"]*100 for b in bias_values], 
                 yerr=[processed_struct[b]["layer_std"]*100 for b in bias_values], 
                 label="OFS-Tune (Layer-wise, 48-D)", marker="s", color="#d62728", capsize=4, lw=1.5)
    ax2.errorbar(bias_array, [processed_struct[b]["poly2_mean"]*100 for b in bias_values], 
                 yerr=[processed_struct[b]["poly2_std"]*100 for b in bias_values], 
                 label="OFS-Tune (Poly-Val $d=2$, 12-D)", marker="^", color="#2ca02c", capsize=4, lw=1.5)
    ax2.errorbar(bias_array, [processed_struct[b]["poly1_mean"]*100 for b in bias_values], 
                 yerr=[processed_struct[b]["poly1_std"]*100 for b in bias_values], 
                 label="OFS-Tune (Poly-Val $d=1$, 8-D)", marker="o", color="#1f77b4", capsize=4, lw=2)
    ax2.errorbar(bias_array, [processed_struct[b]["gt_mean"]*100 for b in bias_values], 
                 yerr=[processed_struct[b]["gt_std"]*100 for b in bias_values], 
                 label="OFS-Tune (GT-Merge, 4-D)", marker="v", color="#ff7f0e", capsize=4, lw=1.5)
                 
    ax2.set_title("B: Structured Late-Layer (Classification Head) Bias", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Validation Shift / Target Bias Scale (%)", fontsize=10)
    ax2.set_ylabel("Simulated Multi-Task Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9, loc="lower left")
    
    plt.tight_layout()
    plt.savefig("validation_bias_robustness.png", dpi=300)
    plt.close()
    print("Multi-panel validation bias plots successfully saved to validation_bias_robustness.png")

if __name__ == "__main__":
    main()
