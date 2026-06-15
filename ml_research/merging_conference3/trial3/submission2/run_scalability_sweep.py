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
K_LIST = [4, 8, 16, 32, 64]
SEEDS = list(range(42, 47))  # 5 seeds for speed: 42 to 46 inclusive
M = 10  # validation sample size

# Base layer sensitivity matrix
def get_sensitivity_matrix():
    s = np.zeros(L)
    s[0:4] = 0.6   # Early layers: low sensitivity
    s[4:8] = 1.0   # Middle layers: moderate sensitivity
    s[8:12] = 1.6  # Late layers: high sensitivity
    
    Sigma = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            Sigma[i, j] = np.sqrt(s[i] * s[j]) * (0.5 ** abs(i - j))
    
    Sigma_inv = np.linalg.inv(Sigma)
    return Sigma, Sigma_inv

SIGMA, SIGMA_INV = get_sensitivity_matrix()
SIGMA_INV_TORCH = torch.tensor(SIGMA_INV, dtype=torch.float32)

# Procedural generation of true targets, baselines, and deltas
def get_task_parameters(k):
    # Standard 4 tasks
    if k == 0:
        baseline = 0.9271
        delta = 0.015
        l_bar = np.linspace(0, 1, L)
        target = 0.5 - 0.25 * l_bar
    elif k == 1:
        baseline = 0.8164
        delta = 0.040
        l_bar = np.linspace(0, 1, L)
        target = 0.2 + 0.35 * np.sin(np.pi * l_bar)
    elif k == 2:
        baseline = 0.9017
        delta = 0.025
        l_bar = np.linspace(0, 1, L)
        target = 0.1 + 0.45 * (l_bar ** 2)
    elif k == 3:
        baseline = 0.7324
        delta = 0.055
        l_bar = np.linspace(0, 1, L)
        target = 0.4 - 0.35 * ((l_bar - 0.5) ** 2)
    else:
        # Procedurally generated task params based on seed k
        rng = np.random.default_rng(k * 1000)
        baseline = rng.uniform(0.70, 0.95)
        delta = rng.uniform(0.01, 0.06)
        
        l_bar = np.linspace(0, 1, L)
        freq = rng.choice([1, 2, 3])
        phase = rng.uniform(0, np.pi)
        amp = rng.uniform(0.15, 0.45)
        bias = rng.uniform(0.20, 0.50)
        target = bias + amp * np.sin(freq * np.pi * l_bar + phase)
        target = np.clip(target, 0.05, 0.95)
        
    return baseline, delta, target

def get_accuracy(lambda_val, k, true_targets, baselines, deltas):
    l_k = np.clip(lambda_val[k], 0.0, 1.0)
    d_k = l_k - true_targets[k]
    d_0 = 0.3 - true_targets[k]
    
    dist_k = d_k.T @ SIGMA_INV @ d_k
    dist_0 = d_0.T @ SIGMA_INV @ d_0
    
    acc = baselines[k] + deltas[k] * (1.0 - dist_k / dist_0)
    return acc

def get_validation_loss(lambda_val, k, seed, true_targets):
    rng = np.random.default_rng(seed + k * 500 + M)
    # Target shift
    nu = rng.normal(0, 0.15 / np.sqrt(M), size=L)
    t_val = true_targets[k] + nu
    
    # Covariance distortion
    B = rng.normal(0, 0.1 / np.sqrt(M), size=(L, L))
    E = B.T @ B
    Sigma_val = SIGMA + E
    Sigma_val_inv = np.linalg.inv(Sigma_val)
    
    l_k = np.clip(lambda_val[k], 0.0, 1.0)
    diff = l_k - t_val
    loss = diff.T @ Sigma_val_inv @ diff
    return loss

def synthesize_lambdas(params, search_space, K):
    lambdas = np.zeros((K, L))
    if search_space == "poly_2":
        # params size: K * 3
        l_bar = np.linspace(0, 1, L)
        for k in range(K):
            lambdas[k, :] = params[k*3] + params[k*3+1] * l_bar + params[k*3+2] * (l_bar ** 2)
    elif search_space == "layer_wise":
        # params size: K * L
        lambdas = params.reshape(K, L)
    else:
        raise ValueError(f"Unknown search space {search_space}")
    return np.clip(lambdas, 0.0, 1.0)

# Optimization routines
def run_ofs_tune_nm(search_space, seed, K, true_targets, baselines, deltas):
    if search_space == "poly_2":
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
        lambdas = synthesize_lambdas(params, search_space, K)
        total_loss = 0.0
        for k in range(K):
            total_loss += get_validation_loss(lambdas, k, seed, true_targets)
        return total_loss / K

    res = minimize(
        val_objective,
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 150}
    )
    
    best_lambdas = synthesize_lambdas(res.x, search_space, K)
    accuracies = [get_accuracy(best_lambdas, k, true_targets, baselines, deltas) for k in range(K)]
    return res.fun, np.mean(accuracies)

def run_ofs_tune_adam(search_space, seed, K, true_targets, baselines, deltas, lr=0.05, steps=150):
    if search_space == "poly_2":
        params_t = torch.zeros((K, 3), requires_grad=True)
        with torch.no_grad():
            params_t[:, 0] = 0.3
    elif search_space == "layer_wise":
        params_t = torch.full((K, L), 0.3, requires_grad=True)
    else:
        raise ValueError(f"Unknown search space {search_space}")
        
    optimizer = optim.Adam([params_t], lr=lr)
    
    t_vals_t = []
    Sigma_val_invs_t = []
    for k in range(K):
        rng = np.random.default_rng(seed + k * 500 + M)
        nu = rng.normal(0, 0.15 / np.sqrt(M), size=L)
        t_val = true_targets[k] + nu
        
        B = rng.normal(0, 0.1 / np.sqrt(M), size=(L, L))
        E = B.T @ B
        Sigma_val = SIGMA + E
        Sigma_val_inv = np.linalg.inv(Sigma_val)
        
        t_vals_t.append(torch.tensor(t_val, dtype=torch.float32))
        Sigma_val_invs_t.append(torch.tensor(Sigma_val_inv, dtype=torch.float32))
        
    l_bar = torch.linspace(0, 1, L)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        if search_space == "poly_2":
            lambdas_list = []
            for k in range(K):
                lambdas_list.append(params_t[k, 0] + params_t[k, 1] * l_bar + params_t[k, 2] * (l_bar ** 2))
            lambdas = torch.stack(lambdas_list)
        elif search_space == "layer_wise":
            lambdas = params_t
            
        lambdas_clamped = torch.clamp(lambdas, 0.0, 1.0)
        
        loss = 0.0
        for k in range(K):
            diff = lambdas_clamped[k] - t_vals_t[k]
            loss += torch.dot(diff, torch.mv(Sigma_val_invs_t[k], diff))
        loss /= K
        
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        if search_space == "poly_2":
            best_lambdas = torch.zeros((K, L))
            for k in range(K):
                best_lambdas[k, :] = params_t[k, 0] + params_t[k, 1] * l_bar + params_t[k, 2] * (l_bar ** 2)
        elif search_space == "layer_wise":
            best_lambdas = params_t
            
        best_lambdas = torch.clamp(best_lambdas, 0.0, 1.0).numpy()
        
    accuracies = [get_accuracy(best_lambdas, k, true_targets, baselines, deltas) for k in range(K)]
    final_loss = loss.item()
    return final_loss, np.mean(accuracies)

def main():
    print("Starting Task Scalability Sweep...")
    
    results = {}
    for K in K_LIST:
        print(f"\n--- Running Sweep for K = {K} Tasks ---")
        
        # Build task parameters for this K
        baselines = []
        deltas = []
        true_targets = []
        for k in range(K):
            base, d, tgt = get_task_parameters(k)
            baselines.append(base)
            deltas.append(d)
            true_targets.append(tgt)
            
        baselines = np.array(baselines)
        deltas = np.array(deltas)
        true_targets = np.stack(true_targets)
        
        # Baselines
        uniform_lambdas = np.full((K, L), 0.3)
        uniform_accs = [get_accuracy(uniform_lambdas, k, true_targets, baselines, deltas) for k in range(K)]
        mean_uniform_acc = np.mean(uniform_accs)
        
        print(f"Uniform Baseline Accuracy: {mean_uniform_acc:.4f}")
        
        results[K] = {
            "uniform_acc": mean_uniform_acc,
            "nm_poly2_loss": [], "nm_poly2_acc": [],
            "nm_layerwise_loss": [], "nm_layerwise_acc": [],
            "adam_poly2_loss": [], "adam_poly2_acc": [],
            "adam_layerwise_loss": [], "adam_layerwise_acc": []
        }
        
        for seed in SEEDS:
            # Nelder-Mead on Poly-Val (d=2)
            nm_p2_loss, nm_p2_acc = run_ofs_tune_nm("poly_2", seed, K, true_targets, baselines, deltas)
            results[K]["nm_poly2_loss"].append(nm_p2_loss)
            results[K]["nm_poly2_acc"].append(nm_p2_acc)
            
            # Nelder-Mead on Layer-wise
            nm_lw_loss, nm_lw_acc = run_ofs_tune_nm("layer_wise", seed, K, true_targets, baselines, deltas)
            results[K]["nm_layerwise_loss"].append(nm_lw_loss)
            results[K]["nm_layerwise_acc"].append(nm_lw_acc)
            
            # Adam on Poly-Val (d=2)
            adam_p2_loss, adam_p2_acc = run_ofs_tune_adam("poly_2", seed, K, true_targets, baselines, deltas)
            results[K]["adam_poly2_loss"].append(adam_p2_loss)
            results[K]["adam_poly2_acc"].append(adam_p2_acc)
            
            # Adam on Layer-wise
            adam_lw_loss, adam_lw_acc = run_ofs_tune_adam("layer_wise", seed, K, true_targets, baselines, deltas)
            results[K]["adam_layerwise_loss"].append(adam_lw_loss)
            results[K]["adam_layerwise_acc"].append(adam_lw_acc)
            
        print(f"Nelder-Mead Poly-Val d=2 Acc: {np.mean(results[K]['nm_poly2_acc']):.4f} | Loss: {np.mean(results[K]['nm_poly2_loss']):.4f}")
        print(f"Nelder-Mead Layer-wise Acc:  {np.mean(results[K]['nm_layerwise_acc']):.4f} | Loss: {np.mean(results[K]['nm_layerwise_loss']):.4f}")
        print(f"Adam Poly-Val d=2 Acc:        {np.mean(results[K]['adam_poly2_acc']):.4f} | Loss: {np.mean(results[K]['adam_poly2_loss']):.4f}")
        print(f"Adam Layer-wise Acc:         {np.mean(results[K]['adam_layerwise_acc']):.4f} | Loss: {np.mean(results[K]['adam_layerwise_loss']):.4f}")

    # Process final statistics
    processed_results = {}
    for K, data in results.items():
        processed_results[K] = {
            "uniform_acc": data["uniform_acc"],
            "nm_poly2_acc_mean": np.mean(data["nm_poly2_acc"]),
            "nm_poly2_acc_std": np.std(data["nm_poly2_acc"]),
            "nm_poly2_loss_mean": np.mean(data["nm_poly2_loss"]),
            "nm_layerwise_acc_mean": np.mean(data["nm_layerwise_acc"]),
            "nm_layerwise_acc_std": np.std(data["nm_layerwise_acc"]),
            "nm_layerwise_loss_mean": np.mean(data["nm_layerwise_loss"]),
            "adam_poly2_acc_mean": np.mean(data["adam_poly2_acc"]),
            "adam_poly2_acc_std": np.std(data["adam_poly2_acc"]),
            "adam_poly2_loss_mean": np.mean(data["adam_poly2_loss"]),
            "adam_layerwise_acc_mean": np.mean(data["adam_layerwise_acc"]),
            "adam_layerwise_acc_std": np.std(data["adam_layerwise_acc"]),
            "adam_layerwise_loss_mean": np.mean(data["adam_layerwise_loss"])
        }
        
    with open("scalability_results.json", "w") as f:
        json.dump(processed_results, f, indent=2)
        
    print("\nResults saved to scalability_results.json.")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Test Accuracy
    plt.plot(K_LIST, [processed_results[k]["nm_poly2_acc_mean"]*100 for k in K_LIST], 'o-', label="OFS-Tune NM (Poly-Val $d=2$)", color="darkorange", linewidth=2)
    plt.plot(K_LIST, [processed_results[k]["nm_layerwise_acc_mean"]*100 for k in K_LIST], 's-', label="OFS-Tune NM (Layer-wise)", color="chocolate", linestyle="--", linewidth=1.5)
    plt.plot(K_LIST, [processed_results[k]["adam_poly2_acc_mean"]*100 for k in K_LIST], 'd-', label="OFS-Tune Adam (Poly-Val $d=2$)", color="royalblue", linewidth=2)
    plt.plot(K_LIST, [processed_results[k]["adam_layerwise_acc_mean"]*100 for k in K_LIST], 'x-', label="OFS-Tune Adam (Layer-wise)", color="forestgreen", linestyle="--", linewidth=1.5)
    plt.plot(K_LIST, [processed_results[k]["uniform_acc"]*100 for k in K_LIST], 'k:', label="Uniform Baseline", linewidth=2)
    
    plt.title("OFS-Tune Task Scalability Analysis ($K \in \{4, 8, 16, 32, 64\}$)", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Merged Tasks ($K$)", fontsize=12)
    plt.ylabel("Multi-Task Average Simulated Accuracy (%)", fontsize=12)
    plt.xscale("log")
    plt.xticks(K_LIST, [str(k) for k in K_LIST])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower left")
    
    plt.tight_layout()
    plt.savefig("scalability_comparison.png", dpi=300)
    print("Plot saved to scalability_comparison.png.")

if __name__ == "__main__":
    main()
