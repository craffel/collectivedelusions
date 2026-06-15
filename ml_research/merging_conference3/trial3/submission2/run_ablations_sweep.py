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
SEEDS = list(range(42, 72))  # 30 seeds: 42 to 71 inclusive
M = 10  # Validation sample size for OFS-Tune

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
SIGMA_INV_TORCH = torch.tensor(SIGMA_INV, dtype=torch.float32)

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

# Multi-scale online TTA noise generation
def generate_tta_noise(k, seed):
    rng = np.random.default_rng(seed + k * 100)
    z = rng.normal(0, 0.12)
    eta_alt = z * ((-1.0) ** np.arange(L))
    eta_white = rng.normal(0, 0.08, size=L)
    eps = rng.normal(0, 0.08, size=L)
    eta_brown = np.cumsum(eps)
    eta = 0.5 * eta_alt + 0.3 * eta_white + 0.2 * eta_brown
    return eta

# Simulated Accuracy with Domain Diversity (Task Interference)
def get_accuracy(lambda_val, k, domain_diversity=0.0):
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
    
    # Task interference penalty proportional to domain diversity and normalized weight distance
    penalty = domain_diversity * (dist_k / dist_0)
    acc = acc - penalty
    return max(0.0, acc)

# Simulated Noisy Labeled Validation Set Loss
def get_validation_loss(lambda_val, k, seed, M, sigma_val=0.15, sigma_cov=0.1):
    rng = np.random.default_rng(seed + k * 500 + M)
    nu = rng.normal(0, sigma_val / np.sqrt(M), size=L)
    t_val = TRUE_TARGETS[k] + nu
    
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
    if search_space == "poly_1":
        l_bar = np.linspace(0, 1, L)
        for k in range(K):
            lambdas[k, :] = params[k*2] + params[k*2+1] * l_bar
    elif search_space == "layer_wise":
        lambdas = params.reshape(K, L)
    else:
        raise ValueError(f"Unknown search space {search_space}")
    return np.clip(lambdas, 0.0, 1.0)

# Offline Few-Shot Validation Tuning (OFS-Tune)
def run_ofs_tune(search_space, M, seed):
    if search_space == "poly_1":
        dim = K * 2
        bounds = []
        x0 = []
        for _ in range(K):
            bounds.extend([(0.0, 1.0), (-1.0, 1.0)])
            x0.extend([0.3, 0.0])
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
            total_loss += get_validation_loss(lambdas, k, seed, M)
        return total_loss / K

    res = minimize(
        val_objective,
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 150}
    )
    return synthesize_lambdas(res.x, search_space)

# Online TTA Loss with Custom Cosine Frequency
class TTALoss(nn.Module):
    def __init__(self, targets, sigma_inv_torch, cos_weight=0.03, freq_factor=10.0):
        super().__init__()
        self.targets = targets
        self.sigma_inv = sigma_inv_torch
        self.cos_weight = cos_weight
        self.freq_factor = freq_factor
        
    def forward(self, lambdas):
        loss = 0.0
        for k in range(K):
            e_k = lambdas[k] - self.targets[k]
            quad = 0.5 + 1.5 * torch.dot(e_k, torch.mv(self.sigma_inv, e_k))
            cos_term = self.cos_weight * torch.sum(1.0 - torch.cos(self.freq_factor * np.pi * e_k))
            loss += quad + cos_term
        return loss

# Online TTA Adaptation
def run_online_tta(method, seed, steps=100, lr=0.01, cos_weight=0.03, freq_factor=10.0):
    if method == "layer_wise":
        lambdas_t = torch.full((K, L), 0.3, requires_grad=True)
        params_to_opt = [lambdas_t]
    elif method == "poly_2":
        alphas_t = torch.zeros((K, 3), requires_grad=True)
        with torch.no_grad():
            alphas_t[:, 0] = 0.3
        params_to_opt = [alphas_t]
    else:
        raise ValueError(f"Unknown method {method}")
        
    optimizer = optim.Adam(params_to_opt, lr=lr)
    sigma_inv_torch = torch.tensor(SIGMA_INV, dtype=torch.float32)
    
    noise = np.stack([generate_tta_noise(k, seed) for k in range(K)])
    targets_np = TRUE_TARGETS + noise
    targets_t = torch.tensor(targets_np, dtype=torch.float32)
    loss_fn = TTALoss(targets_t, sigma_inv_torch, cos_weight=cos_weight, freq_factor=freq_factor)
    
    l_bar = torch.linspace(0, 1, L)
    for step in range(steps):
        optimizer.zero_grad()
        if method == "layer_wise":
            lambdas_active = lambdas_t
        else:
            lambdas_active = torch.zeros((K, L))
            for k in range(K):
                for j in range(3):
                    lambdas_active[k] += alphas_t[k, j] * (l_bar ** j)
                    
        loss = loss_fn(lambdas_active)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        if method == "layer_wise":
            final_lambdas = lambdas_t.detach().numpy()
        else:
            final_lambdas = np.zeros((K, L))
            alphas_np = alphas_t.detach().numpy()
            for k in range(K):
                for j in range(3):
                    final_lambdas[k] += alphas_np[k, j] * (np.linspace(0, 1, L) ** j)
                    
    return np.clip(final_lambdas, 0.0, 1.0)

def main():
    print("Initializing Ablations & Sensitivity Sweeps...")
    
    # Sweep 1: Domain Diversity Sweep (Task Interference Analysis)
    # Sweep domain_diversity D in [0.0, 0.05, 0.10, 0.15, 0.20]
    div_values = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    # We will evaluate across 30 seeds
    uniform_results = {d: [] for d in div_values}
    ofs_results = {d: [] for d in div_values}
    tta_results = {d: [] for d in div_values}
    
    print("\nRunning Domain Diversity Sweep over 30 seeds...")
    for seed in SEEDS:
        # Pre-run optimization once for OFS-Tune Poly-Val (d=1, M=10) on this seed
        ofs_lambdas = run_ofs_tune("poly_1", M, seed)
        # Pre-run optimization once for Online AdaMerging (Layer-wise) on this seed
        tta_lambdas = run_online_tta("layer_wise", seed)
        uniform_lambdas = np.full((K, L), 0.3)
        
        for d in div_values:
            # Uniform Accuracy
            uniform_accs = [get_accuracy(uniform_lambdas, k, domain_diversity=d) for k in range(K)]
            uniform_results[d].append(np.mean(uniform_accs))
            
            # OFS-Tune Accuracy
            ofs_accs = [get_accuracy(ofs_lambdas, k, domain_diversity=d) for k in range(K)]
            ofs_results[d].append(np.mean(ofs_accs))
            
            # TTA Accuracy
            tta_accs = [get_accuracy(tta_lambdas, k, domain_diversity=d) for k in range(K)]
            tta_results[d].append(np.mean(tta_accs))
            
    print("Domain Diversity Sweep complete!")
    
    # Sweep 2: Cosine Penalty Frequency (Entropy Roughness Analysis)
    # Sweep freq_factor F in [1.0, 2.0, 5.0, 10.0, 20.0]
    freq_values = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    # We will evaluate across 30 seeds
    tta_freq_results = {f: [] for f in freq_values}
    poly_freq_results = {f: [] for f in freq_values}
    
    print("\nRunning Cosine Frequency Sweep over 30 seeds...")
    for seed in SEEDS:
        for f in freq_values:
            # Online AdaMerging (Layer-wise)
            tta_lambdas = run_online_tta("layer_wise", seed, freq_factor=f)
            tta_accs = [get_accuracy(tta_lambdas, k, domain_diversity=0.0) for k in range(K)]
            tta_freq_results[f].append(np.mean(tta_accs))
            
            # Online PolyMerge (d=2)
            poly_lambdas = run_online_tta("poly_2", seed, freq_factor=f)
            poly_accs = [get_accuracy(poly_lambdas, k, domain_diversity=0.0) for k in range(K)]
            poly_freq_results[f].append(np.mean(poly_accs))
            
    print("Cosine Frequency Sweep complete!")
    
    # Process Statistics
    processed_div = {
        d: {
            "uniform_mean": np.mean(uniform_results[d]), "uniform_std": np.std(uniform_results[d]),
            "ofs_mean": np.mean(ofs_results[d]), "ofs_std": np.std(ofs_results[d]),
            "tta_mean": np.mean(tta_results[d]), "tta_std": np.std(tta_results[d])
        } for d in div_values
    }
    
    processed_freq = {
        f: {
            "tta_mean": np.mean(tta_freq_results[f]), "tta_std": np.std(tta_freq_results[f]),
            "poly_mean": np.mean(poly_freq_results[f]), "poly_std": np.std(poly_freq_results[f])
        } for f in freq_values
    }
    
    # Save results to file
    with open("ablations_results.json", "w") as f:
        json.dump({"domain_diversity_sweep": processed_div, "cosine_frequency_sweep": processed_freq}, f, indent=2)
    print("Saved results to ablations_results.json")
    
    # Plotting: Two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel 1: Domain Diversity Sweep
    div_array = np.array(div_values)
    ax1.errorbar(div_array * 100, [processed_div[d]["uniform_mean"]*100 for d in div_values], 
                 yerr=[processed_div[d]["uniform_std"]*100 for d in div_values], 
                 label="Uniform (Task Arithmetic)", marker="x", color="black", linestyle=":", capsize=4, lw=1.5)
    ax1.errorbar(div_array * 100, [processed_div[d]["tta_mean"]*100 for d in div_values], 
                 yerr=[processed_div[d]["tta_std"]*100 for d in div_values], 
                 label="Online AdaMerging (Layer)", marker="s", color="#d62728", capsize=4, lw=1.5)
    ax1.errorbar(div_array * 100, [processed_div[d]["ofs_mean"]*100 for d in div_values], 
                 yerr=[processed_div[d]["ofs_std"]*100 for d in div_values], 
                 label="OFS-Tune (Poly-Val $d=1$, $M=10$) [Ours]", marker="o", color="#1f77b4", capsize=4, lw=2)
    
    ax1.set_title("A: Sensitivity to Domain Diversity / Interference", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Domain Diversity / Interference Level (%)", fontsize=10)
    ax1.set_ylabel("Multi-Task Average Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9, loc="lower left")
    
    # Panel 2: Cosine Frequency Sweep
    freq_array = np.array(freq_values)
    # Add Uniform baseline as reference line (which has no cosine term)
    ax2.axhline(y=np.mean(uniform_results[0.0])*100, color="black", linestyle="--", label="Task Arithmetic Baseline", lw=1.5)
    ax2.errorbar(freq_array, [processed_freq[f]["tta_mean"]*100 for f in freq_values], 
                 yerr=[processed_freq[f]["tta_std"]*100 for f in freq_values], 
                 label="Online AdaMerging (Layer-wise)", marker="s", color="#d62728", capsize=4, lw=1.5)
    ax2.errorbar(freq_array, [processed_freq[f]["poly_mean"]*100 for f in freq_values], 
                 yerr=[processed_freq[f]["poly_std"]*100 for f in freq_values], 
                 label="Online PolyMerge ($d=2$)", marker="^", color="#2ca02c", capsize=4, lw=1.5)
    
    ax2.set_title("B: Online TTA Sensitivity to Landscape Roughness", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Cosine Entropy Roughness Frequency Factor ($F$)", fontsize=10)
    ax2.set_ylabel("Multi-Task Average Accuracy (%)", fontsize=10)
    ax2.set_xscale("log")
    ax2.set_xticks(freq_values)
    ax2.set_xticklabels([f"{f}x" for f in freq_values])
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9, loc="lower left")
    
    plt.tight_layout()
    plt.savefig("ablations_analysis.png", dpi=300)
    plt.close()
    print("Ablation plots successfully saved to ablations_analysis.png")

if __name__ == "__main__":
    main()
