import os
import json
import numpy as np
import torch

# Set up results directory
os.makedirs("results", exist_ok=True)

L = 12
datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
seeds = list(range(42, 72))
val_seeds = list(range(10, 15))

def get_optimal_profile(dataset, l):
    x = l / (L - 1)
    if dataset == 'MNIST':
        return 0.5 - 0.25 * x
    elif dataset == 'FashionMNIST':
        return 0.2 + 0.35 * np.sin(np.pi * x)
    elif dataset == 'CIFAR10':
        return 0.1 + 0.45 * (x ** 2)
    elif dataset == 'SVHN':
        return 0.4 - 0.35 * ((x - 0.5) ** 2)
    return 0.3 * np.ones_like(x)

def get_layer_covariance():
    sens = np.zeros(L)
    for l in range(L):
        if l < 4:
            sens[l] = 0.6
        elif l < 8:
            sens[l] = 1.0
        else:
            sens[l] = 1.6
            
    corr_coef = 0.5
    R = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            R[i, j] = corr_coef ** abs(i - j)
            
    D = np.diag(np.sqrt(sens))
    Sigma = D @ R @ D
    return Sigma

Sigma = get_layer_covariance()
Sigma_inv = np.linalg.inv(Sigma)

def get_overfitting_noise_complex(dataset, seed):
    np.random.seed(seed + hash(dataset) % 1000)
    hf = np.random.randn(L) * 0.10 * ((-1) ** np.arange(L))
    white = np.random.randn(L) * 0.08
    steps = np.random.randn(L) * 0.06
    brownian = np.cumsum(steps)
    brownian = brownian - np.mean(brownian)
    return hf + white + brownian

def evaluate_accuracy_coupled(lambdas, dataset):
    opt_profile = np.array([get_optimal_profile(dataset, l) for l in range(L)])
    lambdas_ta = np.ones(L) * 0.3
    
    diff_ta = lambdas_ta - opt_profile
    dist_TA = diff_ta.T @ Sigma_inv @ diff_ta
    
    diff = lambdas - opt_profile
    dist = diff.T @ Sigma_inv @ diff
    
    ta_baselines = {'MNIST': 0.9271, 'FashionMNIST': 0.8164, 'CIFAR10': 0.9017, 'SVHN': 0.7324}
    delta = {'MNIST': 0.015, 'FashionMNIST': 0.040, 'CIFAR10': 0.025, 'SVHN': 0.055}
    
    acc = ta_baselines[dataset] + delta[dataset] * (1.0 - dist / dist_TA)
    return float(np.clip(acc, 0.0, 1.0))

def optimize_adam_coupled(dataset, seed, degree=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if degree is None:
        params_raw = torch.ones(L) * 0.3
        params_raw = params_raw.detach().requires_grad_(True)
    else:
        params_raw = torch.zeros(degree + 1)
        with torch.no_grad():
            params_raw[0] = 0.3
        params_raw = params_raw.detach().requires_grad_(True)
        
    optimizer = torch.optim.Adam([params_raw], lr=1e-2)
    
    opt_profile = torch.tensor([get_optimal_profile(dataset, l) for l in range(L)], dtype=torch.float32)
    noise = torch.tensor(get_overfitting_noise_complex(dataset, seed), dtype=torch.float32)
    target = opt_profile + noise
    
    Sigma_inv_torch = torch.tensor(Sigma_inv, dtype=torch.float32)
    
    for step in range(500):
        optimizer.zero_grad()
        
        if degree is None:
            lambdas = torch.clamp(params_raw, 0.0, 1.0)
        else:
            l_idx = torch.arange(L, dtype=torch.float32) / (L - 1)
            lambdas = torch.zeros(L, dtype=torch.float32)
            for d in range(degree + 1):
                lambdas += params_raw[d] * (l_idx ** d)
            lambdas = torch.clamp(lambdas, 0.0, 1.0)
            
        diff = lambdas - target
        dist = torch.dot(diff, torch.mv(Sigma_inv_torch, diff))
        ripples = 0.03 * torch.sum(1.0 - torch.cos(10.0 * np.pi * diff))
        loss = 0.5 + 1.5 * dist + ripples
        
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        if degree is None:
            final_lambdas = torch.clamp(params_raw, 0.0, 1.0).numpy()
        else:
            l_idx = torch.arange(L, dtype=torch.float32) / (L - 1)
            final_lambdas = torch.zeros(L, dtype=torch.float32)
            for d in range(degree + 1):
                final_lambdas += params_raw[d] * (l_idx ** d)
            final_lambdas = torch.clamp(final_lambdas, 0.0, 1.0).numpy()
            
    return final_lambdas

def optimize_adam_coupled_tv(dataset, seed, beta=5.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    params_raw = torch.ones(L) * 0.3
    params_raw = params_raw.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([params_raw], lr=1e-2)
    
    opt_profile = torch.tensor([get_optimal_profile(dataset, l) for l in range(L)], dtype=torch.float32)
    noise = torch.tensor(get_overfitting_noise_complex(dataset, seed), dtype=torch.float32)
    target = opt_profile + noise
    
    Sigma_inv_torch = torch.tensor(Sigma_inv, dtype=torch.float32)
    
    for step in range(500):
        optimizer.zero_grad()
        lambdas = torch.clamp(params_raw, 0.0, 1.0)
        
        diff = lambdas - target
        dist = torch.dot(diff, torch.mv(Sigma_inv_torch, diff))
        ripples = 0.03 * torch.sum(1.0 - torch.cos(10.0 * np.pi * diff))
        loss = 0.5 + 1.5 * dist + ripples
        
        tv_penalty = torch.mean((lambdas[1:] - lambdas[:-1]) ** 2)
        loss_total = loss + beta * tv_penalty
        
        loss_total.backward()
        optimizer.step()
        
    with torch.no_grad():
        final_lambdas = torch.clamp(params_raw, 0.0, 1.0).numpy()
    return final_lambdas

# 1. Sweep beta for TV regularization
print("Sweeping beta for TV regularization on validation seeds...")
sweep_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
best_beta = 5.0
best_tv_acc = 0.0
for beta_val in sweep_grid:
    all_accs = []
    for dataset in datasets:
        for val_seed in val_seeds:
            lambdas = optimize_adam_coupled_tv(dataset, val_seed, beta=beta_val)
            all_accs.append(evaluate_accuracy_coupled(lambdas, dataset))
    mean_acc = np.mean(all_accs)
    if mean_acc > best_tv_acc:
        best_tv_acc = mean_acc
        best_beta = beta_val

print(f"Optimal beta: {best_beta:.1f}")

# 2. Run full evaluation
results_realistic = {}
methods = [
    ('task_arithmetic', lambda d, s: np.ones(L) * 0.3),
    ('unconstrained_adam', lambda d, s: optimize_adam_coupled(d, s, degree=None)),
    ('tv_reg_adam', lambda d, s: optimize_adam_coupled_tv(d, s, beta=best_beta)),
    ('poly_d0_adam', lambda d, s: optimize_adam_coupled(d, s, degree=0)),
    ('poly_d1_adam', lambda d, s: optimize_adam_coupled(d, s, degree=1)),
    ('poly_d2_adam', lambda d, s: optimize_adam_coupled(d, s, degree=2)),
    ('poly_d3_adam', lambda d, s: optimize_adam_coupled(d, s, degree=3))
]

print("Evaluating all methods under coupled non-convex landscape...")
for name, fn in methods:
    results_realistic[name] = {d: [] for d in datasets}
    for dataset in datasets:
        for seed in seeds:
            lambdas = fn(dataset, seed)
            acc = evaluate_accuracy_coupled(lambdas, dataset)
            results_realistic[name][dataset].append(acc)

# Format the metrics
metrics_json = {}
for method, task_data in results_realistic.items():
    metrics_json[method] = {}
    for dataset, accs in task_data.items():
        metrics_json[method][dataset] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs))
        }

metrics_json['averages'] = {}
for method in results_realistic.keys():
    all_accs = []
    for dataset in datasets:
        all_accs.extend(results_realistic[method][dataset])
    metrics_json['averages'][method] = {
        "mean": float(np.mean(all_accs)),
        "std": float(np.std(all_accs))
    }

# Perform t-test between poly_d2_adam and tv_reg_adam under coupled non-convex landscape
import scipy.stats as stats
poly_accs = []
tv_accs = []
for dataset in datasets:
    poly_accs.extend(results_realistic['poly_d2_adam'][dataset])
    tv_accs.extend(results_realistic['tv_reg_adam'][dataset])
t_stat, p_val = stats.ttest_rel(poly_accs, tv_accs)
metrics_json['t_test_poly_vs_tv'] = {
    "t_statistic": float(t_stat),
    "p_value": float(p_val)
}
print(f"Realistic t-test PolyMerge d=2 vs TV: t={t_stat:.6f}, p={p_val:.6e}")

# Save realistic metrics
with open("results/realistic_metrics.json", "w") as f:
    json.dump(metrics_json, f, indent=2)
print("Saved results/realistic_metrics.json")
