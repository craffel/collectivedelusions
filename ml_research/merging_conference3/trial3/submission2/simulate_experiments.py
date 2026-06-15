import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_num_threads(1)
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import json
import os
import multiprocessing as mp

# Constants
L = 12  # Number of layers
K = 4   # Number of tasks (MNIST, FashionMNIST, CIFAR-10, SVHN)
SEEDS = list(range(42, 72))  # 30 seeds: 42 to 71 inclusive

# Dataset names, baselines and sensitivities from literature
DATASETS = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
BASELINES = np.array([0.9271, 0.8164, 0.9017, 0.7324])
DELTAS = np.array([0.015, 0.040, 0.025, 0.055])

# Layer sensitivity matrix configuration
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

# True optimal target profiles
def get_true_target(k):
    l_bar = np.linspace(0, 1, L)
    if k == 0:    # MNIST: early-layer focus
        return 0.5 - 0.25 * l_bar
    elif k == 1:  # FashionMNIST: mid-layer peak
        return 0.2 + 0.35 * np.sin(np.pi * l_bar)
    elif k == 2:  # CIFAR-10: late-layer focus
        return 0.1 + 0.45 * (l_bar ** 2)
    elif k == 3:  # SVHN: mid-layer strength
        return 0.4 - 0.35 * ((l_bar - 0.5) ** 2)
    else:
        raise ValueError(f"Unknown task {k}")

TRUE_TARGETS = np.stack([get_true_target(k) for k in range(K)])

# Multi-scale online TTA noise generation
def generate_tta_noise(k, seed):
    rng = np.random.default_rng(seed + k * 100)
    
    # Alternating sign noise
    z = rng.normal(0, 0.12)
    eta_alt = z * ((-1.0) ** np.arange(L))
    
    # White noise
    eta_white = rng.normal(0, 0.08, size=L)
    
    # Brownian motion noise
    eps = rng.normal(0, 0.08, size=L)
    eta_brown = np.cumsum(eps)
    
    eta = 0.5 * eta_alt + 0.3 * eta_white + 0.2 * eta_brown
    return eta

# Simulated Accuracy (Generalization)
def get_accuracy(lambda_val, k):
    # lambda_val is a vector of size L (or K x L, in which case we index k)
    if lambda_val.ndim == 2:
        l_k = lambda_val[k]
    else:
        l_k = lambda_val
    
    # Clamp to [0, 1] to keep within valid weight range
    l_k = np.clip(l_k, 0.0, 1.0)
    
    d_k = l_k - TRUE_TARGETS[k]
    d_0 = 0.3 - TRUE_TARGETS[k]
    
    dist_k = d_k.T @ SIGMA_INV @ d_k
    dist_0 = d_0.T @ SIGMA_INV @ d_0
    
    acc = BASELINES[k] + DELTAS[k] * (1.0 - dist_k / dist_0)
    return acc

# Simulated Noisy Labeled Validation Set Loss
def get_validation_loss(lambda_val, k, seed, M, sigma_val=0.15, sigma_cov=0.1):
    # Generate static validation noise for this seed, task, and size M
    rng = np.random.default_rng(seed + k * 500 + M)
    
    # 1. Target shift
    nu = rng.normal(0, sigma_val / np.sqrt(M), size=L)
    t_val = TRUE_TARGETS[k] + nu
    
    # 2. Covariance distortion matrix E_k = B_k^T B_k
    B = rng.normal(0, sigma_cov / np.sqrt(M), size=(L, L))
    E = B.T @ B
    
    # Distorted inverse sensitivity matrix
    Sigma_val = SIGMA + E
    Sigma_val_inv = np.linalg.inv(Sigma_val)
    
    # Calculate quadratic validation loss
    l_k = np.clip(lambda_val[k], 0.0, 1.0) if lambda_val.ndim == 2 else np.clip(lambda_val, 0.0, 1.0)
    diff = l_k - t_val
    loss = diff.T @ Sigma_val_inv @ diff
    return loss

# PyTorch implementation of Online TTA Loss (Simulated Entropy Surrogate)
class TTALoss(nn.Module):
    def __init__(self, targets, sigma_inv_torch, cos_weight=0.03):
        super().__init__()
        self.targets = targets
        self.sigma_inv = sigma_inv_torch
        self.cos_weight = cos_weight
        
    def forward(self, lambdas):
        # lambdas: K x L
        loss = 0.0
        for k in range(K):
            e_k = lambdas[k] - self.targets[k]
            # Quadratic term (distance to noisy target)
            quad = 0.5 + 1.5 * torch.dot(e_k, torch.mv(self.sigma_inv, e_k))
            # Non-convex cosine terms (sharp local minima)
            cos_term = self.cos_weight * torch.sum(1.0 - torch.cos(10.0 * np.pi * e_k))
            loss += quad + cos_term
        return loss

# Synthesis function for search spaces
def synthesize_lambdas(params, search_space):
    # params: 1D array of parameters
    lambdas = np.zeros((K, L))
    if search_space == "gt_merge":
        # params size: K (4 parameters)
        for k in range(K):
            lambdas[k, :] = params[k]
    elif search_space == "poly_1":
        # params size: K * 2 (8 parameters)
        l_bar = np.linspace(0, 1, L)
        for k in range(K):
            lambdas[k, :] = params[k*2] + params[k*2+1] * l_bar
    elif search_space == "poly_2":
        # params size: K * 3 (12 parameters)
        l_bar = np.linspace(0, 1, L)
        for k in range(K):
            lambdas[k, :] = params[k*3] + params[k*3+1] * l_bar + params[k*3+2] * (l_bar ** 2)
    elif search_space == "poly_3":
        # params size: K * 4 (16 parameters)
        l_bar = np.linspace(0, 1, L)
        for k in range(K):
            lambdas[k, :] = params[k*4] + params[k*4+1] * l_bar + params[k*4+2] * (l_bar ** 2) + params[k*4+3] * (l_bar ** 3)
    elif search_space == "layer_wise":
        # params size: K * L (48 parameters)
        lambdas = params.reshape(K, L)
    else:
        raise ValueError(f"Unknown search space {search_space}")
    return np.clip(lambdas, 0.0, 1.0)

# 1. Offline Few-Shot Validation Tuning (OFS-Tune)
def run_ofs_tune(search_space, M, seed):
    # Determine bounds and parameter dimensions
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
    elif search_space == "poly_3":
        dim = K * 4
        bounds = []
        x0 = []
        for _ in range(K):
            bounds.extend([(0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])
            x0.extend([0.3, 0.0, 0.0, 0.0])
    elif search_space == "layer_wise":
        dim = K * L
        bounds = [(0.0, 1.0)] * dim
        x0 = [0.3] * dim
    else:
        raise ValueError(f"Unknown search space {search_space}")
    
    # Offline objective function: minimize mean validation loss across all tasks
    def val_objective(params):
        lambdas = synthesize_lambdas(params, search_space)
        total_loss = 0.0
        for k in range(K):
            total_loss += get_validation_loss(lambdas, k, seed, M)
        return total_loss / K

    # Run Nelder-Mead for ultra-fast and stable local search
    from scipy.optimize import minimize
    res = minimize(
        val_objective,
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 150}
    )
    
    best_lambdas = synthesize_lambdas(res.x, search_space)
    accuracies = [get_accuracy(best_lambdas, k) for k in range(K)]
    return best_lambdas, accuracies

# 1.1. OFS-Tune with PyTorch Adam (Differentiable Validation Optimization)
def run_ofs_tune_adam(search_space, M, seed, lr=0.05, steps=150):
    if search_space == "gt_merge":
        params_t = torch.full((K,), 0.3, requires_grad=True)
    elif search_space == "poly_1":
        params_t = torch.zeros((K, 2), requires_grad=True)
        with torch.no_grad():
            params_t[:, 0] = 0.3
    elif search_space == "poly_2":
        params_t = torch.zeros((K, 3), requires_grad=True)
        with torch.no_grad():
            params_t[:, 0] = 0.3
    elif search_space == "poly_3":
        params_t = torch.zeros((K, 4), requires_grad=True)
        with torch.no_grad():
            params_t[:, 0] = 0.3
    elif search_space == "layer_wise":
        params_t = torch.full((K, L), 0.3, requires_grad=True)
    else:
        raise ValueError(f"Unknown search space {search_space}")
        
    optimizer = optim.Adam([params_t], lr=lr)
    
    # Pre-compute target shift and covariance distortion tensors
    t_vals_t = []
    Sigma_val_invs_t = []
    for k in range(K):
        rng = np.random.default_rng(seed + k * 500 + M)
        nu = rng.normal(0, 0.15 / np.sqrt(M), size=L)
        t_val = TRUE_TARGETS[k] + nu
        
        B = rng.normal(0, 0.1 / np.sqrt(M), size=(L, L))
        E = B.T @ B
        Sigma_val = SIGMA + E
        Sigma_val_inv = np.linalg.inv(Sigma_val)
        
        t_vals_t.append(torch.tensor(t_val, dtype=torch.float32))
        Sigma_val_invs_t.append(torch.tensor(Sigma_val_inv, dtype=torch.float32))
        
    l_bar = torch.linspace(0, 1, L)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Synthesize lambdas to avoid in-place assignment issues
        if search_space == "gt_merge":
            lambdas_list = []
            for k in range(K):
                lambdas_list.append(params_t[k].expand(L))
            lambdas = torch.stack(lambdas_list)
        elif search_space == "poly_1":
            lambdas_list = []
            for k in range(K):
                lambdas_list.append(params_t[k, 0] + params_t[k, 1] * l_bar)
            lambdas = torch.stack(lambdas_list)
        elif search_space == "poly_2":
            lambdas_list = []
            for k in range(K):
                lambdas_list.append(params_t[k, 0] + params_t[k, 1] * l_bar + params_t[k, 2] * (l_bar ** 2))
            lambdas = torch.stack(lambdas_list)
        elif search_space == "poly_3":
            lambdas_list = []
            for k in range(K):
                lambdas_list.append(params_t[k, 0] + params_t[k, 1] * l_bar + params_t[k, 2] * (l_bar ** 2) + params_t[k, 3] * (l_bar ** 3))
            lambdas = torch.stack(lambdas_list)
        elif search_space == "layer_wise":
            lambdas = params_t
            
        lambdas_clamped = torch.clamp(lambdas, 0.0, 1.0)
        
        # Compute loss
        loss = 0.0
        for k in range(K):
            diff = lambdas_clamped[k] - t_vals_t[k]
            loss += torch.dot(diff, torch.mv(Sigma_val_invs_t[k], diff))
        loss /= K
        
        loss.backward()
        optimizer.step()
        
        # Clamp parameters to physical ranges
        with torch.no_grad():
            if search_space == "layer_wise":
                params_t.clamp_(0.0, 1.0)
            elif search_space == "gt_merge":
                params_t.clamp_(0.0, 1.0)
            else:
                params_t[:, 0].clamp_(0.0, 1.0)
                params_t[:, 1:].clamp_(-1.0, 1.0)
                
    # Final synthesized lambdas
    with torch.no_grad():
        if search_space == "gt_merge":
            lambdas_list = []
            for k in range(K):
                lambdas_list.append(params_t[k].expand(L))
            final_lambdas_t = torch.stack(lambdas_list)
        elif search_space == "poly_1":
            lambdas_list = []
            for k in range(K):
                lambdas_list.append(params_t[k, 0] + params_t[k, 1] * l_bar)
            final_lambdas_t = torch.stack(lambdas_list)
        elif search_space == "poly_2":
            lambdas_list = []
            for k in range(K):
                lambdas_list.append(params_t[k, 0] + params_t[k, 1] * l_bar + params_t[k, 2] * (l_bar ** 2))
            final_lambdas_t = torch.stack(lambdas_list)
        elif search_space == "poly_3":
            lambdas_list = []
            for k in range(K):
                lambdas_list.append(params_t[k, 0] + params_t[k, 1] * l_bar + params_t[k, 2] * (l_bar ** 2) + params_t[k, 3] * (l_bar ** 3))
            final_lambdas_t = torch.stack(lambdas_list)
        elif search_space == "layer_wise":
            final_lambdas_t = params_t
            
    final_lambdas = np.clip(final_lambdas_t.detach().numpy(), 0.0, 1.0)
    accuracies = [get_accuracy(final_lambdas, k) for k in range(K)]
    return final_lambdas, accuracies

# 1.2. OFS-Tune with Random Search (Hyperparameter Optimization Baseline)
def run_ofs_tune_random(search_space, M, seed, N_samples=100):
    if search_space == "gt_merge":
        dim = K
        bounds = [(0.0, 1.0)] * dim
    elif search_space == "poly_1":
        dim = K * 2
        bounds = []
        for _ in range(K):
            bounds.extend([(0.0, 1.0), (-1.0, 1.0)])
    elif search_space == "poly_2":
        dim = K * 3
        bounds = []
        for _ in range(K):
            bounds.extend([(0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])
    elif search_space == "poly_3":
        dim = K * 4
        bounds = []
        for _ in range(K):
            bounds.extend([(0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])
    elif search_space == "layer_wise":
        dim = K * L
        bounds = [(0.0, 1.0)] * dim
    else:
        raise ValueError(f"Unknown search space {search_space}")
        
    rng = np.random.default_rng(seed + 1000)
    best_loss = float('inf')
    best_params = None
    
    for _ in range(N_samples):
        candidate = []
        for b in bounds:
            candidate.append(rng.uniform(b[0], b[1]))
        candidate = np.array(candidate)
        
        lambdas = synthesize_lambdas(candidate, search_space)
        total_loss = 0.0
        for k in range(K):
            total_loss += get_validation_loss(lambdas, k, seed, M)
        val_loss = total_loss / K
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = candidate
            
    best_lambdas = synthesize_lambdas(best_params, search_space)
    accuracies = [get_accuracy(best_lambdas, k) for k in range(K)]
    return best_lambdas, accuracies

# 2. Online Test-Time Adaptation (TTA) Under Various Conditions
def run_online_tta(method, condition, seed, steps=100, lr=0.01, cos_weight=0.03, grad_noise_std=0.5, use_ema=False, use_lr_decay=False, noise_scale=1.0):
    # Initialize coefficients to uniform 0.3
    if method == "layer_wise" or method == "reg_cal":
        # Optimize raw lambdas directly (shape: K x L)
        lambdas_t = torch.full((K, L), 0.3, requires_grad=True)
        params_to_opt = [lambdas_t]
    elif method.startswith("poly_"):
        # Optimize polynomial parameters (shape: K x (d+1))
        d = int(method.split("_")[1])
        alphas_t = torch.zeros((K, d+1), requires_grad=True)
        with torch.no_grad():
            alphas_t[:, 0] = 0.3  # Initialize to uniform 0.3
        params_to_opt = [alphas_t]
    else:
        raise ValueError(f"Unknown method {method}")
        
    optimizer = optim.Adam(params_to_opt, lr=lr)
    
    # Optional learning rate decay scheduler
    if use_lr_decay:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    else:
        scheduler = None
        
    sigma_inv_torch = torch.tensor(SIGMA_INV, dtype=torch.float32)
    
    # Generate noisy TTA targets
    noise = np.stack([generate_tta_noise(k, seed) for k in range(K)])
    targets_np = TRUE_TARGETS + noise_scale * noise
    
    # Adversarial Condition: Extreme Label Shift (LS)
    # LS introduces a systematic bias vector b_LS
    if condition == "label_shift":
        rng = np.random.default_rng(seed + 999)
        b_LS = rng.normal(0.15, 0.03, size=(K, L))
        # Random sign for each task's bias
        signs = rng.choice([-1.0, 1.0], size=(K, 1))
        b_LS = b_LS * signs
        targets_np += b_LS
        
    targets_t = torch.tensor(targets_np, dtype=torch.float32)
    loss_fn = TTALoss(targets_t, sigma_inv_torch, cos_weight=cos_weight)
    
    # Projection matrix for Vandermonde system if polynomial
    l_bar = torch.linspace(0, 1, L)
    
    # Initialize EMA tracking if used
    if use_ema:
        # We'll accumulate the synthesized lambdas EMA
        lambdas_ema = np.full((K, L), 0.3)
        beta_ema = 0.95
    else:
        lambdas_ema = None
        
    for step in range(steps):
        optimizer.zero_grad()
        
        # Synthesize lambdas from active parameters
        if method == "layer_wise" or method == "reg_cal":
            lambdas_active = lambdas_t
        else:
            # Poly synthesis
            d = int(method.split("_")[1])
            lambdas_active = torch.zeros((K, L))
            for k in range(K):
                for j in range(d+1):
                    lambdas_active[k] += alphas_t[k, j] * (l_bar ** j)
                    
        # Apply projection constraints or regularizations for specific methods
        if method == "reg_cal":
            # Elasticsearch/TV regularization: add TV roughness penalty to loss
            tv_penalty = 0.0
            for k in range(K):
                diffs = lambdas_active[k, 1:] - lambdas_active[k, :-1]
                tv_penalty += torch.sum(diffs ** 2)
            # Add L2 penalty (distance to uniform 0.3)
            l2_penalty = torch.sum((lambdas_active - 0.3) ** 2)
            # TV weight beta=5.0, L2 weight mu=5.0
            reg = 5.0 * (tv_penalty / (L-1)) + 5.0 * (l2_penalty / L)
        else:
            reg = 0.0
            
        # Bursty Task Streams (Temporal Shift)
        # At step s, only compute loss for a single task block
        if condition == "bursty":
            loss = 0.0
            # Route tasks sequentially: steps 0-24 -> task 0, steps 25-49 -> task 1, etc.
            active_k = (step // (steps // K)) % K
            e_k = lambdas_active[active_k] - targets_t[active_k]
            quad = 0.5 + 1.5 * torch.dot(e_k, torch.mv(sigma_inv_torch, e_k))
            cos_term = cos_weight * torch.sum(1.0 - torch.cos(10.0 * np.pi * e_k))
            loss = quad + cos_term + reg
        else:
            loss = loss_fn(lambdas_active) + reg
            
        loss.backward()
        
        # Adversarial Condition: Ultra-small Batch Sizes (SB)
        # Add gradient noise to parameters before optimizer step
        current_noise_std = grad_noise_std if condition == "small_batch" else 0.0
        if current_noise_std > 0.0:
            with torch.no_grad():
                for p in params_to_opt:
                    if p.grad is not None:
                        noise_grad = torch.normal(0, current_noise_std, size=p.grad.shape)
                        p.grad.add_(noise_grad)
                        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        # Clamp parameters to keep them physically valid
        with torch.no_grad():
            if method == "layer_wise" or method == "reg_cal":
                lambdas_t.clamp_(0.0, 1.0)
            else:
                # Clamp first coeff alpha0 to [0,1], others to [-1, 1]
                alphas_t[:, 0].clamp_(0.0, 1.0)
                alphas_t[:, 1:].clamp_(-1.0, 1.0)
                
        # Accumulate to EMA
        if use_ema:
            with torch.no_grad():
                if method == "layer_wise" or method == "reg_cal":
                    curr_lambdas_np = lambdas_t.detach().numpy()
                else:
                    d = int(method.split("_")[1])
                    curr_lambdas_np = np.zeros((K, L))
                    alphas_np = alphas_t.detach().numpy()
                    l_bar_np = np.linspace(0, 1, L)
                    for k in range(K):
                        for j in range(d+1):
                            curr_lambdas_np[k] += alphas_np[k, j] * (l_bar_np ** j)
                lambdas_ema = beta_ema * lambdas_ema + (1.0 - beta_ema) * np.clip(curr_lambdas_np, 0.0, 1.0)

    if use_ema:
        final_lambdas = lambdas_ema
    else:
        # Final synthesized lambdas evaluation on true target profiles
        with torch.no_grad():
            if method == "layer_wise" or method == "reg_cal":
                final_lambdas = lambdas_t.detach().numpy()
            else:
                d = int(method.split("_")[1])
                final_lambdas = np.zeros((K, L))
                alphas_np = alphas_t.detach().numpy()
                l_bar_np = np.linspace(0, 1, L)
                for k in range(K):
                    for j in range(d+1):
                        final_lambdas[k] += alphas_np[k, j] * (l_bar_np ** j)
                    
    final_lambdas = np.clip(final_lambdas, 0.0, 1.0)
    accuracies = [get_accuracy(final_lambdas, k) for k in range(K)]
    return final_lambdas, accuracies

# Parallel Worker function for a single seed
def process_seed(seed):
    torch.set_num_threads(1)
    print(f"Starting seed {seed}...", flush=True)
    seed_results = {}
    
    # 0. Baseline: Task Arithmetic (Uniform 0.3)
    uniform_lambdas = np.full((K, L), 0.3)
    uniform_accs = [get_accuracy(uniform_lambdas, k) for k in range(K)]
    seed_results["task_arithmetic"] = uniform_accs
    
    # 1. OFS-Tune (Offline Few-Shot Validation Tuning - Nelder-Mead)
    seed_results["ofs_tune"] = {}
    for M in [5, 10, 20, 50]:
        seed_results["ofs_tune"][M] = {}
        for ss in ["gt_merge", "poly_1", "poly_2", "poly_3", "layer_wise"]:
            _, accs = run_ofs_tune(ss, M, seed)
            seed_results["ofs_tune"][M][ss] = accs

    # 1.1. OFS-Tune (Offline Few-Shot Validation Tuning - PyTorch Adam)
    seed_results["ofs_tune_adam"] = {}
    for M in [5, 10, 20, 50]:
        seed_results["ofs_tune_adam"][M] = {}
        for ss in ["gt_merge", "poly_1", "poly_2", "poly_3", "layer_wise"]:
            _, accs = run_ofs_tune_adam(ss, M, seed)
            seed_results["ofs_tune_adam"][M][ss] = accs

    # 1.2. OFS-Tune (Offline Few-Shot Validation Tuning - Random Search)
    seed_results["ofs_tune_random"] = {}
    for M in [5, 10, 20, 50]:
        seed_results["ofs_tune_random"][M] = {}
        for ss in ["gt_merge", "poly_1", "poly_2", "poly_3", "layer_wise"]:
            _, accs = run_ofs_tune_random(ss, M, seed)
            seed_results["ofs_tune_random"][M][ss] = accs
            
    # 2. Online TTA (Standard and Adversarial Conditions)
    # Sweep Methods: layer_wise, poly_0, poly_1, poly_2, poly_3, reg_cal
    # Sweep Conditions: standard, label_shift, bursty, small_batch
    seed_results["online_tta"] = {}
    for condition in ["standard", "label_shift", "bursty", "small_batch"]:
        seed_results["online_tta"][condition] = {}
        for method in ["layer_wise", "poly_0", "poly_1", "poly_2", "poly_3", "reg_cal"]:
            _, accs = run_online_tta(method, condition, seed)
            seed_results["online_tta"][condition][method] = accs
            
    # 3. Online TTA Ablations & Mitigation Evaluations
    seed_results["online_tta_ablation"] = {}
    
    # 3.1. Perfectly Benign setting: No target noise, No non-convex cosine penalty
    seed_results["online_tta_ablation"]["benign"] = {}
    for method in ["layer_wise", "poly_2", "reg_cal"]:
        _, accs = run_online_tta(method, "standard", seed, cos_weight=0.0, noise_scale=0.0)
        seed_results["online_tta_ablation"]["benign"][method] = accs
        
    # 3.2. Standard setting but with TTA mitigations (EMA coefficient smoothing + LR Cosine Decay)
    seed_results["online_tta_ablation"]["mitigated"] = {}
    for method in ["layer_wise", "poly_2", "reg_cal"]:
        _, accs = run_online_tta(method, "standard", seed, cos_weight=0.03, noise_scale=1.0, use_ema=True, use_lr_decay=True)
        seed_results["online_tta_ablation"]["mitigated"][method] = accs

    # 3.3. Gradient noise standard deviation sweep under "small_batch" condition for unconstrained layer_wise
    seed_results["online_tta_ablation"]["noise_sweep"] = {}
    for g_std in [0.0, 0.1, 0.25, 0.5]:
        _, accs = run_online_tta("layer_wise", "small_batch", seed, cos_weight=0.03, grad_noise_std=g_std)
        seed_results["online_tta_ablation"]["noise_sweep"][str(g_std)] = accs
            
    return seed, seed_results

# Main Orchestrator
if __name__ == "__main__":
    print("Initializing parallel simulation runs across 30 seeds...")
    
    # Run multiprocessing over seeds
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = mp.cpu_count()
    num_cpus = min(max(1, num_cpus), 4)  # Clamp to at most 4 to be safe and avoid over-subscription
    print(f"Using {num_cpus} CPUs...")
    with mp.Pool(processes=num_cpus) as pool:
        results_list = pool.map(process_seed, SEEDS)
        
    # Aggregate results
    all_results = {}
    for seed, res in results_list:
        all_results[seed] = res
        
    print("Simulation runs complete! Aggregating and summarizing metrics...")
    
    # Structure for summarized statistics
    summary = {
        "task_arithmetic": {},
        "ofs_tune": {},
        "ofs_tune_adam": {},
        "ofs_tune_random": {},
        "online_tta": {},
        "online_tta_ablation": {
            "benign": {},
            "mitigated": {},
            "noise_sweep": {}
        }
    }
    
    # Aggregate Task Arithmetic
    ta_accs = np.array([all_results[s]["task_arithmetic"] for s in SEEDS])
    summary["task_arithmetic"] = {
        task: {"mean": float(np.mean(ta_accs[:, k])), "std": float(np.std(ta_accs[:, k]))}
        for k, task in enumerate(DATASETS)
    }
    summary["task_arithmetic"]["Average"] = {
        "mean": float(np.mean(ta_accs)), "std": float(np.std(np.mean(ta_accs, axis=1)))
    }
    
    # Aggregate OFS-Tune results
    for M in [5, 10, 20, 50]:
        summary["ofs_tune"][M] = {}
        summary["ofs_tune_adam"][M] = {}
        summary["ofs_tune_random"][M] = {}
        for ss in ["gt_merge", "poly_1", "poly_2", "poly_3", "layer_wise"]:
            # Nelder-Mead
            accs = np.array([all_results[s]["ofs_tune"][M][ss] for s in SEEDS])
            summary["ofs_tune"][M][ss] = {
                task: {"mean": float(np.mean(accs[:, k])), "std": float(np.std(accs[:, k]))}
                for k, task in enumerate(DATASETS)
            }
            summary["ofs_tune"][M][ss]["Average"] = {
                "mean": float(np.mean(accs)), "std": float(np.std(np.mean(accs, axis=1)))
            }
            
            # PyTorch Adam
            accs_adam = np.array([all_results[s]["ofs_tune_adam"][M][ss] for s in SEEDS])
            summary["ofs_tune_adam"][M][ss] = {
                task: {"mean": float(np.mean(accs_adam[:, k])), "std": float(np.std(accs_adam[:, k]))}
                for k, task in enumerate(DATASETS)
            }
            summary["ofs_tune_adam"][M][ss]["Average"] = {
                "mean": float(np.mean(accs_adam)), "std": float(np.std(np.mean(accs_adam, axis=1)))
            }
            
            # Random Search
            accs_random = np.array([all_results[s]["ofs_tune_random"][M][ss] for s in SEEDS])
            summary["ofs_tune_random"][M][ss] = {
                task: {"mean": float(np.mean(accs_random[:, k])), "std": float(np.std(accs_random[:, k]))}
                for k, task in enumerate(DATASETS)
            }
            summary["ofs_tune_random"][M][ss]["Average"] = {
                "mean": float(np.mean(accs_random)), "std": float(np.std(np.mean(accs_random, axis=1)))
            }
            
    # Aggregate Online TTA results
    for condition in ["standard", "label_shift", "bursty", "small_batch"]:
        summary["online_tta"][condition] = {}
        for method in ["layer_wise", "poly_0", "poly_1", "poly_2", "poly_3", "reg_cal"]:
            accs = np.array([all_results[s]["online_tta"][condition][method] for s in SEEDS])
            summary["online_tta"][condition][method] = {
                task: {"mean": float(np.mean(accs[:, k])), "std": float(np.std(accs[:, k]))}
                for k, task in enumerate(DATASETS)
            }
            summary["online_tta"][condition][method]["Average"] = {
                "mean": float(np.mean(accs)), "std": float(np.std(np.mean(accs, axis=1)))
            }
            
    # Aggregate Online TTA Ablations & Mitigations
    # Benign
    for method in ["layer_wise", "poly_2", "reg_cal"]:
        accs = np.array([all_results[s]["online_tta_ablation"]["benign"][method] for s in SEEDS])
        summary["online_tta_ablation"]["benign"][method] = {
            task: {"mean": float(np.mean(accs[:, k])), "std": float(np.std(accs[:, k]))}
            for k, task in enumerate(DATASETS)
        }
        summary["online_tta_ablation"]["benign"][method]["Average"] = {
            "mean": float(np.mean(accs)), "std": float(np.std(np.mean(accs, axis=1)))
        }
        
    # Mitigated
    for method in ["layer_wise", "poly_2", "reg_cal"]:
        accs = np.array([all_results[s]["online_tta_ablation"]["mitigated"][method] for s in SEEDS])
        summary["online_tta_ablation"]["mitigated"][method] = {
            task: {"mean": float(np.mean(accs[:, k])), "std": float(np.std(accs[:, k]))}
            for k, task in enumerate(DATASETS)
        }
        summary["online_tta_ablation"]["mitigated"][method]["Average"] = {
            "mean": float(np.mean(accs)), "std": float(np.std(np.mean(accs, axis=1)))
        }
        
    # Noise Sweep
    for g_std in [0.0, 0.1, 0.25, 0.5]:
        accs = np.array([all_results[s]["online_tta_ablation"]["noise_sweep"][str(g_std)] for s in SEEDS])
        summary["online_tta_ablation"]["noise_sweep"][str(g_std)] = {
            task: {"mean": float(np.mean(accs[:, k])), "std": float(np.std(accs[:, k]))}
            for k, task in enumerate(DATASETS)
        }
        summary["online_tta_ablation"]["noise_sweep"][str(g_std)]["Average"] = {
            "mean": float(np.mean(accs)), "std": float(np.std(np.mean(accs, axis=1)))
        }
            
    # Save raw aggregated results to file
    with open("all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    # Save summarized statistics to file
    with open("summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Files 'all_results.json' and 'summary_metrics.json' written successfully!")
    
    # Generate Plots
    print("Generating figures and plots...")
    
    # Plot 1: OFS-Tune Sample Complexity (M) vs. Average Accuracy
    plt.figure(figsize=(8, 5))
    search_spaces_plot = ["gt_merge", "poly_1", "poly_2", "poly_3", "layer_wise"]
    labels_plot = ["GT-Merge (d=0)", "Poly-Val (d=1)", "Poly-Val (d=2)", "Poly-Val (d=3)", "Layer-wise Search"]
    markers = ["o", "s", "^", "D", "x"]
    M_values = [5, 10, 20, 50]
    
    for ss, label, mkr in zip(search_spaces_plot, labels_plot, markers):
        means = [summary["ofs_tune"][M][ss]["Average"]["mean"] * 100 for M in M_values]
        stds = [summary["ofs_tune"][M][ss]["Average"]["std"] * 100 for M in M_values]
        plt.errorbar(M_values, means, yerr=stds, label=label, marker=mkr, capsize=5, lw=1.5)
        
    # Add Task Arithmetic as reference line
    ta_mean = summary["task_arithmetic"]["Average"]["mean"] * 100
    plt.axhline(y=ta_mean, color="black", linestyle="--", label="Task Arithmetic Baseline", lw=1.5)
    
    plt.title("Offline Few-Shot Validation Tuning (OFS-Tune): Sample Complexity vs. Accuracy", fontsize=11, fontweight="bold")
    plt.xlabel("Validation Samples per Task (M)", fontsize=10)
    plt.ylabel("Multi-Task Average Accuracy (%)", fontsize=10)
    plt.xscale("log")
    plt.xticks(M_values, M_values)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig("ofs_tune_sample_complexity.png", dpi=150)
    plt.close()
    
    # Plot 2: Robustness Comparison Under Adversarial Stream Conditions
    plt.figure(figsize=(10, 5.5))
    conditions_plot = ["standard", "label_shift", "bursty", "small_batch"]
    cond_labels = ["Standard Stream", "Extreme Label Shift", "Bursty Task Stream", "Small Batch Size (Noise)"]
    
    # Choose representative methods
    methods_plot = ["layer_wise", "poly_2", "reg_cal"]
    method_labels = ["Online AdaMerging (Layer)", "Online PolyMerge (d=2)", "Online RegCalMerge"]
    
    x = np.arange(len(conditions_plot))
    width = 0.2
    
    # Plot OFS-Tune (d=1) with M=10 as a robust reference
    ofs_accs = [summary["ofs_tune"][10]["poly_1"]["Average"]["mean"] * 100] * len(conditions_plot)
    plt.bar(x - 1.5 * width, [summary["online_tta"][c]["layer_wise"]["Average"]["mean"] * 100 for c in conditions_plot], width, label="Online AdaMerging (Layer)", color="#d62728", hatch="//")
    plt.bar(x - 0.5 * width, [summary["online_tta"][c]["reg_cal"]["Average"]["mean"] * 100 for c in conditions_plot], width, label="Online RegCalMerge", color="#ff7f0e", hatch="\\")
    plt.bar(x + 0.5 * width, [summary["online_tta"][c]["poly_2"]["Average"]["mean"] * 100 for c in conditions_plot], width, label="Online PolyMerge (d=2)", color="#2ca02c", hatch=".")
    plt.bar(x + 1.5 * width, [summary["ofs_tune"][10]["poly_1"]["Average"]["mean"] * 100] * len(conditions_plot), width, label="OFS-Tune (d=1, M=10) [Ours]", color="#1f77b4")
    
    # Reference baseline
    plt.axhline(y=ta_mean, color="black", linestyle="--", label="Task Arithmetic Baseline", lw=1.2)
    
    plt.title("Robustness Comparison under Adversarial Stream Conditions", fontsize=11, fontweight="bold")
    plt.xticks(x, cond_labels, fontsize=9)
    plt.ylabel("Multi-Task Average Accuracy (%)", fontsize=10)
    plt.ylim(ta_mean - 10.0, ofs_accs[0] + 2.0)
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig("robustness_stress_test.png", dpi=150)
    plt.close()
    
    print("Figures saved: 'ofs_tune_sample_complexity.png' and 'robustness_stress_test.png'.")
    print("Simulation orchestrator finished successfully!")
