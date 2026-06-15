import os
import torch
import torch.nn as nn
import numpy as np

# Set parameters
D = 192
K = 4
L = 14
d = 48
sigmas = [0.05, 0.15, 0.40, 1.20]
biases = [0.0, 0.0, -0.90, -2.30]
gamma_val = 0.05

def get_signatures(rho=0.0):
    v_orth = np.zeros((K, D))
    for k in range(K):
        v_orth[k, k*d:(k+1)*d] = 1.0 / np.sqrt(d)
        
    if rho > 0.0:
        Sigma = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                Sigma[i, j] = rho ** abs(i - j)
        U, S, Vt = np.linalg.svd(Sigma)
        Sigma_half = U @ np.diag(np.sqrt(S)) @ Vt
        
        v = np.zeros((K, D))
        for k in range(K):
            v[k] = Sigma_half @ v_orth[k]
            v[k] /= np.linalg.norm(v[k])
    else:
        v = v_orth.copy()
    return v

def generate_dataset(v, num_samples_per_task=250, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    samples = []
    labels = []
    for k in range(K):
        for _ in range(num_samples_per_task):
            eps = np.random.normal(0, sigmas[k], D)
            samples.append(v[k] + eps)
            labels.append(k)
            
    return torch.tensor(np.array(samples), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def run_sable(h0, v_tensor, biases_tensor, tau=0.05):
    h_norm = h0 / torch.norm(h0, dim=-1, keepdim=True).clamp(min=1e-8)
    v_norm = v_tensor / torch.norm(v_tensor, dim=-1, keepdim=True).clamp(min=1e-8)
    sims = h_norm @ v_norm.T
    alpha = torch.softmax(sims / tau, dim=-1)
    
    h = h0
    for l in range(4, L + 1):
        update = torch.zeros_like(h)
        for k in range(K):
            update += alpha[:, k:k+1] * gamma_val * (v_tensor[k] - h)
        h = h + update
        
    h_expanded = h.unsqueeze(1)
    v_expanded = v_tensor.unsqueeze(0)
    dists = torch.sum((h_expanded - v_expanded)**2, dim=-1)
    logits = -dists + biases_tensor
    return logits, alpha

def run_chemmerge(h0, v_tensor, biases_tensor, tau=0.01, dt=1.5, k_decay=0.3):
    C = torch.ones(h0.shape[0], K) * 0.25
    h = h0
    alphas = []
    
    for l in range(4, L + 1):
        h_norm = h / torch.norm(h, dim=-1, keepdim=True).clamp(min=1e-8)
        v_norm = v_tensor / torch.norm(v_tensor, dim=-1, keepdim=True).clamp(min=1e-8)
        sims = h_norm @ v_norm.T
        
        rates = torch.softmax(sims / tau, dim=-1)
        C_next = C + dt * (rates * (1.0 - C) - k_decay * C)
        C = torch.clamp(C_next, 0.0, 1.0)
        
        alpha = C / torch.sum(C, dim=-1, keepdim=True).clamp(min=1e-8)
        alphas.append(alpha.clone())
        
        update = torch.zeros_like(h)
        for k in range(K):
            update += alpha[:, k:k+1] * gamma_val * (v_tensor[k] - h)
        h = h + update
        
    h_expanded = h.unsqueeze(1)
    v_expanded = v_tensor.unsqueeze(0)
    dists = torch.sum((h_expanded - v_expanded)**2, dim=-1)
    logits = -dists + biases_tensor
    return logits, alphas

def run_ema_sable(h0, v_tensor, biases_tensor, tau=0.05, beta=0.7):
    alpha = torch.ones(h0.shape[0], K) * 0.25
    h = h0
    alphas = []
    
    for l in range(4, L + 1):
        h_norm = h / torch.norm(h, dim=-1, keepdim=True).clamp(min=1e-8)
        v_norm = v_tensor / torch.norm(v_tensor, dim=-1, keepdim=True).clamp(min=1e-8)
        sims = h_norm @ v_norm.T
        
        alpha_target = torch.softmax(sims / tau, dim=-1)
        alpha = beta * alpha + (1.0 - beta) * alpha_target
        alphas.append(alpha.clone())
        
        update = torch.zeros_like(h)
        for k in range(K):
            update += alpha[:, k:k+1] * gamma_val * (v_tensor[k] - h)
        h = h + update
        
    h_expanded = h.unsqueeze(1)
    v_expanded = v_tensor.unsqueeze(0)
    dists = torch.sum((h_expanded - v_expanded)**2, dim=-1)
    logits = -dists + biases_tensor
    return logits, alphas

# Test run for rho = 0.3
rho = 0.3
v = get_signatures(rho)
test_samples, test_labels = generate_dataset(v, num_samples_per_task=250, seed=42)

v_tensor = torch.tensor(v, dtype=torch.float32)
biases_tensor = torch.tensor(biases, dtype=torch.float32)

def eval_method(func, *args):
    logits, alphas = func(test_samples, v_tensor, biases_tensor, *args)
    acc = (torch.argmax(logits, dim=-1) == test_labels).float().mean().item()
    
    # Calculate jitter
    if isinstance(alphas, list):
        jitter_sum = 0.0
        for l in range(1, len(alphas)):
            diff = torch.norm(alphas[l] - alphas[l-1], p=2, dim=-1)
            jitter_sum += diff.mean().item()
        jitter = jitter_sum / (L - 4)
    else:
        jitter = 0.0
        
    return acc, jitter

acc_s, j_s = eval_method(run_sable, 0.05)
print(f"SABLE (tau=0.05): Acc={acc_s*100:.2f}%, Jitter={j_s:.4f}")

acc_c, j_c = eval_method(run_chemmerge, 0.01, 1.5, 0.3)
print(f"ChemMerge (tau=0.01, dt=1.5): Acc={acc_c*100:.2f}%, Jitter={j_c:.4f}")

for beta in [0.3, 0.5, 0.7, 0.8, 0.9]:
    acc_e, j_e = eval_method(run_ema_sable, 0.05, beta)
    print(f"EMA-SABLE (beta={beta}, tau=0.05): Acc={acc_e*100:.2f}%, Jitter={j_e:.4f}")
