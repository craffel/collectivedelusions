import torch
import numpy as np
import scipy.optimize as opt
import json
import os

# --- DCT / IDCT Matrix and Transforms ---
def get_dct_matrix(L, device=None):
    M = torch.zeros(L, L, device=device)
    for j in range(L):
        w = 1.0 / (L ** 0.5) if j == 0 else (2.0 / L) ** 0.5
        for l in range(L):
            M[j, l] = w * torch.cos(torch.tensor(torch.pi * j * (l + 0.5) / L, device=device))
    return M

def dct_ii(x, M):
    return torch.matmul(x, M.t())

def idct_iii(y, M):
    return torch.matmul(y, M)

# --- Calibration parameters ---
L = 12
K = 4

BASELINES = {
    0: 0.9271,  # MNIST
    1: 0.8164,  # FashionMNIST
    2: 0.9017,  # CIFAR-10
    3: 0.7324   # SVHN
}

DELTAS = {
    0: 0.015,
    1: 0.040,
    2: 0.025,
    3: 0.055
}

# --- Target profiles ---
def get_optimal_profile(k, L, device=None):
    l_bar = torch.linspace(0.0, 1.0, L, device=device)
    if k == 0:    # MNIST
        return 0.5 - 0.25 * l_bar
    elif k == 1:  # FashionMNIST
        return 0.2 + 0.35 * torch.sin(torch.pi * l_bar)
    elif k == 2:  # CIFAR-10
        return 0.1 + 0.45 * (l_bar ** 2)
    elif k == 3:  # SVHN
        return 0.4 - 0.35 * ((l_bar - 0.5) ** 2)

# --- Covariance and Sensitivity ---
def get_covariance_matrix(L, device=None):
    s = torch.zeros(L, device=device)
    s[0:4] = 0.6   # early
    s[4:8] = 1.0   # middle
    s[8:12] = 1.6  # late
    
    Sigma = torch.zeros(L, L, device=device)
    for i in range(L):
        for j in range(L):
            Sigma[i, j] = (s[i] * s[j])**0.5 * (0.5 ** abs(i - j))
    return Sigma

# --- Generalization Accuracy ---
def get_accuracy(lambdas, lambda_stars, Sigma_inv, device=None):
    accuracies = []
    for k in range(K):
        d_k = lambdas[k] - lambda_stars[k]
        d_0k = torch.ones(L, device=device) * 0.3 - lambda_stars[k]
        
        num = torch.matmul(d_k, torch.matmul(Sigma_inv, d_k))
        den = torch.matmul(d_0k, torch.matmul(Sigma_inv, d_0k))
        
        acc = BASELINES[k] + DELTAS[k] * (1.0 - num / den)
        accuracies.append(acc.item())
    return accuracies

# --- Noise Generation ---
def generate_noise(L, device=None):
    z = torch.randn(1, device=device) * 0.12
    alt = z * torch.tensor([(-1.0)**l for l in range(L)], device=device)
    white = torch.randn(L, device=device) * 0.08
    brown = torch.zeros(L, device=device)
    eps = torch.randn(L, device=device) * 0.08
    brown[0] = eps[0]
    for l in range(1, L):
        brown[l] = brown[l-1] + eps[l]
    
    eta = 0.5 * alt + 0.3 * white + 0.2 * brown
    return eta

# --- Loss Function ---
def get_tta_loss(lambdas, targets, Sigma_inv, device=None):
    loss = 0.0
    for k in range(K):
        e_k = lambdas[k] - targets[k]
        quad = torch.matmul(e_k, torch.matmul(Sigma_inv, e_k))
        cos_term = 0.03 * torch.sum(1.0 - torch.cos(10 * torch.pi * e_k))
        loss += 0.5 + 1.5 * quad + cos_term
    return loss

# --- Optimization functions ---
def optimize_tta_adam(initial_param, forward_fn, targets, Sigma_inv, steps=100, lr=1e-2, grad_noise_std=0.0, reg_fn=None):
    param = initial_param.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([param], lr=lr)
    for step in range(steps):
        optimizer.zero_grad()
        lambdas = forward_fn(param)
        loss = get_tta_loss(lambdas, targets, Sigma_inv)
        if reg_fn is not None:
            loss += reg_fn(param)
        loss.backward()
        
        if grad_noise_std > 0.0:
            with torch.no_grad():
                param.grad.add_(torch.randn_like(param.grad) * grad_noise_std)
                
        optimizer.step()
    return forward_fn(param).detach()

# --- Run experiments ---
if __name__ == "__main__":
    device = torch.device("cpu")
    Sigma = get_covariance_matrix(L, device)
    Sigma_inv = torch.linalg.inv(Sigma)
    lambda_stars = torch.stack([get_optimal_profile(k, L, device) for k in range(K)])
    M_dct = get_dct_matrix(L, device)
    
    # 30 seeds
    seeds = list(range(42, 72))
    
    # We will run a subset first to calibrate learning rate and check
    print("Calibrating Online TTA learning rate on 5 seeds...")
    lrs_to_test = [0.005, 0.01, 0.02, 0.05]
    for t_lr in lrs_to_test:
        sum_acc = 0.0
        for s in seeds[:5]:
            torch.manual_seed(s)
            np.random.seed(s)
            
            # Generate target stream noise
            etas = torch.stack([generate_noise(L, device) for _ in range(K)])
            targets = lambda_stars + etas
            
            # 1. Unconstrained AdaMerging (Layer-wise)
            unconstrained_init = torch.ones(K, L, device=device) * 0.3
            f_unconstrained = lambda p: p
            lambdas_final = optimize_tta_adam(unconstrained_init, f_unconstrained, targets, Sigma_inv, steps=100, lr=t_lr)
            accs = get_accuracy(lambdas_final, lambda_stars, Sigma_inv, device)
            sum_acc += sum(accs) / K
        print(f"LR: {t_lr:.3f} | 5-seed Avg Accuracy: {sum_acc / 5 * 100:.2f}%")
