import torch
import numpy as np
from run_experiments import model, test_x, test_task, centroids_layer3, gmm_means, gmm_vars

model.eval()
with torch.no_grad():
    for k in range(4):
        mask = (test_task == k)
        bx = test_x[mask][:5] # first 5 samples
        
        # Run up to Layer 3
        h = bx
        for block in model.blocks[:3]:
            W = block.W_base
            max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
            S = max_val / 7.0
            S = torch.clamp(S, min=1e-8)
            Q = torch.round(torch.clamp(W / S, -7, 7))
            W_dequant = Q * S
            h = h @ W_dequant
            
        # Quantize h
        max_h = torch.max(torch.abs(h), dim=-1, keepdim=True)[0]
        S_h = max_h / 127.0
        S_h = torch.clamp(S_h, min=1e-8)
        Q_h = torch.round(torch.clamp(h / S_h, -127, 127))
        h_q = Q_h * S_h
        
        # Compute coordinates
        coords = []
        for j in range(3):
            mu = centroids_layer3[j]
            max_mu = torch.max(torch.abs(mu))
            S_mu = max_mu / 127.0
            S_mu = torch.clamp(S_mu, min=1e-8)
            Q_mu = torch.round(torch.clamp(mu / S_mu, -127, 127))
            mu_q = Q_mu * S_mu
            
            dot_product = torch.sum(h_q * mu_q, dim=-1)
            norm_h = torch.norm(h_q, p=2, dim=-1)
            norm_mu = torch.norm(mu_q, p=2)
            sim = dot_product / (norm_h * norm_mu + 1e-8)
            coords.append(sim)
        coords = torch.stack(coords, dim=1) # (5, 3)
        
        print(f"\nTask {k} (SVHN is OOD=3) - Coordinates:")
        print(coords)
        
        # Compute GMM log-likelihood
        log_probs = []
        for i in range(3):
            m = gmm_means[i]
            v = gmm_vars[i]
            diff = coords - m
            log_density = -0.5 * torch.sum(torch.log(2 * np.pi * v) + (diff ** 2) / v, dim=-1)
            log_probs.append(log_density)
        log_probs = torch.stack(log_probs, dim=1)
        log_lh = torch.logsumexp(log_probs, dim=1) - np.log(3)
        print(f"Task {k} Log-Likelihoods:")
        print(log_lh)
