import torch
import numpy as np
from run_experiments import model, test_x, test_task, centroids_layer3, gmm_means, gmm_vars, test_log_likelihoods, test_is_ood

print("--- GMM MEANS & VARS ---")
for k, m in gmm_means.items():
    print(f"Task {k} Mean coords: {m}")
    print(f"Task {k} Var coords: {gmm_vars[k]}")

# Let's inspect coordinates of test samples
id_coords = []
with torch.no_grad():
    for k in range(4):
        mask = (test_task == k)
        bx = test_x[mask]
        
        h = bx
        for block in model.blocks[:3]:
            W = block.W_base
            max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
            S = max_val / 7.0
            S = torch.clamp(S, min=1e-8)
            Q = torch.round(torch.clamp(W / S, -7, 7))
            W_dequant = Q * S
            h = h @ W_dequant
            
        max_h = torch.max(torch.abs(h), dim=-1, keepdim=True)[0]
        S_h = max_h / 127.0
        S_h = torch.clamp(S_h, min=1e-8)
        Q_h = torch.round(torch.clamp(h / S_h, -127, 127))
        h_q = Q_h * S_h
        
        coords = []
        for j in range(4):
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
        coords = torch.stack(coords, dim=1) # (250, 4)
        print(f"Task {k} Test Mean Coords: {coords.mean(dim=0)}")
        print(f"Task {k} Test Std Coords: {coords.std(dim=0)}")

# Let's inspect coordinates of OOD samples
ood_x = torch.randn(250, 192)
with torch.no_grad():
    h_ood = ood_x
    for block in model.blocks[:3]:
        W = block.W_base
        max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
        S = max_val / 7.0
        S = torch.clamp(S, min=1e-8)
        Q = torch.round(torch.clamp(W / S, -7, 7))
        W_dequant = Q * S
        h_ood = h_ood @ W_dequant
        
    max_h = torch.max(torch.abs(h_ood), dim=-1, keepdim=True)[0]
    S_h = max_h / 127.0
    S_h = torch.clamp(S_h, min=1e-8)
    Q_h = torch.round(torch.clamp(h_ood / S_h, -127, 127))
    h_q = Q_h * S_h
    
    coords = []
    for j in range(4):
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
    coords = torch.stack(coords, dim=1) # (250, 4)
    print(f"OOD Test Mean Coords: {coords.mean(dim=0)}")
    print(f"OOD Test Std Coords: {coords.std(dim=0)}")

id_lh = test_log_likelihoods[~test_is_ood]
ood_lh = test_log_likelihoods[test_is_ood]
print(f"ID Mean LH: {id_lh.mean().item():.3f} | Std: {id_lh.std().item():.3f}")
print(f"OOD Mean LH: {ood_lh.mean().item():.3f} | Std: {ood_lh.std().item():.3f}")
